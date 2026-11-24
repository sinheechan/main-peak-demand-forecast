# 관제 대시보드 (노트북 한 화면)
#   상단: 제목, 현장, 기준 시각
#   1행: 현재 전력 | 15분 예상 수요(계산식 + ML) | ESS | 현장 연결
#   2행: 15분 수요 막대(최근 3시간, 예측 점) | 전력 추이(최근 30분)
#   3행: 제어 이력 | 알림 | 오늘 ML 성적표
#   - 색, 여백은 .streamlit/config.toml 테마에서 설정 (CSS 없음)
#   - ml.* 테이블은 serving/serve.py가 생성, 없으면 ML 영역만 "대기" 표시

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

SITE = "plant1"
REFRESH = 5          # 화면 갱신 주기 (초)
OFFLINE_SEC = 30     # 연결 끊김 기준 (초)
GAP_SEC = 3          # 빈 구간 기준 (초)
WARN = 0.9           # 경고 기준 (목표 대비)
SOC_MIN = 20
V_MIN = 342
V_MAX = 418
HZ_MIN = 59.8
HZ_MAX = 60.2
PF_MIN = 0.90
ML_OLD_SEC = 90      # ML 예측 지연 한계 (초과 시 serve.py 중단으로 간주)

CHART_H = 230        # 차트 높이
CHART_BOX_H = 310    # 차트 칸 높이 (제목 + 차트)
LOG_BOX_H = 270      # 기록 칸 높이
ROWS = 6             # 기록 표 행 수

# 차트 색 (config.toml 다크 팔레트와 같은 값)
BLUE = "#60A5FA"
RED = "#F87171"
GRAY = "#94A3B8"
VIOLET = "#A78BFA"   # ML

ACTIONS = {"shed": "부하 차단", "restore": "부하 복귀", "ess_on": "ESS 방전", "ess_off": "ESS 정지"}
KINDS = {
    "no_data": "수신 끊김", "data_resumed": "수신 재개", "edge_offline": "현장 오프라인",
    "delay": "수신 지연", "duplicate": "중복 수신", "kwh_backward": "전력량 역행",
    "voltage": "전압 이탈", "frequency": "주파수 이탈", "power_factor": "역률 저하",
    "peak_exceeded": "피크 초과",
}

st.set_page_config(page_title="공장 전력 모니터링", page_icon=":material/bolt:", layout="wide")


@st.cache_resource
def engine():
    return create_engine(os.environ["DB_URL"], pool_pre_ping=True)


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


def query(sql, **params):
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def show_tiles(t, r, d, has_ml):
    cols = st.columns(4)

    # 1. 현재 전력 + 전력 품질
    ok = V_MIN <= r["voltage"] <= V_MAX and HZ_MIN <= r["hz"] <= HZ_MAX and r["pf"] >= PF_MIN
    with cols[0].container(border=True, height="stretch"):
        st.metric(
            "현재 전력", f"{r['kw']:,.0f} kW",
            "품질 정상" if ok else "품질 이상", delta_color="green" if ok else "red", delta_arrow="off",
            icon=":material/bolt:",
        )
        st.caption(f"{r['voltage']:.0f} V · 역률 {r['pf']:.2f} · {r['hz']:.2f} Hz")

    # 2. 15분 예상 수요 (계산식) + ML 예측
    with cols[1].container(border=True, height="stretch"):
        if d is None:
            st.metric("15분 예상 수요", None, icon=":material/query_stats:")
            st.caption("구간 데이터 없음")
        else:
            ratio = d["projected_kw"] / d["target_kw"]
            if ratio >= 1:
                state, color = "위험", "red"
            elif ratio >= WARN:
                state, color = "경고", "orange"
            else:
                state, color = "정상", "green"
            ws = d["window_start"]

            st.metric(
                "15분 예상 수요 (계산식)", f"{d['projected_kw']:,.0f} kW",
                f"{state} · 목표 대비 {ratio:.0%}", delta_color=color, delta_arrow="off",
                delta_description=f"목표 {d['target_kw']:,.0f} kW",
                icon=":material/query_stats:",
            )

            # 현재 구간 최신 ML 예측 (ML_OLD_SEC 초과분 제외)
            ml = None
            if has_ml:
                f = query("SELECT ts, ml_kw FROM ml.forecast WHERE window_start = :ws ORDER BY ts DESC LIMIT 1", ws=ws)
                if not f.empty and (pd.Timestamp(t) - f.iloc[0]["ts"]).total_seconds() <= ML_OLD_SEC:
                    ml = f.iloc[0]["ml_kw"]

            window = f"구간 {ws:%H:%M}~{ws + timedelta(minutes=15):%H:%M}"
            if not has_ml:
                st.caption(f"{window} · ML 예측 대기 (serve.py 미실행)")
            elif ml is None:
                st.caption(f"{window} · ML 예측 대기 (구간 시작 2분 후부터)")
            else:
                st.caption(f"{window} · :violet[ML 예측 **{ml:,.0f} kW**] "
                           f"(목표 대비 {ml / d['target_kw']:.0%}, 계산식과 {ml - d['projected_kw']:+,.0f} kW)")

    # 3. ESS
    discharging = r["ess_kw"] > 0
    with cols[2].container(border=True, height="stretch"):
        st.metric(
            "ESS 충전율", f"{r['soc']:.0f} %",
            "방전 중" if discharging else "대기", delta_color="orange" if discharging else "gray", delta_arrow="off",
            icon=":material/battery_charging_full:",
        )
        st.progress(min(max(r["soc"] / 100, 0.0), 1.0))
        st.caption(f"방전 {r['ess_kw']:.0f} kW · 방전 하한 {SOC_MIN}%")

    # 4. 현장 연결 상태 (최근 1시간 빈 구간, 재전송 건수)
    lag = (pd.Timestamp(t) - r["ts"]).total_seconds()
    h = query("""
        SELECT count(*) FILTER (WHERE buffered) AS resent,
               coalesce(sum(gap) FILTER (WHERE gap > :g), 0) AS gap_sec
        FROM (
            SELECT buffered,
                   extract(epoch FROM ts - lag(ts) OVER (ORDER BY ts)) AS gap
            FROM raw.meter_reading
            WHERE ts >= :s
        ) x""", s=t - timedelta(hours=1), g=GAP_SEC).iloc[0]
    online = lag < OFFLINE_SEC
    with cols[3].container(border=True, height="stretch"):
        st.metric(
            "현장 연결", "정상" if online else "끊김",
            "ONLINE" if online else "OFFLINE", delta_color="green" if online else "red", delta_arrow="off",
            delta_description=f"{lag:.0f}초 전 수신",
            icon=":material/sensors:",
        )
        st.caption(f"1시간 빈 구간 {float(h['gap_sec']):.0f}초 · 재전송 {int(h['resent'])}건")


# 목표선 (echarts markLine)
def target_line(target):
    if target is None:
        return {}
    return {"silent": True, "symbol": "none", "lineStyle": {"type": "dashed", "color": GRAY},
            "label": {"formatter": "목표"}, "data": [{"yAxis": target}]}


# 15분 수요 막대 (최근 3시간)
#   - 막대: 확정 구간은 평균, 진행 중 구간은 예상값 (목표 초과는 빨강, 진행 중은 회색)
#   - 점: 구간 내 30초 예측의 평균 (ML 성적표와 같은 기준)
def show_demand_chart(t, target, has_ml):
    st.subheader("15분 수요", icon=":material/bar_chart:")
    st.caption("최근 3시간 · 진행 중 구간은 예상값 · 점선 = 목표")

    since = t - timedelta(hours=3)
    df = query("""
        SELECT window_start,
               CASE WHEN final THEN avg_kw ELSE projected_kw END AS kw,
               target_kw, final
        FROM ops.demand_15min
        WHERE window_start >= :s
        ORDER BY window_start""", s=since)
    if df.empty:
        st.caption("데이터 없음")
        return

    if has_ml:
        pred = query("""
            SELECT window_start, avg(formula_kw) AS formula_kw, avg(ml_kw) AS ml_kw
            FROM ml.forecast
            WHERE window_start >= :s
            GROUP BY window_start""", s=since)
        df = df.merge(pred, on="window_start", how="left")
    else:
        df["formula_kw"] = float("nan")
        df["ml_kw"] = float("nan")

    # 막대 색을 나누려고 열을 셋으로 분리 (한 행에 하나만 값이 있음, 같은 stack으로 쌓음)
    over = df["kw"] >= df["target_kw"]
    chart = pd.DataFrame({
        "구간": df["window_start"].dt.strftime("%H:%M"),
        "확정": df["kw"].where(df["final"] & ~over),
        "목표 초과": df["kw"].where(df["final"] & over),
        "진행 중": df["kw"].where(~df["final"]),
        "계산식 예측": df["formula_kw"],
        "ML 예측": df["ml_kw"],
    })

    series = [
        {"type": "bar", "name": "확정", "stack": "kw", "barMaxWidth": 22,
         "itemStyle": {"color": BLUE}, "markLine": target_line(target)},
        {"type": "bar", "name": "목표 초과", "stack": "kw", "barMaxWidth": 22, "itemStyle": {"color": RED}},
        {"type": "bar", "name": "진행 중", "stack": "kw", "barMaxWidth": 22,
         "itemStyle": {"color": GRAY, "opacity": 0.5}},
    ]
    if chart["ML 예측"].notna().any():
        series.append({"type": "scatter", "name": "계산식 예측", "symbol": "diamond", "symbolSize": 10,
                       "itemStyle": {"color": GRAY}})
        series.append({"type": "scatter", "name": "ML 예측", "symbol": "circle", "symbolSize": 10,
                       "itemStyle": {"color": VIOLET}})
    else:
        chart = chart.drop(columns=["계산식 예측", "ML 예측"])

    st.echarts_chart(
        {
            "dataset": {"source": chart},
            "legend": {"top": 0},
            "grid": {"left": 50, "right": 16, "top": 32, "bottom": 24},
            "tooltip": {"trigger": "axis", "valueFormatter": "%,d kW"},
            "xAxis": {"type": "category"},
            "yAxis": {"type": "value", "name": "kW"},
            "series": series,
        },
        height=CHART_H,
        alt="최근 3시간 15분 구간별 수요와 목표, 예측값 비교",
    )


def show_power_chart(t, target):
    st.subheader("전력 추이", icon=":material/show_chart:")
    st.caption("최근 30분 · 점선 = 목표")

    df = query("SELECT ts, kw FROM raw.meter_reading WHERE ts >= :s ORDER BY ts", s=t - timedelta(minutes=30))
    if df.empty:
        st.caption("데이터 없음")
        return

    st.echarts_chart(
        {
            "dataset": {"source": df},
            "grid": {"left": 50, "right": 16, "top": 16, "bottom": 24},
            "tooltip": {"trigger": "axis", "valueFormatter": "%,d kW"},
            "xAxis": {"type": "time"},
            "yAxis": {"type": "value", "name": "kW", "scale": True},
            "series": [{"type": "line", "name": "전력", "showSymbol": False, "lineStyle": {"width": 1.4},
                        "itemStyle": {"color": BLUE}, "markLine": target_line(target)}],
        },
        height=CHART_H,
        alt="최근 30분 전력 추이와 목표",
    )


def show_control_log():
    st.subheader("제어 이력", icon=":material/tune:")

    df = query("SELECT ts, action, target, reason, projected_kw FROM ops.control_event ORDER BY ts DESC LIMIT :n",
               n=ROWS)
    if df.empty:
        st.caption("기록 없음")
        return

    df["action"] = df["action"].map(lambda a: ACTIONS.get(a, a))
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "ts": st.column_config.DatetimeColumn("시각", format="HH:mm:ss"),
            "action": "동작",
            "target": "대상",
            "reason": st.column_config.TextColumn("사유", width="large"),
            "projected_kw": st.column_config.NumberColumn("예상 kW", format="%,d"),
        },
        alt=f"최근 제어 이력 {ROWS}건",
    )


def show_alerts():
    st.subheader("알림", icon=":material/notifications:")

    df = query("SELECT ts, kind, message FROM ops.alert ORDER BY ts DESC LIMIT :n", n=ROWS)
    if df.empty:
        st.caption("기록 없음")
        return

    df["kind"] = df["kind"].map(lambda k: KINDS.get(k, k))
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "ts": st.column_config.DatetimeColumn("시각", format="HH:mm:ss"),
            "kind": "종류",
            "message": st.column_config.TextColumn("내용", width="large"),
        },
        alt=f"최근 알림 {ROWS}건",
    )


# 오늘 마감 구간의 계산식 vs ML 오차
def show_ml_score(t, has_ml):
    st.subheader("ML 성적표", icon=":material/fact_check:")
    st.caption("오늘 · 확정 구간 기준")
    if not has_ml:
        st.caption("ML 서빙 대기 · serving/serve.py 실행 필요")
        return

    s = query("""
        SELECT count(*) AS n,
               avg(formula_mae) AS formula_mae,
               avg(ml_mae) AS ml_mae,
               count(*) FILTER (WHERE ml_mae < formula_mae) AS ml_wins
        FROM ml.window_eval
        WHERE window_start >= :s""", s=t.replace(hour=0, minute=0, second=0, microsecond=0)).iloc[0]
    n = int(s["n"])
    if n == 0:
        st.caption("오늘 평가된 구간 없음 · 첫 구간 마감 후 표시")
        return

    f_mae = float(s["formula_mae"])
    m_mae = float(s["ml_mae"])
    wins = int(s["ml_wins"])
    gain = (f_mae - m_mae) / f_mae if f_mae > 0 else 0.0

    last = query("SELECT window_start, formula_mae, ml_mae FROM ml.window_eval ORDER BY window_start DESC LIMIT 1").iloc[0]
    last_win = "ML" if last["ml_mae"] < last["formula_mae"] else "계산식"

    st.table(
        {
            "평가 구간": f"{n}개",
            "계산식 평균 오차": f"{f_mae:,.1f} kW",
            "ML 평균 오차": f"{m_mae:,.1f} kW",
            "ML 오차 개선": f"{gain:+.1%}",
            "ML 승률": f"{wins / n:.0%} ({wins}/{n})",
            "최근 구간": f"{last['window_start']:%H:%M} · {last_win} 승",
        },
        border="horizontal",
        alt="오늘 계산식과 ML 예측 오차 비교",
    )


@st.fragment(run_every=REFRESH)
def live():
    t = now()

    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.title("공장 전력 모니터링 · 피크 제어", icon=":material/factory:", width="content")
        st.caption(f"{SITE} · {t:%Y-%m-%d %H:%M:%S} 기준 · {REFRESH}초 갱신", text_alignment="right")

    last = query("SELECT * FROM raw.meter_reading ORDER BY ts DESC LIMIT 1")
    if last.empty:
        st.info("아직 수신된 데이터가 없습니다.", icon=":material/hourglass_empty:")
        return
    r = last.iloc[0]

    dem = query("SELECT * FROM ops.demand_15min ORDER BY window_start DESC LIMIT 1")
    d = None
    target = None
    if not dem.empty:
        d = dem.iloc[0]
        target = float(d["target_kw"])

    # ml 테이블 존재 여부 (serve.py 실행 이력)
    has_ml = bool(query("SELECT to_regclass('ml.forecast') IS NOT NULL "
                        "AND to_regclass('ml.window_eval') IS NOT NULL AS ok").iloc[0]["ok"])

    show_tiles(t, r, d, has_ml)

    # 칸 높이 고정 (좌우 정렬, 넘침 방지)
    cols = st.columns([3, 2])
    with cols[0].container(border=True, height=CHART_BOX_H):
        show_demand_chart(t, target, has_ml)
    with cols[1].container(border=True, height=CHART_BOX_H):
        show_power_chart(t, target)

    cols = st.columns([4, 4, 3])
    with cols[0].container(border=True, height=LOG_BOX_H):
        show_control_log()
    with cols[1].container(border=True, height=LOG_BOX_H):
        show_alerts()
    with cols[2].container(border=True, height=LOG_BOX_H):
        show_ml_score(t, has_ml)


live()
