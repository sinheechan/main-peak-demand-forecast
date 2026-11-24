# 관제 대시보드 (현장 모니터 한 화면)
#   0행: 제목, 현장, 기준 시각
#   1행: 지금 상태 안내 (정상 / 주의 / 위험 / 끊김) - 누구나 읽을 수 있는 한 줄 + 제어기 대응
#   2행: 수전 전력 | 15분 예상 수요 | ESS 충전율 | 현장 연결 (카드마다 숫자·상태 하나씩, 같은 높이)
#   3행: 15분 수요 막대(최근 3시간) | 전력 추이(최근 30분, ESS 효과)
#   4행: 제어 이력 | 현황 | AI 예측 정확도
#   - 색, 글꼴, 여백은 .streamlit/config.toml 테마 (CSS 없음)
#   - 용어 설명은 각 카드 제목 옆 (?) 툴팁
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
WARN = 0.9           # 주의 기준 (목표 대비)
SOC_MIN = 20
V_MIN = 342
V_MAX = 418
HZ_MIN = 59.8
HZ_MAX = 60.2
PF_MIN = 0.90
ML_OLD_SEC = 90      # ML 예측 지연 한계 (초과 시 serve.py 중단으로 간주)
LOAD_HOURS = 2       # 부하 차단 상태를 판단할 제어 이력 범위 (시간)


CHART_H = 240        # 차트 높이
CHART_BOX_H = 330    # 차트 칸 높이 (제목 + 차트)
LOG_BOX_H = 310      # 기록 칸 높이
ROWS = 6             # 기록 표 행 수

# 차트·표 색 (config.toml 라이트 팔레트와 같은 값)
BLUE = "#3B82F6"
GREEN = "#059669"
RED = "#DC2626"
ORANGE = "#EA580C"
GRAY = "#64748B"
VIOLET = "#7C3AED"   # ML

ACTIONS = {"shed": "부하 차단", "restore": "부하 복귀", "ess_on": "ESS 방전", "ess_off": "ESS 정지"}
ACTION_ORDER = ["ESS 방전", "부하 차단", "부하 복귀", "ESS 정지"]
ACTION_COLORS = [ORANGE, RED, GREEN, GRAY]

KINDS = {
    "no_data": "수신 끊김", "data_resumed": "수신 재개", "edge_offline": "현장 오프라인",
    "delay": "수신 지연", "duplicate": "중복 수신", "kwh_backward": "전력량 역행",
    "voltage": "전압 이탈", "frequency": "주파수 이탈", "power_factor": "역률 저하",
    "peak_exceeded": "피크 초과",
}
# 현황 심각도: 위험 = 바로 확인, 주의 = 지켜보기, 정보 = 기록용
SEVERITY = {
    "peak_exceeded": "위험", "edge_offline": "위험", "no_data": "위험",
    "voltage": "주의", "frequency": "주의", "power_factor": "주의", "delay": "주의", "kwh_backward": "주의",
    "data_resumed": "정보", "duplicate": "정보",
}
SEVERITY_ORDER = ["위험", "주의", "정보"]
SEVERITY_COLORS = [RED, ORANGE, GRAY]

st.set_page_config(page_title="공장 전력 모니터링", page_icon=":material/bolt:", layout="wide")


@st.cache_resource
def engine():
    return create_engine(os.environ["DB_URL"], pool_pre_ping=True)


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


def query(sql, **params):
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# 남은 시간 표시 (예: 6분 12초)
def mmss(seconds):
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}분 {seconds % 60:02d}초"


# 목표 대비 비율 -> (상태, 색)
def demand_state(ratio):
    if ratio >= 1:
        return "위험", "red"
    if ratio >= WARN:
        return "주의", "orange"
    return "정상", "green"


# 지금 차단된 부하 번호 (최근 LOAD_HOURS 시간 제어 이력의 마지막 동작이 차단인 것)
#   - 제어기 재시작 시 부하는 모두 복귀되지만 이력에는 남지 않음 -> 재시작 직후엔 실제와 다를 수 있음
def shed_loads(t):
    df = query("""
        SELECT DISTINCT ON (target) target, action
        FROM ops.control_event
        WHERE target IN ('load1', 'load2', 'load3') AND ts >= :s
        ORDER BY target, ts DESC""", s=t - timedelta(hours=LOAD_HOURS))
    shed = []
    for _, row in df.iterrows():
        if row["action"] == "shed":
            shed.append(row["target"])
    return shed


# ---------------------------------------------------------------------------
# 1행: 지금 상태 안내
# ---------------------------------------------------------------------------
def show_status(t, r, d, lag, shed):
    if lag >= OFFLINE_SEC:
        st.error(f"마지막 수신 {lag:,.0f}초 전 · 제어기·네트워크 확인", title="현장 데이터 끊김")
        return

    if d is None:
        st.info("첫 15분 구간 계산 중", title="데이터 대기")
        return

    ws = d["window_start"]
    remain = (ws + timedelta(minutes=15) - pd.Timestamp(t)).total_seconds()
    projected = float(d["projected_kw"])
    target = float(d["target_kw"])
    ratio = projected / target
    state, _ = demand_state(ratio)

    # 제어기가 지금 하고 있는 일 (없으면 생략)
    acts = []
    if r["ess_kw"] > 0:
        acts.append(f"ESS 방전 {r['ess_kw']:,.0f} kW")
    for name in shed:
        acts.append(f"{name.replace('load', '부하 ')} 차단")

    body = f"남은 {mmss(remain)} · 예상 {projected:,.0f} / 목표 {target:,.0f} kW"
    if acts:
        body = body + " · " + ", ".join(acts)

    if state == "위험":
        st.error(body, title=f"위험 · 목표 초과 예상 +{projected - target:,.0f} kW")
    elif state == "주의":
        st.warning(body, title=f"주의 · 여유 {target - projected:,.0f} kW")
    else:
        st.success(body, title=f"정상 · 여유 {target - projected:,.0f} kW")


# ---------------------------------------------------------------------------
# 2행: 핵심 카드 4개
# ---------------------------------------------------------------------------
def show_tiles(t, r, d, has_ml, lag, shed):
    # 카드마다 숫자 1개 + 상태 1개 (그래프·게이지 없이 같은 높이). 세부 수치는 (?) 툴팁으로
    cols = st.columns(4)

    # 1. 수전 전력
    ok = V_MIN <= r["voltage"] <= V_MAX and HZ_MIN <= r["hz"] <= HZ_MAX and r["pf"] >= PF_MIN
    quality = "품질 정상"
    q_color = "green"
    if not ok:
        quality = "품질 이상"
        q_color = "red"
    cols[0].metric(
        "수전 전력", f"{r['kw']:,.0f} kW", quality, delta_color=q_color, delta_arrow="off",
        border=True, height="stretch",
        help=(f"한전에서 공장으로 들어오는 전력  \n"
              f"전압 {r['voltage']:.0f} V · 역률 {r['pf']:.2f} · 주파수 {r['hz']:.2f} Hz"),
    )

    # 2. 15분 예상 수요 + AI 예측 (있을 때만 보조 문구)
    help_text = "이번 15분 평균 전력을 미리 계산한 값. 이 값이 전기요금의 최대수요전력을 정합니다."
    if d is None:
        cols[1].metric("15분 예상 수요", None, border=True, height="stretch", help=help_text)
    else:
        projected = float(d["projected_kw"])
        target = float(d["target_kw"])
        ratio = projected / target
        _, color = demand_state(ratio)

        # 현재 구간 최신 AI 예측 (ML_OLD_SEC 초과분은 표시 안 함)
        ai_text = None
        if has_ml:
            f = query("SELECT ts, ml_kw FROM ml.forecast WHERE window_start = :ws ORDER BY ts DESC LIMIT 1",
                      ws=d["window_start"])
            if not f.empty and (pd.Timestamp(t) - f.iloc[0]["ts"]).total_seconds() <= ML_OLD_SEC:
                ai_text = f"AI 예측 {f.iloc[0]['ml_kw']:,.0f} kW"

        cols[1].metric(
            "15분 예상 수요", f"{projected:,.0f} kW",
            f"목표 {target:,.0f} kW의 {ratio:.0%}", delta_color=color, delta_arrow="off",
            delta_description=ai_text, border=True, height="stretch", help=help_text,
        )

    # 3. ESS 충전율 + 차단된 부하 (있을 때만 보조 문구)
    ess_state = "대기"
    ess_color = "gray"
    if r["ess_kw"] > 0:
        ess_state = f"방전 {r['ess_kw']:,.0f} kW"
        ess_color = "orange"
    shed_text = None
    if shed:
        names = []
        for name in shed:
            names.append(name.replace("load", "부하 "))
        shed_text = ", ".join(names) + " 차단"
    cols[2].metric(
        "ESS 충전율", f"{r['soc']:.0f} %", ess_state, delta_color=ess_color, delta_arrow="off",
        delta_description=shed_text, border=True, height="stretch",
        help=(f"피크가 예상되면 배터리로 전력을 대신 공급해 수전 전력을 줄입니다. 방전 하한 {SOC_MIN}%  \n"
              "그래도 부족하면 부하 3 → 2 → 1 순으로 잠시 끕니다."),
    )

    # 4. 현장 연결 (세부는 툴팁: 최근 1시간 빈 구간, 재전송 건수)
    h = query("""
        SELECT count(*) FILTER (WHERE buffered) AS resent,
               coalesce(sum(gap) FILTER (WHERE gap > :g), 0) AS gap_sec
        FROM (
            SELECT buffered,
                   extract(epoch FROM ts - lag(ts) OVER (ORDER BY ts)) AS gap
            FROM raw.meter_reading
            WHERE ts >= :s
        ) x""", s=t - timedelta(hours=1), g=GAP_SEC).iloc[0]
    conn_value = "정상"
    conn_color = "green"
    if lag >= OFFLINE_SEC:
        conn_value = "끊김"
        conn_color = "red"
    cols[3].metric(
        "현장 연결", conn_value, f"{lag:.0f}초 전 수신", delta_color=conn_color, delta_arrow="off",
        border=True, height="stretch",
        help=(f"제어기 → 서버 1초 전송 상태  \n"
              f"최근 1시간 빈 구간 {float(h['gap_sec']):,.0f}초 · 재전송 {int(h['resent']):,}건"),
    )


# ---------------------------------------------------------------------------
# 3행: 차트
# ---------------------------------------------------------------------------
# 목표선 + 주의선 (echarts markLine)
def target_lines(target):
    if target is None:
        return {}
    return {
        "silent": True, "symbol": "none",
        "data": [
            {"yAxis": target, "lineStyle": {"type": "dashed", "color": RED, "width": 1.2},
             "label": {"formatter": f"목표 {target:,.0f}", "color": RED, "position": "end"}},
            {"yAxis": target * WARN, "lineStyle": {"type": "dotted", "color": ORANGE, "width": 1},
             "label": {"formatter": f"주의 {target * WARN:,.0f}", "color": ORANGE, "position": "end"}},
        ],
    }


# 15분 수요 막대 (최근 3시간)
#   - 막대: 확정 구간은 평균, 진행 중 구간은 예상값 (목표 초과는 빨강, 진행 중은 회색)
#   - 점: 구간 내 30초 예측의 평균 (AI 예측 정확도와 같은 기준)
def show_demand_chart(t, target, has_ml):
    st.subheader("15분 평균 수요", 
                 help="막대 하나 = 15분 평균 전력. 빨간 막대가 요금 기준(최대수요전력)을 올리는 구간입니다.")
    st.caption("최근 3시간")

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
        "기존 계산": df["formula_kw"],
        "AI 예측": df["ml_kw"],
    })

    series = [
        {"type": "bar", "name": "확정", "stack": "kw", "barMaxWidth": 26,
         "itemStyle": {"color": BLUE, "borderRadius": [3, 3, 0, 0]}, "markLine": target_lines(target)},
        {"type": "bar", "name": "목표 초과", "stack": "kw", "barMaxWidth": 26,
         "itemStyle": {"color": RED, "borderRadius": [3, 3, 0, 0]}},
        {"type": "bar", "name": "진행 중", "stack": "kw", "barMaxWidth": 26,
         "itemStyle": {"color": GRAY, "opacity": 0.45, "borderRadius": [3, 3, 0, 0]}},
    ]
    if chart["AI 예측"].notna().any():
        series.append({"type": "scatter", "name": "기존 계산", "symbol": "diamond", "symbolSize": 9,
                       "itemStyle": {"color": GRAY}})
        series.append({"type": "scatter", "name": "AI 예측", "symbol": "circle", "symbolSize": 9,
                       "itemStyle": {"color": VIOLET}})
    else:
        chart = chart.drop(columns=["기존 계산", "AI 예측"])

    st.echarts_chart(
        {
            "dataset": {"source": chart},
            "legend": {"top": 0, "right": 0, "itemWidth": 12, "itemHeight": 8},
            "grid": {"left": 52, "right": 70, "top": 32, "bottom": 24},
            "tooltip": {"trigger": "axis", "valueFormatter": "%,d kW"},
            "xAxis": {"type": "category"},
            "yAxis": {"type": "value", "name": "kW"},
            "series": series,
        },
        height=CHART_H,
        alt="최근 3시간 15분 구간별 평균 수요와 목표, 예측값 비교",
    )


# 전력 추이 (최근 30분)
#   - 파랑 실선: 실제 수전 전력 / 회색 점선: ESS가 없었다면 (수전 + ESS 방전)
#   - 두 선 사이 간격 = ESS가 줄여 준 전력
def show_power_chart(t, target):
    st.subheader("전력 추이", 
                 help="회색 점선은 ESS가 없었을 때의 전력입니다. 파란 선과의 간격만큼 ESS가 피크를 깎아 준 것입니다.")
    st.caption("최근 30분")

    df = query("SELECT ts, kw, ess_kw FROM raw.meter_reading WHERE ts >= :s ORDER BY ts",
               s=t - timedelta(minutes=30))
    if df.empty:
        st.caption("데이터 없음")
        return

    chart = pd.DataFrame({
        "시각": df["ts"],
        "실제 수전": df["kw"],
        "ESS 없을 때": df["kw"] + df["ess_kw"],
    })
    st.echarts_chart(
        {
            "dataset": {"source": chart},
            "legend": {"top": 0, "right": 0, "itemWidth": 14, "itemHeight": 6},
            "grid": {"left": 52, "right": 70, "top": 32, "bottom": 24},
            "tooltip": {"trigger": "axis", "valueFormatter": "%,d kW"},
            "xAxis": {"type": "time"},
            "yAxis": {"type": "value", "name": "kW", "scale": True},
            "dataZoom": [{"type": "inside"}],
            "series": [
                {"type": "line", "name": "ESS 없을 때", "showSymbol": False,
                 "encode": {"x": "시각", "y": "ESS 없을 때"},
                 "lineStyle": {"width": 1, "type": "dashed", "color": GRAY}, "itemStyle": {"color": GRAY}},
                {"type": "line", "name": "실제 수전", "showSymbol": False,
                 "encode": {"x": "시각", "y": "실제 수전"},
                 "lineStyle": {"width": 1.6, "color": BLUE}, "itemStyle": {"color": BLUE},
                 "areaStyle": {"color": BLUE, "opacity": 0.08}, "markLine": target_lines(target)},
            ],
        },
        height=CHART_H,
        alt="최근 30분 수전 전력과 ESS가 없었을 때의 전력, 목표선",
    )


# ---------------------------------------------------------------------------
# 4행: 기록
# ---------------------------------------------------------------------------
def show_control_log():
    st.subheader("제어 이력", 
                 help="제어기가 피크를 막으려고 자동으로 한 일입니다. 예상 kW는 그 순간의 15분 예상 수요입니다.")

    df = query("SELECT ts, action, target, reason, projected_kw FROM ops.control_event ORDER BY ts DESC LIMIT :n",
               n=ROWS)
    if df.empty:
        st.caption("기록 없음")
        return

    # 동작 이름을 한글로, 색 라벨로 보이게 리스트로 감쌈
    labels = []
    for action in df["action"]:
        labels.append([ACTIONS.get(action, action)])
    df["action"] = labels
    df["target"] = df["target"].str.replace("load", "부하 ", regex=False).str.replace("ess", "ESS", regex=False)

    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "ts": st.column_config.DatetimeColumn("시각", format="HH:mm:ss", width="small"),
            "action": st.column_config.MultiselectColumn("동작", options=ACTION_ORDER, color=ACTION_COLORS,
                                                         width="small"),
            "target": st.column_config.TextColumn("대상", width="small"),
            "reason": st.column_config.TextColumn("사유", width="medium"),
            "projected_kw": st.column_config.NumberColumn("예상 kW", format="%,d"),
        },
        alt=f"최근 제어 이력 {ROWS}건",
    )


def show_alerts(t):
    st.subheader("현황", 
                 help="위험 = 바로 확인 필요 · 주의 = 지켜보기 · 정보 = 기록용")

    df = query("SELECT ts, kind, message FROM ops.alert ORDER BY ts DESC LIMIT :n", n=ROWS)
    if df.empty:
        st.caption("특이 사항 없음")
        return

    levels = []
    names = []
    for kind in df["kind"]:
        levels.append([SEVERITY.get(kind, "정보")])
        names.append(KINDS.get(kind, kind))
    show = pd.DataFrame({"ts": df["ts"], "level": levels, "kind": names, "message": df["message"]})

    st.dataframe(
        show,
        hide_index=True,
        column_config={
            "ts": st.column_config.DatetimeColumn("시각", format="HH:mm:ss", width="small"),
            "level": st.column_config.MultiselectColumn("심각도", options=SEVERITY_ORDER, color=SEVERITY_COLORS,
                                                        width="small"),
            "kind": st.column_config.TextColumn("종류", width="small"),
            "message": st.column_config.TextColumn("내용", width="medium"),
        },
        alt=f"최근 현황 {ROWS}건",
    )


# AI 예측 정확도: 오늘 마감된 구간에서 AI가 15분 평균을 얼마나 가깝게 맞혔는지
#   - 정확도 = 1 - (평균 오차 / 평균 수요)
#   - 기존 계산 방식 오차는 비교 기준으로만 표시
def show_ml_score(t, has_ml):
    st.subheader("AI 예측 정확도",
                 help="AI가 15분 평균 전력을 구간이 끝나기 전에 미리 맞힌 정도입니다. "
                      "실제 제어는 기존 계산 방식으로 합니다.")
    st.caption("오늘 마감된 구간 기준")
    if not has_ml:
        st.caption("AI 예측 준비 중 (serving/serve.py 실행 필요)")
        return

    s = query("""
        SELECT count(*) AS n,
               avg(actual_kw) AS actual_kw,
               avg(formula_mae) AS formula_mae,
               avg(ml_mae) AS ml_mae
        FROM ml.window_eval
        WHERE window_start >= :s""", s=t.replace(hour=0, minute=0, second=0, microsecond=0)).iloc[0]
    n = int(s["n"])
    if n == 0:
        st.caption("첫 15분 구간 마감 후 표시")
        return

    actual = float(s["actual_kw"])
    f_mae = float(s["formula_mae"])
    m_mae = float(s["ml_mae"])
    accuracy = 0.0
    if actual > 0:
        accuracy = max(1 - m_mae / actual, 0.0)

    st.metric("정확도", f"{accuracy:.1%}",
              help="100%에 가까울수록 실제 15분 평균과 예측이 가깝습니다.")
    st.table(
        {
            "AI 예측 평균 오차": f"{m_mae:,.1f} kW",
            "기존 계산 방식 평균 오차": f"{f_mae:,.1f} kW",
            "확인한 구간": f"{n}개",
        },
        border="horizontal",
        alt="오늘 AI 예측과 기존 계산 방식의 평균 오차",
    )


@st.fragment(run_every=REFRESH)
def live():
    t = now()

    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.title("공장 전력 모니터링 · 피크 제어", width="content")
        st.caption(f"{SITE} · {t:%Y-%m-%d %H:%M:%S} 기준 · {REFRESH}초마다 자동 갱신", text_alignment="right")

    last = query("SELECT * FROM raw.meter_reading ORDER BY ts DESC LIMIT 1")
    if last.empty:
        st.info("제어기가 켜지면 표시됩니다", title="수신 데이터 없음")
        return
    r = last.iloc[0]
    lag = (pd.Timestamp(t) - r["ts"]).total_seconds()

    # 지금 진행 중인 구간만 사용 (끝난 구간이면 None)
    dem = query("SELECT * FROM ops.demand_15min ORDER BY window_start DESC LIMIT 1")
    d = None
    target = None
    if not dem.empty:
        target = float(dem.iloc[0]["target_kw"])
        if dem.iloc[0]["window_start"] + timedelta(minutes=15) > pd.Timestamp(t):
            d = dem.iloc[0]

    # ml 테이블 존재 여부 (serve.py 실행 이력)
    has_ml = bool(query("SELECT to_regclass('ml.forecast') IS NOT NULL "
                        "AND to_regclass('ml.window_eval') IS NOT NULL AS ok").iloc[0]["ok"])

    shed = shed_loads(t)

    show_status(t, r, d, lag, shed)
    show_tiles(t, r, d, has_ml, lag, shed)

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
        show_alerts(t)
    with cols[2].container(border=True, height=LOG_BOX_H):
        show_ml_score(t, has_ml)


live()
