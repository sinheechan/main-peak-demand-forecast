# HIPE 공장 전체 전력 재생 (simulator에 1초 단위로 전달)
#   - 반환값: 전력(kW), 전압 변동 비율, 원본 시각
#   - 약 5.5초 간격 -> 1초 간격으로 확장 (직전 값 유지, 최대 10초)
#   - 10초 넘는 공백은 건너뜀

import numpy as np
import pandas as pd


class Replay:

    # p95_kw: 규모 기준 (03 전처리와 같은 배율 계산)
    # start_date: 재생 시작일 (학습 기간 제외용)
    # - 전압은 상전압(약 238V)이라 중앙값 대비 비율만 사용
    def __init__(self, path, p95_kw, hour, minute, local_tz, start_date=None):
        print(f"재생 데이터 로드 중: {path}")
        df = pd.read_parquet(path)
        has_volt = "V1_V" in df.columns

        index = pd.DatetimeIndex(df["tmp_timestamp"])
        kw = pd.Series(df["P_kW"].to_numpy(dtype="float64"), index=index).clip(lower=0)
        running = kw[kw > kw.max() * 0.05]
        self.scale = p95_kw / np.percentile(running, 95)

        # 1초 간격 확장 (UTC 기준이라 서머타임 전환에도 중복 없음)
        frame = pd.DataFrame({"kw": kw * self.scale})
        if has_volt:
            frame["v"] = df["V1_V"].to_numpy(dtype="float64")
        frame = frame[~frame.index.duplicated()]
        frame = frame.resample("1s").mean()
        frame = frame.ffill(limit=10)
        frame = frame.dropna(subset=["kw"])

        if has_volt:
            v = frame["v"].to_numpy()
            v_med = np.nanmedian(v[v > 0])
            self.v_ratio = np.where(v > 0, v / v_med, 1.0).astype("float32")
        else:
            self.v_ratio = np.ones(len(frame), dtype="float32")

        local = frame.index.tz_convert(local_tz).tz_localize(None)   # 현지 시각 (표시, 시작 시각 맞춤용)
        self.kw = frame["kw"].to_numpy(dtype="float32")
        self.ts = local.to_numpy()

        # 재생 시작 위치 (끝에 도달하면 여기로 복귀)
        self.first = 0
        if start_date:
            idx = np.flatnonzero(local >= pd.Timestamp(start_date))
            if len(idx) > 0:
                self.first = int(idx[0])
            else:
                print(f"{start_date} 이후 데이터 없음 -> 전체 재생")

        # 현재 시:분과 같은 첫 행부터 시작
        minutes = local.hour * 60 + local.minute
        idx = np.flatnonzero(minutes == hour * 60 + minute)
        idx = idx[idx >= self.first]
        self.i = self.first
        if len(idx) > 0:
            self.i = int(idx[0])
        del df, frame

        print(f"재생 준비 완료: {len(self.kw):,}초 · 배율 x{self.scale:.1f} · "
              f"범위 {pd.Timestamp(self.ts[self.first]):%Y-%m-%d}~{pd.Timestamp(self.ts[-1]):%Y-%m-%d} · "
              f"시작 {pd.Timestamp(self.ts[self.i]):%Y-%m-%d %H:%M:%S}")

    # 1초 분량 반환 (끝에 도달하면 처음으로)
    def next(self):
        kw = float(self.kw[self.i])
        v_ratio = float(self.v_ratio[self.i])
        src_ts = pd.Timestamp(self.ts[self.i])

        self.i += 1
        if self.i >= len(self.kw):
            self.i = self.first
        return kw, v_ratio, src_ts
