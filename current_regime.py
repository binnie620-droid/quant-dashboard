"""
UCHIDA V3 - 현재 시점 국면 확인
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from config import DATES
from data.loader import load_all
from data.features import build_features
from models.baseline import (
    compute_flags, compute_score, classify_regime,
    BaselineConfig, WEIGHT_MAP,
)


def main():
    print("="*70)
    print("UCHIDA V3 - 현재 시점 국면")
    print("="*70)

    df = load_all(start=DATES.train_start, use_cache=True)
    features_raw = build_features(df)
    SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
    features = features_raw.dropna(subset=[c for c in SIGNALS if c in features_raw.columns])

    cfg = BaselineConfig()
    flags = compute_flags(features, cfg)
    score = compute_score(flags, cfg)
    regime = classify_regime(score, cfg)

    # 최근 10일
    last_n = 10
    print(f"\n[최근 {last_n}거래일 국면]")
    print(f"{'날짜':>12s} | {'점수':>6s} | {'국면':>9s} | {'CPI_z':>7s} {'스프레드':>8s} {'10Y-2Y':>8s} {'VIX':>6s} {'MA200':>7s}")
    print("-"*90)

    last_dates = regime.tail(last_n).index
    for d in last_dates:
        sc = score.loc[d]
        rg = regime.loc[d]
        cpi  = features.loc[d, "cpi_z"]
        spr  = features.loc[d, "credit_spread"]
        t10y = features.loc[d, "t10y2y"]
        vix  = features.loc[d, "vix"]
        ma   = features.loc[d, "dist_ma200_QQQ"]
        print(f"{d.strftime('%Y-%m-%d'):>12s} | {sc:>6.2f} | {rg:>9s} | "
              f"{cpi:>7.2f} {spr:>8.2f} {t10y:>8.2f} {vix:>6.1f} {ma:>+7.2%}")

    # 오늘 시점 비중
    today = regime.index[-1]
    today_regime = regime.iloc[-1]
    print(f"\n[현재 권장 비중] {today.strftime('%Y-%m-%d')} | 국면: {today_regime}")
    print("-"*40)
    for asset, w in WEIGHT_MAP[today_regime].items():
        print(f"  {asset:6s}: {w*100:5.1f}%")

    # 임계값 비교
    print(f"\n[임계값 위반 여부]")
    print("-"*40)
    today_row = features.iloc[-1]
    today_flags = flags.iloc[-1]
    checks = [
        ("CPI Z-score > 2.3",         today_row["cpi_z"],          2.3, ">"),
        ("신용스프레드 > 3.5%",         today_row["credit_spread"],  3.5, ">"),
        ("10Y-2Y < 0%",                today_row["t10y2y"],         0.0, "<"),
        ("VIX > 22",                   today_row["vix"],            22.0, ">"),
        ("QQQ MA200거리 < -3%",        today_row["dist_ma200_QQQ"], -0.03, "<"),
    ]
    for name, val, thr, op in checks:
        triggered = (val > thr) if op == ">" else (val < thr)
        mark = "ON " if triggered else "off"
        if op == ">":
            print(f"  [{mark}] {name:25s} | 현재 {val:>7.2f} (임계 {thr})")
        else:
            print(f"  [{mark}] {name:25s} | 현재 {val:>7.2f} (임계 {thr})")

    print(f"\n현재 점수: {score.iloc[-1]:.2f}")
    print(f"  점수 < 1.5  → ATTACK")
    print(f"  1.5~3.0     → DEFENSE")
    print(f"  점수 ≥ 3.0  → CRISIS")


if __name__ == "__main__":
    main()
