"""
UCHIDA V3 - Level 1 재설계용 사전 데이터 분석

Opus 세션에서 위기 분류 체계를 설계할 때 사용할 raw 데이터 추출.

[분석 대상]
1. 위기별 신호 발동 패턴 (어떤 신호가 어떤 위기에 켜지는가)
2. 신호 간 상관관계 (중복 vs 보완)
3. 각 신호의 위기 일치율 (위기 시 켜져 있는 시간 비율)
4. 신호 raw 값의 분위수 분포 (임계값 재설정용)
5. 위기 시점 신호 강도 (연속 점수화 가능성)

[출력]
콘솔 + CSV 파일 (analysis/level1_redesign_data/*.csv)

[실행]
python prep_for_redesign.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from datetime import date

from config import DATES
from data.loader import load_all
from data.features import build_features


# ==========================================
# 위기 구간 정의 (학계 일반 견해 + 실증 검증)
# ==========================================
# NBER recession 기록 + 시장 폭락 시점
# 위기 유형 분류는 잠정 — Opus가 재검토할 것
CRISIS_PERIODS = {
    "2008 GFC (신용)":      ("2008-09-01", "2009-06-30", "credit"),
    "2011 유럽 부채 (신용)":  ("2011-07-01", "2011-12-31", "credit"),
    "2015-16 차이나 (성장)":  ("2015-08-01", "2016-02-29", "growth"),
    "2018 Q4 (긴축)":        ("2018-10-01", "2018-12-31", "rate"),
    "2020 COVID (패닉)":     ("2020-02-15", "2020-05-31", "panic"),
    "2022 인플레이션":         ("2022-01-01", "2022-10-31", "inflation"),
}

# 평시 비교용 (양호한 시장)
CALM_PERIODS = {
    "2013-14 평온":  ("2013-01-01", "2014-12-31"),
    "2017 저변동":    ("2017-01-01", "2017-12-31"),
    "2024 회복":      ("2024-01-01", "2024-12-31"),
}

# 5개 핵심 신호 (현재 baseline)
SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]

# 신호 방향 (gt=클수록 위기, lt=작을수록 위기)
SIGNAL_DIRECTION = {
    "cpi_z":          "gt",
    "credit_spread":  "gt",
    "t10y2y":         "lt",
    "vix":            "gt",
    "dist_ma200_QQQ": "lt",
}

OUT_DIR = "analysis/level1_redesign_data"


def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================
# 분석 1. 위기별 신호 발동 패턴
# ==========================================
def analyze_crisis_signals(features):
    """각 위기 구간에서 각 신호의 raw 값 통계."""
    print("\n" + "=" * 90)
    print("[분석 1] 위기별 신호 발동 패턴")
    print("=" * 90)

    rows = []
    for crisis_name, (start, end, ctype) in CRISIS_PERIODS.items():
        sub = features.loc[start:end]
        if len(sub) == 0:
            continue
        for sig in SIGNALS:
            if sig not in sub.columns:
                continue
            s = sub[sig].dropna()
            if len(s) == 0:
                continue
            rows.append({
                "crisis":    crisis_name,
                "type":      ctype,
                "signal":    sig,
                "n_days":    len(s),
                "min":       s.min(),
                "p25":       s.quantile(0.25),
                "median":    s.median(),
                "p75":       s.quantile(0.75),
                "max":       s.max(),
                "mean":      s.mean(),
            })

    df = pd.DataFrame(rows)

    # 콘솔 출력 (피벗 테이블)
    for sig in SIGNALS:
        sub = df[df["signal"] == sig]
        if len(sub) == 0:
            continue
        direction = "↑클수록 위기" if SIGNAL_DIRECTION[sig] == "gt" else "↓작을수록 위기"
        print(f"\n  [{sig}] ({direction})")
        print(f"  {'위기':<25} {'유형':<10} {'min':>8} {'median':>8} {'max':>8}")
        print("  " + "-" * 70)
        for _, row in sub.iterrows():
            print(f"  {row['crisis']:<25} {row['type']:<10} "
                  f"{row['min']:>8.2f} {row['median']:>8.2f} {row['max']:>8.2f}")

    df.to_csv(f"{OUT_DIR}/01_crisis_signals.csv", index=False)
    print(f"\n  → {OUT_DIR}/01_crisis_signals.csv")
    return df


# ==========================================
# 분석 2. 신호별 위기 일치율
# ==========================================
def analyze_signal_hit_rate(features, current_thresholds):
    """
    각 신호가 위기 시 얼마나 자주 발동하는가? 평시엔?
    Precision/Recall 관점.
    """
    print("\n" + "=" * 90)
    print("[분석 2] 신호별 위기 일치율 (현재 임계값 기준)")
    print("=" * 90)

    # 위기 구간 통합 mask
    crisis_dates = []
    for _, (start, end, _) in CRISIS_PERIODS.items():
        crisis_dates.extend(pd.date_range(start, end, freq="B"))
    crisis_set = set(pd.to_datetime(crisis_dates))

    # 평시 mask
    calm_dates = []
    for _, (start, end) in CALM_PERIODS.items():
        calm_dates.extend(pd.date_range(start, end, freq="B"))
    calm_set = set(pd.to_datetime(calm_dates))

    rows = []
    for sig in SIGNALS:
        if sig not in features.columns:
            continue
        thr, direction = current_thresholds[sig]
        s = features[sig].dropna()

        if direction == "gt":
            flagged = s > thr
        else:
            flagged = s < thr

        # 전체 발동 비율
        overall_rate = flagged.mean()

        # 위기 구간 내 발동 비율 (recall)
        in_crisis = flagged.index.isin(crisis_set)
        crisis_flagged = flagged[in_crisis]
        recall = crisis_flagged.mean() if len(crisis_flagged) > 0 else np.nan

        # 평시 구간 내 발동 비율 (false positive rate)
        in_calm = flagged.index.isin(calm_set)
        calm_flagged = flagged[in_calm]
        fpr = calm_flagged.mean() if len(calm_flagged) > 0 else np.nan

        # 발동 시 위기일 확률 (precision)
        if flagged.sum() > 0:
            flagged_dates = flagged[flagged].index
            in_crisis_when_flagged = flagged_dates.isin(crisis_set).mean()
            precision = in_crisis_when_flagged
        else:
            precision = np.nan

        rows.append({
            "signal":       sig,
            "threshold":    thr,
            "direction":    direction,
            "overall_rate": overall_rate,
            "recall":       recall,   # 위기 시 발동률
            "precision":    precision, # 발동 시 위기일 확률
            "fpr":          fpr,      # 평시 오작동률
        })

    df = pd.DataFrame(rows)
    print(f"\n  {'신호':<20} {'임계값':>8} {'전체%':>8} {'위기시%':>8} {'평시%':>8} {'정밀도':>8}")
    print("  " + "-" * 75)
    for _, row in df.iterrows():
        print(f"  {row['signal']:<20} {row['threshold']:>8.2f} "
              f"{row['overall_rate']*100:>7.1f}% "
              f"{row['recall']*100:>7.1f}% "
              f"{row['fpr']*100:>7.1f}% "
              f"{row['precision']*100:>7.1f}%")

    df.to_csv(f"{OUT_DIR}/02_signal_hit_rate.csv", index=False)
    print(f"\n  → {OUT_DIR}/02_signal_hit_rate.csv")
    return df


# ==========================================
# 분석 3. 신호 raw 값 분위수 분포 (전체 기간)
# ==========================================
def analyze_signal_distribution(features):
    """각 신호의 분포 — 임계값 재설정의 객관적 근거."""
    print("\n" + "=" * 90)
    print("[분석 3] 신호 raw 값 전체 분포 (임계값 재설정용)")
    print("=" * 90)

    rows = []
    for sig in SIGNALS:
        if sig not in features.columns:
            continue
        s = features[sig].dropna()
        rows.append({
            "signal":  sig,
            "n":       len(s),
            "min":     s.min(),
            "p05":     s.quantile(0.05),
            "p10":     s.quantile(0.10),
            "p25":     s.quantile(0.25),
            "p50":     s.median(),
            "p75":     s.quantile(0.75),
            "p90":     s.quantile(0.90),
            "p95":     s.quantile(0.95),
            "p99":     s.quantile(0.99),
            "max":     s.max(),
            "mean":    s.mean(),
            "std":     s.std(),
        })
    df = pd.DataFrame(rows)
    print(f"\n  {'신호':<20} {'p10':>8} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}")
    print("  " + "-" * 70)
    for _, row in df.iterrows():
        print(f"  {row['signal']:<20} "
              f"{row['p10']:>8.2f} {row['p50']:>8.2f} "
              f"{row['p90']:>8.2f} {row['p95']:>8.2f} {row['p99']:>8.2f}")

    df.to_csv(f"{OUT_DIR}/03_signal_distribution.csv", index=False)
    print(f"\n  → {OUT_DIR}/03_signal_distribution.csv")
    return df


# ==========================================
# 분석 4. 신호 간 상관관계
# ==========================================
def analyze_signal_correlation(features):
    """신호가 서로 중복되는가, 보완되는가?"""
    print("\n" + "=" * 90)
    print("[분석 4] 신호 간 상관관계 (방향 부호 보정)")
    print("=" * 90)

    # 모든 신호를 "클수록 위기" 방향으로 통일
    aligned = pd.DataFrame(index=features.index)
    for sig in SIGNALS:
        if sig not in features.columns:
            continue
        if SIGNAL_DIRECTION[sig] == "gt":
            aligned[sig] = features[sig]
        else:
            aligned[sig] = -features[sig]  # 부호 뒤집기

    corr = aligned.corr()
    print(f"\n  Pearson 상관계수 (방향 정렬 후, +1 = 같이 위기 신호):")
    print(corr.round(3).to_string())

    corr.to_csv(f"{OUT_DIR}/04_signal_correlation.csv")
    print(f"\n  → {OUT_DIR}/04_signal_correlation.csv")
    return corr


# ==========================================
# 분석 5. 위기별 신호 패턴 (어떤 위기에 어떤 신호?)
# ==========================================
def analyze_crisis_patterns(features, current_thresholds):
    """위기 유형별로 어떤 신호가 발동했는가?"""
    print("\n" + "=" * 90)
    print("[분석 5] 위기 유형별 신호 발동 패턴")
    print("=" * 90)

    rows = []
    for crisis_name, (start, end, ctype) in CRISIS_PERIODS.items():
        sub = features.loc[start:end]
        if len(sub) == 0:
            continue

        row = {"crisis": crisis_name, "type": ctype, "n_days": len(sub)}
        for sig in SIGNALS:
            if sig not in sub.columns:
                row[sig] = np.nan
                continue
            thr, direction = current_thresholds[sig]
            s = sub[sig].dropna()
            if direction == "gt":
                hit_rate = (s > thr).mean()
            else:
                hit_rate = (s < thr).mean()
            row[sig] = hit_rate
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  현재 임계값 기준 위기 시 발동률 (%):")
    print(f"  {'위기':<25} {'유형':<10}", end="")
    for sig in SIGNALS:
        print(f"{sig[:12]:>14}", end="")
    print()
    print("  " + "-" * 100)
    for _, row in df.iterrows():
        print(f"  {row['crisis']:<25} {row['type']:<10}", end="")
        for sig in SIGNALS:
            v = row[sig] * 100 if not pd.isna(row[sig]) else 0
            print(f"{v:>13.1f}%", end="")
        print()

    df.to_csv(f"{OUT_DIR}/05_crisis_patterns.csv", index=False)
    print(f"\n  → {OUT_DIR}/05_crisis_patterns.csv")
    return df


# ==========================================
# 분석 6. 위기 시점 점수 분해
# ==========================================
def analyze_score_decomposition(features, current_thresholds, current_weights):
    """위기 정점 시점의 점수 분해 — 어떤 신호가 점수의 몇 % 기여?"""
    print("\n" + "=" * 90)
    print("[분석 6] 위기 정점 시점 점수 분해 (현재 baseline 가중치 기준)")
    print("=" * 90)

    rows = []
    for crisis_name, (start, end, ctype) in CRISIS_PERIODS.items():
        sub = features.loc[start:end]
        if len(sub) == 0:
            continue

        # 각 시점 점수 계산
        score = pd.Series(0.0, index=sub.index)
        contributions = {sig: pd.Series(0.0, index=sub.index) for sig in SIGNALS}
        for sig in SIGNALS:
            if sig not in sub.columns:
                continue
            thr, direction = current_thresholds[sig]
            if direction == "gt":
                flag = (sub[sig] > thr).astype(int)
            else:
                flag = (sub[sig] < thr).astype(int)
            w = current_weights[sig]
            contributions[sig] = flag * w
            score = score + flag * w

        # 점수 최고점 시점 찾기
        if score.max() == 0:
            continue
        peak_date = score.idxmax()
        peak_score = score.max()

        row = {
            "crisis":   crisis_name,
            "peak_date": peak_date.strftime("%Y-%m-%d"),
            "peak_score": peak_score,
        }
        for sig in SIGNALS:
            row[f"contrib_{sig}"] = contributions[sig].loc[peak_date]
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  위기 정점 시점 (현재 5점 만점, 임계값 3.0 = CRISIS):")
    print(f"  {'위기':<25} {'정점일':<12} {'점수':>6}  {'기여':<40}")
    print("  " + "-" * 90)
    for _, row in df.iterrows():
        contribs = []
        for sig in SIGNALS:
            v = row[f"contrib_{sig}"]
            if v > 0:
                contribs.append(f"{sig.replace('dist_ma200_','ma_')[:8]}({v:.1f})")
        print(f"  {row['crisis']:<25} {row['peak_date']:<12} "
              f"{row['peak_score']:>6.1f}  {', '.join(contribs)}")

    df.to_csv(f"{OUT_DIR}/06_score_decomposition.csv", index=False)
    print(f"\n  → {OUT_DIR}/06_score_decomposition.csv")
    return df


# ==========================================
# 메인
# ==========================================
def main():
    ensure_outdir()
    print("=" * 90)
    print("UCHIDA V3 - Level 1 재설계용 데이터 분석")
    print("출력: " + OUT_DIR + "/")
    print("=" * 90)

    # 데이터 로드
    print("\n[데이터 로드...]")
    df = load_all(start=DATES.train_start, use_cache=True)
    features = build_features(df)
    BASELINE_SIGNALS = SIGNALS
    available = [c for c in BASELINE_SIGNALS if c in features.columns]
    features = features.dropna(subset=available)
    print(f"  기간: {features.index[0].date()} ~ {features.index[-1].date()}")

    # 현재 baseline 임계값 / 가중치
    current_thresholds = {
        "cpi_z":          (2.3,    "gt"),
        "credit_spread":  (3.5,    "gt"),
        "t10y2y":         (0.0,    "lt"),
        "vix":            (22.0,   "gt"),
        "dist_ma200_QQQ": (-0.03,  "lt"),
    }
    current_weights = {
        "cpi_z":          1.5,
        "credit_spread":  1.5,
        "t10y2y":         1.0,
        "vix":            0.5,
        "dist_ma200_QQQ": 0.5,
    }

    # 분석 실행
    analyze_crisis_signals(features)
    analyze_signal_hit_rate(features, current_thresholds)
    analyze_signal_distribution(features)
    analyze_signal_correlation(features)
    analyze_crisis_patterns(features, current_thresholds)
    analyze_score_decomposition(features, current_thresholds, current_weights)

    print("\n" + "=" * 90)
    print("[완료]")
    print(f"  총 6개 CSV 파일이 {OUT_DIR}/ 에 저장됨")
    print("  Opus 세션에 이 CSV들을 함께 첨부할 것")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ 오류: {e}")
        traceback.print_exc()
