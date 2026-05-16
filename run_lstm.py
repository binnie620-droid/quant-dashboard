"""
UCHIDA V3 - LSTM Walk-Forward 학습 및 백테스트

[실행 순서]
1. 데이터 + features + labels 준비
2. Walk-forward LSTM 학습 (각 fold마다 재학습)
3. 전체 기간 비중 생성
4. 백테스트 실행
5. Baseline과 성과 비교

[실행 방법]
    cd uchida_v3
    python run_lstm.py

[필요 패키지]
    pip install torch scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date

import torch

from config import DATES, INITIAL_CAPITAL
from data.loader import load_all, get_close_prices, get_open_prices
from data.features import build_features
from labels.labeler import label_market_regimes
from models.baseline import predict_weights as baseline_weights
from models.lstm import LSTMConfig, LSTMTrainer, probs_to_weights, run_walk_forward
from backtest.engine import run_backtest
from backtest.metrics import compare_strategies
from backtest.isa_simulator import compare_isa_scenarios


# ==========================================
# 설정
# ==========================================
BACKTEST_START = date(2015, 1, 1)
BACKTEST_END   = date.today()
INITIAL        = float(INITIAL_CAPITAL)

# LSTM 설정 (작은 값으로 시작 — 데이터 적어서 큰 모델 overfitting)
LSTM_CFG = LSTMConfig(
    seq_len      = 60,
    hidden1      = 64,
    hidden2      = 32,
    dense_dim    = 16,
    dropout      = 0.3,
    lr           = 1e-3,
    batch_size   = 32,
    max_epochs   = 100,
    es_patience  = 10,
    lr_patience  = 5,
    val_ratio    = 0.2,
)


# ==========================================
# 1. 데이터 준비
# ==========================================
def prepare_data():
    print("\n" + "="*60)
    print("STEP 1. 데이터 + Feature + Label 준비")
    print("="*60)

    # 데이터 로드 (캐시 우선)
    df = load_all(start=DATES.train_start, use_cache=True)
    print(f"  기간: {df.index[0].date()} ~ {df.index[-1].date()}")

    # Feature 생성
    # LSTM은 147개 전체 feature를 입력으로 사용하므로 전체 dropna 필요.
    # (baseline과 달리 2008 데이터 일부 손실은 불가피 — 긴 lookback feature 때문)
    # Baseline 단독 백테스트는 run_backtest.py 사용 권장.
    features = build_features(df).dropna()
    print(f"  Feature: {len(features.columns)}개, {len(features)}일")

    # 라벨 생성
    close_prices = get_close_prices(df)
    labels_df = label_market_regimes(
        close_prices, baseline="QQQ", k=1.0, horizon=21  # SPY는 ASSETS에 미포함, QQQ 대체
    )
    labels = labels_df["label"]

    # 공통 날짜 기준 정렬
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels   = labels.loc[common]

    # 라벨 분포
    dist = labels.value_counts().sort_index()
    names = {0: "CRISIS", 1: "DEFENSE", 2: "ATTACK"}
    print(f"  라벨 분포:")
    for cls, cnt in dist.items():
        print(f"    {names[cls]:10s}: {cnt}일 ({cnt/len(labels)*100:.1f}%)")

    return df, features, labels


# ==========================================
# 2. LSTM Walk-Forward 학습
# ==========================================
def train_lstm(features, labels):
    print("\n" + "="*60)
    print("STEP 2. LSTM Walk-Forward 학습")
    print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*60)

    weights_df, fold_metrics = run_walk_forward(
        features=features,
        labels=labels,
        cfg=LSTM_CFG,
        verbose=True,
    )

    # Fold 성능 요약
    print(f"\n[Walk-Forward 성능 요약]")
    fm_df = pd.DataFrame(fold_metrics)
    if len(fm_df) > 0:
        print(f"  평균 정확도: {fm_df['accuracy'].mean():.3f}")
        print(f"  최소 정확도: {fm_df['accuracy'].min():.3f}")
        print(f"  최대 정확도: {fm_df['accuracy'].max():.3f}")
        print(f"  Fold 수: {len(fm_df)}")

    print(f"\n  예측 비중 생성 완료: {len(weights_df)}일")

    return weights_df


# ==========================================
# 3. 백테스트
# ==========================================
def run_strategy_backtest(df, target_weights, label="LSTM"):
    print(f"\n  [{label}] 백테스트 실행 중...")

    close_prices = get_close_prices(df)
    open_prices  = get_open_prices(df)

    # 백테스트 기간
    close_bt = close_prices.loc[BACKTEST_START:]
    open_bt  = open_prices.loc[BACKTEST_START:]

    # 공통 날짜
    common_idx = target_weights.index.intersection(close_bt.index)
    if len(common_idx) == 0:
        print(f"  ⚠ [{label}] 공통 날짜 없음")
        return None, None

    tw = target_weights.loc[common_idx]
    cb = close_bt.loc[common_idx]
    ob = open_bt.loc[common_idx]

    # 자산 정렬 및 정규화
    common_assets = [a for a in tw.columns if a in cb.columns]
    tw = tw[common_assets]
    cb = cb[common_assets]
    ob = ob[common_assets]
    row_sums = tw.sum(axis=1).replace(0, 1)
    tw = tw.div(row_sums, axis=0)

    result = run_backtest(
        prices_close=cb,
        prices_open=ob,
        target_weights=tw,
    )
    pv = result.portfolio_value * INITIAL

    print(f"  [{label}] 거래 횟수: {len(result.trades)}회")
    return result, pv


# ==========================================
# 4. 비교 리포트
# ==========================================
def report(results: dict, pv_dict: dict):
    print("\n" + "="*70)
    print("STEP 4. 성과 비교 리포트")
    print("="*70)

    strategies = {name: r.daily_returns for name, r in results.items()}
    metrics_df = compare_strategies(strategies, risk_free_rate=0.03)

    names = list(results.keys())
    col_w = 16
    header = f"{'':20s}" + "".join(f"{n:>{col_w}s}" for n in names)
    print(f"\n{header}")
    print("-" * (20 + col_w * len(names)))

    rows = [
        ("최종 자산",    "pretax_final",  False),
        ("CAGR (%)",     "cagr",          True),
        ("MDD (%)",      "mdd",           True),
        ("Sharpe",       "sharpe",        False),
        ("Sortino",      "sortino",       False),
        ("Calmar",       "calmar",        False),
        ("CE α=1 (%)",   "ce_a1.0",       True),
    ]

    def fmt(val, is_pct):
        if pd.isna(val): return "N/A"
        return f"{val*100:.2f}%" if is_pct else f"{val:.3f}"

    for label, col, is_pct in rows:
        if col == "pretax_final":
            line = f"  {label:18s}" + "".join(f"{pv_dict[n].iloc[-1]:>{col_w},.0f}" for n in names)
        elif col in metrics_df.columns:
            line = f"  {label:18s}" + "".join(f"{fmt(metrics_df.loc[n, col], is_pct):>{col_w}s}" for n in names)
        else:
            continue
        print(line)

    # Entropic sensitivity
    print(f"\n[Entropic Risk Sensitivity (α 변화) — 작을수록 좋음]")
    hdr = f"  {'α':>4s} |" + "".join(f"{n:>{col_w}s}" for n in names)
    print(hdr)
    print(f"  {'-'*(6 + col_w * len(names))}")
    for a in [0.5, 1.0, 2.0, 5.0]:
        col = f"entropic_a{a}"
        if col in metrics_df.columns:
            line = f"  {a:>4.1f} |" + "".join(f"{metrics_df.loc[n, col]*100:>{col_w-1}.2f}%" for n in names)
            print(line)

    # 승패 판정
    print(f"\n[LSTM (월간) vs Baseline 판정]")
    lstm_key = "LSTM (월간)" if "LSTM (월간)" in names else "LSTM (일간)"
    if lstm_key in names and "Baseline" in names:
        checks = {
            "CAGR":   metrics_df.loc[lstm_key,"cagr"]    > metrics_df.loc["Baseline","cagr"],
            "MDD":    abs(metrics_df.loc[lstm_key,"mdd"]) < abs(metrics_df.loc["Baseline","mdd"]),
            "Sharpe": metrics_df.loc[lstm_key,"sharpe"]  > metrics_df.loc["Baseline","sharpe"],
            "Calmar": metrics_df.loc[lstm_key,"calmar"]  > metrics_df.loc["Baseline","calmar"],
        }
        wins = sum(checks.values())
        for k, v in checks.items():
            print(f"  {k:8s}: {'LSTM ✓' if v else 'Baseline ✓'}")
        print(f"\n  최종: LSTM (월간)이 {wins}/4 지표에서 우세")
        if wins >= 3:
            print("  → LSTM (월간) 채택 권장")
        else:
            print("  → Baseline 유지 권장")

    # ISA 세후
    print(f"\n[ISA 세후]")
    isa_df = compare_isa_scenarios(pv_dict, initial_deposit=INITIAL)
    for name in isa_df.index:
        row = isa_df.loc[name]
        print(f"  {name}: 세후 {row['ISA 세후 최종 자산 (원)']:>15,.0f}원  "
              f"(CAGR {row['ISA 세후 CAGR (%)']:.2f}%)")

    print(f"\n{'='*70}")
    return metrics_df


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("UCHIDA V3 - LSTM Walk-Forward 학습 및 백테스트")
    print(f"초기 자본: {INITIAL:,.0f}원")
    print("="*60)

    try:
        # 1. 데이터 준비
        df, features, labels = prepare_data()

        # 2. LSTM 학습
        lstm_weights = train_lstm(features, labels)

        # 3. Baseline 비중 (비교용)
        print("\n" + "="*60)
        print("STEP 3. 백테스트 실행")
        print("="*60)

        feat_bt = features.loc[BACKTEST_START:]
        baseline_w = baseline_weights(feat_bt)

        results = {}
        pv_dict = {}

        # Baseline
        r, pv = run_strategy_backtest(df, baseline_w, "Baseline")
        if r is not None:
            results["Baseline"] = r
            pv_dict["Baseline"] = pv

        # LSTM 비중을 월간으로 resample (거래 횟수 감소 목적)
        # 매월 마지막 영업일의 비중만 사용 → 나머지 날은 forward fill
        lstm_weights_monthly = lstm_weights.resample("BME").last().reindex(
            lstm_weights.index
        ).ffill()
        # 앞쪽 NaN 제거
        lstm_weights_monthly = lstm_weights_monthly.dropna()

        # LSTM (월간)
        r, pv = run_strategy_backtest(df, lstm_weights_monthly, "LSTM (월간)")
        if r is not None:
            results["LSTM (월간)"] = r
            pv_dict["LSTM (월간)"] = pv

        # LSTM (일간, 기존)
        r, pv = run_strategy_backtest(df, lstm_weights, "LSTM (일간)")
        if r is not None:
            results["LSTM (일간)"] = r
            pv_dict["LSTM (일간)"] = pv

        # QQQ B&H
        close_prices = get_close_prices(df)
        open_prices  = get_open_prices(df)
        close_bt = close_prices.loc[BACKTEST_START:]
        open_bt  = open_prices.loc[BACKTEST_START:]
        common   = close_bt.index.intersection(open_bt.index)
        close_bt = close_bt.loc[common]
        open_bt  = open_bt.loc[common]
        common_assets = [a for a in close_bt.columns if a in open_bt.columns]
        qqq_w = pd.DataFrame(0.0, index=common, columns=common_assets)
        if "QQQ" in common_assets:
            qqq_w["QQQ"] = 1.0
        r_qqq = run_backtest(close_bt, open_bt, qqq_w, cost_one_way=0.0, tolerance_band=0.0)
        results["QQQ B&H"] = r_qqq
        pv_dict["QQQ B&H"] = r_qqq.portfolio_value * INITIAL

        # 4. 리포트
        metrics = report(results, pv_dict)

    except Exception as e:
        import traceback
        print(f"\n❌ 오류: {e}")
        traceback.print_exc()
