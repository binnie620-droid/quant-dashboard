"""
UCHIDA V3 - Triple-Barrier Labeling

López de Prado (2018), Advances in Financial Machine Learning, Ch.3.

각 시점 t에서 향후 T일 동안 기준 자산(SPY)의 누적 log return 경로를 추적,
다음 셋 중 어느 barrier에 먼저 닿는지로 라벨링:
    상단 +k·σ_t  먼저 도달 → ATTACK  (2)
    하단 -k·σ_t  먼저 도달 → CRISIS  (0)
    T일 만료              → DEFENSE (1)

[변동성 스케일링 핵심]
σ_t는 시점 t의 60일 realized volatility (연율화) → barrier가 시장 상황에 맞춰
자동으로 넓어지거나 좁아짐. 평온기엔 좁은 barrier, 변동기엔 넓은 barrier.

[Look-ahead 처리]
- σ_t는 [t-60, t] 데이터만 사용 (안전)
- 라벨 자체는 [t+1, t+T] 미래 정보 사용 (정상: 라벨은 정답, 학습 후엔 미래 안 봄)
- 학습 시 시점 t의 feature와 시점 t의 라벨을 매칭 → feature는 [..., t]만 사용,
  라벨은 미래 정보 → look-ahead bias 없음

[클래스 인코딩]
    0 = CRISIS  (방어 자산 비중 ↑)
    1 = DEFENSE (중립)
    2 = ATTACK  (성장 자산 비중 ↑)
숫자 순서가 "방어→공격"이라 모델 출력 해석 직관적.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================
# 상수
# ==========================================
# 클래스 라벨
CRISIS  = 0
DEFENSE = 1
ATTACK  = 2

CLASS_NAMES = {CRISIS: "CRISIS", DEFENSE: "DEFENSE", ATTACK: "ATTACK"}

# 기본 파라미터
DEFAULT_BASELINE = "SPY"
DEFAULT_VOL_WINDOW = 60     # σ 계산 윈도우 (영업일)
DEFAULT_K = 1.0             # barrier 배수
DEFAULT_T = 21              # 시간 윈도우 (영업일)
TRADING_DAYS_PER_YEAR = 252


# ==========================================
# 1. 변동성 계산
# ==========================================
def compute_volatility(
    prices: pd.Series,
    window: int = DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """
    Annualized realized volatility from daily log returns.

    σ_t = std(log returns over [t-window+1, t]) × √252

    Parameters
    ----------
    prices : pd.Series
        기준 자산 가격 (Adjusted Close)
    window : int
        rolling window 길이 (영업일)

    Returns
    -------
    pd.Series
        연율화된 변동성. 앞쪽 window-1 시점은 NaN.
    """
    daily_log_ret = np.log(prices / prices.shift(1))
    sigma = daily_log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sigma


# ==========================================
# 2. Triple-Barrier 라벨링 (핵심)
# ==========================================
def triple_barrier_labels(
    prices: pd.Series,
    sigma: pd.Series,
    k: float = DEFAULT_K,
    horizon: int = DEFAULT_T,
) -> pd.DataFrame:
    """
    각 시점에 Triple-Barrier 라벨 부여.

    Parameters
    ----------
    prices : pd.Series
        기준 자산 가격 (log return 계산용)
    sigma : pd.Series
        연율화된 변동성 (compute_volatility 출력)
    k : float
        barrier 배수 (k × σ × √(T/252))
    horizon : int
        시간 윈도우 T (영업일)

    Returns
    -------
    pd.DataFrame
        index: 날짜 (라벨 부여 가능한 시점만)
        columns:
            - label: int (0=CRISIS, 1=DEFENSE, 2=ATTACK)
            - barrier_up: float (그 시점의 +barrier 값, log return 단위)
            - barrier_dn: float (그 시점의 -barrier 값)
            - touch_day: int (몇 일째에 barrier touch했는지; DEFENSE면 horizon)
            - realized_ret: float (라벨 시점의 실제 누적 수익률)
    """
    log_prices = np.log(prices)

    n = len(prices)
    labels = np.full(n, np.nan)
    barriers_up = np.full(n, np.nan)
    barriers_dn = np.full(n, np.nan)
    touch_days = np.full(n, np.nan)
    realized_rets = np.full(n, np.nan)

    # 시간 스케일링 (T일 기간으로 환산)
    time_scale = np.sqrt(horizon / TRADING_DAYS_PER_YEAR)

    # 라벨 부여 가능한 시점: σ가 유효 & 미래 T일 데이터 존재
    valid_start = sigma.first_valid_index()
    if valid_start is None:
        return pd.DataFrame(columns=["label", "barrier_up", "barrier_dn",
                                      "touch_day", "realized_ret"])
    start_idx = prices.index.get_loc(valid_start)

    for i in range(start_idx, n - horizon):
        sigma_t = sigma.iloc[i]
        if np.isnan(sigma_t):
            continue

        # 그 시점의 barrier 폭 (log return 단위)
        b = k * sigma_t * time_scale
        barriers_up[i] = b
        barriers_dn[i] = -b

        # 향후 horizon일의 누적 log return 경로
        # path[j] = log(P_{t+j} / P_t), j = 1, ..., horizon
        future_log_prices = log_prices.iloc[i+1 : i+1+horizon].values
        path = future_log_prices - log_prices.iloc[i]

        # 어느 barrier에 먼저 닿는지 탐색
        up_hits = np.where(path >= b)[0]
        dn_hits = np.where(path <= -b)[0]

        up_first = up_hits[0] if len(up_hits) > 0 else np.inf
        dn_first = dn_hits[0] if len(dn_hits) > 0 else np.inf

        if up_first < dn_first:
            labels[i] = ATTACK
            touch_days[i] = up_first + 1   # 1-indexed (며칠째)
            realized_rets[i] = path[int(up_first)]
        elif dn_first < up_first:
            labels[i] = CRISIS
            touch_days[i] = dn_first + 1
            realized_rets[i] = path[int(dn_first)]
        else:
            # 시간 만료 (둘 다 inf 또는 동시 도달의 경우 시간 우선)
            labels[i] = DEFENSE
            touch_days[i] = horizon
            realized_rets[i] = path[-1]

    result = pd.DataFrame({
        "label": labels,
        "barrier_up": barriers_up,
        "barrier_dn": barriers_dn,
        "touch_day": touch_days,
        "realized_ret": realized_rets,
    }, index=prices.index)

    # NaN인 행 제거 (라벨 부여 못 한 시점)
    result = result.dropna(subset=["label"])
    result["label"] = result["label"].astype(int)
    result["touch_day"] = result["touch_day"].astype(int)

    return result


# ==========================================
# 3. 메인 인터페이스
# ==========================================
def label_market_regimes(
    df_prices: pd.DataFrame,
    baseline: str = DEFAULT_BASELINE,
    k: float = DEFAULT_K,
    horizon: int = DEFAULT_T,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> pd.DataFrame:
    """
    가격 DataFrame에서 기준 자산을 골라 Triple-Barrier 라벨링.

    Parameters
    ----------
    df_prices : pd.DataFrame
        가격 DataFrame (loader.load_all() 출력 또는 가격만 슬라이싱한 것)
        baseline 컬럼이 반드시 존재해야 함.
    baseline : str
        기준 자산 티커 (기본 SPY)
    k : float
        barrier 배수
    horizon : int
        시간 윈도우 T (영업일)
    vol_window : int
        σ 계산 rolling window

    Returns
    -------
    pd.DataFrame
        triple_barrier_labels의 출력과 동일.

    Raises
    ------
    KeyError
        baseline 컬럼이 df_prices에 없을 때.
    """
    if baseline not in df_prices.columns:
        raise KeyError(f"기준 자산 '{baseline}'이 가격 DataFrame에 없습니다. "
                       f"사용 가능 컬럼: {list(df_prices.columns)}")

    prices = df_prices[baseline]
    sigma = compute_volatility(prices, window=vol_window)
    return triple_barrier_labels(prices, sigma, k=k, horizon=horizon)


# ==========================================
# 4. 분포 요약 유틸리티
# ==========================================
def label_distribution(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    라벨 분포를 클래스별로 요약.

    Returns
    -------
    pd.DataFrame
        index: 클래스 이름
        columns: count, ratio, avg_touch_day, avg_realized_ret
    """
    summary = []
    for cls, name in CLASS_NAMES.items():
        sub = labels_df[labels_df["label"] == cls]
        summary.append({
            "class": name,
            "count": len(sub),
            "ratio": len(sub) / len(labels_df) if len(labels_df) > 0 else 0,
            "avg_touch_day": sub["touch_day"].mean() if len(sub) > 0 else np.nan,
            "avg_realized_ret": sub["realized_ret"].mean() if len(sub) > 0 else np.nan,
        })
    return pd.DataFrame(summary).set_index("class")


# ==========================================
# 5. 자체 검증 (수동 실행 시)
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("labels/labeler.py 단독 실행 테스트 (시뮬레이션 데이터)")
    print("=" * 70)

    # 시뮬레이션 가격: 평온기 + 폭락기 + 회복기 (라벨 분포 검증용)
    np.random.seed(42)
    n_days = 2000
    idx = pd.date_range("2015-01-01", periods=n_days, freq="B")

    # GBM 시뮬레이션: 일부 구간에 폭락 삽입
    daily_ret = np.random.normal(0.0005, 0.012, n_days)  # 평시: 평균 0.05%, std 1.2%
    # 폭락 구간 인위 삽입 (인덱스 800~830)
    daily_ret[800:830] = np.random.normal(-0.02, 0.03, 30)
    # 고변동성 구간 (1500~1550)
    daily_ret[1500:1550] = np.random.normal(0.0, 0.025, 50)

    prices = pd.Series(100 * np.exp(np.cumsum(daily_ret)), index=idx, name="SPY")
    df = pd.DataFrame({"SPY": prices})

    # 라벨링 실행
    labels_df = label_market_regimes(df, baseline="SPY", k=1.0, horizon=21)

    # ----- 결과 출력 -----
    print(f"\n[데이터 기간] {prices.index[0].date()} ~ {prices.index[-1].date()} ({n_days} 영업일)")
    print(f"[라벨 부여 시점] {len(labels_df)}개 (앞쪽 vol_window, 뒤쪽 horizon 제외)")

    print(f"\n[라벨 분포]")
    dist = label_distribution(labels_df)
    print(dist.round(4).to_string())

    # ----- σ 변동 확인 (변동성 스케일링 검증) -----
    sigma = compute_volatility(prices, window=60)
    print(f"\n[변동성 σ 변동 확인]")
    print(f"  평시 평균:    {sigma.iloc[100:500].mean():.4f}")
    print(f"  폭락기 평균:  {sigma.iloc[820:870].mean():.4f}  (커야 정상)")

    # ----- 라벨 시계열 샘플 -----
    print(f"\n[샘플: 폭락 구간 전후 라벨 (인덱스 790~815일 근처)]")
    crash_start = idx[790]
    crash_end = idx[815]
    around_crash = labels_df.loc[crash_start:crash_end]
    print(around_crash[["label", "barrier_up", "realized_ret", "touch_day"]].round(4).to_string())

    print("\n✓ labeler.py 검증 완료")
