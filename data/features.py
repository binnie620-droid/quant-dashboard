"""
UCHIDA V3 - Feature Engineering

loader.py가 반환한 raw DataFrame에서 ML 모델 입력용 feature를 생성.

[Look-ahead bias 방지 원칙]
- t일의 feature는 반드시 [t-window+1, t] 데이터만 사용
- rolling() 연산은 자동으로 안전
- CPI는 45일 발표 지연 적용 (shift(CPI_PUBLICATION_LAG_DAYS))
- backward fill 절대 사용 금지

[Feature 군]
1. 가격 기반: log returns, realized vol, momentum
2. 기술적 지표: RSI, MACD, Bollinger Band, MA 거리
3. 매크로: CPI, yield curve, credit spread, VIX
4. 자산간 관계: rolling correlation, ratio
5. 달러 인덱스: DXY 수준 및 변화율
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS, CPI_PUBLICATION_LAG_DAYS


# ==========================================
# 0. 상수 정의
# ==========================================
# 가격 기반 feature에 사용할 미국 티커 목록
PRICE_TICKERS = [a.ticker_us for a in ASSETS.values()]

# 기술적 지표 rolling window (단위: 영업일)
WIN_SHORT  = 20    # 1개월
WIN_MID    = 60    # 3개월
WIN_LONG   = 200   # 약 10개월 (MA200)

# 모멘텀 window (단위: 영업일)
# 12-1 모멘텀: Jegadeesh & Titman (1993) 표준
MOM_12_1 = (252 - 21, 21)   # (start_lag, skip_lag)
MOM_6_1  = (126 - 21, 21)
MOM_3_1  = (63  - 21, 21)

# RSI window
RSI_WIN = 14

# MACD 파라미터
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIGN  = 9

# Bollinger Band window
BOLL_WIN = 20

# 자산간 correlation window
CORR_WIN = 60


# ==========================================
# 1. 가격 기반 Feature
# ==========================================
def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns (1/5/20/60일).
    log(P_t / P_{t-n}) 형태 → 정규분포에 가깝고 additivity 장점.
    """
    out = {}
    for t in [1, 5, 20, 60]:
        out[f"ret_{t}d"] = np.log(prices / prices.shift(t))
    return pd.concat(out, axis=1)


def _realized_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Realized volatility (20/60일 rolling std of daily log returns).
    연율화 (× sqrt(252)).
    """
    daily_ret = np.log(prices / prices.shift(1))
    out = {}
    for w in [WIN_SHORT, WIN_MID]:
        out[f"rvol_{w}d"] = daily_ret.rolling(w).std() * np.sqrt(252)
    return pd.concat(out, axis=1)


def _momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price momentum: 12-1, 6-1, 3-1 month.
    skip 1개월(21일)은 단기 반전 효과 제거.
    근거: Jegadeesh & Titman (1993), Journal of Finance.
    """
    out = {}
    for name, (total_lag, skip_lag) in [
        ("mom_12_1", MOM_12_1),
        ("mom_6_1",  MOM_6_1),
        ("mom_3_1",  MOM_3_1),
    ]:
        # t-total_lag 시점 가격 대비 t-skip_lag 시점 가격의 수익률
        out[name] = np.log(prices.shift(skip_lag) / prices.shift(total_lag))
    return pd.concat(out, axis=1)


# ==========================================
# 2. 기술적 지표
# ==========================================
def _rsi(prices: pd.DataFrame, window: int = RSI_WIN) -> pd.DataFrame:
    """
    RSI (Relative Strength Index).
    0~100 범위. 70 이상 과매수, 30 이하 과매도.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(columns={c: f"rsi_{c}" for c in rsi.columns})


def _macd(prices: pd.DataFrame) -> pd.DataFrame:
    """
    MACD = EMA(12) - EMA(26).
    Signal = EMA(9) of MACD.
    Histogram = MACD - Signal.
    정규화: prices로 나눠서 자산간 비교 가능하게.
    """
    ema_fast = prices.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = prices.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / prices  # 정규화
    signal = macd_line.ewm(span=MACD_SIGN, adjust=False).mean()
    hist = macd_line - signal

    out = pd.concat([
        macd_line.rename(columns={c: f"macd_{c}" for c in macd_line.columns}),
        hist.rename(columns={c: f"macd_hist_{c}" for c in hist.columns}),
    ], axis=1)
    return out


def _bollinger(prices: pd.DataFrame, window: int = BOLL_WIN) -> pd.DataFrame:
    """
    Bollinger Band %B = (Price - Lower) / (Upper - Lower).
    0~1 범위 (밴드 내 상대 위치). 1 초과 = 과매수, 0 미만 = 과매도.
    """
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    pct_b = (prices - lower) / (upper - lower).replace(0, np.nan)
    return pct_b.rename(columns={c: f"boll_b_{c}" for c in pct_b.columns})


def _bollinger_width(prices: pd.DataFrame, window: int = BOLL_WIN) -> pd.DataFrame:
    """
    Bollinger Band Width = (Upper - Lower) / MA.
    낮을수록 변동성 압축 (박스권 신호).
    정규화: MA로 나눠서 가격 수준 무관.
    """
    ma  = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return width.rename(columns={c: f"boll_w_{c}" for c in width.columns})


def _adx(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ADX (Average Directional Index, Wilder 1978).
    추세 강도 지표. 방향 무관. 0~100 범위.
    < 20: 박스권 (추세 없음)
    > 25: 추세 있음

    근사 계산 (고가/저가 없이 종가만 사용):
    - True Range ≈ |Close(t) - Close(t-1)|
    - DX = |+DI - -DI| / (+DI + -DI)
    - ADX = EWM(DX, window)
    """
    diff = prices.diff().abs()
    atr  = diff.ewm(span=window, adjust=False).mean()

    # +DM, -DM 근사 (종가 기반)
    up   = prices.diff().clip(lower=0)
    down = (-prices.diff()).clip(lower=0)

    plus_di  = up.ewm(span=window, adjust=False).mean()   / atr.replace(0, np.nan)
    minus_di = down.ewm(span=window, adjust=False).mean() / atr.replace(0, np.nan)

    dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(span=window, adjust=False).mean()
    return adx.rename(columns={c: f"adx_{c}" for c in adx.columns})


def _ma_distance(prices: pd.DataFrame) -> pd.DataFrame:
    """
    현재 가격과 MA200, MA50의 거리 (비율).
    (Price / MA - 1). 양수 = 이평선 위, 음수 = 이평선 아래.
    TREND 신호로 기존 UCHIDA가 쓰던 지표의 연속.
    """
    out = {}
    for win, name in [(WIN_LONG, "ma200"), (50, "ma50")]:
        ma = prices.rolling(win).mean()
        out[f"dist_{name}"] = (prices / ma) - 1
    # MA50/MA200 ratio (골든크로스/데드크로스 포착)
    ma50  = prices.rolling(50).mean()
    ma200 = prices.rolling(WIN_LONG).mean()
    out["ma_cross"] = (ma50 / ma200) - 1
    return pd.concat(out, axis=1)


# ==========================================
# 3. 매크로 Feature
# ==========================================
def _macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    매크로 지표 feature 생성.

    입력 컬럼 (FRED):
        CPIAUCSL, BAA10Y, T10Y2Y, DGS10, UNRATE, VIXCLS, DTWEXBGS

    Look-ahead 처리:
        CPI: 45일 발표 지연 shift 적용
        나머지: 발표 즉시 반영 (주/월 단위 데이터는 ffill로 이미 처리됨)
    """
    out = {}

    # --- CPI ---
    # 45일 지연 적용 후 YoY 변화율, Z-score
    cpi = df["CPIAUCSL"].shift(CPI_PUBLICATION_LAG_DAYS)
    cpi_yoy = cpi.pct_change(252) * 100
    cpi_mean = cpi_yoy.rolling(252).mean()
    cpi_std  = cpi_yoy.rolling(252).std()
    out["cpi_yoy"]    = cpi_yoy
    out["cpi_z"]      = (cpi_yoy - cpi_mean) / cpi_std.replace(0, np.nan)
    out["cpi_mom"]    = cpi.pct_change(21) * 100   # 월간 변화율 (단기 인플레 압력)

    # --- Yield Curve ---
    out["t10y2y"]     = df["T10Y2Y"]               # 10Y-2Y spread (역전 = 경기침체 신호)
    out["dgs10"]      = df["DGS10"]                 # 10년 금리 수준
    out["dgs10_chg"]  = df["DGS10"].diff(21)        # 금리 변화 (1개월)

    # --- Credit Spread ---
    # BAA10Y는 FRED에서 이미 "BAA - 10Y Treasury" 스프레드로 제공됨.
    # DGS10 재차감 금지 (이중 차감 버그 수정. 2026-05).
    out["credit_spread"]     = df["BAA10Y"]
    out["credit_spread_chg"] = out["credit_spread"].diff(21)

    # --- 실업률 ---
    out["unrate"]     = df["UNRATE"]
    out["unrate_chg"] = df["UNRATE"].diff(63)       # 3개월 변화 (추세)

    # --- VIX ---
    out["vix"]        = df["VIXCLS"]
    vix_mean = df["VIXCLS"].rolling(252).mean()
    vix_std  = df["VIXCLS"].rolling(252).std()
    out["vix_z"]      = (df["VIXCLS"] - vix_mean) / vix_std.replace(0, np.nan)
    out["vix_chg"]    = df["VIXCLS"].diff(5)        # 주간 VIX 변화

    # --- 달러 인덱스 (DXY) ---
    out["dxy"]        = df["DTWEXBGS"]
    out["dxy_ret_5d"] = np.log(df["DTWEXBGS"] / df["DTWEXBGS"].shift(5))

    return pd.DataFrame(out, index=df.index)


# ==========================================
# 4. 자산간 관계 Feature
# ==========================================
def _cross_asset(prices: pd.DataFrame) -> pd.DataFrame:
    """
    자산간 rolling correlation 및 ratio.

    - QQQ-TLT correlation: 위기 시 음의 상관 → 채권 헤지 유효성
    - QQQ-GLD correlation: 인플레이션 국면 탐지
    - QQQ-OIL correlation: 경기 확장/수축 신호
    - QQQ/SPY ratio: 성장주 vs 가치주 상대 강도
    """
    out = {}
    qqq = prices["QQQ"]

    # Rolling correlations (60일)
    for other in ["TLT", "GLD", "USO"]:
        if other in prices.columns:
            out[f"corr_qqq_{other.lower()}"] = (
                qqq.rolling(CORR_WIN).corr(prices[other])
            )

    # QQQ/SPY ratio (성장주 프리미엄)
    if "SPY" in prices.columns:
        out["qqq_spy_ratio"] = np.log(qqq / prices["SPY"])

    # IEF/QQQ ratio (채권 vs 주식 상대 강도 — 위기 시 IEF↑)
    if "IEF" in prices.columns:
        out["ief_qqq_ratio"] = np.log(prices["IEF"] / qqq)

    return pd.DataFrame(out, index=prices.index)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex 컬럼을 flat string으로 변환. (예: ("ret_1d", "QQQ") → "ret_1d_QQQ")"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{a}_{b}" if b else str(a) for a, b in df.columns]
    return df
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    loader.load_all()의 출력을 받아 전체 feature DataFrame 반환.

    Parameters
    ----------
    df : pd.DataFrame
        loader.load_all()의 출력.
        columns: PRICE_TICKERS + MACRO_FRED_IDS

    Returns
    -------
    pd.DataFrame
        index: 날짜 (df와 동일)
        columns: 모든 feature (약 130개)
        - 앞쪽 행은 rolling window 때문에 NaN 포함 (dropna는 호출자 책임)

    Notes
    -----
    - NaN을 여기서 dropna 안 함: 라벨링 모듈이 날짜 정렬 책임져야 함
    - backward fill 사용 안 함 (look-ahead bias 방지)
    - 가격은 종가(Close)만 사용. 시가(Open)는 백테스트 체결에만 사용.
    """
    # loader.load_all()은 "Close_QQQ", "Open_QQQ" 형태 컬럼으로 반환.
    # features는 종가만 필요. Open은 백테스트 엔진에서만 사용.
    close_cols = [c for c in df.columns if c.startswith("Close_")]
    if close_cols:
        prices = df[close_cols].copy()
        prices.columns = [c.replace("Close_", "") for c in close_cols]
    else:
        # 후방호환: 컬럼이 이미 ticker만 있는 경우 (단위 테스트 등)
        prices = df[[c for c in df.columns if c in [a.ticker_us for a in ASSETS.values()]]].copy()

    # ASSETS 순서대로 정렬
    price_cols = [a.ticker_us for a in ASSETS.values()]
    available_prices = [c for c in price_cols if c in prices.columns]
    prices = prices[available_prices]

    # 각 feature 군 계산
    feat_parts = [
        _flatten_columns(_log_returns(prices)),
        _flatten_columns(_realized_vol(prices)),
        _flatten_columns(_momentum(prices)),
        _flatten_columns(_rsi(prices)),
        _flatten_columns(_macd(prices)),
        _flatten_columns(_bollinger(prices)),
        _flatten_columns(_bollinger_width(prices)),   # 박스권 신호용 (D-019)
        _flatten_columns(_adx(prices)),               # 박스권 신호용 (D-019)
        _flatten_columns(_ma_distance(prices)),
        _macro_features(df),
        _cross_asset(prices),
    ]

    features = pd.concat(feat_parts, axis=1)

    # MultiIndex 컬럼 → flat string (예: ("ret_1d", "QQQ") → "ret_1d_QQQ")
    if isinstance(features.columns, pd.MultiIndex):
        features.columns = [
            f"{a}_{b}" if b else str(a)
            for a, b in features.columns
        ]

    # 컬럼명 중복 검사 (설계 오류 조기 탐지)
    dupes = features.columns[features.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"중복 feature 컬럼 발견: {dupes}")

    return features


# ==========================================
# 6. Feature 목록 조회 유틸리티
# ==========================================
def get_feature_names() -> list:
    """
    feature 이름 목록 반환 (실제 데이터 없이 목록만 필요할 때).
    더미 데이터로 build_features 호출하여 컬럼명 추출.
    """
    idx = pd.date_range("2015-01-01", periods=300, freq="B")
    dummy_prices = pd.DataFrame(
        np.random.lognormal(0, 0.01, (300, len(PRICE_TICKERS))),
        index=idx,
        columns=PRICE_TICKERS,
    ).cumprod()

    macro_cols = ["CPIAUCSL", "BAA10Y", "T10Y2Y", "DGS10", "UNRATE", "VIXCLS", "DTWEXBGS"]
    dummy_macro = pd.DataFrame(
        np.random.randn(300, len(macro_cols)) * 0.1 + 1,
        index=idx,
        columns=macro_cols,
    )
    dummy_df = pd.concat([dummy_prices, dummy_macro], axis=1)
    return build_features(dummy_df).columns.tolist()


# ==========================================
# 7. 자체 검증 (수동 실행 시)
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("data/features.py 단독 실행 테스트 (더미 데이터)")
    print("=" * 60)

    # 더미 데이터 생성 (실제 데이터 없이 구조 검증)
    # loader.load_all() 출력 구조를 흉내냄: Close_TICKER, Open_TICKER, 매크로 컬럼
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2013-01-01", periods=n, freq="B")

    price_cols = PRICE_TICKERS
    macro_cols = ["CPIAUCSL", "BAA10Y", "T10Y2Y", "DGS10", "UNRATE", "VIXCLS", "DTWEXBGS"]

    # Close 가격
    dummy_close = pd.DataFrame(
        np.random.lognormal(0, 0.01, (n, len(price_cols))),
        index=idx, columns=[f"Close_{t}" for t in price_cols],
    ).cumprod() * 100
    # Open 가격 (Close 근처에서 약간 변동)
    dummy_open = dummy_close.shift(1).fillna(dummy_close.iloc[0])
    dummy_open.columns = [f"Open_{t}" for t in price_cols]

    dummy_macro = pd.DataFrame({
        "CPIAUCSL": np.random.lognormal(5, 0.005, n),
        "BAA10Y":   np.random.uniform(3, 6, n),
        "T10Y2Y":   np.random.uniform(-1, 2, n),
        "DGS10":    np.random.uniform(1, 5, n),
        "UNRATE":   np.random.uniform(3, 10, n),
        "VIXCLS":   np.random.uniform(10, 40, n),
        "DTWEXBGS": np.random.uniform(90, 130, n),
    }, index=idx)

    dummy_df = pd.concat([dummy_close, dummy_open, dummy_macro], axis=1)

    # Feature 생성
    feat = build_features(dummy_df)

    print(f"\n[Feature 수]   총 {len(feat.columns)}개")
    print(f"[데이터 기간]  {feat.index[0].date()} ~ {feat.index[-1].date()}")
    print(f"[NaN 비율]     앞쪽 rolling 기간 제외 후: "
          f"{feat.iloc[WIN_LONG:].isna().mean().mean()*100:.2f}%")

    print(f"\n[Feature 목록]")
    groups = {
        "가격 returns":   [c for c in feat.columns if c.startswith("ret_")],
        "실현 변동성":    [c for c in feat.columns if c.startswith("rvol_")],
        "모멘텀":         [c for c in feat.columns if c.startswith("mom_")],
        "RSI":            [c for c in feat.columns if c.startswith("rsi_")],
        "MACD":           [c for c in feat.columns if c.startswith("macd_")],
        "볼린저":         [c for c in feat.columns if c.startswith("boll_")],
        "MA 거리":        [c for c in feat.columns if c.startswith("dist_") or c.startswith("ma_")],
        "매크로":         [c for c in feat.columns if c in ["cpi_yoy","cpi_z","cpi_mom",
                           "t10y2y","dgs10","dgs10_chg","credit_spread","credit_spread_chg",
                           "unrate","unrate_chg","vix","vix_z","vix_chg","dxy","dxy_ret_5d"]],
        "자산간 관계":    [c for c in feat.columns if c.startswith("corr_") or c.endswith("_ratio")],
    }
    for g, cols in groups.items():
        print(f"  {g:15s} ({len(cols):3d}개): {cols[:3]}{'...' if len(cols)>3 else ''}")

    print(f"\n[샘플: 마지막 2행 일부 컬럼]")
    sample_cols = ["ret_1d_QQQ", "rvol_20d_QQQ", "mom_12_1_QQQ",
                   "rsi_QQQ", "cpi_z", "t10y2y", "vix", "corr_qqq_tlt"]
    avail = [c for c in sample_cols if c in feat.columns]
    print(feat[avail].tail(2).round(4).to_string())

    print("\n✓ features.py 검증 완료")
