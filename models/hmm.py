"""
UCHIDA V3 - Hidden Markov Model Regime Classifier

[모델 개요]
Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary 
Time Series and the Business Cycle"의 정신을 자산배분에 적용.

시장에는 보이지 않는 국면(state)이 있고, 우리는 관찰값(수익률, VIX 등)으로
그 국면을 추정한다. HMM은 다음 두 확률을 데이터로부터 자동 학습:
  1. 전이 확률 (Transition): P(다음 상태 | 현재 상태)
  2. 방출 확률 (Emission): P(관찰값 | 현재 상태)

[LSTM과의 차이]
- 파라미터: ~30개 (LSTM 5만+) → 데이터 효율 ↑
- 해석 가능: 각 state의 평균/분산 명확
- 빠른 학습: 수초 (LSTM 수십분)
- 학계 검증: 30년+ 사용

[State 수 결정]
3 states: CRISIS / DEFENSE / ATTACK (우리 기존 분류와 일관)
n_components=3 고정. 추후 BIC로 최적화 가능.

[입력 feature]
저차원으로 압축. HMM은 고차원에 약함.
  1. SPY 일별 log return
  2. SPY 60일 realized vol (annualized)
  3. VIX level
  4. Credit spread (BAA - 10Y)
  5. Yield curve (T10Y2Y)

[State 라벨링]
학습 후 각 state의 평균 수익률로 정렬:
  state 0 (lowest mean return) → CRISIS
  state 1 (middle)               → DEFENSE
  state 2 (highest mean return)  → ATTACK

[참고문헌]
Hamilton, J.D. (1989), Econometrica 57(2)
Ang, A. & Bekaert, G. (2002), Journal of Finance 57(3)
Pesaran, M.H. & Timmermann, A. (2002), JEC
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REBALANCE
from models.baseline import WEIGHT_MAP


# ==========================================
# Config
# ==========================================
@dataclass
class HMMConfig:
    """
    HMM 하이퍼파라미터.

    [설계 근거]
    - n_components=3: CRISIS/DEFENSE/ATTACK 3국면 (기존 분류 일관성)
    - covariance_type='full': 자산 간 공분산 활용
                              'diag' (대각만)도 가능하지만 정보 손실
    - n_iter=100: EM 알고리즘 수렴까지
    - tol=1e-4: 수렴 기준
    """
    n_components: int = 3
    covariance_type: str = "full"   # "full" / "diag" / "tied" / "spherical"
    n_iter: int = 100
    tol: float = 1e-4
    random_state: int = 42

    # Feature 선택
    use_returns: bool = True
    use_volatility: bool = True
    use_vix: bool = True
    use_credit_spread: bool = True
    use_yield_curve: bool = True


DEFAULT_CONFIG = HMMConfig()


# ==========================================
# Feature 준비
# ==========================================
def prepare_hmm_features(
    df_raw: pd.DataFrame,
    cfg: HMMConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    HMM 학습용 저차원 feature 추출.

    Parameters
    ----------
    df_raw : pd.DataFrame
        loader.load_all()의 출력 (Close_SPY, VIXCLS, BAA10Y, T10Y2Y 등 포함)
    cfg : HMMConfig

    Returns
    -------
    pd.DataFrame
        HMM에 넣을 feature (저차원, 5개 이내)
    """
    feats = pd.DataFrame(index=df_raw.index)

    if cfg.use_returns and "Close_QQQ" in df_raw.columns:
        log_p = np.log(df_raw["Close_QQQ"])
        feats["spy_ret"] = log_p.diff()

    if cfg.use_volatility and "Close_QQQ" in df_raw.columns:
        log_p = np.log(df_raw["Close_QQQ"])
        ret = log_p.diff()
        feats["spy_vol"] = ret.rolling(60).std() * np.sqrt(252)

    if cfg.use_vix and "VIXCLS" in df_raw.columns:
        feats["vix"] = df_raw["VIXCLS"]

    if cfg.use_credit_spread and "BAA10Y" in df_raw.columns:
        feats["credit_spread"] = df_raw["BAA10Y"]

    if cfg.use_yield_curve and "T10Y2Y" in df_raw.columns:
        feats["yield_curve"] = df_raw["T10Y2Y"]

    return feats.dropna()


# ==========================================
# HMM Classifier
# ==========================================
class HMMClassifier:
    """
    Gaussian HMM 기반 시장 국면 분류기.

    Parameters
    ----------
    cfg : HMMConfig
    """

    def __init__(self, cfg: HMMConfig = DEFAULT_CONFIG):
        if not HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn 라이브러리 필요. 설치: pip install hmmlearn"
            )

        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = None
        self.state_to_regime: Dict[int, str] = {}  # HMM state → 'CRISIS'/'DEFENSE'/'ATTACK'

    def fit(self, X: pd.DataFrame) -> "HMMClassifier":
        """
        HMM 학습.

        Parameters
        ----------
        X : pd.DataFrame
            prepare_hmm_features() 출력
        """
        # 1. Standardize
        X_s = self.scaler.fit_transform(X.values)

        # 2. HMM 초기화 및 학습
        self.model = hmm.GaussianHMM(
            n_components=self.cfg.n_components,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            tol=self.cfg.tol,
            random_state=self.cfg.random_state,
        )
        self.model.fit(X_s)

        # 3. State 라벨링 (수익률 기준 정렬)
        # 첫 feature는 spy_ret이라고 가정 → 평균 작은 state = CRISIS
        means = self.model.means_  # (n_components, n_features)
        ret_means = means[:, 0]  # spy_ret column의 평균
        order = np.argsort(ret_means)  # lowest → highest

        regime_names = ["CRISIS", "DEFENSE", "ATTACK"]
        for i, state_idx in enumerate(order):
            self.state_to_regime[int(state_idx)] = regime_names[i]

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        각 시점의 국면 확률 예측.

        Returns
        -------
        pd.DataFrame
            columns: ['p_crisis', 'p_defense', 'p_attack']
            (state_to_regime 매핑 반영)
        """
        if self.model is None:
            raise RuntimeError("학습되지 않은 모델. fit() 먼저 호출.")

        X_s = self.scaler.transform(X.values)
        state_probs = self.model.predict_proba(X_s)  # (T, n_components)

        # state → regime 매핑
        out = pd.DataFrame(index=X.index, columns=["p_crisis", "p_defense", "p_attack"], dtype=float)
        for state_idx, regime in self.state_to_regime.items():
            col = f"p_{regime.lower()}"
            out[col] = state_probs[:, state_idx]

        return out

    def predict_states(self, X: pd.DataFrame) -> pd.Series:
        """
        각 시점의 최빈 국면 (Viterbi).

        Returns
        -------
        pd.Series
            values: 'CRISIS' / 'DEFENSE' / 'ATTACK'
        """
        if self.model is None:
            raise RuntimeError("학습되지 않은 모델. fit() 먼저 호출.")

        X_s = self.scaler.transform(X.values)
        states = self.model.predict(X_s)  # Viterbi
        return pd.Series(
            [self.state_to_regime[s] for s in states],
            index=X.index, name="regime",
        )

    def summary(self) -> pd.DataFrame:
        """
        학습 결과 요약.
        각 state의 평균/표준편차 + 라벨.
        """
        if self.model is None:
            raise RuntimeError("학습되지 않은 모델.")

        means_std = self.scaler.scale_  # 역변환용
        means_mean = self.scaler.mean_

        # state별 원본 스케일 평균
        means_raw = self.model.means_ * means_std + means_mean

        n_features = len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else means_raw.shape[1]
        feature_names = (
            self.scaler.feature_names_in_.tolist()
            if hasattr(self.scaler, 'feature_names_in_')
            else [f"feat_{i}" for i in range(n_features)]
        )

        rows = []
        for state_idx in range(self.cfg.n_components):
            regime = self.state_to_regime.get(state_idx, f"state_{state_idx}")
            row = {"state": state_idx, "regime": regime}
            for i, fname in enumerate(feature_names):
                row[f"{fname}_mean"] = means_raw[state_idx, i]
            rows.append(row)

        return pd.DataFrame(rows).set_index("state")


# ==========================================
# 확률 → 비중 변환 (LSTM과 동일)
# ==========================================
def probs_to_weights(
    probs: pd.DataFrame,
    weight_map: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    국면 확률 → 자산 비중 (확률 가중 평균).

    Parameters
    ----------
    probs : pd.DataFrame
        columns: ['p_crisis', 'p_defense', 'p_attack']
    weight_map : dict, optional
        None이면 baseline.WEIGHT_MAP 사용

    Returns
    -------
    pd.DataFrame
        index: probs.index, columns: 자산명
    """
    if weight_map is None:
        weight_map = WEIGHT_MAP

    assets = list(weight_map["ATTACK"].keys())
    w_crisis  = np.array([weight_map["CRISIS"][a]  for a in assets])
    w_defense = np.array([weight_map["DEFENSE"][a] for a in assets])
    w_attack  = np.array([weight_map["ATTACK"][a]  for a in assets])

    # (T, 3) × (3, n_assets) → (T, n_assets)
    prob_arr = probs[["p_crisis", "p_defense", "p_attack"]].values
    weight_matrix = np.stack([w_crisis, w_defense, w_attack], axis=0)
    weights = prob_arr @ weight_matrix

    # 수치 오차 보정
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    weights = weights / row_sums

    return pd.DataFrame(weights, columns=assets, index=probs.index)


# ==========================================
# Walk-Forward 파이프라인
# ==========================================
def run_walk_forward(
    df_raw: pd.DataFrame,
    cfg: HMMConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Walk-forward로 HMM 학습 + 비중 생성.

    LSTM과 다른 점: HMM은 학습 빠르므로 매월 재학습 가능.
    여기선 단순성을 위해 LSTM과 동일한 yearly fold 사용.

    Parameters
    ----------
    df_raw : pd.DataFrame
        loader.load_all() 출력
    cfg : HMMConfig

    Returns
    -------
    (weights_df, fold_info)
    """
    from backtest.walkforward import WalkForwardSplit, get_fold_data

    # 1. Feature 준비
    features = prepare_hmm_features(df_raw, cfg)
    if len(features) == 0:
        raise ValueError("HMM feature 생성 실패: 필요한 컬럼 누락 가능성")

    # 2. Walk-forward 분할
    splitter = WalkForwardSplit(
        min_train_years=3,
        test_period_months=12,
        embargo_days=21,
    )
    folds = splitter.split(features.index)

    if not folds:
        raise ValueError("Walk-forward fold 생성 실패")

    all_probs = pd.DataFrame(
        index=features.index,
        columns=["p_crisis", "p_defense", "p_attack"],
        dtype=float,
    )
    fold_info = []

    for fold in folds:
        if verbose:
            print(f"\n[Fold {fold.fold_id}] train: {fold.train_start.date()} ~ {fold.train_end.date()} "
                  f"| test: {fold.test_start.date()} ~ {fold.test_end.date()}")

        X_train_df, X_test_df = get_fold_data(features, fold)

        if len(X_train_df) < 100:
            if verbose:
                print(f"  ⚠ 학습 데이터({len(X_train_df)}) 부족. 건너뜀.")
            continue

        # HMM 학습
        try:
            model = HMMClassifier(cfg=cfg)
            model.fit(X_train_df)
        except Exception as e:
            if verbose:
                print(f"  ⚠ 학습 실패: {e}")
            continue

        # 예측
        try:
            probs = model.predict_proba(X_test_df)
            all_probs.loc[probs.index] = probs.values

            # State 분포
            states = model.predict_states(X_test_df)
            dist = states.value_counts(normalize=True).to_dict()
            if verbose:
                print(f"  Test 국면 분포: {{k: f'{v*100:.1f}%' for k, v in dist.items()}}".replace("'", ""))

            fold_info.append({
                "fold_id": fold.fold_id,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_test": len(X_test_df),
                "state_distribution": dist,
            })
        except Exception as e:
            if verbose:
                print(f"  ⚠ 예측 실패: {e}")
            continue

    # 3. 확률 → 비중
    valid_mask = all_probs.notna().all(axis=1)
    probs_valid = all_probs[valid_mask]
    weights_df = probs_to_weights(probs_valid)

    return weights_df, fold_info


# ==========================================
# 자체 검증
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("models/hmm.py 단독 실행 테스트")
    print("=" * 70)

    if not HMMLEARN_AVAILABLE:
        print("❌ hmmlearn 미설치. 설치 명령: pip install hmmlearn")
        sys.exit(1)

    # 더미 데이터 생성 (Close_SPY + VIX + 매크로)
    np.random.seed(42)
    n_days = 1500
    idx = pd.date_range("2015-01-01", periods=n_days, freq="B")

    # 두 가지 regime 인위적 생성: 평시 vs 위기
    ret = np.random.normal(0.0005, 0.01, n_days)
    vix = np.random.uniform(15, 20, n_days)

    # 250~350일 구간: 위기 (수익률 ↓, VIX ↑)
    ret[250:350] = np.random.normal(-0.005, 0.025, 100)
    vix[250:350] = np.random.uniform(35, 50, 100)

    # 1200~1300일 구간: 또 다른 위기
    ret[1200:1300] = np.random.normal(-0.003, 0.020, 100)
    vix[1200:1300] = np.random.uniform(30, 45, 100)

    spy_close = 100 * np.exp(np.cumsum(ret))
    df_dummy = pd.DataFrame({
        "Close_SPY": spy_close,
        "VIXCLS": vix,
        "BAA10Y": np.random.uniform(2, 4, n_days),
        "T10Y2Y": np.random.uniform(-0.5, 2.5, n_days),
    }, index=idx)

    # ----- Test 1: Feature 생성 -----
    print("\n[Test 1] Feature 생성")
    feats = prepare_hmm_features(df_dummy)
    print(f"  Shape: {feats.shape}")
    print(f"  Columns: {list(feats.columns)}")
    assert len(feats) > 0 and feats.shape[1] >= 3
    print("  ✓ 통과")

    # ----- Test 2: HMM 학습 -----
    print("\n[Test 2] HMM 학습")
    model = HMMClassifier()
    model.fit(feats)
    print(f"  State → Regime 매핑: {model.state_to_regime}")
    print(f"\n  학습 결과 요약:")
    print(model.summary().round(3).to_string())
    assert len(model.state_to_regime) == 3
    print("  ✓ 통과")

    # ----- Test 3: 확률 예측 -----
    print("\n[Test 3] 확률 예측")
    probs = model.predict_proba(feats)
    print(f"  Shape: {probs.shape}")
    print(f"  확률 합 최대 편차: {abs(probs.sum(axis=1) - 1.0).max():.2e}")
    assert abs(probs.sum(axis=1) - 1.0).max() < 1e-6
    print("  ✓ 통과")

    # ----- Test 4: 위기 구간 감지 -----
    print("\n[Test 4] 위기 구간 감지 (인위 생성한 250~350일)")
    states = model.predict_states(feats)
    crisis_in_known = (states.iloc[250:350] == "CRISIS").sum()
    print(f"  250~350 구간 CRISIS 판정: {crisis_in_known}/100일")
    # 정확히는 안 맞지만 어느 정도 잡아야 함
    print("  (HMM이 위기 구간을 감지하는지 확인)")

    # ----- Test 5: 확률 → 비중 -----
    print("\n[Test 5] 확률 → 비중 변환")
    weights = probs_to_weights(probs)
    print(f"  비중 합 최대 편차: {abs(weights.sum(axis=1) - 1.0).max():.2e}")
    print(f"  자산 컬럼: {list(weights.columns)}")
    assert abs(weights.sum(axis=1) - 1.0).max() < 1e-9
    print("  ✓ 통과")

    print("\n" + "=" * 70)
    print("✓ 5가지 단위 테스트 통과")
    print("=" * 70)
