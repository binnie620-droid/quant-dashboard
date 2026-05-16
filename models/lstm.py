"""
UCHIDA V3 - LSTM Regime Classifier

과거 60일 feature 시계열 → 향후 21일 시장 국면 확률 예측.

[모델 구조]
    Input  (B, 60, 132)
        ↓  LayerNorm
    LSTM   (hidden=64, dropout=0.3)
        ↓
    LSTM   (hidden=32, dropout=0.3)
        ↓  마지막 timestep
    Linear (32 → 16) + GELU + Dropout(0.3)
        ↓
    Linear (16 → 3)
        ↓
    Softmax → [P_crisis, P_defense, P_attack]

[확률 → 비중 변환]
    옵션 B: 확률 가중 평균
    w = P_crisis × W_crisis + P_defense × W_defense + P_attack × W_attack

[학습 전략]
    - Loss: CrossEntropy + class_weight (CRISIS에 높은 가중치)
    - Optimizer: Adam (lr=1e-3)
    - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
    - Early stopping: patience=10 (val_loss 기준)
    - Walk-forward: walkforward.py의 Fold 구조 사용

[Label 인코딩]
    0 = CRISIS, 1 = DEFENSE, 2 = ATTACK
    (labels/labeler.py의 CRISIS=0, DEFENSE=1, ATTACK=2 와 일치)

[참고문헌]
    Fischer & Krauss (2018), "Deep learning with long short-term memory
    networks for financial market predictions", EJOR.
    Srivastava et al. (2014), "Dropout: A Simple Way to Prevent Neural
    Networks from Overfitting", JMLR.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REBALANCE
from models.baseline import WEIGHT_MAP


# ==========================================
# 0. 하이퍼파라미터
# ==========================================
@dataclass
class LSTMConfig:
    """
    LSTM 하이퍼파라미터.
    모두 파라미터로 노출 → sensitivity analysis 가능.

    [설계 근거]
    - seq_len=60: 3개월. 분기 매크로 사이클 포착. 너무 길면 vanishing gradient.
    - hidden1=64, hidden2=32: 샘플 ~150개에 대해 충분히 작음.
      128 이상은 overfitting 위험. 32→16 bottleneck으로 정보 압축.
    - dropout=0.3: Srivastava et al. (2014) 권장 범위 0.2~0.5의 중간.
    - lr=1e-3: Adam 기본값. ReduceLROnPlateau가 자동 조정.
    - batch_size=32: 소규모 데이터에서 표준.
    """
    seq_len:      int   = 60
    hidden1:      int   = 64
    hidden2:      int   = 32
    dense_dim:    int   = 16
    dropout:      float = 0.3
    lr:           float = 1e-3
    batch_size:   int   = 32
    max_epochs:   int   = 100
    es_patience:  int   = 10    # early stopping patience
    lr_patience:  int   = 5     # lr scheduler patience
    lr_factor:    float = 0.5
    n_classes:    int   = 3
    val_ratio:    float = 0.2   # 학습 데이터 중 검증 비율


DEFAULT_CONFIG = LSTMConfig()


# ==========================================
# 1. Dataset
# ==========================================
class RegimeDataset(Dataset):
    """
    Sliding window dataset.

    Parameters
    ----------
    X : np.ndarray, shape (T, n_features)
        Standardized feature array
    y : np.ndarray, shape (T,)
        Label array (0/1/2)
    seq_len : int
        Sliding window 길이
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 60):
        self.seq_len = seq_len
        # sample i: X[i : i+seq_len] → y[i+seq_len-1]
        self.n_samples = len(X) - seq_len + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"데이터 길이({len(X)})가 seq_len({seq_len})보다 짧습니다."
            )
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]          # (seq_len, n_features)
        label = self.y[idx + self.seq_len - 1]            # 윈도우 끝 시점의 라벨
        return x_seq, label


# ==========================================
# 2. 모델
# ==========================================
class LSTMClassifier(nn.Module):
    """
    2-layer LSTM + Dense → 3-class 국면 분류기.

    Parameters
    ----------
    input_size : int
        Feature 수 (기본 132)
    cfg : LSTMConfig
        하이퍼파라미터
    """

    def __init__(self, input_size: int, cfg: LSTMConfig = DEFAULT_CONFIG):
        super().__init__()
        self.cfg = cfg

        # Input normalization (학습 안정성 + 스케일 무관)
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM layers (batch_first=True: input shape (B, seq, feature))
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden1,
            batch_first=True,
            dropout=0.0,    # 단층이라 dropout 미적용 (다층 LSTM 사이에서만 의미있음)
        )
        self.drop1 = nn.Dropout(cfg.dropout)

        self.lstm2 = nn.LSTM(
            input_size=cfg.hidden1,
            hidden_size=cfg.hidden2,
            batch_first=True,
            dropout=0.0,
        )
        self.drop2 = nn.Dropout(cfg.dropout)

        # Dense layers
        self.fc1 = nn.Linear(cfg.hidden2, cfg.dense_dim)
        self.act1 = nn.GELU()
        self.drop3 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.dense_dim, cfg.n_classes)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, orthogonal for LSTM."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
            elif "weight" in name and param.data.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (B, 3)
            Softmax 확률 [P_crisis, P_defense, P_attack]
        """
        # Input normalization
        x = self.input_norm(x)

        # LSTM 1
        out, _ = self.lstm1(x)          # (B, seq_len, hidden1)
        out = self.drop1(out)

        # LSTM 2
        out, _ = self.lstm2(out)        # (B, seq_len, hidden2)
        out = self.drop2(out)

        # 마지막 timestep만 사용 (many-to-one)
        out = out[:, -1, :]             # (B, hidden2)

        # Dense
        out = self.fc1(out)             # (B, dense_dim)
        out = self.act1(out)
        out = self.drop3(out)
        out = self.fc2(out)             # (B, n_classes)

        return torch.softmax(out, dim=-1)


# ==========================================
# 3. 학습 루프
# ==========================================
class LSTMTrainer:
    """
    LSTM 학습/예측 인터페이스.

    Parameters
    ----------
    cfg : LSTMConfig
    device : str
        'cuda' | 'mps' | 'cpu' (자동 감지)
    """

    def __init__(self, cfg: LSTMConfig = DEFAULT_CONFIG, device: str = None):
        self.cfg = cfg
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.scaler = StandardScaler()
        self.model: Optional[LSTMClassifier] = None
        self.train_history: List[Dict] = []

    def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """
        역빈도 기반 class weight.
        CRISIS가 적을수록 높은 가중치 → 모델이 CRISIS를 무시하지 않게.
        """
        counts = np.bincount(y, minlength=self.cfg.n_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)  # 0 방지
        weights = 1.0 / counts
        weights = weights / weights.mean()            # 평균 1로 정규화
        return torch.FloatTensor(weights).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> "LSTMTrainer":
        """
        학습.

        Parameters
        ----------
        X_train : np.ndarray, shape (T, n_features)
            Raw feature (Standardize는 내부에서)
        y_train : np.ndarray, shape (T,)
            Label (0/1/2)
        """
        cfg = self.cfg

        # 1. Standardize (train 통계만 사용)
        X_s = self.scaler.fit_transform(X_train)
        X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Train / Val split (시간 순서 유지 — shuffle 안 함)
        n_val = max(cfg.seq_len + 1, int(len(X_s) * cfg.val_ratio))
        X_tr, X_vl = X_s[:-n_val], X_s[-n_val:]
        y_tr, y_vl = y_train[:-n_val], y_train[-n_val:]

        # 3. Dataset & DataLoader
        tr_ds = RegimeDataset(X_tr, y_tr, cfg.seq_len)
        vl_ds = RegimeDataset(X_vl, y_vl, cfg.seq_len)
        tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=False)
        vl_dl = DataLoader(vl_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        # 4. 모델 초기화
        n_features = X_s.shape[1]
        self.model = LSTMClassifier(input_size=n_features, cfg=cfg).to(self.device)

        # 5. Loss, Optimizer, Scheduler
        class_weights = self._compute_class_weights(y_tr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.lr_patience,
            factor=cfg.lr_factor,
        )

        # 6. 학습 루프
        best_val_loss = float("inf")
        best_state = None
        es_counter = 0
        self.train_history = []

        for epoch in range(cfg.max_epochs):
            # Train
            self.model.train()
            tr_loss = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                probs = self.model(xb)
                loss = criterion(probs, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_ds)

            # Validation
            self.model.eval()
            vl_loss = 0.0
            vl_correct = 0
            with torch.no_grad():
                for xb, yb in vl_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    probs = self.model(xb)
                    vl_loss += criterion(probs, yb).item() * len(xb)
                    vl_correct += (probs.argmax(1) == yb).sum().item()
            vl_loss /= len(vl_ds)
            vl_acc = vl_correct / len(vl_ds)

            scheduler.step(vl_loss)
            self.train_history.append({
                "epoch": epoch + 1,
                "tr_loss": tr_loss,
                "vl_loss": vl_loss,
                "vl_acc": vl_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | tr_loss={tr_loss:.4f} "
                      f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping
            if vl_loss < best_val_loss - 1e-5:
                best_val_loss = vl_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= cfg.es_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Best 모델 복원
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        확률 예측.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)

        Returns
        -------
        np.ndarray, shape (T - seq_len + 1, 3)
            각 시점의 [P_crisis, P_defense, P_attack]
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_s = self.scaler.transform(X)
        X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
        y_dummy = np.zeros(len(X_s), dtype=int)

        ds = RegimeDataset(X_s, y_dummy, self.cfg.seq_len)
        dl = DataLoader(ds, batch_size=128, shuffle=False)

        self.model.eval()
        probs_list = []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(self.device)
                probs_list.append(self.model(xb).cpu().numpy())

        return np.vstack(probs_list)   # (N, 3)

    def save(self, path: str):
        """모델 + 스케일러 저장."""
        import pickle
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "cfg": self.cfg,
            "train_history": self.train_history,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "LSTMTrainer":
        """저장된 모델 로드."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        trainer = cls(cfg=state["cfg"])
        trainer.scaler = state["scaler"]
        trainer.train_history = state["train_history"]
        if state["model_state"] is not None:
            n_features = state["scaler"].n_features_in_
            trainer.model = LSTMClassifier(input_size=n_features, cfg=state["cfg"])
            trainer.model.load_state_dict(state["model_state"])
            trainer.model.to(trainer.device)
        return trainer


# ==========================================
# 4. 확률 → 비중 변환
# ==========================================
def probs_to_weights(
    probs: np.ndarray,
    weight_map: Dict[str, Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    옵션 B: 확률 가중 평균으로 자산 비중 결정.

    Parameters
    ----------
    probs : np.ndarray, shape (N, 3)
        [P_crisis, P_defense, P_attack] (각 행 합=1)
    weight_map : dict
        국면별 자산 비중. None이면 baseline.WEIGHT_MAP 사용.

    Returns
    -------
    pd.DataFrame, shape (N, n_assets)
        각 행이 목표 비중 (합=1)
    """
    if weight_map is None:
        weight_map = WEIGHT_MAP

    assets = list(weight_map["ATTACK"].keys())
    w_crisis  = np.array([weight_map["CRISIS"][a]  for a in assets])
    w_defense = np.array([weight_map["DEFENSE"][a] for a in assets])
    w_attack  = np.array([weight_map["ATTACK"][a]  for a in assets])

    # (N, 3) × (3, n_assets) → (N, n_assets)
    weight_matrix = np.stack([w_crisis, w_defense, w_attack], axis=0)  # (3, n_assets)
    weights = probs @ weight_matrix                                      # (N, n_assets)

    # 수치 오차 보정 (합이 1.0 ± 1e-10 이어야 함)
    weights = weights / weights.sum(axis=1, keepdims=True)

    return pd.DataFrame(weights, columns=assets)


# ==========================================
# 5. Walk-forward 학습 파이프라인
# ==========================================
def run_walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    cfg: LSTMConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Walk-forward validation 전체 실행.

    Parameters
    ----------
    features : pd.DataFrame
        build_features() 출력
    labels : pd.Series
        labeler.label_market_regimes() 출력의 'label' 컬럼
    cfg : LSTMConfig

    Returns
    -------
    (all_weights, fold_metrics) tuple
        all_weights: 전체 기간 목표 비중 DataFrame
        fold_metrics: fold별 검증 성능 리스트
    """
    from backtest.walkforward import WalkForwardSplit, get_fold_data

    # features와 labels 날짜 정렬
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels   = labels.loc[common_idx]

    splitter = WalkForwardSplit(
        min_train_years=3,
        test_period_months=12,
        embargo_days=21,
    )
    folds = splitter.split(features.index)

    if not folds:
        raise ValueError("Walk-forward fold가 생성되지 않았습니다. 데이터 기간을 확인하세요.")

    all_probs   = pd.DataFrame(index=features.index, columns=["p_crisis","p_defense","p_attack"], dtype=float)
    fold_metrics = []

    for fold in folds:
        if verbose:
            print(f"\n[Fold {fold.fold_id}] train: {fold.train_start.date()} ~ {fold.train_end.date()} "
                  f"| test: {fold.test_start.date()} ~ {fold.test_end.date()}")

        X_train_df, X_test_df = get_fold_data(features, fold)
        y_train_s,  y_test_s  = get_fold_data(labels,   fold)

        X_train = X_train_df.values
        y_train = y_train_s.values.astype(int)
        X_test  = X_test_df.values
        y_test  = y_test_s.values.astype(int)

        # test 데이터가 seq_len보다 짧으면 건너뜀
        if len(X_test) < cfg.seq_len:
            if verbose:
                print(f"  ⚠ test 데이터({len(X_test)}일) < seq_len({cfg.seq_len}). 건너뜀.")
            continue

        # 학습
        trainer = LSTMTrainer(cfg=cfg)
        trainer.fit(X_train, y_train, verbose=verbose)

        # 예측 (test set)
        probs = trainer.predict_proba(X_test)  # (N_test, 3)

        # 예측 결과를 전체 index에 매핑
        # predict_proba는 seq_len-1개 앞부분 skip → 날짜 정렬
        test_dates = X_test_df.index[cfg.seq_len - 1:]
        if len(test_dates) == len(probs):
            all_probs.loc[test_dates] = probs

        # Fold 성능 (정확도)
        pred_labels = probs.argmax(axis=1)
        y_test_aligned = y_test[cfg.seq_len - 1:]
        if len(y_test_aligned) == len(pred_labels):
            acc = (pred_labels == y_test_aligned).mean()
        else:
            acc = np.nan

        fold_metrics.append({
            "fold_id":    fold.fold_id,
            "test_start": fold.test_start,
            "test_end":   fold.test_end,
            "accuracy":   acc,
            "n_test":     len(probs),
        })
        if verbose:
            print(f"  Test accuracy: {acc:.3f} ({len(probs)} samples)")

    # 확률 → 비중
    valid_mask = all_probs.notna().all(axis=1)
    probs_array = all_probs[valid_mask].values.astype(float)
    weights_df = probs_to_weights(probs_array)
    weights_df.index = all_probs[valid_mask].index

    return weights_df, fold_metrics


# ==========================================
# 6. 자체 검증 (더미 데이터)
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("models/lstm.py 단독 실행 테스트 (더미 데이터)")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # 더미 feature (500일, 20개 feature — 빠른 테스트용)
    n_days, n_feat = 500, 20
    idx = pd.date_range("2015-01-01", periods=n_days, freq="B")
    X_dummy = np.random.randn(n_days, n_feat)

    # 더미 라벨 (3-class, 약간 불균형)
    y_dummy = np.random.choice([0, 1, 2], size=n_days, p=[0.2, 0.4, 0.4])

    # ----- Test 1: 기본 학습 -----
    print("\n[Test 1] 기본 학습 및 예측")
    cfg = LSTMConfig(
        seq_len=20,       # 빠른 테스트용 (실전은 60)
        hidden1=16,
        hidden2=8,
        dense_dim=8,
        max_epochs=5,
        es_patience=3,
        batch_size=16,
    )
    trainer = LSTMTrainer(cfg=cfg)
    trainer.fit(X_dummy[:400], y_dummy[:400], verbose=False)

    probs = trainer.predict_proba(X_dummy[400:])
    print(f"  예측 확률 shape: {probs.shape}")
    print(f"  확률 합 (최대 편차): {abs(probs.sum(axis=1) - 1.0).max():.2e}")
    assert probs.shape[1] == 3, "출력이 3-class여야 함"
    assert abs(probs.sum(axis=1) - 1.0).max() < 1e-5, "확률 합 != 1"
    print("  ✓ 통과")

    # ----- Test 2: 확률 → 비중 변환 -----
    print("\n[Test 2] 확률 → 비중 변환 (옵션 B)")
    weights = probs_to_weights(probs)
    weight_sums = weights.sum(axis=1)
    max_dev = (weight_sums - 1.0).abs().max()
    print(f"  비중 합 최대 편차: {max_dev:.2e}")
    print(f"  비중 음수 여부: {(weights < -1e-9).any().any()}")
    assert max_dev < 1e-9, "비중 합 != 1"
    assert not (weights < -1e-9).any().any(), "음수 비중 발생"
    print(f"  샘플 비중 (첫 행): {weights.iloc[0].round(3).to_dict()}")
    print("  ✓ 통과")

    # ----- Test 3: 저장/로드 -----
    print("\n[Test 3] 모델 저장 및 로드")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    trainer.save(path)
    loaded = LSTMTrainer.load(path)
    probs2 = loaded.predict_proba(X_dummy[400:])
    diff = abs(probs - probs2).max()
    print(f"  저장/로드 후 예측 차이: {diff:.2e}")
    assert diff < 1e-6, "저장/로드 후 결과 불일치"
    os.unlink(path)
    print("  ✓ 통과")

    # ----- Test 4: LayerNorm 입력 스케일 불변 -----
    print("\n[Test 4] 스케일이 다른 입력에도 안정적 예측")
    X_scaled = X_dummy[400:] * 1000   # 1000배 스케일
    probs_scaled = trainer.predict_proba(X_scaled)
    # StandardScaler가 처리하므로 결과 동일해야 함
    diff_scaled = abs(probs - probs_scaled).max()
    print(f"  1000배 스케일 입력 예측 차이: {diff_scaled:.2e}")
    assert diff_scaled < 1e-4, "스케일 불변 실패"
    print("  ✓ 통과")

    print("\n" + "=" * 70)
    print("✓ 4가지 단위 테스트 모두 통과")
    print(f"  device: {trainer.device}")
    print(f"  model params: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print("=" * 70)
