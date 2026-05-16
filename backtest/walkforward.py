"""
UCHIDA V3 - Walk-Forward Validation

학습/검증 기간을 시간 순서대로 분할하여 미래 데이터 누출 방지.

[핵심 원칙]
1. 학습 데이터는 항상 검증 데이터보다 과거
2. Embargo period: Triple-Barrier 라벨이 T=21일 미래를 참조하므로
   학습 끝 ~ 검증 시작 사이 21일 gap 필수 (López de Prado 2018, Ch.7)
3. Expanding window: 학습 데이터 누적 증가 (Rolling보다 데이터 효율적)

[분할 구조]
    |──────train_0──────|embargo|──test_0──|
    |──────────train_1──────────|embargo|──test_1──|
    |────────────────train_2────────────────|embargo|──test_2──|
    ...

[참고문헌]
López de Prado (2018), Advances in Financial Machine Learning, Ch.7.
"Purged K-Fold Cross-Validation with Embargo"

[사용 예]
    from backtest.walkforward import WalkForwardSplit

    splits = WalkForwardSplit(
        min_train_years=3,
        test_period_months=12,
        embargo_days=21,
    )
    for fold in splits.split(df_features, df_labels):
        X_train = df_features.loc[fold.train_start:fold.train_end]
        y_train = df_labels.loc[fold.train_start:fold.train_end]
        X_test  = df_features.loc[fold.test_start:fold.test_end]
        y_test  = df_labels.loc[fold.test_start:fold.test_end]
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Iterator, Optional
from datetime import date


# ==========================================
# Fold 컨테이너
# ==========================================
@dataclass
class Fold:
    """단일 walk-forward 분할."""
    fold_id:     int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp    # 학습 마지막 날 (embargo 시작 전)
    embargo_end: pd.Timestamp    # embargo 마지막 날 (검증 시작 전)
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp

    @property
    def train_days(self) -> int:
        return int((self.train_end - self.train_start).days)

    @property
    def test_days(self) -> int:
        return int((self.test_end - self.test_start).days)

    def __repr__(self):
        return (
            f"Fold({self.fold_id}: "
            f"train [{self.train_start.date()} ~ {self.train_end.date()}] "
            f"| embargo [{self.train_end.date()} ~ {self.embargo_end.date()}] "
            f"| test [{self.test_start.date()} ~ {self.test_end.date()}])"
        )


# ==========================================
# Walk-Forward Splitter
# ==========================================
class WalkForwardSplit:
    """
    Expanding-window walk-forward 분할기.

    Parameters
    ----------
    min_train_years : float
        최소 학습 기간 (년). 3년 기본.
        근거: 금융 사이클 최소 단위. 3년 미만이면 강세장/약세장 둘 다 못 봄.
    test_period_months : int
        검증 기간 (개월). 12개월 기본.
        근거: 계절성 포착 + 평가 안정성. 3개월 미만은 noisy.
    embargo_days : int
        학습-검증 사이 gap (영업일). Triple-Barrier T와 동일하게 21일 기본.
        근거: 라벨이 T=21일 미래 참조 → 그 기간만큼 gap 필요.
    """

    def __init__(
        self,
        min_train_years: float = 3.0,
        test_period_months: int = 12,
        embargo_days: int = 21,
    ):
        self.min_train_years = min_train_years
        self.test_period_months = test_period_months
        self.embargo_days = embargo_days

    def split(self, index: pd.DatetimeIndex) -> List[Fold]:
        """
        날짜 인덱스에서 Fold 목록 생성.

        Parameters
        ----------
        index : pd.DatetimeIndex
            전체 데이터의 날짜 인덱스 (영업일, 정렬 필요).

        Returns
        -------
        List[Fold]
            시간 순서대로 정렬된 Fold 목록.
        """
        index = index.sort_values()
        start = index[0]
        end   = index[-1]

        # 최소 학습 기간 끝 날짜
        min_train_end = start + pd.DateOffset(years=self.min_train_years)

        # 검증 기간 오프셋
        test_delta = pd.DateOffset(months=self.test_period_months)
        embargo_delta = pd.offsets.BusinessDay(self.embargo_days)

        folds = []
        fold_id = 0
        test_start_cursor = min_train_end + embargo_delta

        while True:
            test_end_cursor = test_start_cursor + test_delta

            # 검증 끝이 전체 데이터 끝을 넘으면 종료
            if test_end_cursor > end:
                break

            # 학습 끝 = 검증 시작 - embargo
            train_end = test_start_cursor - embargo_delta

            # 인덱스에서 가장 가까운 날짜 snap
            train_end_snap   = _snap(index, train_end,   side="before")
            embargo_end_snap = _snap(index, test_start_cursor - pd.offsets.BusinessDay(1), side="before")
            test_start_snap  = _snap(index, test_start_cursor, side="after")
            test_end_snap    = _snap(index, test_end_cursor,   side="before")

            if train_end_snap is None or test_start_snap is None or test_end_snap is None:
                break
            if train_end_snap >= test_start_snap:
                break

            folds.append(Fold(
                fold_id=fold_id,
                train_start=start,
                train_end=train_end_snap,
                embargo_end=embargo_end_snap if embargo_end_snap else train_end_snap,
                test_start=test_start_snap,
                test_end=test_end_snap,
            ))

            fold_id += 1
            # 다음 검증 기간: 이번 검증 끝 + 1일 다음 영업일
            test_start_cursor = test_end_cursor + pd.offsets.BusinessDay(1)

        return folds

    def summary(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        모든 Fold 요약 DataFrame.
        학습 전 시각화/검토용.
        """
        folds = self.split(index)
        rows = []
        for f in folds:
            rows.append({
                "fold":         f.fold_id,
                "train_start":  f.train_start.date(),
                "train_end":    f.train_end.date(),
                "embargo_end":  f.embargo_end.date(),
                "test_start":   f.test_start.date(),
                "test_end":     f.test_end.date(),
                "train_days":   f.train_days,
                "test_days":    f.test_days,
            })
        return pd.DataFrame(rows).set_index("fold")


# ==========================================
# 유틸리티
# ==========================================
def _snap(
    index: pd.DatetimeIndex,
    dt: pd.Timestamp,
    side: str = "before",
) -> Optional[pd.Timestamp]:
    """
    dt에 가장 가까운 인덱스 날짜 반환.
    side="before": dt 이하 중 최대
    side="after":  dt 이상 중 최소
    """
    if side == "before":
        mask = index <= dt
        if not mask.any():
            return None
        return index[mask][-1]
    else:
        mask = index >= dt
        if not mask.any():
            return None
        return index[mask][0]


def apply_embargo(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    DataFrame에서 embargo 기간을 제거한 학습용 슬라이스 반환.
    Triple-Barrier 라벨의 미래 참조로 인한 누출 방지.

    Parameters
    ----------
    df : pd.DataFrame
        학습 데이터 (index: DatetimeIndex)
    train_end : pd.Timestamp
        학습 기간 마지막 날
    test_start : pd.Timestamp
        검증 기간 시작 날

    Returns
    -------
    pd.DataFrame
        train_end까지만 잘린 df. test_start 이후 데이터 없음.
    """
    return df.loc[:train_end]


def get_fold_data(
    df: pd.DataFrame,
    fold: Fold,
) -> tuple:
    """
    단일 Fold에서 학습/검증 데이터 추출.

    Returns
    -------
    (df_train, df_test) tuple
    """
    df_train = df.loc[fold.train_start:fold.train_end]
    df_test  = df.loc[fold.test_start:fold.test_end]
    return df_train, df_test


# ==========================================
# 자체 검증
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("backtest/walkforward.py 단독 실행 테스트")
    print("=" * 70)

    # 2010-01-01 ~ 2025-12-31 영업일 인덱스
    idx = pd.date_range("2010-01-01", "2025-12-31", freq="B")
    print(f"\n전체 기간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)} 영업일)")

    splitter = WalkForwardSplit(
        min_train_years=3,
        test_period_months=12,
        embargo_days=21,
    )

    summary = splitter.summary(idx)
    print(f"\n[Walk-Forward 분할 요약] 총 {len(summary)} Fold")
    print(summary.to_string())

    # 검증: 각 fold의 학습-검증 사이에 embargo가 있는지
    folds = splitter.split(idx)
    print(f"\n[Embargo 검증]")
    all_ok = True
    for f in folds:
        gap_days = (f.test_start - f.train_end).days
        ok = gap_days >= splitter.embargo_days
        if not ok:
            print(f"  ✗ Fold {f.fold_id}: gap={gap_days}일 (필요: {splitter.embargo_days}일)")
            all_ok = False

    if all_ok:
        print(f"  ✓ 모든 Fold에 embargo({splitter.embargo_days}일) 이상 gap 확인")

    # 검증: 학습-검증 겹침 없는지
    print(f"\n[학습-검증 겹침 검증]")
    all_ok = True
    for f in folds:
        if f.train_end >= f.test_start:
            print(f"  ✗ Fold {f.fold_id}: 학습 끝({f.train_end.date()}) ≥ 검증 시작({f.test_start.date()})")
            all_ok = False
    if all_ok:
        print(f"  ✓ 모든 Fold에 학습-검증 겹침 없음")

    # 검증: Expanding window (학습 기간이 단조 증가)
    print(f"\n[Expanding window 검증]")
    train_lengths = [f.train_days for f in folds]
    is_increasing = all(train_lengths[i] <= train_lengths[i+1] for i in range(len(train_lengths)-1))
    if is_increasing:
        print(f"  ✓ 학습 기간 단조 증가 확인 ({train_lengths[0]}일 → {train_lengths[-1]}일)")
    else:
        print(f"  ✗ 학습 기간이 단조 증가하지 않음")

    print("\n✓ walkforward.py 검증 완료")
