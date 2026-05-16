# UCHIDA V3 — Level 1 재설계 인계 문서 (Sonnet → Opus)

**작성일**: 2026-05-14
**작성 세션**: Sonnet (진단 + 사전 분석)
**다음 세션**: Opus (Level 1 재설계 설계)

---

## 1. 이번 세션에서 발견한 것 (요약)

### 1-1. BUG-1: credit_spread 이중 차감 (수정 완료)

**파일**: `data/features.py` 203줄

```python
# 수정 전 (버그)
out["credit_spread"] = df["BAA10Y"] - df["DGS10"]

# 수정 후
out["credit_spread"] = df["BAA10Y"]
```

**근거**: FRED BAA10Y 시리즈 정의에 따르면 BAA10Y는 이미 "Moody's Baa - 10Y Treasury yield"의 **스프레드**. 여기서 DGS10을 한 번 더 빼면 `BAA - 2×DGS10`이 되어 의미가 무너짐.

**영향**:
- 2008.10 BAA10Y ≈ 5.5%, DGS10 ≈ 3.8% → 버그로 1.7% 계산됨 → 임계값 3.5 미달, flag=0
- 2022 인플레이션기 credit_spread가 음수로 나옴 (현실에서 불가능)

### 1-2. BUG-2: 2008 데이터 누락 (수정 완료)

**파일**: `run_sensitivity.py`, `run_backtest.py`, `run_compare_all.py`

```python
# 수정 전
features = build_features(df).dropna()
# → 147개 feature 중 mom_12_1(231일 lookback) 등의 NaN 때문에 2009-03까지 데이터 삭제

# 수정 후
BASELINE_SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
features = build_features(df).dropna(subset=BASELINE_SIGNALS)
# → 5개 핵심 신호 기준으로만 NaN 제거. 2007-01-11부터 사용 가능.
```

**참고**: `run_lstm.py`는 147개 전체 feature가 필요하므로 전체 dropna 유지 (의도 주석 추가).

### 1-3. DESIGN-1: CRISIS 분류 0% (미해결 — Opus 과제)

두 버그 수정 후에도 baseline A 시나리오에서 **CRISIS 분류가 22년간 0회** 발동.

**근본 원인**:
- crisis_threshold=3.0인데 22년간 점수 최대값=2.5
- 5개 신호 중 CPI(w=1.5) + credit_spread(w=1.5) 동시 발동이 필요한 구조
- 그런데 실증: 2008(신용위기, CPI 정상), 2020(보건위기, CPI 정상), 2022(인플레, credit_spread 정상) 등 **위기 유형이 모두 다름**

**위기 유형별 발동 패턴** (현재 임계값 기준):

| 위기 | 유형 | 발동 신호 | 정점 점수 |
|---|---|---|---|
| 2008 GFC | 신용 | credit_spread, vix, trend | ≤ 2.5 |
| 2020 COVID | 패닉 | credit_spread(겨우), vix, trend | 2.5 |
| 2022 인플레 | 인플레 | yield, vix, trend | 2.0 |

**결론**: 단일 가중합 점수로 모든 위기 유형을 잡으려는 시도 자체가 구조적으로 실패.

---

## 2. 현재 baseline 성과 (BUG 수정 후 실측)

| 지표 | QQQ B&H | A.현재 | E.Static-60/40 | F.Static-ATTACK |
|---|---|---|---|---|
| CAGR | 16.52% | 16.41% | 13.93% | 14.97% |
| MDD | -49.40% | -32.36% | -37.83% | -45.29% |
| Sharpe | 0.604 | **0.795** | 0.686 | 0.602 |
| Calmar | 0.334 | **0.507** | 0.368 | 0.330 |
| CE α=5 | 5.21% | **9.45%** | 7.95% | 6.03% |

**baseline A의 가치 (입증됨)**:
- vs Static-ATTACK: CAGR +1.44%p, MDD -12.93%p, Sharpe +0.193 — **신호 시스템 자체의 알파 확인**
- vs Static-60/40: CAGR +2.48%p, MDD +5.47%p — 동적 조정의 우월성 입증

**baseline A의 한계**:
- ARCHITECTURE.md 채택 기준 4개 중 2개만 통과 (CAGR ≥ 15% ✅, vs QQQ 알파 ≥ -3%p ✅, MDD ≤ -22% ❌, Sharpe ≥ 0.85 ❌)
- CRISIS 0% 발동 → 진짜 위기 회피 미작동

---

## 3. Opus가 결정해야 할 것

### 3-1. 위기 분류 체계

**현재 구조 (가중합)의 문제**:
- 모든 위기를 같은 점수 척도로 측정
- 위기 유형 다양성 무시 (Reinhart & Rogoff 2009의 일반적 견해 — 정확 인용 미확인)

**선택지**:

(a) **OR 게이트**: "한 신호라도 강하면 CRISIS"
- 장점: 다양한 위기 유형 포착
- 단점: false positive 증가 우려

(b) **위기 유형별 분리 점수**:
- 인플레 점수: cpi_z + 부수 신호
- 신용 점수: credit_spread + vix
- 패닉 점수: vix_z + dist_ma200
- 어떤 하나라도 임계값 넘으면 CRISIS

(c) **연속 점수화 (flag 폐기)**:
- 0/1 flag 대신 신호 강도를 0~1 sigmoid로 매핑
- 가중합으로 연속 점수
- 점수 구간으로 ATTACK/DEFENSE/CRISIS 분류

**Opus 결정 사항**: 위 중 어떤 구조? 학계 근거?

### 3-2. 신호 시점 처리 (선행 vs 동행 vs 후행)

5개 신호의 시점 특성이 다름:

| 신호 | 시점 특성 | 위기 시 상태 |
|---|---|---|
| T10Y2Y | **선행** (위기 1~2년 전) | 위기 시점엔 역전 풀림, flag=0 |
| CPI z-score | **인플레 위기에만 동행** | 신용/패닉 위기엔 무용 |
| Credit spread | **동행** | 신용 위기에만 강함 |
| VIX | **동행** | 모든 패닉에 강함 |
| MA200 거리 | **후행/동행** | 추세 위기에 강함 |

**Opus 결정 사항**:
- 선행 지표(T10Y2Y)를 CRISIS 트리거에 포함시킬지?
- 또는 ATTACK → DEFENSE 사전 전환에만 사용할지?
- 신호별로 다른 역할 부여?

### 3-3. 임계값 재설정 근거

`analysis/level1_redesign_data/03_signal_distribution.csv`에 각 신호의 분위수 분포 있음.

**현재 임계값의 적정성 검토 필요**:
- cpi_z > 2.3: 현재 4.8% 발동 (99퍼센타일) — 너무 엄격?
- credit_spread > 3.5: 현재 0.0% 발동 (BUG-1 수정 전 기준이었음) — 재산정 필요
- vix > 22: 현재 26.3% 발동 — 적정한가?

**Opus 결정 사항**: 학계 일반 견해 + 분위수 근거로 새 임계값.

### 3-4. WEIGHT_MAP 재검토 여부

현재 WEIGHT_MAP (DECISION_LOG D-009)은 사용자 성향("MDD 희생해도 CAGR")을 반영.
CRISIS가 작동하기 시작하면 CRISIS 비중도 검토 필요.

**Opus 결정 사항**: WEIGHT_MAP 그대로 둘지, 미세 조정할지.

---

## 4. 절대 건드리지 말 것 (확정된 것)

- **자산 universe 8개** (D-004) — 변경 불가
- **WEIGHT_MAP의 SCHD CRISIS 20%** — 사용자 명시 선호 (D-009)
- **거래비용 0.115%** (D-010)
- **5% no-trade band + 월간 리밸런싱** (D-011, D-016)
- **Triple-Barrier 라벨링 파라미터** (D-001) — LSTM/HMM용
- **ARCHITECTURE.md의 목표 함수 4가지** — 단, MDD -22%는 달성 가능성 재검토 가능

---

## 5. 사용 가능한 자료 (이번 세션 산출물)

### 5-1. 데이터 분석 결과
**위치**: `analysis/level1_redesign_data/` (prep_for_redesign.py 실행 후 생성)

| 파일 | 내용 |
|---|---|
| `01_crisis_signals.csv` | 위기별 신호 raw 값 통계 |
| `02_signal_hit_rate.csv` | 신호별 precision/recall/FPR |
| `03_signal_distribution.csv` | 전체 기간 신호 분위수 분포 |
| `04_signal_correlation.csv` | 신호 간 Pearson 상관 (방향 정렬) |
| `05_crisis_patterns.csv` | 위기 유형별 신호 발동 패턴 |
| `06_score_decomposition.csv` | 위기 정점 시점 점수 분해 |

### 5-2. 진단 스크립트
**위치**: `diagnose_crisis.py` (이미 실행, 결과는 채팅 로그 참조)

### 5-3. 사용자 코드
- `data/features.py` (BUG-1 수정됨)
- `models/baseline.py` (변경 없음)
- `run_sensitivity.py` (BUG-2 수정됨, Static 벤치마크 추가됨)
- `run_backtest.py`, `run_compare_all.py`, `run_lstm.py` (BUG-2 수정됨)

---

## 6. Opus 세션 권장 작업 순서

1. **이 문서 + ARCHITECTURE.md + DECISION_LOG.md + INSTRUCTIONS.md 읽기**
2. **6개 CSV 분석** — 위기 유형별 패턴 확인
3. **3가지 구조(a/b/c) 학술적 평가** — 어떤 게 학계 근거 강한가?
4. **설계안 1~2개 제시** — 사용자에게 선택권
5. **선택된 설계 → 구현 명세** (Sonnet으로 전환 권장)

---

## 7. INSTRUCTIONS.md 준수 확인 (이번 세션)

- ✅ 사용자 코드 부분 수정 (BUG-1: 1줄, BUG-2: 각 파일 3~5줄)
- ✅ 비중 합 검증 가능 (변경 없음)
- ✅ AI 판단 명시 (BASELINE_SIGNALS 선택, dropna subset 방식 등)
- ✅ 학계 근거 인용 (FRED BAA10Y 정의 — 공식 페이지 확인됨)
- ✅ 모델 분리 (Sonnet=진단+코드 / Opus=설계)

### 미해결 (Opus가 결정):
- WEIGHT_MAP CRISIS 비중의 학계 근거 (현재 D-009 "AI 판단")
- 새 임계값/가중치의 학계 근거
- 위기 분류 체계의 학계 근거

---

## 8. 솔직한 짚을 점 (사용자에게 전달)

1. **이전 sensitivity 결과의 +3.25%p 알파는 가짜였음** — 2008 GFC 누락 때문. 진짜 알파 vs QQQ는 -0.10%p.

2. **그러나 baseline의 진짜 알파는 vs Static-ATTACK 비교에서 드러남** — +1.44%p CAGR, -12.93%p MDD. 즉 "QQQ 대비 알파"가 아닌 "같은 자산으로 더 효율적 운용" 관점에서 가치 있음.

3. **ARCHITECTURE.md 목표 함수 재검토 필요할 수 있음** — "QQQ 대비 알파"보다 "동일 자산 정적 보유 대비 알파"가 더 정확한 기준일 수 있음.

4. **MDD -22% 달성 가능성은 미지수** — Level 1 재설계 후 CRISIS가 정상 작동해도 -22%는 야심찬 목표.
