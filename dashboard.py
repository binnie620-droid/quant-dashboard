"""
UCHIDA V3 - 일일 추적 Streamlit 대시보드

[기능]
1. 한국 ETF 보유 입력 (평단, 수량)
2. pykrx로 한국 ETF 현재가 자동 조회
3. 5개 매크로 신호 + 국면 판정 (loader/features/baseline 재사용)
4. 목표 비중 vs 현재 비중 비교
5. 5% 밴드 초과 또는 국면 전환 시 매매 지시 (계산만, 매매는 사용자 직접)
6. 일별 기록 CSV 누적 저장

[실행 시각 처리]
- 한국 시간 16:00 이후: 오늘 종가 사용
- 한국 시간 16:00 이전: 어제 종가 사용 (한국 장 마감 전)

[실행 방법]
    pip install streamlit pykrx
    streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from config import DATES, ASSETS
from data.loader import load_all
from data.features import build_features
from models.baseline import (
    compute_flags, compute_score, classify_regime,
    BaselineConfig, WEIGHT_MAP,
)

# pykrx (한국 ETF 가격)
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False


# ==========================================
# 상수
# ==========================================
STATE_FILE = Path(__file__).parent / "dashboard_state.json"   # 보유 정보 저장
LOG_FILE   = Path(__file__).parent / "dashboard_log.csv"      # 일별 기록 누적
REBALANCE_BAND = 0.05  # 5% no-trade band

# 한국 ETF 코드 매핑 (config.ASSETS 참조)
KR_CODES = {k: v.code_kr for k, v in ASSETS.items()}
KR_NAMES = {k: v.name_kr for k, v in ASSETS.items()}


# ==========================================
# 유틸 함수
# ==========================================
def get_reference_date():
    """
    한국 시간 기준 가격 참조 일자.
    - 16:00 이후: 오늘
    - 16:00 이전: 어제 (단 어제 영업일)
    - 주말이면 직전 금요일
    """
    now = datetime.now()
    if now.hour < 16:
        ref = now.date() - timedelta(days=1)
    else:
        ref = now.date()
    # 주말 처리
    while ref.weekday() >= 5:  # 5=토, 6=일
        ref -= timedelta(days=1)
    return ref


@st.cache_data(ttl=3600)
def fetch_kr_prices(ref_date, codes_tuple):
    """
    pykrx로 한국 ETF 종가 조회.
    
    Returns: dict {asset_key: price_krw} 또는 빈 dict (실패 시)
    """
    if not PYKRX_AVAILABLE:
        return {}
    
    prices = {}
    date_str = ref_date.strftime("%Y%m%d")
    for asset_key, code in codes_tuple:
        try:
            df = krx.get_market_ohlcv_by_date(date_str, date_str, code)
            if not df.empty:
                prices[asset_key] = float(df["종가"].iloc[-1])
            else:
                # 당일 데이터 없으면 직전 영업일 5일 시도
                start = (ref_date - timedelta(days=7)).strftime("%Y%m%d")
                df2 = krx.get_market_ohlcv_by_date(start, date_str, code)
                if not df2.empty:
                    prices[asset_key] = float(df2["종가"].iloc[-1])
        except Exception as e:
            st.warning(f"{asset_key} ({code}) 가격 조회 실패: {e}")
    return prices


@st.cache_data(ttl=3600)
def compute_signals():
    """
    Baseline 5개 신호 + 국면 계산.
    
    Returns: dict (signals + regime)
    """
    df = load_all(start=DATES.train_start, use_cache=True)
    features_raw = build_features(df)
    SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
    available = [c for c in SIGNALS if c in features_raw.columns]
    features = features_raw.dropna(subset=available)
    
    cfg = BaselineConfig()
    flags = compute_flags(features, cfg)
    score = compute_score(flags, cfg)
    regime = classify_regime(score, cfg)
    
    latest = features.index[-1]
    return {
        "date":           latest,
        "cpi_z":          float(features.loc[latest, "cpi_z"]),
        "credit_spread":  float(features.loc[latest, "credit_spread"]),
        "t10y2y":         float(features.loc[latest, "t10y2y"]),
        "vix":            float(features.loc[latest, "vix"]),
        "dist_ma200_QQQ": float(features.loc[latest, "dist_ma200_QQQ"]),
        "score":          float(score.iloc[-1]),
        "regime":         str(regime.iloc[-1]),
        "thresholds": {
            "cpi_z":          (cfg.cpi_z_threshold,             ">"),
            "credit_spread":  (cfg.credit_spread_threshold,     ">"),
            "t10y2y":         (cfg.t10y2y_threshold,            "<"),
            "vix":            (cfg.vix_threshold,               ">"),
            "dist_ma200_QQQ": (cfg.ma_dist_threshold,           "<"),
        },
    }


def load_holdings():
    """보유 정보 로드 (없으면 빈 dict)."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_holdings(holdings):
    """보유 정보 저장."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(holdings, f, ensure_ascii=False, indent=2)


def append_log(row):
    """일별 기록 누적."""
    df_row = pd.DataFrame([row])
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    # 같은 날짜 중복 시 마지막만 유지
    df = df.drop_duplicates(subset=["date"], keep="last")
    df.to_csv(LOG_FILE, index=False)


# ==========================================
# Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="UCHIDA V3 Daily Tracker", layout="wide")
    st.title("UCHIDA V3 - 일일 추적")
    
    # ---- 1. 환경 체크 ----
    if not PYKRX_AVAILABLE:
        st.error("pykrx가 설치되지 않았습니다: `pip install pykrx`")
        return
    
    ref_date = get_reference_date()
    now = datetime.now()
    st.caption(f"현재 시각: {now.strftime('%Y-%m-%d %H:%M')} | 가격 참조일: {ref_date} "
               f"({'16시 이후 → 오늘 종가' if now.hour >= 16 else '16시 이전 → 직전 영업일 종가'})")
    
    # ---- 2. 보유 입력 ----
    st.header("1. 보유 정보")
    holdings = load_holdings()
    
    cols = st.columns(len(KR_CODES))
    new_holdings = {}
    for i, (asset_key, code) in enumerate(KR_CODES.items()):
        with cols[i]:
            st.subheader(f"{asset_key}")
            st.caption(f"{KR_NAMES[asset_key]} ({code})")
            avg = st.number_input(
                f"평단 (원) - {asset_key}",
                min_value=0.0, value=float(holdings.get(asset_key, {}).get("avg_price", 0)),
                step=10.0, key=f"avg_{asset_key}",
            )
            shares = st.number_input(
                f"수량 (주) - {asset_key}",
                min_value=0, value=int(holdings.get(asset_key, {}).get("shares", 0)),
                step=1, key=f"shares_{asset_key}",
            )
            new_holdings[asset_key] = {
                "avg_price": avg,
                "shares":    shares,
                "code":      code,
            }
    
    if st.button("보유 정보 저장"):
        save_holdings(new_holdings)
        st.success("저장 완료")
        holdings = new_holdings
    
    # ---- 3. 현재가 + 평가액 ----
    st.header("2. 평가 현황")
    
    # 보유한 자산만 가격 조회
    held_codes = tuple((k, v["code"]) for k, v in new_holdings.items() if v["shares"] > 0)
    
    if not held_codes:
        st.info("보유 정보가 입력되지 않았습니다. 위에서 평단과 수량을 입력하세요.")
        return
    
    with st.spinner(f"pykrx에서 {ref_date} 종가 조회 중..."):
        prices = fetch_kr_prices(ref_date, held_codes)
    
    if not prices:
        st.error("가격 조회 실패. 네트워크 또는 pykrx 상태를 확인하세요.")
        return
    
    # 평가표 계산
    rows = []
    total_value = 0
    total_cost  = 0
    for asset_key, info in new_holdings.items():
        if info["shares"] == 0:
            continue
        price = prices.get(asset_key, 0)
        value = price * info["shares"]
        cost  = info["avg_price"] * info["shares"]
        pnl   = value - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0
        rows.append({
            "자산":     asset_key,
            "코드":     info["code"],
            "수량":     info["shares"],
            "평단":     info["avg_price"],
            "현재가":   price,
            "평가액":   value,
            "원금":     cost,
            "손익":     pnl,
            "손익%":    pnl_pct,
        })
        total_value += value
        total_cost  += cost
    
    eval_df = pd.DataFrame(rows)
    
    # 비중 계산
    if total_value > 0:
        eval_df["현재비중%"] = eval_df["평가액"] / total_value * 100
    
    st.dataframe(
        eval_df.style.format({
            "평단":     "{:,.0f}원",
            "현재가":   "{:,.0f}원",
            "평가액":   "{:,.0f}원",
            "원금":     "{:,.0f}원",
            "손익":     "{:+,.0f}원",
            "손익%":    "{:+.2f}%",
            "현재비중%": "{:.1f}%",
        }),
        use_container_width=True,
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 평가액", f"{total_value:,.0f}원")
    col2.metric("총 원금",   f"{total_cost:,.0f}원")
    total_pnl_pct = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
    col3.metric("총 손익", f"{total_value - total_cost:+,.0f}원",
                f"{total_pnl_pct:+.2f}%")
    
    # ---- 4. 5개 신호 + 국면 ----
    st.header("3. 매크로 신호 + 국면")
    
    with st.spinner("매크로 신호 계산 중..."):
        sig = compute_signals()
    
    st.caption(f"신호 기준일: {sig['date'].strftime('%Y-%m-%d')}")
    
    sig_rows = []
    for name, val in [
        ("CPI Z-score",      sig["cpi_z"]),
        ("신용 스프레드",     sig["credit_spread"]),
        ("10Y-2Y 역전",      sig["t10y2y"]),
        ("VIX",              sig["vix"]),
        ("QQQ MA200 거리",   sig["dist_ma200_QQQ"]),
    ]:
        key = {"CPI Z-score":"cpi_z", "신용 스프레드":"credit_spread",
               "10Y-2Y 역전":"t10y2y", "VIX":"vix", "QQQ MA200 거리":"dist_ma200_QQQ"}[name]
        thr, op = sig["thresholds"][key]
        triggered = (val > thr) if op == ">" else (val < thr)
        sig_rows.append({
            "신호":    name,
            "현재값":  val,
            "임계값":  thr,
            "조건":    op,
            "상태":    "🔴 ON" if triggered else "⚪ off",
        })
    
    sig_df = pd.DataFrame(sig_rows)
    st.dataframe(sig_df.style.format({"현재값": "{:.3f}", "임계값": "{:.2f}"}),
                 use_container_width=True)
    
    col1, col2 = st.columns(2)
    col1.metric("위험 점수", f"{sig['score']:.2f}")
    
    regime_color = {"ATTACK": "🟢", "DEFENSE": "🟡", "CRISIS": "🔴"}.get(sig["regime"], "")
    col2.metric("국면", f"{regime_color} {sig['regime']}")
    
    st.caption("점수 < 1.5 → ATTACK | 1.5 ≤ 점수 < 3.0 → DEFENSE | 점수 ≥ 3.0 → CRISIS")
    
    # ---- 5. 목표 비중 vs 현재 비중 + 매매 지시 ----
    st.header("4. 목표 비중 vs 현재 비중")
    
    target_weights = WEIGHT_MAP[sig["regime"]]
    
    # 비교표
    cmp_rows = []
    for asset_key in target_weights.keys():
        tgt = target_weights[asset_key] * 100
        cur = 0
        if total_value > 0:
            asset_row = eval_df[eval_df["자산"] == asset_key]
            if not asset_row.empty:
                cur = float(asset_row["현재비중%"].iloc[0])
        diff = cur - tgt
        # 트리거 판단: 절댓값 5%p 초과
        triggered = abs(diff) > REBALANCE_BAND * 100
        cmp_rows.append({
            "자산":      asset_key,
            "목표%":     tgt,
            "현재%":     cur,
            "차이%p":   diff,
            "밴드초과": "⚠ YES" if triggered else "  no",
        })
    
    cmp_df = pd.DataFrame(cmp_rows)
    st.dataframe(
        cmp_df.style.format({"목표%": "{:.1f}", "현재%": "{:.1f}", "차이%p": "{:+.2f}"}),
        use_container_width=True,
    )
    
    # 트리거 조건
    band_exceeded = cmp_df["밴드초과"].str.contains("YES").any()
    
    # 이전 국면 확인 (state에 저장)
    prev_regime = holdings.get("__last_regime__", None) if isinstance(holdings.get("__last_regime__"), str) else None
    regime_changed = (prev_regime is not None) and (prev_regime != sig["regime"])
    
    # ---- 6. 매매 지시 (필요 시) ----
    if band_exceeded or regime_changed:
        st.header("5. ⚠ 매매 지시")
        
        if regime_changed:
            st.warning(f"국면 전환 감지: {prev_regime} → {sig['regime']}")
        if band_exceeded:
            st.warning(f"5% 밴드 초과 자산 있음")
        
        st.subheader("목표 도달을 위한 매매")
        action_rows = []
        for asset_key in target_weights.keys():
            tgt_pct = target_weights[asset_key]
            tgt_value = total_value * tgt_pct
            
            cur_shares = 0
            cur_price = 0
            cur_value = 0
            if asset_key in prices:
                cur_price = prices[asset_key]
                # 현재 보유
                info = new_holdings.get(asset_key, {})
                cur_shares = info.get("shares", 0)
                cur_value = cur_price * cur_shares
            
            value_diff = tgt_value - cur_value
            
            # 목표 수량 (현재가 기준)
            tgt_shares = int(round(tgt_value / cur_price)) if cur_price > 0 else 0
            share_diff = tgt_shares - cur_shares
            
            action = ""
            if share_diff > 0:
                action = f"매수 {share_diff:,}주"
            elif share_diff < 0:
                action = f"매도 {abs(share_diff):,}주"
            else:
                action = "유지"
            
            action_rows.append({
                "자산":        asset_key,
                "현재 수량":   cur_shares,
                "목표 수량":   tgt_shares,
                "수량 차이":   share_diff,
                "금액 차이":   value_diff,
                "지시":        action,
            })
        
        action_df = pd.DataFrame(action_rows)
        st.dataframe(
            action_df.style.format({
                "현재 수량": "{:,d}", "목표 수량": "{:,d}",
                "수량 차이": "{:+,d}", "금액 차이": "{:+,.0f}원",
            }),
            use_container_width=True,
        )
        
        st.info("위 매매는 계산만 표시. 실제 매매는 직접 진행하세요.")
    else:
        st.success("✓ 5% 밴드 내 + 국면 유지. 매매 불필요.")
    
    # ---- 7. 일별 기록 저장 ----
    if st.button("오늘 기록 저장"):
        log_row = {
            "date":          str(ref_date),
            "ref_time":      now.strftime("%H:%M"),
            "regime":        sig["regime"],
            "score":         sig["score"],
            "total_value":   total_value,
            "total_cost":    total_cost,
            "total_pnl":     total_value - total_cost,
            "total_pnl_pct": total_pnl_pct,
            "cpi_z":         sig["cpi_z"],
            "credit_spread": sig["credit_spread"],
            "t10y2y":        sig["t10y2y"],
            "vix":           sig["vix"],
            "ma200_dist":    sig["dist_ma200_QQQ"],
        }
        append_log(log_row)
        new_holdings["__last_regime__"] = sig["regime"]
        save_holdings(new_holdings)
        st.success(f"{ref_date} 기록 저장 완료")
    
    # ---- 8. 누적 로그 표시 ----
    if LOG_FILE.exists():
        st.header("6. 일별 기록")
        log_df = pd.read_csv(LOG_FILE)
        st.dataframe(log_df.tail(30), use_container_width=True)


if __name__ == "__main__":
    main()
