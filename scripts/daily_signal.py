"""
UCHIDA V3 - GitHub Actions 일일 신호 계산

매일 16:30 KST에 실행:
1. 5개 매크로 신호 계산
2. 국면 판정
3. data/signals.json 갱신 (HTML이 읽음)
4. 국면 전환 시 텔레그램 알림
"""

import os
import json
import requests
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATES, ASSETS
from data.loader import load_all
from data.features import build_features
from models.baseline import (
    compute_flags, compute_score, classify_regime,
    BaselineConfig,
)

KST = ZoneInfo("Asia/Seoul")
TG_TOKEN   = os.environ.get("TG_TOKEN", "")
USERS_FILE = Path(__file__).parent.parent / "data" / "users.json"
STATE_FILE = Path(__file__).parent.parent / "data" / "bot_state.json"
OUT_FILE   = Path(__file__).parent.parent / "data" / "signals.json"

REGIME_COLORS = {"ATTACK": "🟢", "DEFENSE": "🟡", "CRISIS": "🔴"}


def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def send_telegram(chat_id: str, text: str):
    if not TG_TOKEN or not chat_id:
        print(f"[TG 스킵] chat_id={chat_id}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }, timeout=10)
    if not resp.ok:
        print(f"[TG 오류] {resp.text}")


def fetch_kr_prices():
    """pykrx로 한국 ETF 현재가 조회."""
    prices = {}
    try:
        from pykrx import stock as krx
        today = date.today()
        # 16시 이후 = 오늘, 이전 = 어제
        now = datetime.now(KST)
        ref = today if now.hour >= 16 else today - timedelta(days=1)
        while ref.weekday() >= 5:
            ref -= timedelta(days=1)
        date_str  = ref.strftime("%Y%m%d")
        start_str = (ref - timedelta(days=7)).strftime("%Y%m%d")

        for key, asset in ASSETS.items():
            code = asset.code_kr
            try:
                df = krx.get_market_ohlcv_by_date(date_str, date_str, code)
                if df.empty:
                    df = krx.get_market_ohlcv_by_date(start_str, date_str, code)
                if not df.empty:
                    prices[key] = float(df["종가"].iloc[-1])
            except Exception as e:
                print(f"[가격 오류] {key}: {e}")
    except ImportError:
        print("[pykrx 없음] 가격 조회 스킵")
    return prices


def main():
    print(f"[{datetime.now(KST).strftime('%Y-%m-%d %H:%M')} KST] 신호 계산 시작")

    # 신호 계산
    df = load_all(start=DATES.train_start, use_cache=True)
    features_raw = build_features(df)
    SIGS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
    available = [c for c in SIGS if c in features_raw.columns]
    features = features_raw.dropna(subset=available)

    cfg = BaselineConfig()
    flags  = compute_flags(features, cfg)
    score  = compute_score(flags, cfg)
    regime = classify_regime(score, cfg)

    latest  = features.index[-1]
    cur_regime = str(regime.iloc[-1])
    cur_score  = float(score.iloc[-1])

    sig_data = {
        "cpi_z":        {"value": float(features.loc[latest, "cpi_z"]),
                         "threshold": cfg.cpi_z_threshold, "op": ">",
                         "triggered": bool(flags.loc[latest, "cpi_flag"]) if "cpi_flag" in flags.columns else False,
                         "label": "CPI Z-score"},
        "credit_spread":{"value": float(features.loc[latest, "credit_spread"]),
                         "threshold": cfg.credit_spread_threshold, "op": ">",
                         "triggered": bool(flags.loc[latest, "spread_flag"]) if "spread_flag" in flags.columns else False,
                         "label": "신용스프레드"},
        "t10y2y":       {"value": float(features.loc[latest, "t10y2y"]),
                         "threshold": cfg.t10y2y_threshold, "op": "<",
                         "triggered": bool(flags.loc[latest, "yield_flag"]) if "yield_flag" in flags.columns else False,
                         "label": "10Y-2Y"},
        "vix":          {"value": float(features.loc[latest, "vix"]),
                         "threshold": cfg.vix_threshold, "op": ">",
                         "triggered": bool(flags.loc[latest, "vix_flag"]) if "vix_flag" in flags.columns else False,
                         "label": "VIX"},
        "dist_ma200":   {"value": float(features.loc[latest, "dist_ma200_QQQ"]),
                         "threshold": cfg.ma_dist_threshold, "op": "<",
                         "triggered": bool(flags.loc[latest, "ma_flag"]) if "ma_flag" in flags.columns else False,
                         "label": "QQQ MA200"},
    }

    # 한국 ETF 가격
    prices = fetch_kr_prices()
    print(f"가격 조회: {prices}")

    # signals.json 저장
    output = {
        "date":    latest.strftime("%Y-%m-%d"),
        "updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "regime":  cur_regime,
        "score":   cur_score,
        "signals": sig_data,
        "prices":  prices,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"signals.json 저장: {cur_regime} (점수 {cur_score:.2f})")

    # 국면 전환 감지 → 텔레그램 알림
    state = load_state()
    prev_regime = state.get("last_regime")
    users = load_users()

    if prev_regime and prev_regime != cur_regime:
        old_icon = REGIME_COLORS.get(prev_regime, "")
        new_icon = REGIME_COLORS.get(cur_regime, "")
        msg = (
            f"🚨 *UCHIDA V3 — 국면 전환*\n\n"
            f"{old_icon} {prev_regime} → {new_icon} *{cur_regime}*\n"
            f"점수: {cur_score:.2f}\n\n"
            f"*[신호 현황]*\n"
        )
        for s in sig_data.values():
            icon = "🔴" if s["triggered"] else "⚪"
            val_str = f"{s['value']:+.2%}" if s["label"] == "QQQ MA200" else f"{s['value']:.2f}"
            msg += f"  {icon} {s['label']}: {val_str}\n"
        msg += f"\n대시보드: https://binnie620-droid.github.io/UCHIDA"

        # 모든 사용자에게 알림
        if users:
            for uid, udata in users.items():
                chat_id = udata.get("telegram_chat_id", "")
                send_telegram(chat_id, msg)
                print(f"[TG] {uid} 알림 전송")
        else:
            # users.json 없으면 환경변수 CHAT_ID로
            chat_id = os.environ.get("TG_CHAT_ID", "")
            send_telegram(chat_id, msg)
    else:
        print(f"국면 유지: {cur_regime}")
        # 변화 없어도 매일 간단 리포트 (선택)
        daily_msg = (
            f"*UCHIDA V3 — {latest.strftime('%Y-%m-%d')}*\n\n"
            f"{REGIME_COLORS.get(cur_regime,'')} 국면: *{cur_regime}* (점수 {cur_score:.2f})\n"
            f"매매 불필요 ✅"
        )
        if users:
            for uid, udata in users.items():
                chat_id = udata.get("telegram_chat_id", "")
                send_telegram(chat_id, daily_msg)
        else:
            chat_id = os.environ.get("TG_CHAT_ID", "")
            send_telegram(chat_id, daily_msg)

    # 상태 저장
    save_state({"last_regime": cur_regime, "last_run": str(date.today())})
    print("완료")


if __name__ == "__main__":
    main()
