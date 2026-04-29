import os, json, requests, warnings
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
APP_KEY, APP_SECRET, CANO = os.getenv("KIS_MOCK_APP_KEY"), os.getenv("KIS_MOCK_APP_SECRET"), os.getenv("KIS_CANO")
TG_TOKEN, TG_CHAT_ID = "8754772387:AAEAQYSLm2EReLFR6dRrQz62VbTntH_ZYk4", "8167010481"
URL_BASE = "https://openapi.koreainvestment.com:9443"

def get_token():
    res = requests.post(f"{URL_BASE}/oauth2/tokenP", data=json.dumps({"grant_type":"client_credentials","appkey":APP_KEY,"appsecret":APP_SECRET}))
    return res.json().get("access_token")

def send_msg(text): requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":text})

def main():
    token = get_token()
    with open('meta_target_list.json', 'r', encoding='utf-8') as f: data = json.load(f)

    # [디테일 2] 어깨 매도(Trailing Stop) 원리 설명
    # 실제 KIS API를 통해 현재 잔고 종목의 '매수 후 최고가'를 추적하여 
    # '최고가 - (2 * ATR)'를 하회하면 전량 매도 주문을 쏘는 로직이 여기에 들어갑니다.
    
    buy_log = []
    for t in data['targets']:
        # [디테일 4] KOSPI 100 기반 정수 수량 주문
        qty = int((10000000 * (1 - 0.4 if data['zone'] == "🟡 YELLOW" else 1) * t['kelly_fraction']) / t['price'])
        if qty > 0: buy_log.append(f"🔥 {t['name']} | {qty}주 매수 (진입방어선: {int(t['price'] - 2*t['atr'])}원)")

    msg = f"[{data['date']} 미츠다 마스터 작전]\n존: {data['zone']}\n\n" + "\n".join(buy_log)
    send_msg(msg)

if __name__ == "__main__":
    main()
