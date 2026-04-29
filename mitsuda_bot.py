import os, json, requests, warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

APP_KEY, APP_SECRET, CANO = os.getenv("KIS_MOCK_APP_KEY"), os.getenv("KIS_MOCK_APP_SECRET"), os.getenv("KIS_CANO")
TG_TOKEN, TG_CHAT_ID = "8754772387:AAEAQYSLm2EReLFR6dRrQz62VbTntH_ZYk4", "8167010481"
URL_BASE = "https://openapi.koreainvestment.com:9443"

def get_token():
    res = requests.post(f"{URL_BASE}/oauth2/tokenP", data=json.dumps({"grant_type":"client_credentials","appkey":APP_KEY,"appsecret":APP_SECRET}))
    return res.json().get("access_token")

# [복구: 실전 매수/매도 주문 함수]
def execute_order(token, code, qty, side="buy"):
    tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":tr_id}
    data = {"CANO":CANO, "ACNT_PRDT_CD":"01", "PDNO":code, "ORD_DVSN":"01", "ORD_QTY":str(qty), "ORD_UNPR":"0"}
    res = requests.post(f"{URL_BASE}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, data=json.dumps(data))
    return res.json().get('rt_cd') == '0'

# [복구: 어깨 매도(Trailing Stop) 로직]
def manage_trailing_stop(token):
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":"VTRP6641R"}
    params = {"CANO":CANO, "ACNT_PRDT_CD":"01", "PRDT_TYPE_CD":"01", "CTX_AREA_FK100":"", "CTX_AREA_NK100":""}
    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
    
    sell_log = []
    try:
        holdings = res.json()['output1']
        for stock in holdings:
            code, name, qty = stock['pdno'], stock['prdt_name'], int(stock['hldg_qty'])
            cur_price = int(stock['prpr'])
            # [핵심] 최근 7일 최고가 대비 2*ATR 하락 시 어깨 매도 (여기선 단순화하여 수익률 기반 시뮬레이션)
            evlu_rate = float(stock['evlu_pfit_rt'])
            if evlu_rate <= -5.0: # 예시: 손절선 터치 시
                if execute_order(token, code, qty, "sell"):
                    sell_log.append(f"🛑 어깨 매도 완료: {name}")
    except: pass
    return sell_log

def create_pdf_report():
    if not os.path.exists('vintage_performance.csv'): return None
    df = pd.read_csv('vintage_performance.csv').fillna('')
    f_p = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(f_p): fm.fontManager.addfont(f_p); plt.rcParams['font.family'] = 'NanumGothic'
    fig, ax = plt.subplots(figsize=(12, 8)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    # ... (색상 입히기 생략)
    name = f"MITSUDA_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    with PdfPages(name) as pdf: pdf.savefig(fig, bbox_inches='tight')
    plt.close(); return name

def main():
    token = get_token()
    
    # 1. 어깨 매도 관리 (이미 가진 애들 감시)
    sell_msgs = manage_trailing_stop(token)
    if sell_msgs: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":"\n".join(sell_msgs)})

    # 2. 성과 보고서 PDF 발송
    pdf = create_pdf_report()
    if pdf:
        with open(pdf, 'rb') as f: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument", data={"chat_id":TG_CHAT_ID}, files={"document":f})
        os.remove(pdf)

    # 3. 오늘자 매수 집행
    try:
        with open('meta_target_list.json', 'r', encoding='utf-8') as f: data = json.load(f)
    except: return

    buy_log = []
    for t in data['targets']:
        qty = int((10000000 * t['kelly_fraction']) / t['price']) # 1천만 원 기준 예시
        if qty > 0 and execute_order(token, t['code'], qty, "buy"):
            buy_log.append(f"✅ {t['name']} {qty}주 매수 완료")
    
    if buy_log:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":f"🐆 MITSUDA 작전 보고\n존: {data['zone']}\n" + "\n".join(buy_log)})

if __name__ == "__main__":
    main()
