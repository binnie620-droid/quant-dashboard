import os, json, requests, warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

APP_KEY, APP_SECRET, CANO = os.getenv("KIS_MOCK_APP_KEY"), os.getenv("KIS_MOCK_APP_SECRET"), os.getenv("KIS_CANO")
TG_TOKEN, TG_CHAT_ID = "8754772387:AAEAQYSLm2EReLFR6dRrQz62VbTntH_ZYk4", "8167010481"
URL_BASE = "https://openapi.koreainvestment.com:9443"

def get_token():
    res = requests.post(f"{URL_BASE}/oauth2/tokenP", data=json.dumps({"grant_type":"client_credentials","appkey":APP_KEY,"appsecret":APP_SECRET}))
    return res.json().get("access_token")

def get_balance(token):
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":"VTRP6641R"}
    params = {"CANO":CANO, "ACNT_PRDT_CD":"01", "PRDT_TYPE_CD":"01", "CTX_AREA_FK100":"", "CTX_AREA_NK100":""}
    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
    try: return int(res.json()['output2'][0]['tot_evlu_amt'])
    except: return 10000000

def execute_market_buy(token, code, qty):
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":"VTTC0802U"} # 모의매수
    data = {"CANO":CANO, "ACNT_PRDT_CD":"01", "PDNO":code, "ORD_DVSN":"01", "ORD_QTY":str(qty), "ORD_UNPR":"0"}
    res = requests.post(f"{URL_BASE}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, data=json.dumps(data))
    return res.json().get('rt_cd') == '0'

def create_pdf():
    if not os.path.exists('vintage_performance.csv'): return None
    df = pd.read_csv('vintage_performance.csv').fillna('')
    f_p = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(f_p): fm.fontManager.addfont(f_p); plt.rcParams['font.family'] = 'NanumGothic'
    fig, ax = plt.subplots(figsize=(12, 8)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    for (r, c), cell in table.get_celld().items():
        if r > 0 and c >= 2:
            try:
                val = float(cell.get_text().get_text())
                cell.get_text().set_color('#d32f2f' if val > 0 else '#1976d2')
            except: pass
    name = f"MITSUDA_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    with PdfPages(name) as pdf: pdf.savefig(fig, bbox_inches='tight')
    plt.close(); return name

def main():
    token = get_token()
    total_asset = get_balance(token)
    
    # 1. 과거 성과 리포트 전송
    pdf = create_pdf()
    if pdf:
        with open(pdf, 'rb') as f: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument", data={"chat_id":TG_CHAT_ID}, files={"document":f})
        os.remove(pdf)

    # 2. 오늘자 매수 집행
    try:
        with open('meta_target_list.json', 'r', encoding='utf-8') as f: data = json.load(f)
    except: return

    buy_log = []
    cash_ratio = 0.4 if data['zone'] == "🟡 YELLOW" else (0.8 if data['zone'] == "🔴 RED" else 0.0)
    for t in data['targets']:
        qty = int((total_asset * (1 - cash_ratio) * t['kelly_fraction']) / t['price'])
        if qty > 0 and execute_market_buy(token, t['code'], qty):
            buy_log.append(f"✅ {t['name']} {qty}주 매수 완료")
    
    if buy_log:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":f"🐆 MITSUDA 작전 보고\n존: {data['zone']}\n" + "\n".join(buy_log)})

if __name__ == "__main__":
    main()
