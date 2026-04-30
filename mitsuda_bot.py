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

# --- 🚨 [핵심 디버깅 추가] 예수금 조회 함수 ---
def get_available_cash(token):
    headers = {
        "Content-Type": "application/json", 
        "authorization": f"Bearer {token}", 
        "appkey": APP_KEY, 
        "appsecret": APP_SECRET, 
        "tr_id": "VTTC8436R"
    }
    params = {
        "CANO": CANO, 
        "ACNT_PRDT_CD": "01", 
        "EXPT_SETL_CMPD_DVSN_CD": "00", 
        "INQR_DVSN_1": "00", 
        "INQR_DVSN_2": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }
    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
    
    res_json = res.json()
    print(f"💰 [잔고 조회 API 응답]: {res_json.get('msg1')}")
    
    try:
        cash = int(res_json['output2'][0]['dnca_tot_amt'])
        print(f"💵 [보스의 실제 예수금]: {cash}원")
        return cash
    except Exception as e:
        print(f"🚨 [잔고 조회 실패 원인]: {e}")
        return 0

def cancel_order(token, code):
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":"VTTC0803U"}
    data = {"CANO":CANO, "ACNT_PRDT_CD":"01", "ORGN_ODNO":"", "RVSE_CNCL_DVSN_CD":"02", "ORD_QTY":"0", "ORD_UNPR":"0", "QTY_ALL_ORD_FLG":"Y"}
    requests.post(f"{URL_BASE}/uapi/domestic-stock/v1/trading/order-rvse-cncl", headers=headers, data=json.dumps(data))

def execute_order(token, code, qty, side="buy"):
    tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":tr_id}
    data = {"CANO":CANO, "ACNT_PRDT_CD":"01", "PDNO":code, "ORD_DVSN":"01", "ORD_QTY":str(qty), "ORD_UNPR":"0"}
    res = requests.post(f"{URL_BASE}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, data=json.dumps(data))
    
    res_data = res.json()
    if res_data.get('rt_cd') != '0':
        print(f"🛑 [주문 거절] {code}: {res_data.get('msg1')}")
        
    return res_data.get('rt_cd') == '0'

def create_pdf():
    if not os.path.exists('vintage_performance.csv'): return None
    df = pd.read_csv('vintage_performance.csv').fillna('')
    f_p = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(f_p): fm.fontManager.addfont(f_p); plt.rcParams['font.family'] = 'NanumGothic'
    fig, ax = plt.subplots(figsize=(14, 8)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    for (r, c), cell in table.get_celld().items():
        if r > 0 and c >= 2:
            try:
                val = float(cell.get_text().get_text())
                if val > 0: cell.get_text().set_color('#d32f2f') 
                elif val < 0: cell.get_text().set_color('#1976d2')
            except: pass
    name = f"MITSUDA_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    with PdfPages(name) as pdf: pdf.savefig(fig, bbox_inches='tight')
    plt.close(); return name

def main():
    token = get_token()
    if not token:
        print("🚨 토큰 발급 실패")
        return
        
    pdf = create_pdf()
    if pdf:
        with open(pdf, 'rb') as f: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument", data={"chat_id":TG_CHAT_ID}, files={"document":f})
        os.remove(pdf)

    try:
        with open('meta_target_list.json', 'r', encoding='utf-8') as f: data = json.load(f)
    except: 
        print("🚨 작전 지시서(JSON)를 찾을 수 없습니다.")
        return

    buy_log = []
    
    # 내 계좌 진짜 돈 확인
    total_cash = get_available_cash(token)
    print(f"🔍 [오늘의 투자 가능 총액]: {total_cash}원")
    
    for t in data['targets']:
        invest_amount = total_cash * t['kelly']
        qty = int(invest_amount / t['price'])
        
        print(f"🎯 [타겟 확인] {t['name']} - 필요금액: {invest_amount}원 / 계산된 수량: {qty}주")
        
        if qty > 0:
            cancel_order(token, t['code']) 
            if execute_order(token, t['code'], qty, "buy"):
                buy_log.append(f"✅ {t['name']} ({t['score']}점) {qty}주 매수 (비중: {round(t['kelly']*100, 1)}%)")
    
    msg = f"🐆 MITSUDA 작전 보고 ({data['date']})\n존: {data['zone']}\n\n" + ("\n".join(buy_log) if buy_log else "매수 체결 내역 없음 (관망).")
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":msg})

if __name__ == "__main__":
    main()
