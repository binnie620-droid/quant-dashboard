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

# [복구] 주문 취소 함수
def cancel_all_orders(token, code):
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":"VTTC0803U"}
    data = {"CANO":CANO, "ACNT_PRDT_CD":"01", "ORGN_ODNO":"", "RVSE_CNCL_DVSN_CD":"02", "ORD_QTY":"0", "ORD_UNPR":"0", "QTY_ALL_ORD_FLG":"Y"}
    requests.post(f"{URL_BASE}/uapi/domestic-stock/v1/trading/order-rvse-cncl", headers=headers, data=json.dumps(data))

def execute_order(token, code, qty, side="buy"):
    tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":tr_id}
    data = {"CANO":CANO, "ACNT_PRDT_CD":"01", "PDNO":code, "ORD_DVSN":"01", "ORD_QTY":str(qty), "ORD_UNPR":"0"}
    res = requests.post(f"{URL_BASE}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, data=json.dumps(data))
    return res.json().get('rt_cd') == '0'

# [추가] 실시간 계좌 예수금(주문 가능 현금) 조회 함수
def get_available_cash(token):
    headers = {
        "Content-Type": "application/json", 
        "authorization": f"Bearer {token}", 
        "appkey": APP_KEY, 
        "appsecret": APP_SECRET, 
        "tr_id": "VTTC8436R"  # 모의계좌 잔고조회 TR ID (실전은 TTTC8436R)
    }
    params = {
        "CANO": CANO, 
        "ACNT_PRDT_CD": "01", 
        "EXPT_SETL_CMPD_DVSN_CD": "00", 
        "INQR_DVSN_1": "00", 
        "INQR_DVSN_2": "00"
    }
    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
    try:
        # dnca_tot_amt: 예수금총액 (주문 가능한 실제 현금)
        cash = int(res.json()['output2'][0]['dnca_tot_amt'])
        return cash
    except:
        return 0 # 에러 시 0원 처리

# [복구] 어깨 매도 및 취소 로직
def run_trading_cleanup(token):
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":"VTRP6641R"}
    params = {"CANO":CANO, "ACNT_PRDT_CD":"01", "PRDT_TYPE_CD":"01", "CTX_AREA_FK100":"", "CTX_AREA_NK100":""}
    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
    
    sell_log = []
    try:
        for s in res.json()['output1']:
            code, name, qty = s['pdno'], s['prdt_name'], int(s['hldg_qty'])
            rate = float(s['evlu_pfit_rt'])
            # 손절선(-4%) 혹은 익절 후 꺾임 감지 시
            if rate <= -4.0 or rate >= 10.0:
                cancel_all_orders(token, code)
                if execute_order(token, code, qty, "sell"):
                    sell_log.append(f"🛑 {name} 어깨 매도 완료")
    except: pass
    return sell_log

# [복구] PDF 색깔 입히기
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
                v = float(cell.get_text().get_text())
                cell.get_text().set_color('#d32f2f' if v > 0 else '#1976d2')
            except: pass
    name = f"MITSUDA_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    with PdfPages(name) as pdf: pdf.savefig(fig, bbox_inches='tight')
    plt.close(); return name

def main():
    token = get_token()
    # 1. 매도 감시 및 취소
    sell_msgs = run_trading_cleanup(token)
    if sell_msgs: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":"\n".join(sell_msgs)})
    
    # 2. 레포트 발송
    pdf = create_pdf()
    if pdf:
        with open(pdf, 'rb') as f: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument", data={"chat_id":TG_CHAT_ID}, files={"document":f})
        os.remove(pdf)

    # 3. 매수 집행 및 브리핑
    try:
        with open('meta_target_list.json', 'r', encoding='utf-8') as f: data = json.load(f)
    except: return

    buy_log = []
    
    # 1. 오늘자 내 계좌의 찐 현금을 파악한다
    total_cash = get_available_cash(token) 
    
    for t in data['targets']:
        # 2. 내 실제 현금에 켈리 비중을 곱해서 투자 금액을 산정한다
        invest_amount = total_cash * t['kelly'] 
        qty = int(invest_amount / t['price'])
        
        if qty > 0:
            cancel_order(token, t['code'])
            if execute_order(token, t['code'], qty, "buy"):
                buy_log.append(f"✅ {t['name']} ({t['score']}점) {qty}주 매수 (비중: {round(t['kelly']*100, 1)}%)")
            cancel_all_orders(token, t['code'])
            if execute_order(token, t['code'], qty, "buy"):
                buy_log.append(f"✅ {t['name']} ({t['score']}점) {qty}주 매수")
    
    # [복구] 미츠다 작전 보고 메시지
    msg = f"🐆 MITSUDA 작전 보고 ({data['date']})\n존: {data['zone']}\n\n"
    msg += "\n".join(buy_log) if buy_log else "오늘의 사냥감은 없습니다."
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":msg})

if __name__ == "__main__":
    main()
