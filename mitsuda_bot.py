import os, json, requests, warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import pandas_ta_classic as ta
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

APP_KEY, APP_SECRET, CANO = os.getenv("KIS_MOCK_APP_KEY"), os.getenv("KIS_MOCK_APP_SECRET"), os.getenv("KIS_CANO")
TG_TOKEN, TG_CHAT_ID = "8754772387:AAEAQYSLm2EReLFR6dRrQz62VbTntH_ZYk4", "8167010481"
URL_BASE = "https://openapivts.koreainvestment.com:29443"

def get_token():
    res = requests.post(f"{URL_BASE}/oauth2/tokenP", data=json.dumps({"grant_type":"client_credentials","appkey":APP_KEY,"appsecret":APP_SECRET}))
    return res.json().get("access_token")

def get_available_cash(token):
    headers = {"Content-Type": "application/json", "authorization": f"Bearer {token}", "appkey": APP_KEY, "appsecret": APP_SECRET, "tr_id": "VTTC8434R"}
    # 🚨 [수정 완료] OFLN_YN 오타를 KIS 공식 규격인 OFL_YN으로 수정
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": "01",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "N", 
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
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
    if res_data.get('rt_cd') != '0': print(f"🛑 [주문 거절] {code}: {res_data.get('msg1')}")
    return res_data.get('rt_cd') == '0'

def run_trading_cleanup(token, zone):
    headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":APP_KEY, "appsecret":APP_SECRET, "tr_id":"VTTC8434R"}
    # 🚨 [수정 완료] 매도 감시 쪽 잔고조회 파라미터 동일하게 오타 수정
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": "01",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "N",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }
    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
    
    sell_log = []
    multiplier = 1.2 if zone == "🔴 RED" else 2.0
    print(f"🛡️ [보유종목 감시 시작] 적용 손절 배수: ATR {multiplier}배")
    
    try:
        for s in res.json().get('output1', []):
            code, name, qty = s['pdno'], s['prdt_name'], int(s['hldg_qty'])
            if qty <= 0: continue
            
            entry_price, current_price, rate = float(s['pchs_avg_pric']), float(s['prpr']), float(s['evlu_pfls_rt'])
            
            # 실시간 ATR 동적 계산
            df = fdr.DataReader(code, (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d"))
            if len(df) > 14:
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                current_atr = float(df['ATR'].iloc[-1])
                stop_price = entry_price - (current_atr * multiplier)
                
                if current_price <= stop_price or rate >= 10.0:
                    cancel_order(token, code)
                    if execute_order(token, code, qty, "sell"):
                        reason = f"ATR {multiplier}배 손절" if current_price <= stop_price else "익절"
                        sell_log.append(f"🛑 {name} 매도 완료 ({reason})")
    except Exception as e: print(f"🚨 [매도 감시 에러]: {e}")
    return sell_log

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
    if not token: return
        
    try:
        with open('meta_target_list.json', 'r', encoding='utf-8') as f: data = json.load(f)
        zone = data.get('zone', '🟢 GREEN')
    except: return

    # 1. 매도 감시 및 취소
    sell_msgs = run_trading_cleanup(token, zone)
    if sell_msgs: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":"\n".join(sell_msgs)})

    # 2. 레포트 발송
    pdf = create_pdf()
    if pdf:
        with open(pdf, 'rb') as f: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument", data={"chat_id":TG_CHAT_ID}, files={"document":f})
        os.remove(pdf)

    # 3. 매수 집행 및 보고서 작성
    buy_log = []
    target_log = [] 
    total_cash = get_available_cash(token)
    
    for t in data['targets']:
        target_log.append(f"🎯 {t['name']} ({t['score']}점)")
        invest_amount = total_cash * t['kelly']
        qty = int(invest_amount / t['price']) if t['price'] > 0 else 0
        
        if qty > 0:
            cancel_order(token, t['code']) 
            if execute_order(token, t['code'], qty, "buy"):
                buy_log.append(f"✅ {t['name']} {qty}주 매수")
    
    # 보고서 조립
    msg = f"🐆 MITSUDA 작전 보고 ({data['date']})\n존: {zone}\n\n"
    msg += "--- 오늘의 타겟 (AI 추천) ---\n"
    msg += ("\n".join(target_log) if target_log else "추천 종목 없음.") + "\n\n"
    msg += "--- 실제 매수 체결 ---\n"
    msg += ("\n".join(buy_log) if buy_log else "체결 내역 없음 (현금 부족 또는 관망).")
    
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":msg})

if __name__ == "__main__":
    main()
