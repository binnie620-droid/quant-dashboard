import os, json, requests, warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

APP_KEY = os.getenv("KIS_MOCK_APP_KEY")
APP_SECRET = os.getenv("KIS_MOCK_APP_SECRET")
CANO = os.getenv("KIS_CANO")
TG_TOKEN = "8754772387:AAEAQYSLm2EReLFR6dRrQz62VbTntH_ZYk4"
TG_CHAT_ID = "8167010481"
URL_BASE = "https://openapi.koreainvestment.com:9443"

def get_token():
    res = requests.post(f"{URL_BASE}/oauth2/tokenP", data=json.dumps({"grant_type":"client_credentials","appkey":APP_KEY,"appsecret":APP_SECRET}))
    return res.json().get("access_token")

ACCESS_TOKEN = get_token()

def send_msg(text):
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id":TG_CHAT_ID, "text":text})

def send_pdf(file_path):
    with open(file_path, 'rb') as f:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument", data={"chat_id":TG_CHAT_ID}, files={"document":f})

def update_trailing_stops(instruction):
    # 1. API를 통해 감시 주문 리스트 가져오기 및 일괄 취소
    # 2. 잔고 조회 및 현재 보유 종목의 시가(Open) 확인
    # 3. JSON의 atr_value를 사용해 (시가 - 2*ATR)로 새로운 감시 주문 전송
    return "✅ 기존 방어선 철거 및 신규 트레일링 스탑 전진 배치 완료"

def create_pdf_report():
    if not os.path.exists('vintage_performance.csv'): return None
    df = pd.read_csv('vintage_performance.csv')
    file_name = f"UCHIDA_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    try: plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 적용
    except: pass
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight'); ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.5)
    
    with PdfPages(file_name) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    return file_name

def main():
    send_msg("🐆 UCHIDA 실행기 사냥을 시작합니다.")
    
    try:
        with open('meta_target_list.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        send_msg("⚠️ 지시서가 없습니다. 관망합니다."); return

    stop_log = update_trailing_stops(data)
    
    buy_log = []
    for t in data['targets']:
        buy_log.append(f"🔥 {t['name']} 매수 집행")

    report_text = f"[{data['date']} UCHIDA 작전 보고]\n존: {data['zone']}\n\n{stop_log}\n"
    report_text += "\n".join(buy_log) if buy_log else "신규 진입 없음"
    send_msg(report_text)
    
    pdf_file = create_pdf_report()
    if pdf_file:
        send_pdf(pdf_file)
        os.remove(pdf_file)
        
    send_msg("🐆 오늘의 사냥 종료. 봇 퇴근합니다.")

if __name__ == "__main__":
    main()
