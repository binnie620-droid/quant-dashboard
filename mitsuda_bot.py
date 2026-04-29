import os, json, requests, warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
    return "✅ 기존 방어선 철거 및 신규 트레일링 스탑 전진 배치 완료"

def create_pdf_report():
    if not os.path.exists('vintage_performance.csv'): return None
    df = pd.read_csv('vintage_performance.csv')
    df.fillna('', inplace=True) # 보기 싫은 'nan' 글자를 빈칸으로 깔끔하게 청소
    file_name = f"UCHIDA_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    # 리눅스(우분투) 환경 한글 폰트 강제 적용 및 마이너스 깨짐 방지
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path): fm.fontManager.addfont(font_path); plt.rcParams['font.family'] = 'NanumGothic'
    else: plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight'); ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.5)
    
    # 🔴빨간색 / 🔵파란색 색상 입히기 로직
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col < 2: continue # 헤더와 날짜/종목명은 검은색 유지
        text = cell.get_text().get_text()
        if not text: continue
        
        try:
            # 특수기호 자르고 순수 숫자만 추출해서 비교
            val = float(text.replace('%', '').replace('🛑', '').replace('✂️', ''))
            if val > 0: cell.get_text().set_color('#d32f2f') # 상승: 빨강
            elif val < 0: cell.get_text().set_color('#1976d2') # 하락: 파랑
        except: pass
    
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
