import streamlit as st
import FinanceDataReader as fdr
from pykrx import stock  # ⭐️ 수급 데이터를 가져오는 핵심 라이브러리 추가
import pandas_ta as ta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="KOSPI 50 AI 스윙 봇", layout="wide")
st.title("📈 KOSPI 50 AI 스윙 매매 시스템 (수급 분석 탑재)")

@st.cache_data(ttl=3600)
def get_ai_predictions():
    df_krx = fdr.StockListing('KOSPI')
    top_stocks = df_krx.head(50)[['Code', 'Name']].values 

    df_fx = fdr.DataReader('USD/KRW', '2019-01-01')[['Close']].rename(columns={'Close':'Exchange_Rate'})
    df_kospi = fdr.DataReader('KS11', '2019-01-01')[['Close']]
    df_kospi['KOSPI_Trend_20d'] = df_kospi['Close'].pct_change(20) * 100

    train_data, stock_dfs = [], {}
    today_str = datetime.now().strftime('%Y%m%d')

    # 상태 진행바 추가 (데이터 수집이 오래 걸리므로)
    progress_text = "서버에서 50개 종목의 5년 치 가격 및 수급 데이터를 긁어오고 있습니다. (약 1~2분 소요)..."
    my_bar = st.progress(0, text=progress_text)

    for idx, (code, name) in enumerate(top_stocks):
        try:
            # 1. 가격 데이터
            df = fdr.DataReader(code, '2019-01-01')
            if len(df) < 500: continue
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # ⭐️ 2. 수급 데이터 (pykrx) 추가
            try:
                # 2019년부터 오늘까지의 투자자별 순매수 거래량
                df_investor = stock.get_market_trading_volume_by_date("20190101", today_str, code)
                
                # 전체 거래량 대비 외국인/기관이 얼마나 샀는지 비율(%)로 변환
                df['Inst_Ratio'] = (df_investor['기관합계'] / df['Volume']) * 100
                df['Fore_Ratio'] = (df_investor['외국인'] / df['Volume']) * 100
                time.sleep(0.1) # 한국거래소 서버 차단 방지용 휴식
            except:
                df['Inst_Ratio'] = 0
                df['Fore_Ratio'] = 0
            
            # 3. 기술적 지표
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df['MACD_Pct'] = (macd.iloc[:, 0] / df['Close']) * 100
            bb = ta.bbands(df['Close'])
            df['BB_PctB'] = (df['Close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
            df['BB_Bandwidth'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
            df['OBV_ROC'] = ta.obv(df['Close'], df['Volume']).pct_change(5)
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ATR_Pct'] = (atr / df['Close']) * 100
            
            df = df.join(df_fx).join(df_kospi[['KOSPI_Trend_20d']]).ffill()
            
            # ⭐️ 4. 타겟 변경: 넉넉한 방어선(-2.0)과 거대한 목표(+3.0)
            future_high = df['High'].rolling(window=10, min_periods=1).max().shift(-10)
            future_low = df['Low'].rolling(window=10, min_periods=1).min().shift(-10)
            
            profit_line = df['Close'] + (atr * 3.0) # 목표가 상향
            stop_line = df['Close'] - (atr * 2.0)
            
            df['Target'] = ((future_high >= profit_line) & (future_low > stop_line)).astype(int)
            
            df['Name'] = name
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            stock_dfs[name] = df
            train_data.append(df.iloc[:-10].dropna())
        except: continue
        
        # 진행률 업데이트
        my_bar.progress((idx + 1) / 50, text=f"{name} 데이터 적재 완료... ({idx+1}/50)")

    my_bar.empty() # 로딩바 숨기기

    df_train = pd.concat(train_data)
    
    # ⭐️ 피처(조건)에 수급 데이터 추가
    features = ['Exchange_Rate', 'KOSPI_Trend_20d', 'RSI_14', 'MACD_Pct', 'BB_PctB', 'OBV_ROC', 'Volume_Ratio', 'BB_Bandwidth', 'ATR_Pct', 'Inst_Ratio', 'Fore_Ratio']
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(df_train[features], df_train['Target'].astype(int))
    
    return model, stock_dfs, features

model, stock_dfs, features = get_ai_predictions()

def style_returns(val):
    if isinstance(val, str) and '%' in val:
        try:
            clean_val = val.replace('%', '').replace('+', '')
            num = float(clean_val)
            if num > 0: return 'color: #ef5350; font-weight: bold;' 
            elif num < 0: return 'color: #42a5f5; font-weight: bold;' 
        except: return ''
    return ''

tab1, tab2 = st.tabs(["🎯 오늘의 추천 종목", "📊 과거 성과 매트릭스"])

with tab1:
    st.subheader(f"📅 분석 기준일: {datetime.now().strftime('%Y-%m-%d')}")
    today_picks = []
    for name, df in stock_dfs.items():
        row = df.iloc[[-1]]
        X_today = row[features].fillna(0)
        proba = model.predict_proba(X_today)[0, 1]
        
        if proba >= 0.60:
            current_p = row['Close'].values[0]
            atr_val = current_p * (row['ATR_Pct'].values[0] / 100)
            
            # 수급 데이터 표시용
            inst_r = row['Inst_Ratio'].values[0]
            fore_r = row['Fore_Ratio'].values[0]
            
            today_picks.append({
                '종목명': name, '확신도': f"{proba*100:.1f}%",
                '외인순매수비중': f"{fore_r:+.1f}%", '기관순매수비중': f"{inst_r:+.1f}%", # ⭐️ 수급 정보 화면에 추가
                '매수범위': f"{int(current_p):,} ~ {int(current_p*1.01):,}",
                '목표가': f"{int(current_p + atr_val*3.0):,}", # ⭐️ 3.0 적용
                '손절가': f"{int(current_p - atr_val*2.0):,}",
                'sort': proba
            })
            
    if today_picks:
        df_picks = pd.DataFrame(today_picks).sort_values('sort', ascending=False).drop(columns=['sort'])
        st.dataframe(df_picks, use_container_width=True, hide_index=True)
    else:
        st.info("😴 현재 확신도 60%를 넘는 종목이 없습니다. 확실한 기회가 올 때까지 현금을 보유하세요.")

with tab2:
    st.subheader("🕵️‍♂️ AI 추천 이후 실시간 수익률 매트릭스")
    
    if st.button('매트릭스 정산 시작'):
        with st.status("수급 데이터까지 전수 조사 중입니다...", expanded=True) as status:
            matrix_rows = []
            for name, df in stock_dfs.items():
                if len(df) < 30: continue
                for i in range(15, 0, -1):
                    try:
                        past_row = df.iloc[[-i]]
                        X_past = past_row[features].fillna(0)
                        proba = model.predict_proba(X_past)[0, 1]
                        
                        if proba >= 0.60:
                            entry_p = past_row['Close'].values[0]
                            res = {'추천일': past_row.index[0].strftime('%m/%d'), '종목명': name, '확신도': f"{proba*100:.1f}%"}
                            for d in range(1, 11):
                                target_idx = -i + d
                                if target_idx < 0:
                                    res[f"T+{d}"] = f"{(df.iloc[target_idx]['Close'] / entry_p - 1) * 100:+.2f}%"
                                else: res[f"T+{d}"] = "-"
                            matrix_rows.append(res)
                    except: continue
            status.update(label="정산 완료!", state="complete", expanded=False)

        if matrix_rows:
            df_m = pd.DataFrame(matrix_rows).sort_values(by=['추천일'], ascending=False)
            return_cols = [col for col in df_m.columns if col.startswith('T+')]
            try: styled_df = df_m.style.map(style_returns, subset=return_cols)
            except: styled_df = df_m.style.applymap(style_returns, subset=return_cols)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 최근 15거래일 동안 확신도 60%를 넘긴 엄선된 종목이 없었습니다.")
