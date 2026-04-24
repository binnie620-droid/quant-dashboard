import streamlit as st
import FinanceDataReader as fdr
import pandas_ta as ta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 웹페이지 설정 (모바일 최적화)
st.set_page_config(page_title="나의 퀀트 비서", page_icon="📈", layout="centered")

st.title("🚀 오늘의 KOSPI Top 50 스윙 타점")
st.write("매일 오후 3시, AI가 분석한 최적의 진입 구간을 확인하세요.")

# 분석 시작 버튼
if st.button('데이터 분석 및 AI 모델 가동 시작'):
    with st.spinner('KOSPI 50개 종목의 방대한 데이터를 수집하고 AI가 학습 중입니다...'):
        
        # 1. 데이터 준비
        df_krx = fdr.StockListing('KOSPI')
        top_stocks = df_krx.head(50)[['Code', 'Name']].values 
        
        df_fx = fdr.DataReader('USD/KRW', '2019-01-01')[['Close']].rename(columns={'Close':'Exchange_Rate'})
        df_kospi = fdr.DataReader('KS11', '2019-01-01')[['Close']]
        df_kospi['KOSPI_Trend_20d'] = df_kospi['Close'].pct_change(20) * 100

        train_data, today_data = [], []

        for code, name in top_stocks:
            try:
                df = fdr.DataReader(code, '2019-01-01')
                if len(df) < 500: continue
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                
                # 지표 계산
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
                df['Target'] = ((df['Close'].shift(-10) / df['Close'] - 1) >= 0.025).astype(float)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                today_row = df.iloc[[-1]].copy()
                today_row['Name'] = name
                today_data.append(today_row)
                train_data.append(df.iloc[:-10].dropna())
            except: continue

        df_train = pd.concat(train_data)
        df_today = pd.concat(today_data)
        
        # 2. AI 학습
        features = ['Exchange_Rate', 'KOSPI_Trend_20d', 'RSI_14', 'MACD_Pct', 'BB_PctB', 'OBV_ROC', 'Volume_Ratio', 'BB_Bandwidth', 'ATR_Pct']
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(df_train[features], df_train['Target'].astype(int))

        # 3. 예측 및 브리핑
        probs = model.predict_proba(df_today[features].fillna(0))[:, 1]
        df_today['Proba'] = probs
        picks = df_today[df_today['Proba'] >= 0.55]

        # 4. 화면 출력
        st.subheader(f"📅 분석 기준일: {df_today.index.max().strftime('%Y-%m-%d')}")
        
        if not picks.empty:
            buy_list = []
            for _, row in picks.iterrows():
                atr_val = row['Close'] * (row['ATR_Pct'] / 100)
                buy_list.append({
                    '종목명': row['Name'],
                    '확신도': f"{row['Proba']*100:.1f}%",
                    '진입범위': f"{int(row['Close']):,} ~ {int(row['Close']*1.01):,}",
                    '목표가': f"{int(row['Close'] + atr_val*1.5):,}",
                    '손절가': f"{int(row['Close'] - atr_val*1.0):,}"
                })
            
            st.success(f"🔥 총 {len(buy_list)}개의 종목이 포착되었습니다!")
            st.dataframe(pd.DataFrame(buy_list), use_container_width=True)
            st.info("💡 목표가 도달 시 익절, 손절가 터치 시 칼손절을 엄수하세요.")
        else:
            st.warning("😴 오늘은 확실한 자리가 없습니다. 현금을 보유하세요.")

import streamlit as st
import FinanceDataReader as fdr
import pandas_ta as ta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# [설정] 페이지 레이아웃 및 제목
st.set_page_config(page_title="KOSPI 50 AI 스윙 봇", layout="wide")
st.title("📈 KOSPI Top 50 AI 스윙 트레이딩 시스템")
st.markdown("매일 오후 3시, AI가 학습한 패턴을 바탕으로 다음 10거래일의 승률을 예측합니다.")

# [로직] 데이터 수집 및 모델 학습 함수 (캐싱 적용으로 속도 향상)
@st.cache_data(ttl=3600) # 1시간 동안 결과 기억
def get_ai_predictions():
    df_krx = fdr.StockListing('KOSPI')
    top_stocks = df_krx.head(50)[['Code', 'Name']].values 

    df_fx = fdr.DataReader('USD/KRW', '2019-01-01')[['Close']].rename(columns={'Close':'Exchange_Rate'})
    df_kospi = fdr.DataReader('KS11', '2019-01-01')[['Close']]
    df_kospi['KOSPI_Trend_20d'] = df_kospi['Close'].pct_change(20) * 100

    train_data = []
    stock_dfs = {} # 과거 추적을 위해 전체 데이터 저장

    for code, name in top_stocks:
        try:
            df = fdr.DataReader(code, '2019-01-01')
            if len(df) < 500: continue
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # 지표 계산
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
            df['Target'] = ((df['Close'].shift(-10) / df['Close'] - 1) >= 0.025).astype(float)
            df['Name'] = name
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            stock_dfs[name] = df
            train_data.append(df.iloc[:-10].dropna())
        except: continue

    df_train = pd.concat(train_data)
    features = ['Exchange_Rate', 'KOSPI_Trend_20d', 'RSI_14', 'MACD_Pct', 'BB_PctB', 'OBV_ROC', 'Volume_Ratio', 'BB_Bandwidth', 'ATR_Pct']
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(df_train[features], df_train['Target'].astype(int))
    
    return model, stock_dfs, features

# [실행] 모델 돌리기
model, stock_dfs, features = get_ai_predictions()

# --- 탭 구성 ---
tab1, tab2 = st.tabs(["🎯 오늘의 추천 종목", "📊 과거 성과 매트릭스 (n*10)"])

with tab1:
    st.subheader(f"📅 분석 기준일: {datetime.now().strftime('%Y-%m-%d')} 장 마감 데이터")
    
    today_picks = []
    for name, df in stock_dfs.items():
        row = df.iloc[[-1]]
        X_today = row[features].fillna(0)
        proba = model.predict_proba(X_today)[0, 1]
        
        if proba >= 0.55:
            current_p = row['Close'].values[0]
            atr_p = row['ATR_Pct'].values[0]
            atr_val = current_p * (atr_p / 100)
            
            today_picks.append({
                '종목명': name,
                '확신도': f"{proba*100:.1f}%",
                '매수범위': f"{int(current_p):,} ~ {int(current_p*1.01):,}",
                '목표가': f"{int(current_p + atr_val*1.5):,}",
                '손절가': f"{int(current_p - atr_val*1.0):,}",
                'Proba_Val': proba
            })
            
    if today_picks:
        df_picks = pd.DataFrame(today_picks).sort_values(by='Proba_Val', ascending=False)
        st.dataframe(df_picks.drop(columns=['Proba_Val']), use_container_width=True, hide_index=True)
    else:
        st.write("😴 오늘은 추천 종목이 없습니다.")

with tab2:
    st.subheader("🕵️‍♂️ AI 추천 이후 실제 수익률 추적")
    st.write("과거 10거래일 동안 AI가 추천했던 종목들이 이후 1~10일 동안 어떤 수익률을 기록했는지 보여줍니다.")
    
    if st.button('매트릭스 조회 (데이터 정산 시작)'):
        matrix_rows = []
        
        # 최근 20일 중, '추천 시점'으로 쓸 10일 (T-20 ~ T-10일 전)
        # 각 종목별로 루프를 돌며 과거 시점의 추천 여부 확인
        for name, df in stock_dfs.items():
            if len(df) < 30: continue
            
            # 최근 20거래일 데이터 중 뒤에서 20번째~11번째 날을 '추천일'로 가정
            for i in range(20, 10, -1):
                past_row = df.iloc[[-i]]
                past_date = past_row.index[0].strftime('%m/%d')
                
                X_past = past_row[features].fillna(0)
                proba = model.predict_proba(X_past)[0, 1]
                
                if proba >= 0.55: # 과거 그날 AI가 추천했다면
                    entry_price = past_row['Close'].values[0]
                    returns = {'추천일': past_date, '종목명': name}
                    
                    # 추천일 이후 1~10일간의 수익률 계산
                    for day in range(1, 11):
                        after_idx = -i + day
                        if after_idx < 0: # 아직 오지 않은 날짜 제외
                            current_val = df.iloc[after_idx]['Close']
                            ret = (current_val / entry_price - 1) * 100
                            returns[f"T+{day}"] = f"{ret:+.2f}%"
                        else:
                            returns[f"T+{day}"] = "-"
                    
                    matrix_rows.append(returns)
        
        if matrix_rows:
            df_matrix = pd.DataFrame(matrix_rows).sort_values(by='추천일', ascending=False)
            
            # 가독성을 위한 스타일 적용 (양수는 빨간색, 음수는 파란색)
            def color_returns(val):
                if isinstance(val, str) and '%' in val:
                    num = float(val.replace('%', ''))
                    color = 'red' if num > 0 else 'blue'
                    return f'color: {color}'
                return ''
            
            st.dataframe(df_matrix.style.applymap(color_returns), use_container_width=True, hide_index=True)
        else:
            st.write("해당 기간 동안 AI가 추천했던 종목이 없습니다.")

st.sidebar.write("---")
st.sidebar.info("💡 **트레이딩 가이드**\n1. 매일 15:15분 분석 권장\n2. T+n은 추천 시점 이후 누적 수익률\n3. 손절가 터치 시 무조건 정리")
