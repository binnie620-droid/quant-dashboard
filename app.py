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
