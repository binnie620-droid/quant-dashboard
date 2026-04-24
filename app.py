import streamlit as st
import FinanceDataReader as fdr
import pandas_ta as ta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# [설정] 페이지 레이아웃
st.set_page_config(page_title="KOSPI 50 AI 스윙 봇", layout="wide")
st.title("📈 KOSPI Top 50 AI 스윙 트레이딩 시스템")

# [로직] 데이터 수집 및 모델 학습 (캐싱 적용)
@st.cache_data(ttl=3600)
def get_ai_predictions():
    df_krx = fdr.StockListing('KOSPI')
    top_stocks = df_krx.head(50)[['Code', 'Name']].values 

    df_fx = fdr.DataReader('USD/KRW', '2019-01-01')[['Close']].rename(columns={'Close':'Exchange_Rate'})
    df_kospi = fdr.DataReader('KS11', '2019-01-01')[['Close']]
    df_kospi['KOSPI_Trend_20d'] = df_kospi['Close'].pct_change(20) * 100

    train_data, stock_dfs = [], {}

    for code, name in top_stocks:
        try:
            df = fdr.DataReader(code, '2019-01-01')
            if len(df) < 500: continue
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
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

model, stock_dfs, features = get_ai_predictions()

# 수익률에 색상을 입히는 마법 함수
def style_returns(val):
    if isinstance(val, str) and '%' in val:
        try:
            # + 또는 - 기호를 제거하고 숫자로 변환
            clean_val = val.replace('%', '').replace('+', '')
            num = float(clean_val)
            if num > 0:
                return 'color: #ef5350; font-weight: bold;' # 한국형 상승 (빨강)
            elif num < 0:
                return 'color: #42a5f5; font-weight: bold;' # 한국형 하락 (파랑)
        except:
            return ''
    return ''

tab1, tab2 = st.tabs(["🎯 오늘의 추천 종목", "📊 과거 성과 매트릭스 (n*10)"])

with tab1:
    st.subheader(f"📅 분석 기준일: {datetime.now().strftime('%Y-%m-%d')}")
    today_picks = []
    for name, df in stock_dfs.items():
        row = df.iloc[[-1]]
        X_today = row[features].fillna(0)
        proba = model.predict_proba(X_today)[0, 1]
        
        if proba >= 0.55:
            current_p = row['Close'].values[0]
            atr_val = current_p * (row['ATR_Pct'].values[0] / 100)
            today_picks.append({
                '종목명': name, '확신도': f"{proba*100:.1f}%",
                '매수범위': f"{int(current_p):,} ~ {int(current_p*1.01):,}",
                '목표가': f"{int(current_p + atr_val*1.5):,}", '손절가': f"{int(current_p - atr_val*1.0):,}",
                'sort': proba
            })
            
    if today_picks:
        df_picks = pd.DataFrame(today_picks).sort_values('sort', ascending=False).drop(columns=['sort'])
        st.dataframe(df_picks, use_container_width=True, hide_index=True)
    else:
        st.info("😴 현재 확신도 55%를 넘는 종목이 없습니다. 현금을 보유하며 관망하세요.")

with tab2:
    st.subheader("🕵️‍♂️ AI 추천 이후 실제 수익률 추적")
    st.write("지난 10거래일간 AI가 55% 이상의 확신으로 추천했던 종목들의 실시간 수익 현황입니다.")
    
    if st.button('매트릭스 정산 시작'):
        with st.status("데이터를 전수 조사하고 색상을 입히는 중입니다...", expanded=True) as status:
            matrix_rows = []
            for name, df in stock_dfs.items():
                if len(df) < 30: continue
                for i in range(20, 10, -1):
                    past_row = df.iloc[[-i]]
                    X_past = past_row[features].fillna(0)
                    proba = model.predict_proba(X_past)[0, 1]
                    
                    if proba >= 0.55:
                        entry_p = past_row['Close'].values[0]
                        res = {'추천일': past_row.index[0].strftime('%m/%d'), '종목명': name, '확신도': f"{proba*100:.1f}%"}
                        for d in range(1, 11):
                            idx = -i + d
                            if idx < 0:
                                ret = (df.iloc[idx]['Close'] / entry_p - 1) * 100
                                res[f"T+{d}"] = f"{ret:+.2f}%"
                            else: res[f"T+{d}"] = "-"
                        matrix_rows.append(res)
            status.update(label="정산 및 시각화 완료!", state="complete", expanded=False)

        if matrix_rows:
            df_m = pd.DataFrame(matrix_rows).sort_values(by='추천일', ascending=False)
            # 🎨 스타일 적용 핵심 코드: 모든 셀에 대해 style_returns 함수 적용
            st.dataframe(df_m.style.map(style_returns), use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 최근 10거래일 동안 AI 확신도 55%를 넘긴 추천 종목이 없었습니다.")

st.sidebar.markdown("---")
st.sidebar.write("✅ **색상 가이드**")
st.sidebar.write("🔴 **빨간색**: 추천가 대비 수익 중")
st.sidebar.write("🔵 **파란색**: 추천가 대비 손실 중")
