import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import pandas_ta_classic as ta
from lightgbm import LGBMClassifier
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

APP_KEY, APP_SECRET = os.getenv("KIS_MOCK_APP_KEY"), os.getenv("KIS_MOCK_APP_SECRET")
TODAY = datetime.now()

def get_all_data():
    res = requests.post("https://openapi.koreainvestment.com:9443/oauth2/tokenP", 
                        data=json.dumps({"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}))
    token = res.json().get("access_token")
    if not token: return None, None, None

    top_stocks = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(100)
    stocks = dict(zip(top_stocks['Code'], top_stocks['Name']))
    start_date = (TODAY - timedelta(days=500)).strftime("%Y-%m-%d")
    
    ks = fdr.DataReader('KS11', start_date)
    vix = fdr.DataReader('VIX', start_date) # 혹은 VKOSPI
    ks['KOSPI_MA20'] = ta.sma(ks['Close'], length=20)
    macro_df = pd.concat([ks[['Close', 'KOSPI_MA20']], vix[['Close']]], axis=1).dropna()
    macro_df.columns = ['KOSPI', 'KOSPI_MA20', 'VIX']

    # [디테일 3] VIX 자동 보정 로직 (상위 70%, 90% 퀀타일)
    vix_yellow = macro_df['VIX'].quantile(0.7)
    vix_red = macro_df['VIX'].quantile(0.9)

    data_list = {}
    for code, name in stocks.items():
        try:
            df = fdr.DataReader(code, start_date)
            if len(df) < 75: continue
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
            
            # [디테일 2] 트리플 배리어 학습 (고정 3%가 아닌, ATR 기반 생존 학습)
            entry_p = df['Open'].shift(-1)
            stop_level = entry_p - (df['ATR'] * 2.0)
            df['Target'] = np.where((df['High'].rolling(7).max().shift(-7) / entry_p - 1.0 >= 0.04) & 
                                    (df['Low'].rolling(7).min().shift(-7) > stop_level), 1, 0)
            
            df['Is_Valid'] = (df['Close'] > df['MA20'])
            df['Date'] = df.index; df['Code'] = code; df['Name'] = name
            data_list[code] = df.dropna()
        except: continue
    return data_list, macro_df, stocks, vix_yellow, vix_red

def run_mitsuda_engine():
    data_list, macro_df, stocks, v_y, v_r = get_all_data()
    if not data_list: return

    all_combined = pd.concat(data_list.values())
    features = ['MFI', 'ATR']
    
    curr = macro_df.iloc[-1]
    is_bull = curr['KOSPI'] > curr['KOSPI_MA20']
    
    # [디테일 1 & 3] 0.62 수치 반영 및 동적 존 설정
    if curr['VIX'] >= v_r: zone, threshold = "🔴 RED", 0.95
    elif curr['VIX'] >= v_y or not is_bull: zone, threshold = "🟡 YELLOW", 0.75
    else: zone, threshold = "🟢 GREEN", 0.62 # 보스가 찾은 최적값!
    
    model = LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1)
    train_df = all_combined[all_combined.index <= (TODAY - timedelta(days=15))].dropna()
    model.fit(train_df[features], train_df['Target'])

    export_targets = []
    for code, name in stocks.items():
        df = data_list.get(code)
        if df is None: continue
        row = df.iloc[[-1]]
        if row['Is_Valid'].values[0] and "RED" not in zone:
            p = model.predict_proba(row[features])[0][1]
            if p >= threshold:
                kelly_f = max(0, ((1.5 * p) - (1 - p)) / 1.5)
                export_targets.append({
                    "code": code, "name": name, "score": round(p, 4),
                    "kelly_fraction": round(kelly_f / 2, 4),
                    "price": int(row['Close'].values[0]), "atr": float(row['ATR'].values[0])
                })

    with open('meta_target_list.json', 'w', encoding='utf-8') as f:
        json.dump({"date": TODAY.strftime("%Y-%m-%d"), "zone": zone, "targets": export_targets[:3]}, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_mitsuda_engine()
