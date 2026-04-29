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
APP_KEY = os.getenv("KIS_MOCK_APP_KEY")
APP_SECRET = os.getenv("KIS_MOCK_APP_SECRET")
TODAY = datetime.now()

def get_all_data():
    res = requests.post("https://openapi.koreainvestment.com:9443/oauth2/tokenP", 
                        data=json.dumps({"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}))
    token = res.json().get("access_token")
    if not token:
        print("KIS 토큰 발급 실패")
        return None, None, None

    df_krx = fdr.StockListing('KOSPI')
    cap_col = next((col for col in df_krx.columns if any(kw in col.lower().replace(" ", "") for kw in ['marcap', 'marketcap', '시가총액'])), df_krx.select_dtypes(include=[np.number]).columns[0])
    
    top_stocks = df_krx.sort_values(cap_col, ascending=False).head(100)
    stocks = dict(zip(top_stocks['Code'], top_stocks['Name']))
    start_date = (TODAY - timedelta(days=500)).strftime("%Y-%m-%d")
    
    ks = fdr.DataReader('KS11', start_date)
    try: vix = fdr.DataReader('VKOSPI', start_date)
    except: vix = fdr.DataReader('VIX', start_date)

    ks['KOSPI_MA20'] = ta.sma(ks['Close'], length=20)
    macro_df = pd.concat([ks[['Close', 'KOSPI_MA20']], vix[['Close']]], axis=1).dropna()
    macro_df.columns = ['KOSPI', 'KOSPI_MA20', 'VKOSPI']
    macro_df['Date'] = macro_df.index

    data_list = {}
    for code, name in stocks.items():
        try:
            df = fdr.DataReader(code, start_date)
            if len(df) < 70: continue
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['MA60'] = ta.sma(df['Close'], length=60)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['Dist_MA20'] = (df['Close'] / df['MA20']) - 1.0
            df['Dist_MA60'] = (df['MA20'] / df['MA60']) - 1.0
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
            df['ATR_Ratio'] = df['ATR'] / df['Close'] * 100
            obv = ta.obv(df['Close'], df['Volume'])
            vol_ma20 = ta.sma(df['Volume'], length=20)
            df['OBV_Norm'] = obv.diff() / vol_ma20
            
            entry_price = df['Open'].shift(-1)
            stop_level = entry_price - (df['ATR'] * 2.0)
            future_low_min = df['Low'].rolling(window=7).min().shift(-7)
            future_close = df['Close'].shift(-7)
            
            df['Target'] = np.where(((future_close / entry_price) - 1.0 >= 0.02) & (future_low_min > stop_level), 1, 0)
            df['Is_Valid'] = (df['Close'] > df['MA20'])
            df['Date'] = df.index; df['Code'] = code; df['Name'] = name
            data_list[code] = df.dropna()
        except: continue
    return data_list, macro_df, stocks

def run_mitsuda_engine():
    print("🐆 MITSUDA 분석 엔진 가동 시작...")
    data_list, macro_df, stocks = get_all_data()
    if not data_list: return

    all_combined = pd.concat(data_list.values())
    exclude_cols = ['Date', 'Code', 'Name', 'Target', 'Is_Valid', 'Close', 'Open', 'High', 'Low', 'Volume', 'Change', 'MA20', 'MA60', 'ATR']
    features = [c for c in all_combined.columns if c not in exclude_cols]

    curr = macro_df.iloc[-1]
    vix_val, ks_val, ks_ma20 = curr['VKOSPI'], curr['KOSPI'], curr['KOSPI_MA20']
    is_bull = ks_val > ks_ma20
    if vix_val >= 25.0: zone, threshold = "🔴 RED", 0.95
    elif vix_val >= 22.0 or not is_bull: zone, threshold = "🟡 YELLOW", 0.72
    else: zone, threshold = "🟢 GREEN", 0.63
    
    safe_today = TODAY - timedelta(days=15)
    train_df = all_combined[(all_combined.index <= safe_today) & (all_combined['Is_Valid'] == True)].dropna()
    main_model = LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, class_weight='balanced', random_state=42, verbose=-1)
    main_model.fit(train_df[features], train_df['Target'])

    export_targets = []
    for code, name in stocks.items():
        df = data_list.get(code)
        if df is None or df.empty: continue
        row = df.iloc[[-1]]
        if row['Is_Valid'].values[0] and "RED" not in zone:
            prob = main_model.predict_proba(row[features])[0][1]
            if prob >= threshold:
                export_targets.append({
                    "code": code, "name": name, "score": round(prob, 4),
                    "ref_yesterday_close": int(row['Close'].values[0]),
                    "atr_value": float(row['ATR'].values[0])
                })

    with open('meta_target_list.json', 'w', encoding='utf-8') as f:
        json.dump({"date": TODAY.strftime("%Y-%m-%d"), "zone": zone, "targets": export_targets}, f, ensure_ascii=False, indent=4)

    vintage_data = []
    all_dates = sorted([d for d in macro_df.index.unique() if d.date() < TODAY.date()])
    test_dates = all_dates[-10:]

    for t_date in test_dates:
        m_row = macro_df[macro_df.index == t_date]
        h_vix = m_row['VKOSPI'].values[0]
        h_bull = m_row['KOSPI'].values[0] > m_row['KOSPI_MA20'].values[0]
        h_th = 1.00 if h_vix >= 25.0 else (0.85 if h_vix >= 22.0 or not h_bull else 0.63)
        
        train_pit = all_combined[(all_combined.index <= (t_date - timedelta(days=15))) & (all_combined['Is_Valid'] == True)].dropna()
        if train_pit.empty: continue
        
        wfv_model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, class_weight='balanced', random_state=42, verbose=-1)
        wfv_model.fit(train_pit[features], train_pit['Target'])

        for code, name in stocks.items():
            df = data_list.get(code)
            row = df[df.index == t_date]
            if not row.empty and row['Is_Valid'].values[0] and wfv_model.predict_proba(row[features])[0][1] > h_th:
                future_dates = [d for d in all_dates if d > t_date][:10]
                if not future_dates: continue
                f_df_1 = df[df.index == future_dates[0]]
                if f_df_1.empty: continue
                
                raw_entry_p = f_df_1['Open'].values[0]
                entry_atr = row['ATR'].values[0]
                slippage = entry_atr * 0.10
                actual_entry_p = raw_entry_p + slippage
                current_stop_l = raw_entry_p - (entry_atr * 2.0)
                
                rets = {"날짜": t_date.strftime('%m/%d'), "종목": name}
                is_cut = False
                for j in range(1, 11):
                    if is_cut or j > len(future_dates):
                        rets[f"D+{j}"] = ""
                        continue
                    f_df = df[df.index == future_dates[j-1]]
                    if f_df.empty: continue
                    
                    today_low, today_open, today_close, today_atr = f_df['Low'].values[0], f_df['Open'].values[0], f_df['Close'].values[0], f_df['ATR'].values[0]
                    if today_low <= current_stop_l:
                        is_cut = True
                        worst_price = min(current_stop_l, today_open)
                        actual_exit_p = worst_price - slippage
                        rets[f"D+{j}"] = f"🛑{round(((actual_exit_p/actual_entry_p)-1)*100, 1)}%"
                    else:
                        actual_exit_p = today_close - slippage
                        ret_val = round(((actual_exit_p/actual_entry_p)-1)*100, 1)
                        if j == 7 and ret_val <= 0.0:
                            is_cut = True
                            rets[f"D+{j}"] = f"✂️{ret_val}%"
                        else:
                            rets[f"D+{j}"] = ret_val
                        potential_new_stop = today_close - (today_atr * 2.0)
                        if potential_new_stop > current_stop_l: current_stop_l = potential_new_stop
                vintage_data.append(rets)
    
    if vintage_data:
        pd.DataFrame(vintage_data).to_csv('vintage_performance.csv', index=False)
        print("성과 보고서 데이터 생성 완료.")

if __name__ == "__main__":
    run_mitsuda_engine()
