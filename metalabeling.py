import pandas as pd
import numpy as np
import requests, json, warnings, os
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import pandas_ta_classic as ta
from lightgbm import LGBMClassifier
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

APP_KEY, APP_SECRET = os.getenv("KIS_MOCK_APP_KEY"), os.getenv("KIS_MOCK_APP_SECRET")
TODAY = datetime.now()

def get_all_data():
    res = requests.post("https://openapi.koreainvestment.com:9443/oauth2/tokenP", 
                        data=json.dumps({"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}))
    token = res.json().get("access_token")
    if not token: return None, None, None, 0, 0

    top_stocks = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(100)
    stocks = dict(zip(top_stocks['Code'], top_stocks['Name']))
    start_date = (TODAY - timedelta(days=500)).strftime("%Y-%m-%d")
    
    ks, vix = fdr.DataReader('KS11', start_date), fdr.DataReader('VIX', start_date)
    ks['KOSPI_MA20'] = ta.sma(ks['Close'], length=20)
    macro_df = pd.concat([ks[['Close', 'KOSPI_MA20']], vix[['Close']]], axis=1).dropna()
    macro_df.columns = ['KOSPI', 'KOSPI_MA20', 'VIX']
    
    v_y, v_r = macro_df['VIX'].quantile(0.7), macro_df['VIX'].quantile(0.9)

    data_list = {}
    for code, name in stocks.items():
        try:
            df = fdr.DataReader(code, start_date)
            if len(df) < 75: continue
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
            df['Dist_MA20'] = (df['Close'] / df['MA20']) - 1.0 
            
            entry_p = df['Open'].shift(-1)
            stop_l = entry_p - (df['ATR'] * 2.0)
            df['Target'] = np.where((df['High'].rolling(7).max().shift(-7)/entry_p - 1.0 >= 0.04) & 
                                    (df['Low'].rolling(7).min().shift(-7) > stop_l), 1, 0)
            df['Is_Valid'] = (df['Close'] > df['MA20'])
            df['Date'], df['Code'], df['Name'] = df.index, code, name
            data_list[code] = df.dropna()
        except: continue
    return data_list, macro_df, stocks, v_y, v_r

def run_mitsuda_engine():
    data_list, macro_df, stocks, v_y, v_r = get_all_data()
    if not data_list: return

    all_combined = pd.concat(data_list.values())
    features = ['Dist_MA20', 'MFI', 'ATR']
    all_dates = sorted(macro_df.index.unique())

    # --- [파트 1: 오늘자 사냥 지시서] ---
    model_today = LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1)
    train_today = all_combined[all_combined.index <= (TODAY - timedelta(days=15))].dropna()
    model_today.fit(train_today[features], train_today['Target'])

    curr = macro_df.iloc[-1]
    if curr['VIX'] >= v_r: zone, threshold = "🔴 RED", 0.95
    elif curr['VIX'] >= v_y or curr['KOSPI'] < curr['KOSPI_MA20']: zone, threshold = "🟡 YELLOW", 0.75
    else: zone, threshold = "🟢 GREEN", 0.62 

    export_targets = []
    for code, name in stocks.items():
        row = data_list[code].iloc[[-1]]
        if row['Is_Valid'].values[0] and zone != "🔴 RED":
            p = model_today.predict_proba(row[features])[0][1]
            if p >= threshold:
                k_f = max(0, ((1.5 * p) - (1 - p)) / 1.5)
                export_targets.append({"code": code, "name": name, "score": round(p, 4), "kelly": round(k_f/2, 4), "price": int(row['Close'].values[0]), "atr": float(row['ATR'].values[0])})

    print(f"\n--- 오늘의 미츠다 분석 점수 (Threshold: {threshold}) ---")
    sorted_all = []
    for code, name in stocks.items():
        row = data_list[code].iloc[[-1]]
        if row['Is_Valid'].values[0]:
            p = model_today.predict_proba(row[features])[0][1]
            sorted_all.append((name, p))
    for name, p in sorted(sorted_all, key=lambda x: x[1], reverse=True)[:10]:
        print(f"{name}: {round(p, 4)}")

    with open('meta_target_list.json', 'w', encoding='utf-8') as f:
        json.dump({"date": TODAY.strftime("%Y-%m-%d"), "zone": zone, "targets": export_targets[:3]}, f, ensure_ascii=False, indent=4)

    # --- [파트 2: 타임머신 성과 기록 (Vintage)] ---
    vintage = []
    for t_date in all_dates[-10:]:
        m_pit = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)
        train_pit = all_combined[all_combined.index <= (t_date - timedelta(days=15))]
        if train_pit.empty: continue
        m_pit.fit(train_pit[features], train_pit['Target'])

        # [보스 지시] 해당 시점의 시장 존에 따른 동적 손절 배수 결정
        curr_pit = macro_df[macro_df.index <= t_date].iloc[-1]
        if curr_pit['VIX'] >= v_r: pit_zone = "🔴 RED"
        elif curr_pit['VIX'] >= v_y or curr_pit['KOSPI'] < curr_pit['KOSPI_MA20']: pit_zone = "🟡 YELLOW"
        else: pit_zone = "🟢 GREEN"
        
        multiplier = 1.2 if pit_zone == "🔴 RED" else 2.0

        for code, name in stocks.items():
            df = data_list[code]
            row = df[df.index == t_date]
            if not row.empty and row['Is_Valid'].values[0]:
                p_val = m_pit.predict_proba(row[features])[0][1]
                if p_val >= 0.62:
                    rets = {"날짜": t_date.strftime('%m/%d'), "종목": name}
                    future_dates = [d for d in all_dates if d > t_date][:10]
                    if not future_dates: continue
                    
                    entry_price = df[df.index == future_dates[0]]['Open'].values[0]
                    current_stop_loss = entry_price - (row['ATR'].values[0] * multiplier)
                    highest_price = entry_price
                    is_cut = False
                    
                    for j, f_d in enumerate(future_dates, 1):
                        if is_cut:
                            rets[f"D+{j}"] = "-" 
                            continue
                        f_row = df[df.index == f_d]
                        curr_low, curr_close, curr_high, curr_atr = f_row['Low'].values[0], f_row['Close'].values[0], f_row['High'].values[0], f_row['ATR'].values[0]
                        
                        if curr_low <= current_stop_loss:
                            rets[f"D+{j}"] = f"🛑{round(((current_stop_loss/entry_price)-1)*100, 1)}%"
                            is_cut = True
                        elif j == 7 and curr_close <= entry_price:
                            rets[f"D+{j}"] = f"✂️{round(((curr_close/entry_price)-1)*100, 1)}%"
                            is_cut = True
                        else:
                            rets[f"D+{j}"] = round(((curr_close/entry_price)-1)*100, 1)
                            if curr_high > highest_price:
                                highest_price = curr_high
                                current_stop_loss = highest_price - (curr_atr * multiplier)
                    vintage.append(rets)
    
    if vintage: pd.DataFrame(vintage).to_csv('vintage_performance.csv', index=False)

if __name__ == "__main__":
    run_mitsuda_engine()
