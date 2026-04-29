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
            # [복구] 20일선 이격도 추가
            df['Dist_MA20'] = (df['Close'] / df['MA20']) - 1.0
            
            # [트리플 배리어 레이블링]
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
    features = ['Dist_MA20', 'MFI', 'ATR'] # [복구] 핵심 피처 3종
    curr = macro_df.iloc[-1]
    
    # 0.62 임계값 적용
    if curr['VIX'] >= v_r: zone, threshold = "🔴 RED", 0.95
    elif curr['VIX'] >= v_y or curr['KOSPI'] < curr['KOSPI_MA20']: zone, threshold = "🟡 YELLOW", 0.75
    else: zone, threshold = "🟢 GREEN", 0.62 

    # [15일 시차 학습]
    model = LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1)
    train_df = all_combined[all_combined.index <= (TODAY - timedelta(days=15))].dropna()
    model.fit(train_df[features], train_df['Target'])

    # [작전 지시서 작성]
    export_targets = []
    for code, name in stocks.items():
        row = data_list[code].iloc[[-1]]
        if row['Is_Valid'].values[0] and zone != "🔴 RED":
            p = model.predict_proba(row[features])[0][1]
            if p >= threshold:
                k_f = max(0, ((1.5 * p) - (1 - p)) / 1.5)
                export_targets.append({"code": code, "name": name, "score": round(p, 4), "kelly": round(k_f/2, 4), "price": int(row['Close'].values[0]), "atr": float(row['ATR'].values[0])})

    with open('meta_target_list.json', 'w', encoding='utf-8') as f:
        json.dump({"date": TODAY.strftime("%Y-%m-%d"), "zone": zone, "targets": export_targets[:3]}, f, ensure_ascii=False, indent=4)

    # [복구] 정예 종목 빈티지 레포트 (색깔 입히기용 데이터)
    vintage = []
    all_dates = sorted(macro_df.index.unique())
    for t_date in all_dates[-10:]:
        for code, name in stocks.items():
            df = data_list[code]
            row = df[df.index == t_date]
            if not row.empty and row['Is_Valid'].values[0] and model.predict_proba(row[features])[0][1] > 0.62:
                rets = {"날짜": t_date.strftime('%m/%d'), "종목": name}
                f_dates = [d for d in all_dates if d > t_date][:10]
                entry = df[df.index == f_dates[0]]['Open'].values[0] if f_dates else None
                if entry:
                    for j, f_d in enumerate(f_dates, 1):
                        rets[f"D+{j}"] = round(((df[df.index == f_d]['Close'].values[0]/entry)-1)*100, 1)
                    vintage.append(rets)
    if vintage: pd.DataFrame(vintage).to_csv('vintage_performance.csv', index=False)

if __name__ == "__main__":
    run_mitsuda_engine()
