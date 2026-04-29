# metalabeling.py 의 vintage 생성 부분 수정본

    # --- [파트 2: 타임머신 성과 기록 (Vintage) - 시뮬레이션 강화] ---
    vintage = []
    all_dates = sorted(macro_df.index.unique())
    for t_date in all_dates[-10:]:
        m_pit = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)
        train_pit = all_combined[all_combined.index <= (t_date - timedelta(days=15))]
        if train_pit.empty: continue
        m_pit.fit(train_pit[features], train_pit['Target'])

        for code, name in stocks.items():
            df = data_list[code]
            row = df[df.index == t_date]
            if not row.empty and row['Is_Valid'].values[0]:
                p_val = m_pit.predict_proba(row[features])[0][1]
                if p_val >= 0.62: # 보스의 그린 존 최적값
                    rets = {"날짜": t_date.strftime('%m/%d'), "종목": name}
                    future_dates = [d for d in all_dates if d > t_date][:10]
                    
                    if not future_dates: continue
                    
                    # [시뮬레이션 시작]
                    entry_price = df[df.index == future_dates[0]]['Open'].values[0]
                    current_stop_loss = entry_price - (row['ATR'].values[0] * 2.0)
                    highest_price = entry_price
                    is_cut = False
                    
                    for j, f_d in enumerate(future_dates, 1):
                        if is_cut:
                            rets[f"D+{j}"] = "-" # 이미 청산됨
                            continue
                            
                        f_row = df[df.index == f_d]
                        curr_low = f_row['Low'].values[0]
                        curr_close = f_row['Close'].values[0]
                        curr_high = f_row['High'].values[0]
                        curr_atr = f_row['ATR'].values[0]
                        
                        # 1. 트레일링 스탑 체크
                        if curr_low <= current_stop_loss:
                            rets[f"D+{j}"] = f"🛑{round(((current_stop_loss/entry_price)-1)*100, 1)}%"
                            is_cut = True
                        # 2. 7일 타임컷 체크
                        elif j == 7 and curr_close <= entry_price:
                            rets[f"D+{j}"] = f"✂️{round(((curr_close/entry_price)-1)*100, 1)}%"
                            is_cut = True
                        else:
                            # 수익률 기록 및 트레일링 스탑 상향 조정
                            rets[f"D+{j}"] = round(((curr_close/entry_price)-1)*100, 1)
                            if curr_high > highest_price:
                                highest_price = curr_high
                                current_stop_loss = highest_price - (curr_atr * 2.0)
                                
                    vintage.append(rets)
    
    if vintage: pd.DataFrame(vintage).to_csv('vintage_performance.csv', index=False)
    # metalabeling.py 의 export_targets 저장 직전에 추가
    for t in export_targets: print(f"분석 결과: {t['name']} - 점수: {t['score']}")
