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
            
            # 🎨 최신 Pandas 버전에 맞춰 .map() 사용 (AttributeError 방지)
            try:
                styled_df = df_m.style.map(style_returns)
            except AttributeError:
                # 구버전일 경우를 대비해 applymap도 보험으로 남겨둠
                styled_df = df_m.style.applymap(style_returns)
                
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 최근 10거래일 동안 AI 확신도 55%를 넘긴 추천 종목이 없었습니다.")
