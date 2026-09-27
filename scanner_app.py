import concurrent.futures
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from method1 import run_method_1
from method2 import run_method_2
from method3 import run_method_3

st.set_page_config(page_title="3-in-1 Master AI Stock Scanner", layout="wide", page_icon="🚀")

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

st.title("🤖 3-in-1 Master AI Swingtrade Scanner")
st.caption("Eén overzicht met gecombineerde AI & ML scores uit alle 3 de analyseresults.")

st.sidebar.header("⚙️ Scanner Instellingen")
default_watchlist = "AAPL, NVDA, TSLA, AMD, PLTR, MSFT, HOOD, AMZN, GOOGL, META"
scan_input = st.sidebar.text_area("Watchlist (tickers door komma's/spaties gescheiden):", value=default_watchlist, height=100)
lstm_epochs = st.sidebar.slider("LSTM Epochs (Snelheid vs Accuratheid)", min_value=4, max_value=20, value=8)

def run_full_combined_scan(ticker):
    m1_res = run_method_1(ticker)
    m2_res = run_method_2(ticker)
    m3_res = run_method_3(ticker, epochs=lstm_epochs)

    if not m1_res:
        return None

    m1_score_norm = round(max(1.0, min(10.0, (m1_res["M1_Score"] + 5) * 0.9 + 1)), 1)
    m2_score = m2_res["M2_Score"] if m2_res else 5.0
    m3_score = m3_res["M3_Score"] if m3_res else 5.0

    master_score = round((m1_score_norm * 0.35) + (m2_score * 0.35) + (m3_score * 0.30), 1)
    signal = "🟢 BULLISH" if master_score >= 6.5 else ("🔴 BEARISH" if master_score <= 4.0 else "🟠 NEUTRAAL")

    return {
        "Ticker": ticker,
        "Koers": m1_res["Koers"],
        "Master AI Score": master_score,
        "Signaal": signal,
        "M1 Score": f"{m1_score_norm}/10",
        "M2 Score": f"{m2_score}/10",
        "M3 Score": f"{m3_score}/10",
        "M1 ML Kans": m1_res["M1_ML_Kans"],
        "M2 ML Prob": m2_res["M2_ML_Prob"] if m2_res else "N/A",
        "M3 Ensemble": m3_res["Ensemble_Prob"] if m3_res else "N/A",
        "M3 LSTM Δ": m3_res["LSTM_Change_Pct"] if m3_res else "N/A",
        "Support": m1_res["Support"],
        "Resistance": m1_res["Resistance"],
        "m1_raw": m1_res,
        "m2_raw": m2_res,
        "m3_raw": m3_res
    }

scan_btn = st.sidebar.button("🚀 Start 3-in-1 Multi-Scan", type="primary", use_container_width=True)

if scan_btn or "scan_results_df" not in st.session_state:
    tickers_to_scan = [t.strip().upper() for t in scan_input.replace(',', ' ').split() if t.strip()]
    
    with st.spinner(f"Scannen van {len(tickers_to_scan)} aandelen via Methode 1, 2 en 3 (PyTorch ML/DL)..."):
        results = []
        progress_bar = st.progress(0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_ticker = {executor.submit(run_full_combined_scan, t): t for t in tickers_to_scan}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                if data:
                    results.append(data)
                completed += 1
                progress_bar.progress(completed / len(tickers_to_scan))

        if results:
            df_res = pd.DataFrame(results).sort_values(by="Master AI Score", ascending=False).reset_index(drop=True)
            st.session_state.scan_results_df = df_res

if "scan_results_df" in st.session_state:
    scan_df = st.session_state.scan_results_df

    st.markdown("### 🏆 Master Multi-Scanner Resultaten")
    
    display_df = scan_df[[
        "Ticker", "Koers", "Master AI Score", "Signaal", 
        "M1 Score", "M2 Score", "M3 Score", 
        "M1 ML Kans", "M2 ML Prob", "M3 Ensemble", "M3 LSTM Δ"
    ]].copy()
    
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Interactieve Card View")

    for idx, row in scan_df.iterrows():
        col_t, col_sig, col_m1, col_m2, col_m3, col_mast, col_act = st.columns([1.0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.3])

        col_t.markdown(f"**{row['Ticker']}**<br><small>${row['Koers']}</small>", unsafe_allow_html=True)
        col_sig.markdown(row["Signaal"])
        col_m1.markdown(f"M1 Pattern<br>**{row['M1 Score']}**", unsafe_allow_html=True)
        col_m2.markdown(f"M2 Flow/TA<br>**{row['M2 Score']}**", unsafe_allow_html=True)
        col_m3.markdown(f"M3 LSTM/ML<br>**{row['M3 Score']}**", unsafe_allow_html=True)
        col_mast.markdown(f"🎯 **{row['Master AI Score']} / 10**")

        if col_act.button(f"📊 Analyseer {row['Ticker']}", key=f"btn_{row['Ticker']}_{idx}"):
            st.session_state.selected_ticker = row["Ticker"]
            st.rerun()

        st.markdown("<hr style='margin: 4px 0px; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

st.markdown("---")
ticker = st.session_state.selected_ticker
st.subheader(f"🔍 Diepgaande Details voor: {ticker}")

res = run_full_combined_scan(ticker)

if res:
    m1 = res["m1_raw"]
    m2 = res["m2_raw"]
    m3 = res["m3_raw"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Master AI Score", f"{res['Master AI Score']} / 10", res["Signaal"])
    c2.metric("Methode 1 (Options & Sentiment)", res["M1 Score"], f"ML Prob: {res['M1 ML Kans']}")
    c3.metric("Methode 2 (Money Flow & TA)", res["M2 Score"], f"ML Prob: {res['M2 ML Prob']}")
    c4.metric("Methode 3 (LSTM + Ensemble)", res["M3 Score"], f"LSTM Δ: {res['M3 LSTM Δ']}")

    tab_grafiek, tab_m3_details = st.tabs(["📊 Technische Grafiek", "🧠 Methode 3 (PyTorch LSTM & Ensemble ML)"])

    with tab_grafiek:
        df = m1["df"]
        plot_df = df.tail(90)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"], name="Prijs",
            increasing_line_color="green", decreasing_line_color="red"
        ))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA20"], name="SMA20", line=dict(color="#3b82f6", width=1.5)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA50"], name="SMA50", line=dict(color="#a855f7", width=1.5)))

        if m1["support_val"]:
            fig.add_hline(y=m1["support_val"], line_dash="dash", line_color="green", annotation_text=f"Support ${m1['support_val']:.2f}")
        if m1["resistance_val"]:
            fig.add_hline(y=m1["resistance_val"], line_dash="dash", line_color="red", annotation_text=f"Resistance ${m1['resistance_val']:.2f}")

        fig.update_layout(height=450, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_m3_details:
        if m3 and m3.get("lstm_data"):
            lstm = m3["lstm_data"]
            ens = m3["ens_data"]
            
            st.markdown("#### Model Accuratheid & Voorspellingen")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Ensemble Stijgkans", res["M3 Ensemble"])
            col_b.metric("LSTM Voorspelde Close", f"${m3['LSTM_Pred']}", res["M3 LSTM Δ"])
            col_c.metric("LSTM Directional Accuracy", f"{lstm['dir_acc'] * 100:.1f}%")

            if ens and "feature_imp" in ens:
                avg_imp = ens["feature_imp"]
                imp_fig = go.Figure(go.Bar(x=avg_imp.values[::-1], y=avg_imp.index[::-1], orientation='h', marker_color='steelblue'))
                imp_fig.update_layout(title="Top 10 Important Features (Random Forest + Gradient Boosting)", height=350)
                st.plotly_chart(imp_fig, use_container_width=True)
