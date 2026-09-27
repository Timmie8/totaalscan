import concurrent.futures
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from method1 import run_method_1
from method2 import run_method_2
from method3 import run_method_3

st.set_page_config(page_title="3-in-1 Master AI Stock Scanner", layout="wide", page_icon="🚀")

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

st.title("🤖 3-in-1 Master AI Swingtrade Scanner")
st.caption("Eén overzicht met gecombineerde AI & ML scores uit alle 3 de analysemethodes.")

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
        "AI-TA-Score": m1_score_norm,
        "SST Score": m2_score,
        "AI Combi Score": m3_score,
        "AI-TA Kans": m1_res["M1_ML_Kans"],
        "SST Kans": m2_res["M2_ML_Prob"] if m2_res else "N/A",
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
    
    with st.spinner(f"Scannen van {len(tickers_to_scan)} aandelen via Methode 1, 2 en 3..."):
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

# Helper om percentages te parsen naar getallen (0-100)
def parse_pct_value(val):
    try:
        if isinstance(val, str):
            val_clean = val.replace('%', '').strip()
            v = float(val_clean)
            if v <= 1.0 and '%' not in val:
                v = v * 100
            return v
        return float(val) * 100 if val <= 1.0 else float(val)
    except (ValueError, TypeError):
        return None

# Badge voor Card View op basis van percentage (groen boven 55%)
def get_prob_badge(val_str, label=""):
    pct = parse_pct_value(val_str)
    if pct is None:
        return f"<span>{label}: N/A</span>"
    
    if pct > 55.0:
        bg = "#2e7d32"  # Groen
    elif pct < 45.0:
        bg = "#c62828"  # Rood
    else:
        bg = "#ef6c00"  # Oranje

    return f'<span style="background-color:{bg}; color:#ffffff; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{label}: {val_str}</span>'

# Badge voor Card View op basis van score
def get_score_badge(score, label_prefix="", min_green=6.0):
    try:
        val = float(score)
    except (ValueError, TypeError):
        val = 5.0

    if val >= min_green:
        bg = "#2e7d32"  # Groen
    elif val <= (min_green - 1.6 if min_green <= 6.0 else 4.9):
        bg = "#c62828"  # Rood
    else:
        bg = "#ef6c00"  # Oranje

    return f'<span style="background-color:{bg}; color:#ffffff; padding: 3px 8px; border-radius: 4px; font-weight: bold;">{label_prefix}{val}/10</span>'

if "scan_results_df" in st.session_state:
    scan_df = st.session_state.scan_results_df

    st.markdown("### 🏆 Master Multi-Scanner Resultaten")
    
    display_df = scan_df[[
        "Ticker", "Koers", "Master AI Score", "Signaal", 
        "AI-TA-Score", "SST Score", "AI Combi Score", 
        "AI-TA Kans", "SST Kans", "M3 Ensemble", "M3 LSTM Δ"
    ]].copy()

    # Standaard scores: Groen >= 6.0
    def highlight_standard_scores(val):
        try:
            v = float(val)
            if v >= 6.0:
                return 'background-color: rgba(46, 125, 50, 0.35); font-weight: bold;'
            elif v <= 4.4:
                return 'background-color: rgba(198, 40, 40, 0.35); font-weight: bold;'
            else:
                return 'background-color: rgba(239, 108, 0, 0.25);'
        except (ValueError, TypeError):
            return ''

    # AI-TA Score & SST Score: Pas groen Boven 6.9 (dus >= 7.0)
    def highlight_strict_score(val):
        try:
            v = float(val)
            if v > 6.9:
                return 'background-color: rgba(46, 125, 50, 0.35); font-weight: bold;'
            elif v < 5.0:
                return 'background-color: rgba(198, 40, 40, 0.35); font-weight: bold;'
            else:
                return 'background-color: rgba(239, 108, 0, 0.25);'
        except (ValueError, TypeError):
            return ''

    # ML Kansen > 55%
    def highlight_probs(val):
        pct = parse_pct_value(val)
        if pct is None:
            return ''
        if pct > 55.0:
            return 'background-color: rgba(46, 125, 50, 0.35); font-weight: bold;'
        elif pct < 45.0:
            return 'background-color: rgba(198, 40, 40, 0.35); font-weight: bold;'
        else:
            return 'background-color: rgba(239, 108, 0, 0.25);'

    def highlight_signal_col(val):
        if "BULLISH" in str(val):
            return 'background-color: rgba(46, 125, 50, 0.4); font-weight: bold;'
        elif "BEARISH" in str(val):
            return 'background-color: rgba(198, 40, 40, 0.4); font-weight: bold;'
        else:
            return 'background-color: rgba(239, 108, 0, 0.3);'

    styled_df = display_df.style\
        .map(highlight_standard_scores, subset=["Master AI Score", "AI Combi Score"])\
        .map(highlight_strict_score, subset=["AI-TA-Score", "SST Score"])\
        .map(highlight_probs, subset=["AI-TA Kans", "SST Kans", "M3 Ensemble"])\
        .map(highlight_signal_col, subset=["Signaal"])

    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Interactieve Card View (Scores & ML-Kansen gekleurd)")

    for idx, row in scan_df.iterrows():
        sig = row["Signaal"]
        if "BULLISH" in sig:
            card_border = "#2e7d32"
            card_bg = "rgba(46, 125, 50, 0.08)"
        elif "BEARISH" in sig:
            card_border = "#c62828"
            card_bg = "rgba(198, 40, 40, 0.08)"
        else:
            card_border = "#ef6c00"
            card_bg = "rgba(239, 108, 0, 0.05)"

        with st.container():
            st.markdown(
                f'<div style="border-left: 6px solid {card_border}; background-color: {card_bg}; padding: 12px; border-radius: 6px; margin-bottom: 12px;">',
                unsafe_allow_html=True
            )
            col_t, col_sig, col_m1, col_m2, col_m3, col_mast, col_act = st.columns([1.0, 1.2, 1.4, 1.4, 1.4, 1.3, 1.2])

            col_t.markdown(f"### **{row['Ticker']}**\n**${row['Koers']}**")
            col_sig.markdown(f"**Signaal:**<br>{row['Signaal']}", unsafe_allow_html=True)
            
            # Card badges (AI-TA en SST drempel min_green=7.0)
            col_m1.markdown(f"AI-TA Pattern<br>{get_score_badge(row['AI-TA-Score'], min_green=7.0)}<br><small>{get_prob_badge(row['AI-TA Kans'], 'Kans')}</small>", unsafe_allow_html=True)
            col_m2.markdown(f"SST Flow/TA<br>{get_score_badge(row['SST Score'], min_green=7.0)}<br><small>{get_prob_badge(row['SST Kans'], 'Kans')}</small>", unsafe_allow_html=True)
            col_m3.markdown(f"AI Combi<br>{get_score_badge(row['AI Combi Score'], min_green=6.0)}<br><small>{get_prob_badge(row['M3 Ensemble'], 'Ens')}</small>", unsafe_allow_html=True)
            col_mast.markdown(f"🎯 **Master Score**<br>{get_score_badge(row['Master AI Score'], min_green=6.0)}", unsafe_allow_html=True)

            if col_act.button(f"📊 Details", key=f"btn_{row['Ticker']}_{idx}", use_container_width=True):
                st.session_state.selected_ticker = row["Ticker"]
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

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
    c2.metric("AI-TA-Score (Options & Sentiment)", f"{res['AI-TA-Score']}/10", f"AI-TA Kans: {res['AI-TA Kans']}")
    c3.metric("SST Score (Money Flow & TA)", f"{res['SST Score']}/10", f"SST Kans: {res['SST Kans']}")
    c4.metric("AI Combi Score (LSTM + Ensemble)", f"{res['AI Combi Score']}/10", f"LSTM Δ: {res['M3 LSTM Δ']}")

    tab_grafiek, tab_m3_details = st.tabs(["📊 Technische Grafiek", "🧠 AI Combi Details (PyTorch LSTM & Ensemble ML)"])

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
