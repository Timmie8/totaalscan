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

st.title("🤖 Master AI Stock Scanner")
st.caption("Overzicht gefocust op AI-TA-Score, SST-Score en M3 Ensemble met gewogen totaalscore.")

st.sidebar.header("⚙️ Scanner Instellingen")
default_watchlist = "AAPL, NVDA, TSLA, AMD, PLTR, MSFT, HOOD, AMZN, GOOGL, META"
scan_input = st.sidebar.text_area("Watchlist (tickers door komma's/spaties gescheiden):", value=default_watchlist, height=100)
lstm_epochs = st.sidebar.slider("LSTM Epochs (Snelheid vs Accuratheid)", min_value=4, max_value=20, value=8)

# Helper om percentages om te zetten naar een float (0-100)
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
        return 0.0

def calculate_weighted_score(m1_score_norm, m2_score, m3_ensemble_val):
    """
    Berekening volgens de regel:
    - SST-Score bullish (> 6.9 / >= 7.0)  -> 70%
    - AI-TA-Score bullish (> 6.9 / >= 7.0) -> 10%
    - M3 Ensemble bullish (> 55%)          -> 20%
    Telling gaat alleen in zodra een component groen (bullish) is.
    """
    total = 0.0

    # 1. SST Score check (70%)
    if m2_score > 6.9:
        total += 70.0

    # 2. AI-TA Score check (10%)
    if m1_score_norm > 6.9:
        total += 10.0

    # 3. M3 Ensemble check (20%)
    ens_pct = parse_pct_value(m3_ensemble_val)
    if ens_pct > 55.0:
        total += 20.0

    return round(total, 1)

def run_full_combined_scan(ticker):
    m1_res = run_method_1(ticker)
    m2_res = run_method_2(ticker)
    m3_res = run_method_3(ticker, epochs=lstm_epochs)

    if not m1_res:
        return None

    m1_score_norm = round(max(1.0, min(10.0, (m1_res["M1_Score"] + 5) * 0.9 + 1)), 1)
    m2_score = m2_res["M2_Score"] if m2_res else 5.0
    m3_ensemble = m3_res["Ensemble_Prob"] if m3_res else "N/A"

    # Bereken de gewogen totaalscore vooraan
    totaal_score_pct = calculate_weighted_score(m1_score_norm, m2_score, m3_ensemble)

    if totaal_score_pct >= 70.0:
        signal = "🟢 BULLISH"
    elif totaal_score_pct >= 20.0:
        signal = "🟠 NEUTRAAL"
    else:
        signal = "🔴 BEARISH"

    return {
        "Totaal Score (%)": totaal_score_pct,
        "Ticker": ticker,
        "Koers": m1_res["Koers"],
        "AI-TA-Score": m1_score_norm,
        "SST-Score": m2_score,
        "M3 Ensemble": m3_ensemble,
        "Signaal": signal,
        "m1_raw": m1_res,
        "m2_raw": m2_res,
        "m3_raw": m3_res
    }

scan_btn = st.sidebar.button("🚀 Start Scan", type="primary", use_container_width=True)

if scan_btn or "scan_results_df" not in st.session_state:
    tickers_to_scan = [t.strip().upper() for t in scan_input.replace(',', ' ').split() if t.strip()]
    
    with st.spinner(f"Scannen van {len(tickers_to_scan)} aandelen..."):
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
            df_res = pd.DataFrame(results).sort_values(by="Totaal Score (%)", ascending=False).reset_index(drop=True)
            st.session_state.scan_results_df = df_res

# Helper voor badge weergave in de Card View
def get_score_badge(score, is_pct=False, threshold=7.0):
    try:
        val = float(score)
    except (ValueError, TypeError):
        val = 0.0

    if (is_pct and val > 55.0) or (not is_pct and val > threshold):
        bg = "#2e7d32"  # Groen
    else:
        bg = "#ef6c00"  # Oranje/Neutraal

    suffix = "%" if is_pct else "/10"
    return f'<span style="background-color:{bg}; color:#ffffff; padding: 3px 8px; border-radius: 4px; font-weight: bold;">{score}{suffix}</span>'

if "scan_results_df" in st.session_state:
    scan_df = st.session_state.scan_results_df

    st.markdown("### 🏆 Master Multi-Scanner Resultaten")
    
    # Alleen gevraagde kolommen tonen, Totaal Score vooraan
    display_df = scan_df[[
        "Totaal Score (%)", "Ticker", "Koers", 
        "AI-TA-Score", "SST-Score", "M3 Ensemble", "Signaal"
    ]].copy()

    # Styling voor Totaal Score (%)
    def highlight_totaal_score(val):
        try:
            v = float(val)
            if v >= 70.0:
                return 'background-color: rgba(46, 125, 50, 0.45); font-weight: bold; color: #ffffff;'
            elif v >= 20.0:
                return 'background-color: rgba(239, 108, 0, 0.3); font-weight: bold;'
            else:
                return 'background-color: rgba(198, 40, 40, 0.25);'
        except (ValueError, TypeError):
            return ''

    # Styling voor AI-TA & SST (Groen > 6.9)
    def highlight_strict_score(val):
        try:
            v = float(val)
            if v > 6.9:
                return 'background-color: rgba(46, 125, 50, 0.35); font-weight: bold;'
            else:
                return 'background-color: rgba(239, 108, 0, 0.2);'
        except (ValueError, TypeError):
            return ''

    # Styling voor M3 Ensemble (Groen > 55%)
    def highlight_ensemble(val):
        pct = parse_pct_value(val)
        if pct > 55.0:
            return 'background-color: rgba(46, 125, 50, 0.35); font-weight: bold;'
        else:
            return 'background-color: rgba(239, 108, 0, 0.2);'

    styled_df = display_df.style\
        .map(highlight_totaal_score, subset=["Totaal Score (%)"])\
        .map(highlight_strict_score, subset=["AI-TA-Score", "SST-Score"])\
        .map(highlight_ensemble, subset=["M3 Ensemble"])

    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Overzichtskaarten")

    for idx, row in scan_df.iterrows():
        tot = row["Totaal Score (%)"]
        card_border = "#2e7d32" if tot >= 70.0 else ("#ef6c00" if tot >= 20.0 else "#c62828")
        card_bg = "rgba(46, 125, 50, 0.08)" if tot >= 70.0 else "rgba(239, 108, 0, 0.05)"

        with st.container():
            st.markdown(
                f'<div style="border-left: 6px solid {card_border}; background-color: {card_bg}; padding: 12px; border-radius: 6px; margin-bottom: 12px;">',
                unsafe_allow_html=True
            )
            col_tot, col_t, col_m1, col_m2, col_m3, col_act = st.columns([1.3, 1.2, 1.3, 1.3, 1.3, 1.0])

            col_tot.markdown(f"🎯 **Totaal Score**<br><h2 style='margin:0; color:{card_border};'>{tot}%</h2>", unsafe_allow_html=True)
            col_t.markdown(f"### **{row['Ticker']}**\n**${row['Koers']}**")
            
            col_m1.markdown(f"AI-TA-Score (10%)<br>{get_score_badge(row['AI-TA-Score'], is_pct=False, threshold=6.9)}", unsafe_allow_html=True)
            col_m2.markdown(f"SST-Score (70%)<br>{get_score_badge(row['SST-Score'], is_pct=False, threshold=6.9)}", unsafe_allow_html=True)
            col_m3.markdown(f"M3 Ensemble (20%)<br>{get_score_badge(row['M3 Ensemble'], is_pct=True)}", unsafe_allow_html=True)

            if col_act.button(f"📊 Details", key=f"btn_{row['Ticker']}_{idx}", use_container_width=True):
                st.session_state.selected_ticker = row["Ticker"]
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
ticker = st.session_state.selected_ticker
st.subheader(f"🔍 Technische Details voor: {ticker}")

res = run_full_combined_scan(ticker)

if res:
    m1 = res["m1_raw"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Totaal Score", f"{res['Totaal Score (%)']}%", res["Signaal"])
    c2.metric("AI-TA-Score", f"{res['AI-TA-Score']}/10")
    c3.metric("SST-Score", f"{res['SST-Score']}/10")
    c4.metric("M3 Ensemble", res["M3 Ensemble"])

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
