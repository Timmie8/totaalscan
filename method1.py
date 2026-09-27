import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier

_vader_instance = None

def get_vader():
    global _vader_instance
    if _vader_instance is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _vader_instance = SentimentIntensityAnalyzer()
    return _vader_instance

def get_daily_data_m1(ticker, period="1y"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval="1d", auto_adjust=False)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def get_ticker_info_and_options(ticker):
    short_percent, short_ratio = 0.0, 0.0
    total_call_vol, total_put_vol = 0, 0
    total_call_oi, total_put_oi = 0, 0
    iv_list = []

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        short_percent = info.get("shortPercentOfFloat", 0) or 0
        short_ratio = info.get("shortRatio", 0) or 0
        expirations = t.options
        if expirations:
            for exp in expirations[:3]:
                try:
                    opt = t.option_chain(exp)
                    c_df = opt.calls
                    p_df = opt.puts

                    if c_df is not None and not c_df.empty:
                        c_vol = pd.to_numeric(c_df["volume"], errors="coerce").fillna(0).sum()
                        c_oi = pd.to_numeric(c_df["openInterest"], errors="coerce").fillna(0).sum()
                        total_call_vol += c_vol
                        total_call_oi += c_oi
                        c_iv = pd.to_numeric(c_df["impliedVolatility"], errors="coerce").dropna().mean()
                        if pd.notna(c_iv) and c_iv > 0:
                            iv_list.append(c_iv)

                    if p_df is not None and not p_df.empty:
                        p_vol = pd.to_numeric(p_df["volume"], errors="coerce").fillna(0).sum()
                        p_oi = pd.to_numeric(p_df["openInterest"], errors="coerce").fillna(0).sum()
                        total_put_vol += p_vol
                        total_put_oi += p_oi
                        p_iv = pd.to_numeric(p_df["impliedVolatility"], errors="coerce").dropna().mean()
                        if pd.notna(p_iv) and p_iv > 0:
                            iv_list.append(p_iv)
                except Exception:
                    continue
    except Exception:
        pass

    pc_ratio_vol = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 1.0
    cp_ratio = round(total_call_vol / total_put_vol, 2) if total_put_vol > 0 else (2.0 if total_call_vol > 0 else 1.0)
    pc_ratio_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 1.0
    avg_iv = round(float(np.mean(iv_list)) * 100, 1) if iv_list else 0.0

    return {
        "short_percent": round(short_percent * 100, 2),
        "short_ratio": round(short_ratio, 1),
        "cp_ratio": cp_ratio,
        "pc_ratio_vol": pc_ratio_vol,
        "pc_ratio_oi": pc_ratio_oi,
        "avg_iv": avg_iv,
        "calls_vol": int(total_call_vol),
        "puts_vol": int(total_put_vol),
        "calls_oi": int(total_call_oi),
        "puts_oi": int(total_put_oi),
    }

def get_ai_vader_sentiment(ticker):
    sia = get_vader()
    titles = []
    try:
        t = yf.Ticker(ticker)
        news_items = t.news or []
        for item in news_items:
            if isinstance(item, dict):
                title = item.get("title") or item.get("content", {}).get("title")
                if title:
                    titles.append(title)
    except Exception:
        pass

    if not titles:
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item"):
                    t_text = item.find("title")
                    if t_text is not None and t_text.text:
                        titles.append(t_text.text)
        except Exception:
            pass

    if not titles:
        return 5.0, "Geen recente koppen"

    scores = [sia.polarity_scores(t)["compound"] for t in titles[:10]]
    avg_compound = float(np.mean(scores))
    ai_score = round((avg_compound + 1) * 4.5 + 1, 1)
    label = "Bullish" if ai_score >= 6.0 else ("Bearish" if ai_score <= 4.0 else "Neutraal")
    return ai_score, f"{label} ({len(scores)} koppen)"

def analyze_candles(df):
    if len(df) < 3:
        return {
            "avg_volume": 0, "last_volume": 0,
            "candle_1d_label": "N/A", "candle_1d_bullish": False,
            "candle_3d_label": "N/A", "candle_3d_bullish": False,
        }
    avg_vol_10 = int(df["Volume"].tail(10).mean())
    last_vol = int(df["Volume"].iloc[-1])
    last_row = df.iloc[-1]
    o_1, c_1 = last_row["Open"], last_row["Close"]
    change_1d = ((c_1 - o_1) / o_1) * 100
    is_1d_bullish = c_1 >= o_1
    candle_1d_label = f"🟢 Bullish ({change_1d:+.2f}%)" if is_1d_bullish else f"🔴 Bearish ({change_1d:+.2f}%)"

    last_3 = df.tail(3)
    c_start = last_3["Open"].iloc[0]
    c_end = last_3["Close"].iloc[-1]
    net_3d_change = ((c_end - c_start) / c_start) * 100
    c_flags = [(row["Close"] >= row["Open"]) for _, row in last_3.iterrows()]

    if all(c_flags):
        candle_3d_label = f"🟢 Sterk Bullish (3 Witte Soldaten: {net_3d_change:+.2f}%)"
        is_3d_bullish = True
    elif not any(c_flags):
        candle_3d_label = f"🔴 Sterk Bearish (3 Zwarte Kraaien: {net_3d_change:+.2f}%)"
        is_3d_bullish = False
    elif c_end > c_start:
        candle_3d_label = f"🟢 Bullish Reversal/Trend ({net_3d_change:+.2f}%)"
        is_3d_bullish = True
    else:
        candle_3d_label = f"🔴 Bearish Reversal/Trend ({net_3d_change:+.2f}%)"
        is_3d_bullish = False

    return {
        "avg_volume": avg_vol_10, "last_volume": last_vol,
        "candle_1d_label": candle_1d_label, "candle_1d_bullish": is_1d_bullish,
        "candle_3d_label": candle_3d_label, "candle_3d_bullish": is_3d_bullish,
    }

def calc_support_resistance(df, window=3, lookback=60):
    recent = df.tail(lookback).copy()
    highs, lows = [], []
    h, l = recent["High"].values, recent["Low"].values
    for i in range(window, len(recent) - window):
        if h[i] == max(h[i - window : i + window + 1]):
            highs.append(h[i])
        if l[i] == min(l[i - window : i + window + 1]):
            lows.append(l[i])
    current_price = df["Close"].iloc[-1]
    resistances = sorted([x for x in highs if x > current_price])
    supports = sorted([x for x in lows if x < current_price], reverse=True)
    return (supports[0] if supports else None, resistances[0] if resistances else None)

def compute_ml_swing_prediction(df):
    if len(df) < 80:
        return 50.0, "Onvoldoende data"
    data = df.copy()
    data["Return_1D"] = data["Close"].pct_change(1)
    data["Return_3D"] = data["Close"].pct_change(3)
    data["Return_5D"] = data["Close"].pct_change(5)
    data["Vol_Ratio"] = data["Volume"] / data["Volume"].rolling(10).mean()
    data["SMA20_Dist"] = (data["Close"] - data["Close"].rolling(20).mean()) / data["Close"].rolling(20).mean()
    data["Target"] = (data["Close"].shift(-3) > data["Close"] * 1.015).astype(int)

    feature_cols = ["Return_1D", "Return_3D", "Return_5D", "Vol_Ratio", "SMA20_Dist"]
    clean_data = data.dropna()
    if len(clean_data) < 50:
        return 50.0, "Onvoldoende schone data"

    X, y = clean_data[feature_cols], clean_data["Target"]
    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X[:-3], y[:-3])
    prob_bullish = model.predict_proba(X.iloc[[-1]])[0][1] * 100
    label = "Stijging (3-5d)" if prob_bullish >= 60 else ("Daling/Risico" if prob_bullish <= 40 else "Neutraal")
    return round(prob_bullish, 1), f"AI Pattern: {label}"

def add_technical_indicators_m1(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def run_method_1(t_code):
    df = get_daily_data_m1(t_code)
    if df.empty:
        return None

    df = add_technical_indicators_m1(df)
    opt_short = get_ticker_info_and_options(t_code)
    ai_score, ai_subtext = get_ai_vader_sentiment(t_code)
    ml_prob, ml_subtext = compute_ml_swing_prediction(df)
    support, resistance = calc_support_resistance(df)
    candle_info = analyze_candles(df)

    last = df.iloc[-1]
    price = last["Close"]
    macd_bullish = last["MACD"] > last["MACD_signal"]
    sma_bullish = price > last["SMA20"]

    score = 0
    reasons = []

    if ml_prob >= 60.0:
        score += 1.5
        reasons.append(f"ML High Prob ({ml_prob}%)")
    elif ml_prob <= 40.0:
        score -= 1.5
        reasons.append(f"ML Low Prob ({ml_prob}%)")

    if candle_info["candle_1d_bullish"]:
        score += 0.5
        reasons.append("1D Candle Bullish")
    else:
        score -= 0.5
        reasons.append("1D Candle Bearish")

    if candle_info["candle_3d_bullish"]:
        score += 0.5
        reasons.append("3D Candle Pattern Bullish")
    else:
        score -= 0.5
        reasons.append("3D Candle Pattern Bearish")

    if opt_short["cp_ratio"] > 1.3:
        score += 1
        reasons.append("Options Call Dominant")
    elif opt_short["cp_ratio"] < 0.7:
        score -= 1
        reasons.append("Options Put Dominant")

    if opt_short["short_percent"] > 15.0:
        score += 0.5
        reasons.append("High Short Squeeze Potential")

    if sma_bullish:
        score += 1
        reasons.append("Boven SMA20")
    else:
        score -= 1
        reasons.append("Onder SMA20")

    if macd_bullish:
        score += 1
        reasons.append("MACD Bullish")

    score = round(score, 1)
    signal = "🟢 BULLISH" if score >= 2.5 else ("🔴 BEARISH" if score <= -2.5 else "🟠 NEUTRAAL")

    return {
        "Ticker": t_code,
        "Koers": round(price, 2),
        "M1_Score": score,
        "M1_Signaal": signal,
        "M1_ML_Kans": f"{ml_prob}%",
        "M1_AI_Score": f"{ai_score}/10",
        "Support": f"${support:.2f}" if support else "N/A",
        "Resistance": f"${resistance:.2f}" if resistance else "N/A",
        "Short_Float": f"{opt_short['short_percent']}%",
        "Call_Put_Ratio": opt_short["cp_ratio"],
        "Details": " · ".join(reasons),
        "df": df,
        "support_val": support,
        "resistance_val": resistance,
        "candle_info": candle_info,
        "opt_short": opt_short
    }
