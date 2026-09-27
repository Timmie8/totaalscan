import numpy as np
import pandas as pd
import ta
import yfinance as yf

def calculate_ml_3d_probability(rsi_val, macd_diff, vol_ratio, mfi_val, ad_trend_3d, ema5, ema15, live_price):
    base_prob = 50.0
    if live_price > ema5 > ema15:
        base_prob += 10.0
    elif live_price < ema5 < ema15:
        base_prob -= 10.0

    if macd_diff > 0:
        base_prob += 8.0
    else:
        base_prob -= 6.0

    if vol_ratio >= 1.2 and mfi_val >= 7.5:
        base_prob += 12.0
    elif mfi_val >= 6.0:
        base_prob += 5.0
    elif mfi_val <= 3.0:
        base_prob -= 8.0

    if ad_trend_3d == "Accumulatie 🟢":
        base_prob += 8.0
    elif ad_trend_3d == "Distributie 🔴":
        base_prob -= 8.0

    if 48 <= rsi_val <= 62:
        base_prob += 7.0
    elif rsi_val > 70:
        base_prob -= 8.0
    elif rsi_val < 35:
        base_prob += 3.0

    final_prob = min(95.0, max(15.0, base_prob))
    return round(final_prob, 1)

def calculate_comprehensive_scores(vol_ratio, rsi_val, macd_val, macd_prev, stoch_k, stoch_d, put_call_ratio, short_float, mfi_val, ad_trend_3d, ml_prob):
    vol_score = 5.0
    if vol_ratio >= 1.5: vol_score += 2.0
    elif vol_ratio >= 1.1: vol_score += 1.0

    if mfi_val >= 8.0: vol_score += 3.0
    elif mfi_val >= 6.0: vol_score += 1.5
    elif mfi_val <= 3.0: vol_score -= 2.0
    vol_score = round(min(10.0, max(1.0, vol_score)), 1)

    pcr_score = 5.0
    if put_call_ratio is not None and not np.isnan(put_call_ratio):
        if put_call_ratio < 0.8: pcr_score = 9.0
        elif put_call_ratio <= 1.0: pcr_score = 6.5
        else: pcr_score = 3.0
    pcr_score = round(pcr_score, 1)

    sentiment_score = 5.0
    if short_float is not None and not np.isnan(short_float):
        if short_float < 0.05: sentiment_score += 2.5
        elif short_float > 0.15: sentiment_score -= 2.0
    if vol_ratio > 1.3: sentiment_score += 1.5
    sentiment_score = round(min(10.0, max(1.0, sentiment_score)), 1)

    tech_score = 5.0
    if 55 <= rsi_val <= 70: tech_score += 1.0
    elif rsi_val > 70: tech_score -= 1.0
    if macd_val > 0 and macd_val > macd_prev: tech_score += 1.5
    if stoch_k > stoch_d: tech_score += 1.0
    if ad_trend_3d == "Accumulatie 🟢": tech_score += 1.5
    elif ad_trend_3d == "Distributie 🔴": tech_score -= 1.5
    tech_score = round(min(10.0, max(1.0, tech_score)), 1)

    ml_score = round(ml_prob / 10.0, 1)

    total_score = (
        (vol_score * 0.25) +
        (pcr_score * 0.15) +
        (sentiment_score * 0.15) +
        (tech_score * 0.25) +
        (ml_score * 0.20)
    )
    total_score = round(total_score, 1)

    return total_score, vol_score, pcr_score, sentiment_score, tech_score, ml_score

def run_method_2(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d", interval="1d")
        if df.empty or len(df) < 30:
            return None

        info = ticker.info or {}
        live_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        day_change_pct = ((live_price - prev_close) / prev_close) * 100

        current_volume = df['Volume'].iloc[-1]
        avg_vol_20d = df['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = current_volume / avg_vol_20d if avg_vol_20d > 0 else 1.0

        high_today = df['High'].iloc[-1]
        low_today = df['Low'].iloc[-1]
        close_today = df['Close'].iloc[-1]

        day_range = high_today - low_today
        clv = ((close_today - low_today) - (high_today - close_today)) / day_range if day_range > 0 else 0.0

        mf_raw = (clv * 0.5) + (np.clip(day_change_pct / 3.0, -1, 1) * 0.3) + (np.clip((vol_ratio - 1.0), 0, 1) * 0.2)
        mfi_val = round(np.clip((mf_raw + 1) * 4.5 + 1, 1.0, 10.0), 1)

        clv_series = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
        clv_series = clv_series.fillna(0)
        ad_line = (clv_series * df['Volume']).cumsum()
        ad_diff_3d = ad_line.iloc[-1] - ad_line.iloc[-4]
        ad_trend_3d = "Accumulatie 🟢" if ad_diff_3d > 0 else ("Distributie 🔴" if ad_diff_3d < 0 else "Neutraal 🟡")

        ema5 = ta.trend.ema_indicator(df['Close'], window=5).iloc[-1]
        ema15 = ta.trend.ema_indicator(df['Close'], window=15).iloc[-1]
        rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]

        low_min14 = df['Low'].rolling(window=14).min()
        high_max14 = df['High'].rolling(window=14).max()
        stoch_k_series = 100 * ((df['Close'] - low_min14) / (high_max14 - low_min14))
        stoch_d_series = stoch_k_series.rolling(window=3).mean()
        stoch_k = stoch_k_series.iloc[-1]
        stoch_d = stoch_d_series.iloc[-1]

        macd = ta.trend.MACD(df['Close'])
        macd_val = macd.macd().iloc[-1]
        macd_prev = macd.macd().iloc[-2]
        macd_diff = macd.macd_diff().iloc[-1]

        pcr_volume = None
        try:
            expirations = ticker.options
            if expirations:
                opt = ticker.option_chain(expirations[0])
                c_vol = opt.calls['volume'].fillna(0).sum()
                p_vol = opt.puts['volume'].fillna(0).sum()
                pcr_volume = p_vol / c_vol if c_vol > 0 else 1.0
        except Exception:
            pass

        short_float = info.get("shortPercentOfFloat", None)

        ml_prob = calculate_ml_3d_probability(rsi, macd_diff, vol_ratio, mfi_val, ad_trend_3d, ema5, ema15, live_price)
        total_score, vol_score, pcr_score, sentiment_score, tech_score, ml_score = calculate_comprehensive_scores(
            vol_ratio, rsi, macd_val, macd_prev, stoch_k, stoch_d, pcr_volume, short_float, mfi_val, ad_trend_3d, ml_prob
        )

        return {
            "Ticker": symbol,
            "Koers": round(live_price, 2),
            "M2_Score": total_score,
            "M2_ML_Prob": f"{ml_prob}%",
            "M2_MFI_Score": mfi_val,
            "M2_Tech_Score": tech_score,
            "M2_Vol_Score": vol_score,
            "M2_AD_Trend": ad_trend_3d
        }
    except Exception:
        return None
