import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import ta
import yfinance as yf

def add_technical_indicators_m3(df):
    df = df.copy()
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
    
    df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
    
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

def calc_ensemble(df):
    feature_cols = [
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'ADX',
        'RSI', 'Stoch', 'CCI', 'BB_high', 'BB_low', 'ATR', 'OBV', 'MFI', 'Return'
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df['Target']
    
    if len(X) < 80:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    rf_proba = rf.predict_proba(X_test)[:, 1]
    gb_proba = gb.predict_proba(X_test)[:, 1]
    ensemble_proba = (rf_proba + gb_proba) / 2
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ensemble_pred)
    
    latest_features = X.iloc[[-1]]
    rf_latest = rf.predict_proba(latest_features)[0, 1]
    gb_latest = gb.predict_proba(latest_features)[0, 1]
    ens_latest = (rf_latest + gb_latest) / 2
    
    rf_imp = pd.Series(rf.feature_importances_, index=available)
    gb_imp = pd.Series(gb.feature_importances_, index=available)
    avg_imp = ((rf_imp + gb_imp) / 2).sort_values(ascending=False).head(10)
    
    return {
        'rf_acc': rf_acc,
        'gb_acc': gb_acc,
        'ens_acc': ens_acc,
        'prob_up': ens_latest,
        'ensemble_proba': ensemble_proba,
        'X_test': X_test,
        'feature_imp': avg_imp
    }

class StockDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def calc_lstm(df, epochs=8, seq_len=30):
    if len(df) < 100:
        return None
        
    prices = df['Close'].values.astype(np.float32)
    train_size = int(len(prices) * 0.8)
    
    scaler = StandardScaler()
    train_prices = prices[:train_size].reshape(-1, 1)
    scaler.fit(train_prices)
    
    prices_scaled = scaler.transform(prices.reshape(-1, 1)).flatten()
    
    train_data = prices_scaled[:train_size]
    test_data = prices_scaled[train_size - seq_len:]
    
    train_ds = StockDataset(train_data, seq_len)
    test_ds = StockDataset(test_data, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.unsqueeze(-1).to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.unsqueeze(-1).to(device)
            pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())
    
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals_inv = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    dir_actual = np.diff(actuals_inv) > 0
    dir_pred = np.diff(preds_inv) > 0
    dir_acc = np.mean(dir_actual == dir_pred) if len(dir_actual) > 0 else 0
    
    last_seq = torch.FloatTensor(prices_scaled[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        next_pred_scaled = model(last_seq).cpu().numpy()[0, 0]
    next_pred = scaler.inverse_transform([[next_pred_scaled]])[0, 0]
    current = prices[-1]
    change_pct = (next_pred - current) / current * 100
    
    return {
        'current_price': current,
        'next_pred': next_pred,
        'change_pct': change_pct,
        'dir_acc': dir_acc,
        'losses': losses,
        'preds_inv': preds_inv,
        'actuals_inv': actuals_inv,
        'train_size': train_size
    }

def run_method_3(ticker, epochs=8):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        if df.empty or len(df) < 100:
            return None
            
        df = add_technical_indicators_m3(df)
        ens = calc_ensemble(df)
        lstm = calc_lstm(df, epochs=epochs)
        
        ens_prob = ens['prob_up'] if ens else 0.5
        lstm_pct = lstm['change_pct'] if lstm else 0.0
        
        score_ens = ens_prob * 10.0
        score_lstm = np.clip((lstm_pct + 5.0), 0.0, 10.0)
        
        m3_score = round((score_ens * 0.5) + (score_lstm * 0.5), 1)
        
        return {
            "Ticker": ticker,
            "M3_Score": m3_score,
            "Ensemble_Prob": f"{ens_prob * 100:.1f}%",
            "LSTM_Change_Pct": f"{lstm_pct:+.2f}%",
            "LSTM_Pred": round(lstm['next_pred'], 2) if lstm else "N/A",
            "ens_data": ens,
            "lstm_data": lstm,
            "df": df
        }
    except Exception:
        return None
