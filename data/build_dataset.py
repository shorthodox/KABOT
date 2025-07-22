import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === CONFIG ===
stocks = ['TATAMOTORS.NS', 'HDFCBANK.NS', 'SUZLON.NS', 'ZOMATO.NS']
start_date = '2024-07-21'
end_date = '2025-07-21'

# Trading strategy parameters
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD = 2
ATR_WINDOW = 14

def compute_rsi(series, window=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD manually"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20, std=2):
    """Calculate Bollinger Bands manually"""
    sma = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    return upper, lower

def compute_atr(high, low, close, window=14):
    """Calculate Average True Range manually"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def generate_labels(df):
    """Generate trading signals based on technical indicators"""
    # Initialize all as HOLD
    df['label'] = 'HOLD'
    
    # BUY conditions (need at least 2 to trigger)
    buy_conditions = (
        (df['rsi'] < RSI_OVERSOLD).astype(int) +
        (df['macd'] > df['macd_signal']).astype(int) +
        (df['close_price'] < df['bollinger_lower']).astype(int)
    )
    
    # SELL conditions (need at least 2 to trigger)
    sell_conditions = (
        (df['rsi'] > RSI_OVERBOUGHT).astype(int) +
        (df['macd'] < df['macd_signal']).astype(int) +
        (df['close_price'] > df['bollinger_upper']).astype(int))
    
    # Apply labels
    df.loc[buy_conditions >= 2, 'label'] = 'BUY'
    df.loc[sell_conditions >= 2, 'label'] = 'SELL'
    
    return df

def get_stock_data(symbol):
    print(f"\n[FETCHING] {symbol}")
    try:
        # Download data with 1 day retry for failed attempts
        try:
            df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        except:
            print("Retrying download...")
            df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
            
        if df is None or df.empty:
            print(f"[ERROR] No data fetched for {symbol}")
            return None
        
        df.dropna(inplace=True)
        if len(df) == 0:
            print(f"[ERROR] Empty DataFrame after dropna for {symbol}")
            return None

        # Calculate all indicators manually
        df['rsi'] = compute_rsi(df['Close']).fillna(50)
        
        macd, signal = compute_macd(df['Close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df['macd'] = macd.fillna(0)
        df['macd_signal'] = signal.fillna(0)
        
        df['bollinger_upper'], df['bollinger_lower'] = compute_bollinger_bands(
            df['Close'], BB_WINDOW, BB_STD)
        df['bollinger_upper'] = df['bollinger_upper'].bfill().ffill()
        df['bollinger_lower'] = df['bollinger_lower'].bfill().ffill()
        
        df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean().bfill().ffill()
        
        df['atr'] = compute_atr(df['High'], df['Low'], df['Close'], ATR_WINDOW)
        df['atr'] = df['atr'].fillna(df['atr'].mean() if not df['atr'].isnull().all() else 0)
        
        # Select and rename columns
        df = df[['Close', 'ema_20', 'rsi', 'macd', 'macd_signal', 
                'bollinger_upper', 'bollinger_lower', 'atr', 'Volume']]
        df.columns = ['close_price', 'ema_20', 'rsi', 'macd', 'macd_signal', 
                     'bollinger_upper', 'bollinger_lower', 'atr', 'volume']
        
        # Generate trading signals
        df = generate_labels(df)
        df['stock'] = symbol
        
        print(f"[SUCCESS] Processed {len(df)} rows for {symbol}")
        print("Label distribution:")
        print(df['label'].value_counts())
        
        return df
    
    except Exception as e:
        print(f"[ERROR] Failed to process {symbol}: {str(e)}")
        return None

def main():
    all_data = []
    for stock in stocks:
        df = get_stock_data(stock)
        if df is not None:
            all_data.append(df)
    
    if all_data:
        full_df = pd.concat(all_data)
        full_df.dropna(inplace=True)
        
        # Save to CSV
        save_path = "data/processed/trading_dataset.csv"
        full_df.to_csv(save_path, index=False)
        
        print(f"\n[FINAL RESULT] Saved {len(full_df)} rows to {save_path}")
        print("Final label distribution:")
        print(full_df['label'].value_counts())
    else:
        print("\n[ERROR] No data was processed successfully")

if __name__ == "__main__":
    main()