import os
import pandas as pd
from glob import glob

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET_PROFIT_PCT = 0.02  # 2%
STOPLOSS_PCT = 0.01       # 1%

def label_data(df):
    df = df.dropna()
    df['signal'] = 0

    # Simple buy/sell logic — RSI < 30 → Buy, RSI > 70 → Sell
    df.loc[df['rsi'] < 30, 'signal'] = 1
    df.loc[df['rsi'] > 70, 'signal'] = -1

    # Add target profit & stoploss columns
    df['target_profit'] = df['close_price'] * (1 + TARGET_PROFIT_PCT)
    df['stoploss'] = df['close_price'] * (1 - STOPLOSS_PCT)
    return df

def process_all():
    files = glob(os.path.join(RAW_DIR, "*.csv"))
    for file in files:
        name = os.path.basename(file)
        print(f"Processing {name}...")
        df = pd.read_csv(file)
        df = label_data(df)
        df.to_csv(os.path.join(PROCESSED_DIR, name), index=False)
        print(f"Saved processed: {name}")

if __name__ == "__main__":
    process_all()
