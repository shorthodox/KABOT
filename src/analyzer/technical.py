# src/indicators/technical.py

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def apply_indicators(self):
        df = self.df.copy()
        
        # Ensure numeric values and handle potential errors
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if df['close'].isna().any():
            raise ValueError("Invalid close prices detected after conversion")

        # --- Moving Averages ---
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # --- MACD ---
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # --- RSI ---
        delta = df['close'].diff().astype(float)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # --- Bollinger Bands ---
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        
        # Calculate Bollinger Band Width
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # --- Volume Indicators ---
        if 'volume' in df.columns:
            df['Volume_MA_20'] = df['volume'].rolling(window=20).mean()
            df['Volume_Change'] = df['volume'].pct_change()
        
        # --- Cleanup ---
        # Replace infinite values first
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows with any NA values (from indicators or original data)
        initial_count = len(df)
        df.dropna(inplace=True)
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} rows with NA/Inf values")
        
        if len(df) < 20:
            raise ValueError("Insufficient data remaining after cleaning (need at least 20 rows)")
        
        return df



