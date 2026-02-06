import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import joblib
import sys

# Add project root to path to allow importing src.config when running successfully from root or as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def download_data(ticker='RELIANCE.NS', period='5y'):
    """
    Download daily OHLCV data for a specific ticker.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    
    # yfinance sometimes returns MultiIndex columns (Ticker, Price Type), we flatten them if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure standard column names
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def calculate_technical_indicators(df):
    """
    Calculate Technical Indicators including RSI, MACD, Bollinger Bands, and ATR.
    """
    print("Calculating technical indicators...")
    
    # RSI (14 periods)
    rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    
    # MACD (Fast=12, Slow=26, Signal=9)
    macd_indicator = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['MACD_Diff'] = macd_indicator.macd_diff()
    
    # Bollinger Bands (Window=20, NumStd=2)
    bb_indicator = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()
    df['BB_Mid'] = bb_indicator.bollinger_mavg()
    
    # ATR (14 periods)
    atr_indicator = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr_indicator.average_true_range()
    
    return df

def create_target(df):
    """
    Create a target variable for 'Buy', 'Sell', and 'Hold'.
    Logic: If next day's close > 1% up = Buy (1); < -1% down = Sell (2); otherwise = Hold (0).
    """
    print("Creating target variable...")
    
    # Calculate percentage change for next day
    df['Next_Close'] = df['Close'].shift(-1)
    df['Pct_Change_Next'] = (df['Next_Close'] - df['Close']) / df['Close']
    
    # Drop rows where future data is missing (the last row)
    df.dropna(subset=['Pct_Change_Next'], inplace=True)
    
    conditions = [
        (df['Pct_Change_Next'] > 0.01),  # Buy
        (df['Pct_Change_Next'] < -0.01)  # Sell
    ]
    choices = [1, 2] # 1: Buy, 2: Sell
    
    # Default is 0: Hold
    df['Target'] = np.select(conditions, choices, default=0)
    
    # Drop temporary columns
    df.drop(['Next_Close', 'Pct_Change_Next'], axis=1, inplace=True)
    
    return df

def prepare_data():
    # 1. Download Data
    df = download_data()
    
    if df.empty:
        print("No data downloaded. Exiting.")
        return

    # 2. Calculate Indicators
    df = calculate_technical_indicators(df)
    
    # 3. Create Target
    df = create_target(df)
    
    # 4. Handle Missing Values
    initial_shape = df.shape
    df.dropna(inplace=True)
    print(f"Dropped rows with missing values: {initial_shape[0] - df.shape[0]} rows removed.")
    
    # 5. Split Data (Chronological 80/20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # 6. Normalize Feature Columns
    feature_cols = [col for col in df.columns if col != 'Target']
    
    print("Normalizing features...")
    scaler = MinMaxScaler()
    
    # Fit on Train, Transform Train and Test
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # 7. Save to CSV
    print("Saving to CSV...")
    train_df.to_csv(config.TRAIN_DATA_FILE, index=False)
    test_df.to_csv(config.TEST_DATA_FILE, index=False)
    
    # Save Metadata for Dashboard (Dates and Unscaled Close)
    # Re-download or re-slice original if needed, but here we can just use the indices if preserved?
    # The original download had dates as index.
    # Note: df was modified in place by dropna.
    # We want the unscaled data corresponding to test set.
    # Since we can't easily invert the scaler perfectly without loading it (we have it in memory though),
    # let's just grab the columns from the original process if we had kept a copy, or re-download.
    # Actually, simpler: maintain a copy of unscaled df before scaling?
    # But wait, earlier code did: test_meta = df.iloc[train_size:].copy() BEFORE scaling?
    # Let's check original code:
    # "train_df[feature_cols] = scaler.fit_transform..." -> Modifies train_df.
    # "test_meta = df.iloc[train_size:].copy()" -> This was done AFTER creating train_df/test_df but BEFORE saving test_meta?
    # In original code:
    # 1. split
    # 2. scale train_df and test_df
    # 3. test_meta = df.iloc[train_size:]... 
    # BUT 'df' was NOT scaled in the original code! 'train_df' and 'test_df' were copies.
    # So 'df' still holds unscaled data. Perfect.
    
    test_meta = df.iloc[train_size:].copy()
    test_meta = test_meta[['Open', 'High', 'Low', 'Close', 'Volume']] # Keep OHLCV
    test_meta.to_csv(config.TEST_META_FILE, index=True) # Keep index (Date)
    
    # Save Scaler for Live Prediction
    joblib.dump(scaler, config.SCALER_FILE)
    print(f"Saved 'scaler.pkl' to {config.SCALER_FILE}")
    print(f"Saved 'test_meta.csv' to {config.TEST_META_FILE}")

    print(f"Done! Files 'train_data.csv' and 'test_data.csv' have been created in {config.DATA_DIR}.")

if __name__ == "__main__":
    prepare_data()
