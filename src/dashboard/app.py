import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, GlobalAveragePooling1D, Dropout, LayerNormalization, MultiHeadAttention, Add
import xgboost as xgb
import lightgbm as lgb
from darts import TimeSeries
from darts.models import NBEATSModel
import ta
import yfinance as yf
import os
import sys
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Path Setup
try:
    import src.config as config
except ImportError:
    # If running with streamlit run src/dashboard/app.py, cwd might be root, so this might work
    # Or we need to look up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.append(project_root)
    import src.config as config

# --- Configuration ---
st.set_page_config(page_title="Advanced Stock Prediction System", page_icon="📈", layout="wide")

# Paths from Config
TEST_META_FILE = config.TEST_META_FILE
PREDICTIONS_FILE = config.PREDICTIONS_FILE
SCALER_FILE = config.SCALER_FILE
MODEL_1_FILE = config.MODEL_1_FILE
MODEL_2_FILE = config.MODEL_2_FILE
MODEL_3_FILE = config.MODEL_3_FILE
ENSEMBLE_FILE = config.ENSEMBLE_MODEL_FILE

# Define Weights for Feature Importance Labels
FEATURE_NAMES = ['M1_Hold', 'M1_Buy', 'M1_Sell', 'NBEATS_Pct', 'XGB_Hold', 'XGB_Buy', 'XGB_Sell']

# --- Custom Objects for Model 1 ---
@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int64)
        y_true = tf.one_hot(y_true, depth=y_pred.shape[-1])
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))

# --- Utils ---
def calculate_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    return df

# --- Title ---
st.title("📈 Advanced Algo-Trading System")
st.markdown("### Deep Learning Ensemble: Hybrid CNN-GRU-Transformer + N-BEATS + XGBoost")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "💰 Backtesting Simulator", "🔴 Live Prediction"])

# ==========================================
# TAB 1: MODEL EVALUATION
# ==========================================
with tab1:
    st.header("Model Performance Metrics")
    
    if os.path.exists(PREDICTIONS_FILE) and os.path.exists(ENSEMBLE_FILE):
        df_pred = pd.read_csv(PREDICTIONS_FILE)
        y_true = df_pred['Target']
        y_pred = df_pred['Final_Prediction']
        
        # 1. Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Hold', 'Buy', 'Sell'], 
                        yticklabels=['Hold', 'Buy', 'Sell'], ax=ax_cm)
            ax_cm.set_ylabel('Actual')
            ax_cm.set_xlabel('Predicted')
            st.pyplot(fig_cm)
            
        # 2. Feature Importance
        with col2:
            st.subheader("Ensemble Feature Importance")
            try:
                model = joblib.load(ENSEMBLE_FILE)
                # Check if model has feature_importances_
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    feat_fig, feat_ax = plt.subplots()
                    sns.barplot(x=importances, y=FEATURE_NAMES, ax=feat_ax, hue=FEATURE_NAMES, palette="viridis", legend=False)
                    feat_ax.set_title("Contribution of Base Models")
                    st.pyplot(feat_fig)
                else:
                    st.info("Feature importance not available for this model type.")
            except Exception as e:
                st.error(f"Could not load feature importance: {e}")

        # 3. Detailed Metrics
        st.subheader("Class-wise Metrics")
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=['Hold', 'Buy', 'Sell'], output_dict=True)
        metrics_df = pd.DataFrame(report).transpose()
        st.dataframe(metrics_df.style.highlight_max(axis=0))

    else:
        st.error(f"Prediction files not found at {PREDICTIONS_FILE}. Please run the training pipeline.")


# ==========================================
# TAB 2: BACKTESTING SIMULATOR
# ==========================================
with tab2:
    st.header("Strategy Backtesting")
    
    if os.path.exists(TEST_META_FILE) and os.path.exists(PREDICTIONS_FILE):
        meta_df = pd.read_csv(TEST_META_FILE)
        preds_df = pd.read_csv(PREDICTIONS_FILE)
        
        # Controls
        initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
        
        if len(meta_df) == len(preds_df):
            # Combine
            bt_df = pd.concat([meta_df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
            
            # Simulation Loop
            cash = initial_capital
            shares = 0
            portfolio_values = []
            buy_hold_shares = initial_capital / bt_df.iloc[0]['Close']
            
            trade_log = []
            
            for i, row in bt_df.iterrows():
                price = row['Close']
                signal = row['Final_Prediction'] # 0: Hold, 1: Buy, 2: Sell
                
                # Buy Rule: If Buy signal and we have cash, buy max
                if signal == 1 and cash > 0:
                    shares = cash / price
                    cash = 0
                    trade_log.append({'Date': i, 'Type': 'BUY', 'Price': price, 'Value': shares*price})
                    
                # Sell Rule: If Sell signal and we have shares, sell all
                elif signal == 2 and shares > 0:
                    cash = shares * price
                    shares = 0
                    trade_log.append({'Date': i, 'Type': 'SELL', 'Price': price, 'Value': cash})
                
                # Calculate Daily Value
                current_val = cash + (shares * price)
                portfolio_values.append(current_val)
            
            bt_df['Portfolio_Value'] = portfolio_values
            bt_df['Buy_Hold_Value'] = bt_df['Close'] * buy_hold_shares
            
            # Results
            final_val = portfolio_values[-1]
            bh_val = bt_df.iloc[-1]['Buy_Hold_Value']
            
            strat_ret = (final_val - initial_capital) / initial_capital
            bh_ret = (bh_val - initial_capital) / initial_capital
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Strategy Return", f"{strat_ret:.2%}", f"${final_val - initial_capital:.2f}")
            m2.metric("Buy & Hold Return", f"{bh_ret:.2%}", f"${bh_val - initial_capital:.2f}")
            m3.metric("Trades Executed", len([t for t in trade_log if t['Type']=='SELL']))
            
            # Equity Curve
            st.subheader("Equity Curve")
            st.line_chart(bt_df[['Portfolio_Value', 'Buy_Hold_Value']])
            
            # Detailed Log
            with st.expander("View Trade Log"):
                st.dataframe(pd.DataFrame(trade_log))
                
        else:
            st.error("Data Mismatch between Price Data and Predictions.")
    else:
        st.warning("Data files missing.")

# ==========================================
# TAB 3: LIVE PREDICTION
# ==========================================
with tab3:
    st.header("Live Analysis")
    
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        ticker = st.text_input("Ticker Symbol (Yahoo Finance)", value="RELIANCE.NS")
    with col_btn:
        run_pred = st.button("Analyze Live", type="primary")
        
    if run_pred:
        with st.spinner(f"Connecting to market data for {ticker}..."):
            try:
                # 1. Fetch
                # auto_adjust=False ensures standard OHLC behavior matching training
                data = yf.download(ticker, period='6mo', progress=False, auto_adjust=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                
                if len(data) < 100:
                    st.error("Not enough recent data for this ticker.")
                    st.stop()
                    
                # 2. Indicators
                data = calculate_technical_indicators(data)
                data = data.dropna()
                
                # 3. Chart
                st.subheader(f"Price Action: {ticker}")
                st.line_chart(data['Close'].tail(60))
                
                # 4. Prepare Input
                if not os.path.exists(SCALER_FILE):
                    st.error("Scaler file not found.")
                    st.stop()
                    
                scaler = joblib.load(SCALER_FILE)
                # Feature cols must match training
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'BB_High', 'BB_Low', 'BB_Mid', 'ATR']
                
                input_data = data[feature_cols].copy()
                scaled_data = scaler.transform(input_data)
                
                seq_60 = scaled_data[-60:]
                last_row = scaled_data[-1].reshape(1, -1)
                
                # 5. Inference
                # Model 1
                m1 = load_model(MODEL_1_FILE, custom_objects={'FocalLoss': FocalLoss})
                m1_prob = m1.predict(seq_60.reshape(1, 60, -1), verbose=0)[0]
                
                # Model 2
                m2 = NBEATSModel.load(MODEL_2_FILE)
                close_series = TimeSeries.from_values(input_data['Close'].values[-60:])
                m2_pred_price = m2.predict(n=1, series=close_series, verbose=False).values()[0][0]
                m2_pct = (m2_pred_price - input_data['Close'].iloc[-1]) / input_data['Close'].iloc[-1]
                
                # Model 3
                m3 = joblib.load(MODEL_3_FILE)
                m3_prob = m3.predict_proba(last_row)[0]
                
                # Ensemble
                # Feature names must match training for LightGBM to be happy
                ensemble_cols = ['M1_Prob_Hold', 'M1_Prob_Buy', 'M1_Prob_Sell', 'NBEATS_Pred_PctChange', 'XGB_Prob_Hold', 'XGB_Prob_Buy', 'XGB_Prob_Sell']
                
                ensemble_feat = pd.DataFrame([[
                    m1_prob[0], m1_prob[1], m1_prob[2],
                    m2_pct,
                    m3_prob[0], m3_prob[1], m3_prob[2]
                ]], columns=ensemble_cols)
                
                meta_model = joblib.load(ENSEMBLE_FILE)
                final_pred = meta_model.predict(ensemble_feat)[0]
                final_probs = meta_model.predict_proba(ensemble_feat)[0] # [Hold, Buy, Sell]
                
                # 6. Display Result
                st.divider()
                r1, r2, r3 = st.columns([1, 1, 2])
                
                mapping = {0: "HOLD", 1: "BUY", 2: "SELL"}
                color_map = {0: "gray", 1: "green", 2: "red"}
                
                res = mapping[final_pred]
                conf = final_probs[final_pred]
                
                with r1:
                    st.metric("Recommendation", res)
                with r2:
                    st.metric("Confidence", f"{conf:.2%}")
                with r3:
                    # Individual Model Votes
                    st.caption("Model Consensus")
                    st.progress(float(m1_prob[1]), text=f"Hybrid DL Buy Signal: {m1_prob[1]:.2f}")
                    st.progress(float(m3_prob[1]), text=f"XGBoost Buy Signal: {m3_prob[1]:.2f}")
                    st.info(f"N-BEATS Forecast: {m2_pct:+.2%} price movement")

            except Exception as e:
                st.error(f"Analysis Failed: {e}")
