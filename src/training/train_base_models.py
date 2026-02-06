import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, GlobalAveragePooling1D, Dropout, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight
from darts import TimeSeries
from darts.models import NBEATSModel
import logging
import warnings
import os
import sys

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

# --- Configuration & Setup ---
warnings.filterwarnings('ignore')
logging.getLogger("darts").setLevel(logging.WARNING)
tf.random.set_seed(42)
np.random.seed(42)

# Filenames from Config
TRAIN_DATA_FILE = config.TRAIN_DATA_FILE
TEST_DATA_FILE = config.TEST_DATA_FILE
MODEL_1_FILE = config.MODEL_1_FILE
MODEL_2_FILE = config.MODEL_2_FILE
MODEL_3_FILE = config.MODEL_3_FILE
ENSEMBLE_OUTPUT = config.ENSEMBLE_INPUT_FILE

# --- Model 1 Component: Hybrid CNN-GRU-Transformer ---

@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def get_config(self):
        config_dict = super(FocalLoss, self).get_config()
        config_dict.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config_dict

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int64)
        y_true = tf.one_hot(y_true, depth=y_pred.shape[-1])
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))

def transformer_block(inputs, key_dim, num_heads, dropout=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    x1 = Add()([inputs, attention_output])
    x1 = LayerNormalization(epsilon=1e-6)(x1)
    
    ffn_output = Dense(key_dim, activation="relu")(x1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    x2 = Add()([x1, ffn_output])
    return LayerNormalization(epsilon=1e-6)(x2)

def build_model_1(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = GRU(64, return_sequences=True)(x)
    x = transformer_block(x, key_dim=64, num_heads=4)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(3, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def create_sequences(df, seq_length=60):
    data = df.drop('Target', axis=1).values
    target = df['Target'].values
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(target[i + seq_length - 1])
    return np.array(X), np.array(y)

def train_and_predict_model_1(train_df, test_df):
    print("\n=== Model 1: Hybrid CNN-GRU-Transformer ===")
    seq_length = 60
    
    # Prepare Data
    X_train, y_train = create_sequences(train_df, seq_length)
    X_test_val, y_test_val = create_sequences(test_df, seq_length)
    
    # Class Weights for Imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")
    
    # Train
    model = build_model_1((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])
    
    callbacks = [
        ModelCheckpoint(MODEL_1_FILE, monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)
    ]
    
    print(f"Training Model 1, saving to {MODEL_1_FILE}...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test_val, y_test_val), callbacks=callbacks, verbose=1, class_weight=class_weights_dict)
    
    # Predict on Test Set (Full)
    # Re-construct sequences for the entire test set including the boundary from train
    print("Generating Model 1 predictions for Ensemble...")
    full_data = pd.concat([train_df.iloc[-seq_length:], test_df]).reset_index(drop=True)
    full_values = full_data.drop('Target', axis=1).values
    
    X_ensemble = []
    for i in range(len(test_df)):
        X_ensemble.append(full_values[i : i + seq_length])
    X_ensemble = np.array(X_ensemble)
    
    probs = model.predict(X_ensemble, verbose=0)
    return pd.DataFrame(probs, columns=['M1_Prob_Hold', 'M1_Prob_Buy', 'M1_Prob_Sell'])

# --- Model 2 Component: N-BEATS ---

def train_and_predict_nbeats(train_df, test_df):
    print("\n=== Model 2: N-BEATS (Trend/Seasonality) ===")
    
    # Darts requires continuous index
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    test_df.index = range(len(train_df), len(train_df) + len(test_df))

    train_series = TimeSeries.from_dataframe(train_df, value_cols='Close')
    test_series = TimeSeries.from_dataframe(test_df, value_cols='Close')
    
    model = NBEATSModel(
        input_chunk_length=60,
        output_chunk_length=1,
        n_epochs=15, 
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu"}
    )
    
    print("Training N-BEATS...")
    model.fit(train_series)
    
    print("Forecasting with N-BEATS...")
    # Historical Forecasts on Test Set
    # We append train to test to allow lookback for the first test predictions
    full_series = train_series.append(test_series)
    
    pred_series = model.historical_forecasts(
        series=full_series,
        start=len(train_series),
        forecast_horizon=1,
        retrain=False,
        verbose=False
    )
    
    preds_df = pred_series.to_dataframe()
    preds_df.columns = ['NBEATS_Pred_Price']
    preds_df = preds_df.reset_index(drop=True)
    
    # Calculate % Change from previous close to predict direction
    all_close = pd.concat([train_df['Close'], test_df['Close']]).reset_index(drop=True)
    test_start_idx = len(train_df)
    prev_closes = all_close.iloc[test_start_idx-1 : -1].values
    
    preds_df['NBEATS_Pred_PctChange'] = (preds_df['NBEATS_Pred_Price'] - prev_closes) / prev_closes
    
    # Save Model
    model.save(MODEL_2_FILE)
    print(f"N-BEATS model saved to '{MODEL_2_FILE}'.")
    
    return preds_df[['NBEATS_Pred_PctChange']]

# --- Model 3 Component: XGBoost ---

def train_and_predict_xgboost(train_df, test_df):
    print("\n=== Model 3: XGBoost (Technical Indicators) ===")
    
    feature_cols = [c for c in train_df.columns if c != 'Target']
    X_train = train_df[feature_cols]
    y_train = train_df['Target']
    X_test = test_df[feature_cols]
    
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Simplified Search for Speed
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    
    print("Tuning XGBoost...")
    # Using small cv and iter for speed in this demo
    random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=3, cv=2, scoring='f1_weighted', random_state=42, n_jobs=1)
    random_search.fit(X_train, y_train)
    
    best_xgb = random_search.best_estimator_
    print(f"Best XGB Params: {random_search.best_params_}")
    
    # Save Model
    import joblib
    joblib.dump(best_xgb, MODEL_3_FILE)
    print(f"XGBoost model saved to '{MODEL_3_FILE}'.")
    
    probs = best_xgb.predict_proba(X_test)
    return pd.DataFrame(probs, columns=['XGB_Prob_Hold', 'XGB_Prob_Buy', 'XGB_Prob_Sell'])

# --- Main Execution ---

def main():
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"Error: {TRAIN_DATA_FILE} not found. Please run data preparation first.")
        return

    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)
    
    # 1. Train & Predict Model 1
    m1_probs = train_and_predict_model_1(train_df, test_df)
    
    # 2. Train & Predict Model 2
    m2_preds = train_and_predict_nbeats(train_df, test_df)
    
    # 3. Train & Predict Model 3
    m3_probs = train_and_predict_xgboost(train_df, test_df)
    
    # 4. Consolidate
    print("\n=== Creating Ensemble Dataset ===")
    ensemble_df = pd.concat([
        test_df['Target'].reset_index(drop=True),
        m1_probs,
        m2_preds,
        m3_probs
    ], axis=1)
    
    ensemble_df.to_csv(ENSEMBLE_OUTPUT, index=False)
    print(f"Success! Ensemble data saved to '{ENSEMBLE_OUTPUT}'.")
    print(ensemble_df.head())

if __name__ == "__main__":
    main()
