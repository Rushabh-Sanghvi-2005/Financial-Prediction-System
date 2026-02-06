import os

# Base Directory (Project Root)
# Assumes this config file is in src/config.py, so we go up one level to src, then one level to root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# File Paths - Data
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train_data.csv")
TEST_DATA_FILE = os.path.join(DATA_DIR, "test_data.csv")
TEST_META_FILE = os.path.join(DATA_DIR, "test_meta.csv")
ENSEMBLE_INPUT_FILE = os.path.join(DATA_DIR, "ensemble_input.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "final_predictions.csv")

# File Paths - Models
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_1_FILE = os.path.join(MODELS_DIR, "model_1_hybrid.keras")
MODEL_2_FILE = os.path.join(MODELS_DIR, "nbeats_model.pth")
MODEL_3_FILE = os.path.join(MODELS_DIR, "xgboost_model.pkl")
ENSEMBLE_MODEL_FILE = os.path.join(MODELS_DIR, "ensemble_model.pkl")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
