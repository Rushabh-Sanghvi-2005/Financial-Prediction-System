import os
import sys
import subprocess
import shutil

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# EMBEDDED REQUIREMENTS
REQUIRED_PACKAGES = [
    "pandas", "numpy", "streamlit", "matplotlib", "seaborn",
    "tensorflow", "xgboost", "lightgbm", "darts", "ta",
    "yfinance", "scikit-learn", "joblib", "openpyxl"
]

def log(msg):
    print(f"[*] {msg}")

def run_command(cmd, desc):
    log(f"Running: {desc}...")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except:
        log(f"Failed: {desc}")
        return False

def check_dependencies():
    log("Verifying Libraries...")
    # Generate temporary requirements file
    req_file = os.path.join(PROJECT_ROOT, "temp_requirements.txt")
    with open(req_file, "w") as f:
        f.write("\n".join(REQUIRED_PACKAGES))
    
    cmd = f'"{sys.executable}" -m pip install -r "{req_file}" --disable-pip-version-check'
    run_command(cmd, "Dependency Installation")
    
    if os.path.exists(req_file):
        os.remove(req_file)

def pipeline_manager():
    # 1. Check Data
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(DATA_DIR, "train_data.csv")):
        log("Generating Data...")
        run_command(f'"{sys.executable}" -m src.preparation.prepare_data', "Data Prep")

    # 2. Check Models
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(MODELS_DIR, "model_1_hybrid.keras")):
        log("Training Models...")
        run_command(f'"{sys.executable}" -m src.training.train_base_models', "Model Train")

    # 3. Check Ensemble
    if not os.path.exists(os.path.join(MODELS_DIR, "ensemble_model.pkl")):
        log("Training Ensemble...")
        run_command(f'"{sys.executable}" -m src.training.ensemble', "Ensemble Train")

    # 4. Predictions
    if not os.path.exists(os.path.join(DATA_DIR, "final_predictions.csv")):
        log("Generating Predictions...")
        run_command(f'"{sys.executable}" -m src.training.ensemble', "Predictions")

def launch_dashboard():
    log("Launching Dashboard...")
    app_path = os.path.join(SRC_DIR, "dashboard", "app.py")
    cmd = f'"{sys.executable}" -m streamlit run "{app_path}"'
    os.system(cmd)

def main():
    print("=== FINANCIAL PREDICTION SYSTEM ===")
    check_dependencies()
    pipeline_manager()
    launch_dashboard()

if __name__ == "__main__":
    main()
