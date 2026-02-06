import os
import sys
import subprocess
import shutil
import warnings

# --- WARNING SUPPRESSION (Clean Output) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # Hide TensorFlow Info/Warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'      # Hide OneDNN Custom Ops Msg
warnings.filterwarnings("ignore")              # Hide Python Warnings


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
        # cmd should be a list for shell=False (more robust)
        if isinstance(cmd, str):
            # Fallback for string commands, but discouraged
            subprocess.check_call(cmd, shell=True)
        else:
            subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        log(f"Failed: {desc} (Error code: {e.returncode})")
        return False
    except Exception as e:
        log(f"Failed: {desc} (Exception: {e})")
        return False

def check_dependencies():
    log("Verifying Libraries...")
    req_file = os.path.join(PROJECT_ROOT, "temp_requirements.txt")
    with open(req_file, "w") as f:
        f.write("\n".join(REQUIRED_PACKAGES))
    
    # Use list for robustness against spaces in paths
    cmd = [sys.executable, "-m", "pip", "install", "-r", req_file, "--disable-pip-version-check"]
    run_command(cmd, "Dependency Installation")
    
    if os.path.exists(req_file):
        os.remove(req_file)

def launch_dashboard():
    log("Launching Dashboard...")
    app_path = os.path.join(SRC_DIR, "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log(f"Dashboard crashed with error code {e.returncode}")
    except KeyboardInterrupt:
        log("Dashboard stopped by user.")

def main():
    print("=== FINANCIAL PREDICTION SYSTEM ===")
    check_dependencies()
    # pipeline_manager() removed - Handled by Streamlit App UI
    launch_dashboard()

if __name__ == "__main__":
    main()
