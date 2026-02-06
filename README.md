# 📈 Advanced Financial Prediction System (DL)

A comprehensive Deep Learning pipeline for stock market prediction, utilizing a hybrid architecture of CNN, GRU, and Transformers (Attention), combined with N-BEATS and XGBoost in an ensemble meta-learner.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Key Features

*   **Hybrid Deep Learning Model**: Combines Conv1D (spatial features), GRU (temporal sequences), and Multi-Head Attention (long-range dependencies).
*   **Ensemble Learning**: Stacks predictions from the Hybrid model, N-BEATS, and XGBoost using a LightGBM meta-learner for superior accuracy.
*   **Self-Healing Architecture**: The system automatically generates missing datasets (`train_data.csv`) and retrains models if they are not found.
*   **Interactive Dashboard**: A full-featured Streamlit web app for visualizing stock trends, technical indicators, and model performance metrics.
*   **Zero-Config Deployment**: Runs on any Windows PC without requiring pre-installed Python, thanks to the embedded portable runtime launcher.

## 🛠️ Tech Stack

*   **Core**: Python 3.11
*   **Deep Learning**: TensorFlow/Keras, Darts (N-BEATS)
*   **Machine Learning**: XGBoost, LightGBM, Scikit-Learn
*   **Data Processing**: Pandas, NumPy, TA-Lib (Technical Analysis), YFinance
*   **Visualization**: Streamlit, Matplotlib, Seaborn

## 📦 Installation & Usage

**The Easiest Way:**
1.  **Download `start.bat`** from this repository (click the file above -> Download raw file, or Download ZIP and extract it).
2.  Place `start.bat` in an empty folder on your PC.
3.  **Double-click `start.bat`**.
    *   It will automatically download the rest of this repository.
    *   It will set up a portable Python environment (if needed).
    *   It will train all models and launch the dashboard.

**Manual Setup (For Developers):**
```bash
# 1. Clone the repository
git clone https://github.com/Rushabh-Sanghvi-2005/Financial-Prediction-System.git
cd Financial-Prediction-System

# 2. Run the bootstrapper
python bootstrap.py
```

## 📊 Project Structure

```
├── src/
│   ├── dashboard/      # Streamlit Web App
│   ├── preparation/    # Data fetching (YFinance) & Feature Engineering
│   ├── training/       # Model definitions (Hybrid, N-BEATS, Ensemble)
│   └── config.py       # Central configuration
├── bootstrap.py        # Orchestrator (Auto-install & Run)
└── start.bat           # Portable Launcher Entry Point
```

## 📈 Results

*   **Hybrid Model**: Captures complex market volatility using Attention.
*   **Ensemble**: Improves generalization by combining diverse model architectures.
*   **Backtesting**: Robust evaluation metrics (RMSE, MAE, R2 Coverage).

---
*Created for Sem 6 Project (Deep Learning).*
