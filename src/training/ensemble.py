import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

# Files
INPUT_FILE = config.ENSEMBLE_INPUT_FILE
MODEL_FILE = config.ENSEMBLE_MODEL_FILE
PREDICTIONS_FILE = config.PREDICTIONS_FILE

def train_ensemble():
    print("Loading Ensemble Input Data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run train_base_models.py first.")
        return

    # Features and Target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Split into Train/Test for the Meta-Learner
    # Note: This 'Train' is effectively a validation set from the perspective of the base models.
    # But for the meta-learner, we need to split it again to validate the meta-learner itself.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\n--- Training LightGBM Meta-Learner ---")
    clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Evaluation
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test, y_pred, target_names=['Hold', 'Buy', 'Sell']))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    
    # Save Model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\nEnsemble model saved to '{MODEL_FILE}'.")
    
    # Generate Final Predictions for usage in Dashboard (using the full loaded dataset)
    # in a real scenario, we might want to use the original test dates, but here we just have the ensemble inputs
    print("Generating final predictions for dashboard...")
    final_preds = clf.predict(X)
    df['Final_Prediction'] = final_preds
    
    # Save for Dashboard
    df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Final predictions saved to '{PREDICTIONS_FILE}'.")

if __name__ == "__main__":
    train_ensemble()
