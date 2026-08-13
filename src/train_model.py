"""
train_model.py
Training pipeline for Crop Recommendation System.
Trains 6 algorithms, evaluates them, selects the best model, and exports model artifacts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from src.data_preprocessing import load_dataset, prepare_data, save_scaler
from src.evaluate_model import evaluate_classifier, generate_confusion_matrix_plot

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Crop_recommendation.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
STATIC_IMG_DIR = os.path.join(PROJECT_ROOT, "static", "images")


def get_candidate_models(random_state: int = 42) -> dict:
    """Return dictionary of classification algorithms."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Support Vector Machine (SVM)": SVC(probability=True, kernel='rbf', C=1.0, random_state=random_state),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state, eval_metric='mlogloss')
    return models


def train_and_select_best_model(data_path: str = DATA_PATH, verbose: bool = True):
    """
    Main training execution function.
    """
    if verbose:
        print("==================================================")
        print("🌾 CROP RECOMMENDATION SYSTEM — TRAINING PIPELINE")
        print("==================================================")
        print(f"Loading dataset from: {data_path}")
        
    df = load_dataset(data_path)
    
    # Prepare data & scaler
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = prepare_data(df)
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    save_scaler(scaler, scaler_path)
    if verbose:
        print(f"✅ Scaler saved to {scaler_path}")
        
    # Fit LabelEncoder for XGBoost if needed, or maintain target strings
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    models = get_candidate_models()
    results = []
    trained_models = {}
    
    unique_labels = sorted(df['label'].unique().tolist())
    
    for name, model in models.items():
        if verbose:
            print(f"\n🔄 Training model: {name}...")
            
        if name == "XGBoost":
            model.fit(X_train_scaled, y_train_encoded)
            # Evaluate using string labels for consistency
            y_pred_encoded = model.predict(X_test_scaled)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
        metrics = evaluate_classifier(model, X_test_scaled, y_test, name)
        results.append(metrics)
        trained_models[name] = (model, y_pred)
        
        if verbose:
            print(f"   Accuracy: {metrics['accuracy']*100:.2f}% | F1-Weighted: {metrics['f1_weighted']*100:.2f}%")
            
    # Convert results to DataFrame for comparison
    results_df = pd.DataFrame(results).sort_values(by="f1_weighted", ascending=False)
    
    if verbose:
        print("\n==================================================")
        print("📊 MODEL COMPARISON RESULTS:")
        print("==================================================")
        for _, row in results_df.iterrows():
            print(f"• {row['model_name']:<30} | Accuracy: {row['accuracy']*100:.2f}% | F1: {row['f1_weighted']*100:.2f}%")
            
    # Select best model
    best_row = results_df.iloc[0]
    best_model_name = best_row['model_name']
    best_model, best_y_pred = trained_models[best_model_name]
    
    if verbose:
        print("\n==================================================")
        print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
        print(f"   Accuracy: {best_row['accuracy']*100:.2f}%")
        print(f"   F1-Score: {best_row['f1_weighted']*100:.2f}%")
        print("==================================================")
        
    # Save best model
    best_model_path = os.path.join(MODELS_DIR, "crop_model.pkl")
    joblib.dump(best_model, best_model_path)
    if verbose:
        print(f"✅ Best model saved to {best_model_path}")
        
    # Save label encoder
    label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    joblib.dump(label_encoder, label_encoder_path)
    
    # Generate confusion matrix plot
    cm_path_out = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    cm_path_static = os.path.join(STATIC_IMG_DIR, "confusion_matrix.png")
    generate_confusion_matrix_plot(y_test, best_y_pred, unique_labels, cm_path_out, title=f"Confusion Matrix — {best_model_name}")
    generate_confusion_matrix_plot(y_test, best_y_pred, unique_labels, cm_path_static, title=f"Confusion Matrix — {best_model_name}")
    if verbose:
        print(f"✅ Confusion matrix saved to {cm_path_static}")
        
    # Save metadata JSON
    metadata = {
        "best_model_name": best_model_name,
        "best_accuracy": best_row['accuracy'],
        "best_f1_weighted": best_row['f1_weighted'],
        "features": feature_names,
        "crop_classes": unique_labels,
        "model_comparison": [
            {
                "model_name": r["model_name"],
                "accuracy": r["accuracy"],
                "f1_weighted": r["f1_weighted"],
                "precision_weighted": r["precision_weighted"],
                "recall_weighted": r["recall_weighted"]
            }
            for r in results
        ]
    }
    
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    if verbose:
        print(f"✅ Model metadata saved to {metadata_path}\n")
        
    return best_model, scaler, metadata


if __name__ == "__main__":
    train_and_select_best_model()
