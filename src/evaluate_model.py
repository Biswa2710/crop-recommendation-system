"""
evaluate_model.py
Evaluates classification algorithms and produces performance reports & visualization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a fitted model on test set.
    Returns dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "y_pred": y_pred
    }


def generate_confusion_matrix_plot(y_test, y_pred, labels, output_path: str, title: str = "Confusion Matrix"):
    """Generate and save confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=labels, yticklabels=labels,
                cbar=True)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Predicted Crop', fontsize=12)
    plt.ylabel('Actual Crop', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
