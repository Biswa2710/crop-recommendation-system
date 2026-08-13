"""
data_preprocessing.py
Data loading, validation, and preprocessing pipeline for Crop Recommendation System.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COL = 'label'


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and validate the crop recommendation dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at path: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
        
    # Drop NaNs if any
    df = df.dropna().copy()
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")
        
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and target, perform train-test split, and fit scaler on training data only.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, FEATURE_COLS


def save_scaler(scaler, save_path: str):
    """Save scaler to disk using joblib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)


def load_scaler(load_path: str):
    """Load scaler from disk using joblib."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Scaler file not found at path: {load_path}")
    return joblib.load(load_path)
