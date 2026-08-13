"""
create_notebook.py
Generates notebooks/crop_analysis.ipynb with complete EDA, model comparison, and exports.
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(PROJECT_ROOT, "notebooks", "crop_analysis.ipynb")


def build_notebook():
    cells = []

    def add_md(source_str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_str.split("\n")]
        })

    def add_code(source_str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_str.split("\n")]
        })

    add_md("""# 🌾 Crop Recommendation System — Machine Learning Analysis & Model Selection

This notebook demonstrates the end-to-end Machine Learning pipeline for predicting the optimal crop based on soil nutrients and climatic parameters.

### Features Analyzed:
1. **Nitrogen (N)**: Soil nitrogen ratio
2. **Phosphorus (P)**: Soil phosphorus ratio
3. **Potassium (K)**: Soil potassium ratio
4. **Temperature**: Ambient air temperature (°C)
5. **Humidity**: Relative air humidity (%)
6. **pH**: Soil pH (acidic / alkaline measure)
7. **Rainfall**: Seasonal rainfall (mm)

---""")

    add_code("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
plt.style.use('ggplot')""")

    add_md("## 1. Load Dataset & Integrity Check")

    add_code("""df = pd.read_csv('../data/Crop_recommendation.csv')
print("Dataset Shape:", df.shape)
print("Missing Values:\\n", df.isnull().sum())
df.head()""")

    add_md("## 2. Exploratory Data Analysis (EDA)")

    add_code("""plt.figure(figsize=(10, 5))
df['label'].value_counts().plot(kind='bar', color='emerald')
plt.title('Crop Target Class Distribution (22 Crops)')
plt.xlabel('Crop')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()""")

    add_code("""# Correlation Heatmap
numeric_df = df.drop(columns=['label'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()""")

    add_md("## 3. Train-Test Split & Preprocessing")

    add_code("""X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")""")

    add_md("## 4. Model Training & Evaluation")

    add_code("""models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, m in models.items():
    m.fit(X_train_scaled, y_train)
    preds = m.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name:<25} Accuracy: {acc*100:.2f}%")""")

    add_md("## 5. Best Model & Confusion Matrix")

    add_code("""best_model_name = max(results, key=results.get)
print(f"Selected Best Model: {best_model_name} ({results[best_model_name]*100:.2f}% Accuracy)")

best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Confusion Matrix — {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()""")

    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
    with open(NB_PATH, "w") as f:
        json.dump(nb_dict, f, indent=2)
    print(f"Created notebook at {NB_PATH}")


if __name__ == "__main__":
    build_notebook()
