# 🌾 CropWise.AI — Smart AI Agriculture Platform

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Verified Accuracy](https://img.shields.io/badge/Verified_Accuracy-99.55%25-10b981?style=for-the-badge)

**CropWise.AI** is a production-quality, machine learning-powered smart agriculture platform that analyzes soil chemistry ($N, P, K$, Soil pH) and climatic factors (Temperature, Humidity, Rainfall) to provide precise crop recommendations, field condition health assessments, and general fertilizer guidance.

---

## 🌟 Key Upgraded Features

- **Empirically Verified ML Accuracy (99.55%)**: Tested on a 20% stratified held-out test set (440 samples). 5-Fold Cross-Validation accuracy is **99.32% ± 0.58%**.
- **Data Leakage Audited**: `StandardScaler` is fitted **strictly on training data** after stratified train-test splitting.
- **Field Condition Health Summary**: Classifies every user field measurement as `Optimal`, `Low`, `High`, `Acidic`, `Alkaline`, or `Normal` relative to the target crop's actual requirement bounds.
- **General Fertilizer & Amendment Guidance**: Compares N-P-K readings against target crop requirement profiles to suggest safe, non-hazardous organic and soil management strategies.
- **Structured Crop Profiles Database**: Complete dataset of 22 supported crops stored in `data/crop_profiles.json`.
- **Interactive Crop Explorer & Requirements Search**: Allows users to search any crop name (e.g. *Coffee, Rice, Mango, Watermelon*) and click **`⚡ Auto-Fill Predictor Form`** to populate field inputs instantly.
- **Multi-Model Comparison**: Evaluates Random Forest, Support Vector Machine (SVM), Decision Tree, K-Nearest Neighbors (KNN), and Logistic Regression side-by-side.
- **SaaS Dark Emerald AI Theme**: Custom responsive UI with subtle glassmorphism, responsive cards, accessibility focus states, and zero heavy dependencies.

---

## 📊 Verified Machine Learning Performance

Evaluated on `data/Crop_recommendation.csv` (2,200 total samples, 22 classes, 80/20 stratified split):

| Algorithm | Test Accuracy | F1-Score (Weighted) | Precision (Weighted) | Recall (Weighted) | 5-Fold Cross Validation | Selection |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **99.55%** | **99.55%** | **99.57%** | **99.55%** | **99.32% ± 0.58%** | **🏆 Selected** |
| **Support Vector Machine (SVM)** | 98.41% | 98.40% | 98.56% | 98.41% | 97.95% ± 1.07% | Evaluated |
| **Decision Tree** | 97.95% | 97.94% | 98.10% | 97.95% | 98.41% ± 0.58% | Evaluated |
| **K-Nearest Neighbors (KNN)** | 97.95% | 97.93% | 98.04% | 97.95% | 96.53% ± 1.21% | Evaluated |
| **Logistic Regression** | 97.27% | 97.25% | 97.40% | 97.27% | 96.82% ± 0.66% | Evaluated |

---

## 📁 Architecture & File Structure

```
crop-recommendation-system/
├── app.py                      # Flask App server & REST endpoints
├── requirements.txt            # Python dependencies
├── Procfile                    # Cloud deploy startup configuration
├── vercel.json                 # Vercel serverless deployment config
├── README.md                   # Complete documentation
│
├── data/
│   ├── Crop_recommendation.csv # Dataset (2,200 samples, 22 classes)
│   └── crop_profiles.json      # Structured profiles & ranges for 22 crops
│
├── models/
│   ├── crop_model.pkl          # Trained Random Forest model
│   ├── scaler.pkl              # Fitted StandardScaler object
│   ├── label_encoder.pkl       # Target label encoder
│   └── metadata.json           # Model benchmark metrics
│
├── notebooks/
│   └── crop_analysis.ipynb     # Jupyter Notebook for EDA & Training
│
├── src/
│   ├── data_preprocessing.py   # Dataset loader & scaling pipeline
│   ├── data_validation.py      # Range validation, health summary & fertilizer advice
│   ├── train_model.py          # Multi-model training pipeline
│   ├── evaluate_model.py       # Metrics & confusion matrix rendering
│   ├── prediction.py           # CropPredictor inference engine
│   ├── crop_info.py            # Crop database loader
│   └── create_notebook.py      # Notebook generator script
│
├── templates/
│   └── index.html              # Responsive Dashboard Template
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom Glassmorphic Emerald CSS
│   ├── js/
│   │   └── script.js           # Client controller, search & AJAX engine
│   └── images/
│       └── confusion_matrix.png # Confusion matrix plot
│
└── tests/
    └── test_prediction.py      # Unittest test suite (5/5 tests passing)
```

---

## ⚙️ Running Locally

1. **Activate Environment & Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Test Suite**:
   ```bash
   python3 -m unittest discover -s tests
   ```

3. **Launch Flask Server**:
   ```bash
   PORT=5005 python3 app.py
   ```
   Open **[http://localhost:5005](http://localhost:5005)** in your web browser.

---

## ⚠️ Disclaimer

CropWise.AI is a machine learning-based agricultural recommendation platform built for educational and decision-support purposes. Predictions are derived statistically from training distributions and do not replace professional in-field soil testing or local agricultural extension services.
