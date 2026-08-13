# 🌾 CropWise AI — Crop Recommendation System Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

An end-to-end, production-ready Machine Learning web application designed to empower farmers and agricultural specialists by recommending the optimal crop based on soil nutrient composition (N, P, K) and micro-climatic readings (Temperature, Humidity, Soil pH, Rainfall).

---

## 🚀 Key Features

- **Multi-Model Machine Learning Pipeline**: Trains and benchmarks 5 classification algorithms (Random Forest, SVM, Decision Tree, KNN, Logistic Regression).
- **High Accuracy Prediction**: Achieves **99.55% accuracy** using Random Forest Classifier.
- **Statistical Confidence Score**: Provides model proximity score and secondary alternative crop recommendations.
- **Strict Input Validation**: Protects against invalid numbers, impossible ranges, missing fields, and negative nutrient values with user-friendly error alerts.
- **Separated Agricultural Guidance**: Clearly distinguishes statistical ML output from practical agronomic field advice (soil type, season, fertilizer profile, field tips).
- **Interactive Preset Samples**: Allows 1-click testing with pre-filled sample inputs for Rice, Maize, Coffee, Cotton, Apple, and Watermelon.
- **Modern Responsive Agriculture UI**: Premium glassmorphic interface built with Vanilla CSS, micro-interactions, dark emerald theme, and mobile responsiveness.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6+), FontAwesome |
| **Backend** | Python 3.9+, Flask Web Framework |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib, Matplotlib, Seaborn, XGBoost (Optional) |
| **Testing** | Built-in Unittest Framework |

---

## 📊 Dataset Details

The system uses the **Crop Recommendation Dataset** (`Crop_recommendation.csv`) containing 2,200 records across 22 unique crop categories (100 samples per crop):

- **Nutrient Features**: Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$) in mg/kg.
- **Climatic Features**: Ambient Temperature (°C), Relative Humidity (%), Soil pH, Annual Rainfall (mm).
- **Target Label**: 22 crops including *Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee*.

---

## 🤖 ML Pipeline & Model Evaluation

The training pipeline standardizes inputs using `StandardScaler` fitted strictly on training data (80/20 train-test split, `random_state=42`, stratified).

### Candidate Models Benchmark Comparison:

| Algorithm | Test Accuracy | F1-Score (Weighted) | Precision (Weighted) | Recall (Weighted) | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **99.55%** | **99.55%** | **99.58%** | **99.55%** | **🏆 Selected** |
| **Support Vector Machine (SVM)** | 98.41% | 98.40% | 98.49% | 98.41% | Evaluated |
| **Decision Tree** | 97.95% | 97.94% | 98.01% | 97.95% | Evaluated |
| **K-Nearest Neighbors (KNN)** | 97.95% | 97.93% | 98.05% | 97.95% | Evaluated |
| **Logistic Regression** | 97.27% | 97.25% | 97.43% | 97.27% | Evaluated |

---

## 📁 Project Architecture

```
crop-recommendation-system/
│
├── app.py                      # Flask Server Application & REST API routes
├── requirements.txt            # Python Dependencies
├── README.md                   # Complete Documentation
│
├── data/
│   └── Crop_recommendation.csv # Dataset (2,200 samples, 22 classes)
│
├── models/                     # Saved Model Artifacts
│   ├── crop_model.pkl          # Trained Random Forest Model
│   ├── scaler.pkl              # Fitted StandardScaler object
│   ├── label_encoder.pkl       # Target label encoder object
│   └── metadata.json           # Model benchmark metrics & parameters
│
├── notebooks/
│   └── crop_analysis.ipynb     # Interactive Jupyter Notebook EDA & Training
│
├── src/
│   ├── data_preprocessing.py   # Dataset loading, validation & scaling
│   ├── train_model.py          # Multi-model training execution pipeline
│   ├── evaluate_model.py       # Metrics computation & confusion matrix rendering
│   ├── prediction.py           # CropPredictor inference engine & range validator
│   ├── crop_info.py            # Agronomic guidance database & feature bounds
│   └── create_notebook.py      # Notebook generator script
│
├── templates/
│   └── index.html              # Modern Responsive Frontend UI Template
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom Emerald Glassmorphism Stylesheet
│   ├── js/
│   │   └── script.js           # Client-side validation, Presets & AJAX
│   └── images/
│       └── confusion_matrix.png # Generated confusion matrix heatmap
│
├── outputs/
│   └── confusion_matrix.png    # Evaluation Confusion Matrix
│
└── tests/
    └── test_prediction.py      # Automated Unittest Suite
```

---

## ⚙️ Installation & Running Locally

### 1. Clone & Set Up Project
```bash
git clone https://github.com/your-username/crop-recommendation-system.git
cd crop-recommendation-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training Pipeline (Optional)
The pre-trained model artifacts are already included under `models/`. To re-train from scratch:
```bash
python src/train_model.py
```

### 4. Run Automated Test Suite
```bash
python -m unittest discover -s tests
```

### 5. Launch Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`** (or `http://localhost:5005`).

---

## 🔌 API Endpoints

- `GET /` — Renders main web application.
- `POST /predict` — Accepts form data or JSON (`{N, P, K, temperature, humidity, ph, rainfall}`) and returns predicted crop, confidence, input summary, and agricultural advice.
- `GET /api/crops` — Returns list of all 22 supported crops with icons and seasons.
- `GET /api/info/<crop_name>` — Returns agricultural guidance details for a crop.
- `GET /api/model-info` — Returns model benchmark metrics and comparison data.

---

## 🔮 Future Enhancements

1. **IoT Sensor Integration**: Live telemetry integration with field soil moisture and NPK sensors.
2. **Real-time Weather API**: Automatic fetch of localized temperature, humidity, and forecast rainfall using OpenWeather API.
3. **Market Price & Yield Analytics**: Crop price trends and profitability analysis.
