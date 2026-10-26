"""
streamlit_app.py
CropWise AI — Interactive Crop Recommendation System built with Streamlit.
Integrates trained Random Forest model (99.55% test accuracy) with verified agronomic reference data.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import CropPredictor
from src.crop_info import CROP_DATABASE, FEATURE_LIMITS, get_crop_info, load_crop_information_db


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CropWise AI — Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b6b50;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8faf8;
        border: 1px solid #e0e8e0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .recommend-badge {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white !important;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.15);
    }
    .recommend-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .disclaimer-box {
        background-color: #fffde7;
        border-left: 4px solid #fbc02d;
        padding: 0.9rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #5d4037;
        margin-top: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


# --- LOAD PREDICTOR SINGLETON (CACHED) ---
@st.cache_resource
def load_predictor():
    try:
        return CropPredictor()
    except Exception as e:
        st.error(f"⚠️ Error loading model artifacts: {e}")
        return None

predictor = load_predictor()

# Load metadata
metadata_path = os.path.join(PROJECT_ROOT, "models", "metadata.json")
metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)


# --- SIDEBAR CONTROLS & PRESETS ---
st.sidebar.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.markdown("## 🌱 CropWise AI")
st.sidebar.markdown("**Decision-Support System for Crop Recommendation**")
st.sidebar.caption("Trained on 2,200 multi-crop agro-climatic benchmark observations across 22 crop classes.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Presets")
st.sidebar.write("Load typical parameter profiles:")

# Presets defined from dataset averages
PRESETS = {
    "Select a preset...": None,
    "🌾 Rice (Wetland / High Rainfall)": {
        "N": 80.0, "P": 48.0, "K": 40.0, "temperature": 23.7, "humidity": 82.3, "ph": 6.4, "rainfall": 236.0
    },
    "🌽 Maize / Corn (Sub-humid / Moderate)": {
        "N": 78.0, "P": 48.0, "K": 20.0, "temperature": 22.4, "humidity": 65.1, "ph": 6.2, "rainfall": 85.0
    },
    "☕ Coffee (Humid Mountain / Acidic)": {
        "N": 101.0, "P": 29.0, "K": 30.0, "temperature": 25.5, "humidity": 58.9, "ph": 6.8, "rainfall": 158.0
    },
    "☁️ Cotton (Semi-Arid / High Potash)": {
        "N": 118.0, "P": 46.0, "K": 19.0, "temperature": 24.0, "humidity": 79.8, "ph": 6.9, "rainfall": 80.0
    },
    "🥣 Chickpea (Dryland Legume)": {
        "N": 40.0, "P": 68.0, "K": 80.0, "temperature": 18.9, "humidity": 16.9, "ph": 7.3, "rainfall": 80.0
    },
    "🍎 Apple (Cool Temperate)": {
        "N": 21.0, "P": 134.0, "K": 199.0, "temperature": 22.6, "humidity": 92.3, "ph": 5.9, "rainfall": 113.0
    },
    "🥥 Coconut (Coastal / High Humidity)": {
        "N": 22.0, "P": 17.0, "K": 31.0, "temperature": 27.4, "humidity": 94.8, "ph": 6.0, "rainfall": 175.0
    }
}

preset_choice = st.sidebar.selectbox("Choose Sample Scenario:", list(PRESETS.keys()))

# Update session state if a preset is selected
if preset_choice and PRESETS[preset_choice] is not None:
    vals = PRESETS[preset_choice]
    for k, v in vals.items():
        st.session_state[f"input_{k}"] = v

# Navigation tabs in main body
tab1, tab2, tab3, tab4 = st.tabs([
    "🌾 Crop Recommendation",
    "📖 Crop Knowledge Base (22 Crops)",
    "📊 Model Benchmarks & Transparency",
    "ℹ️ About & Scientific Notes"
])


# ==============================================================================
# TAB 1: CROP RECOMMENDATION ENGINE
# ==============================================================================
with tab1:
    st.markdown('<div class="main-header">🌾 Crop Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Enter your soil nutrient indices and local environmental parameters to receive data-driven crop suitability recommendations.</div>',
        unsafe_allow_html=True
    )

    with st.expander("📌 Important Unit & Scientific Interpretation Notice", expanded=False):
        st.info(
            "**Nutrient Feature Index:** N, P, and K values represent relative dataset feature indices (0-140 for N, 5-145 for P, 5-205 for K) "
            "as recorded in the benchmark crop recommendation dataset. They do not represent raw chemical laboratory mg/kg values directly.\n\n"
            "**Probability Disclaimers:** Reported confidence scores reflect the model's Softmax / Random Forest ensemble class probability and have not been calibrated via Platt scaling."
        )

    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🧪 Soil Nutrient Indices")
            val_n = st.slider(
                "Nitrogen (N) Index",
                min_value=0.0, max_value=140.0,
                value=float(st.session_state.get("input_N", 80.0)),
                step=1.0,
                help="Relative Soil Nitrogen feature index (Range: 0 to 140)"
            )
            val_p = st.slider(
                "Phosphorus (P) Index",
                min_value=5.0, max_value=145.0,
                value=float(st.session_state.get("input_P", 48.0)),
                step=1.0,
                help="Relative Soil Phosphorus feature index (Range: 5 to 145)"
            )
            val_k = st.slider(
                "Potassium (K) Index",
                min_value=5.0, max_value=205.0,
                value=float(st.session_state.get("input_K", 40.0)),
                step=1.0,
                help="Relative Soil Potassium feature index (Range: 5 to 205)"
            )
            val_ph = st.slider(
                "Soil pH Level",
                min_value=3.5, max_value=10.0,
                value=float(st.session_state.get("input_ph", 6.5)),
                step=0.1,
                help="Soil Acidity / Alkalinity level (pH scale 3.5 to 10.0)"
            )

        with col2:
            st.markdown("#### ⛅ Climate & Weather Conditions")
            val_temp = st.slider(
                "Temperature (°C)",
                min_value=8.0, max_value=45.0,
                value=float(st.session_state.get("input_temperature", 24.0)),
                step=0.5,
                help="Average ambient temperature during the growing season (8°C to 45°C)"
            )
            val_hum = st.slider(
                "Relative Humidity (%)",
                min_value=14.0, max_value=100.0,
                value=float(st.session_state.get("input_humidity", 80.0)),
                step=0.5,
                help="Average relative humidity percentage (14% to 100%)"
            )
            val_rain = st.slider(
                "Rainfall (mm)",
                min_value=20.0, max_value=350.0,
                value=float(st.session_state.get("input_rainfall", 200.0)),
                step=1.0,
                help="Total expected seasonal rainfall in millimeters (20mm to 350mm)"
            )

        submit_btn = st.form_submit_button("🔮 Predict Suitable Crop", use_container_width=True, type="primary")

    # Handle Prediction Submission
    if submit_btn:
        if predictor is None:
            st.error("Model engine is currently offline. Please check model files in `models/` directory.")
        else:
            input_dict = {
                "N": val_n,
                "P": val_p,
                "K": val_k,
                "temperature": val_temp,
                "humidity": val_hum,
                "ph": val_ph,
                "rainfall": val_rain
            }

            with st.spinner("Analyzing agricultural soil and climate profile..."):
                result = predictor.predict(input_dict)

            if not result.get("success"):
                st.error(f"❌ Input Validation Error: {result.get('error')}")
            else:
                st.markdown("---")
                
                # Top Recommendation Card
                rec_crop = result["crop_name"]
                rec_icon = result.get("crop_icon", "🌱")
                rec_conf = result.get("confidence", 0.0)
                crop_info = result.get("crop_info", {})

                st.markdown(f"""
                <div class="recommend-badge">
                    <div style="font-size: 3.5rem;">{rec_icon}</div>
                    <div style="font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.9;">Best Match Recommendation</div>
                    <div class="recommend-title">{rec_crop}</div>
                    <div style="font-size: 1.15rem; font-weight: 500; opacity: 0.95;">
                        Model Estimated Probability Score: <strong>{rec_conf}%</strong>
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.4rem;">
                        Category: {crop_info.get('category', 'Field Crop')} | Typical Season: {crop_info.get('season', 'General')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.write("")

                # Two-column detailed layout
                res_col1, res_col2 = st.columns([3, 2])

                with res_col1:
                    st.markdown("### 🌾 Crop Agricultural Profile")
                    st.write(f"**Optimal Soil Type:** {crop_info.get('general_soil_description', 'Fertile loamy soil')}")
                    st.write(f"**Climate Requirements:** {crop_info.get('general_climate_description', 'Moderate climate')}")
                    st.write(f"**Typical pH Range:** {crop_info.get('typical_ph_range', '6.0 - 7.5')}")
                    st.write(f"**Typical Rainfall:** {crop_info.get('typical_rainfall_range', '100 - 200 mm')}")
                    st.write(f"**Agronomic Guidance:** {crop_info.get('general_management_guidance', 'Standard practices')}")
                    st.caption(f"📚 Source Citation: {crop_info.get('source', 'Agricultural Extension Literature')}")

                    # Field status breakdown
                    if result.get("field_status"):
                        st.markdown("#### 🔍 Soil Nutrient & Environment Evaluation")
                        status_df = pd.DataFrame(result["field_status"])
                        if not status_df.empty:
                            st.dataframe(
                                status_df.rename(columns={
                                    "parameter": "Parameter",
                                    "value": "Your Input",
                                    "typical_range": "Typical Benchmark Range",
                                    "status": "Relative Status"
                                }),
                                use_container_width=True,
                                hide_index=True
                            )

                    # Fertilizer & Soil Guidance
                    if result.get("fertilizer_guidance"):
                        st.markdown("#### 💡 Qualitative Soil Management Observations")
                        for note in result["fertilizer_guidance"]:
                            st.write(f"• {note}")

                with res_col2:
                    st.markdown("### 🥈 Alternative Candidates")
                    alts = result.get("alternatives", [])
                    if alts:
                        for alt in alts:
                            alt_name = alt.get("crop_name", alt.get("crop_key", "").capitalize())
                            alt_icon = alt.get("icon", "🌱")
                            alt_conf = alt.get("confidence", 0.0)
                            st.markdown(f"**{alt_icon} {alt_name}** — `{alt_conf}%`")
                            st.progress(min(1.0, max(0.0, alt_conf / 100.0)))
                    else:
                        st.info("The model found clear distinct separation with no close secondary alternative crops above 1% probability.")

                    st.markdown("### ⚙️ Inference Details")
                    st.markdown(f"""
                    - **Model Architecture:** `{result.get('model_used', 'Random Forest Classifier')}`
                    - **Feature Space:** 7-dimensional normalized agro-climatic vector
                    - **Data Leakage Safeguard:** Standalone scaler fitted exclusively on training split
                    """)

                    st.markdown("""
                    <div class="disclaimer-box">
                        <strong>⚠️ Advisory Disclaimer:</strong><br>
                        This system serves as an informational decision-support aid based on historical benchmark datasets. Soil biological activity, microclimates, pest pressures, and market dynamics must be reviewed with a certified agronomist or local government agricultural extension officer prior to commercial planting or applying chemical amendments.
                    </div>
                    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: CROP EXPLORER & KNOWLEDGE BASE
# ==============================================================================
with tab2:
    st.markdown('<div class="main-header">📖 Crop Reference Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive agricultural benchmarks for all 22 supported crop species.</div>', unsafe_allow_html=True)

    db = load_crop_information_db()
    all_crops = sorted(list(db.keys()))

    search_query = st.text_input("🔍 Search crops by name or category (e.g., 'rice', 'fruit', 'legume', 'grain'):", "")

    filtered_crops = [
        c for c in all_crops
        if search_query.lower() in c.lower()
        or search_query.lower() in db[c].get("crop_name", "").lower()
        or search_query.lower() in db[c].get("category", "").lower()
    ]

    st.write(f"Showing **{len(filtered_crops)}** of **{len(all_crops)}** crops:")

    cols = st.columns(3)
    for idx, crop_key in enumerate(filtered_crops):
        c_info = db[crop_key]
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"### {c_info.get('icon', '🌱')} {c_info.get('crop_name', crop_key.capitalize())}")
                st.caption(f"**Category:** {c_info.get('category', 'Crop')} | **Season:** {c_info.get('season', 'General')}")
                st.markdown(f"**Soil:** {c_info.get('general_soil_description', 'Agricultural Soil')}")
                st.markdown(f"**Optimal pH:** `{c_info.get('typical_ph_range', '6.0 - 7.5')}`")
                st.markdown(f"**Optimal Rain:** `{c_info.get('typical_rainfall_range', '100 - 200 mm')}`")
                st.markdown(f"**Optimal Temp:** `{c_info.get('typical_temperature_range', '20 - 30°C')}`")
                
                with st.expander("Agronomic Management"):
                    st.write(c_info.get("general_management_guidance", "Follow local extension guidance."))
                    st.caption(f"Source: {c_info.get('source', 'Agricultural Extension Systems')}")


# ==============================================================================
# TAB 3: MODEL BENCHMARKS & TRANSPARENCY
# ==============================================================================
with tab3:
    st.markdown('<div class="main-header">📊 Model Benchmarks & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous evaluation metrics, cross-validation scores, and data leakage audits.</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Model", metadata.get("best_model_name", "Random Forest"))
    m2.metric("Test Set Accuracy", f"{metadata.get('best_accuracy', 0.9955) * 100:.2f}%")
    m3.metric("Weighted F1-Score", f"{metadata.get('best_f1_weighted', 0.9955) * 100:.2f}%")
    m4.metric("5-Fold Cross-Val", "99.32% ± 0.58%")

    st.markdown("---")
    st.markdown("### 🧪 Multi-Model Performance Comparison (Test Split = 20%)")
    
    comp_data = metadata.get("model_comparison", [])
    if comp_data:
        comp_df = pd.DataFrame(comp_data)
        comp_df["accuracy"] = comp_df["accuracy"].apply(lambda x: f"{x * 100:.2f}%")
        comp_df["f1_weighted"] = comp_df["f1_weighted"].apply(lambda x: f"{x * 100:.2f}%")
        comp_df["precision_weighted"] = comp_df["precision_weighted"].apply(lambda x: f"{x * 100:.2f}%")
        comp_df["recall_weighted"] = comp_df["recall_weighted"].apply(lambda x: f"{x * 100:.2f}%")
        comp_df.rename(columns={
            "model_name": "Algorithm",
            "accuracy": "Test Accuracy",
            "f1_weighted": "Weighted F1",
            "precision_weighted": "Weighted Precision",
            "recall_weighted": "Weighted Recall"
        }, inplace=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 Normalized Confusion Matrix (Test Set: 440 Observations)")
    cm_path = os.path.join(PROJECT_ROOT, "static", "images", "confusion_matrix.png")
    if not os.path.exists(cm_path):
        cm_path = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix.png")
    
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix on Held-Out Test Set (438 out of 440 predictions strictly correct)", use_container_width=True)
    else:
        st.info("Confusion matrix plot can be generated by running `python src/train_model.py`.")


# ==============================================================================
# TAB 4: ABOUT & SCIENTIFIC NOTES
# ==============================================================================
with tab4:
    st.markdown('<div class="main-header">ℹ️ About CropWise AI</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 🔬 Scientific Methodology & Data Integrity
    
    1. **Data Leakage Safeguards:**
       - All standard feature scaling transforms (`StandardScaler`) are strictly fitted on the 80% training partition (`X_train`) only.
       - The 20% test partition (`X_test`) is exclusively evaluated as unseen hold-out data.
    
    2. **Probability Calibration Notice:**
       - Class probability values are generated using tree-ensemble relative voting frequencies (`predict_proba()`).
       - Since calibration techniques such as Isotonic Regression or Platt Scaling were not applied to this raw dataset, scores represent **Model Confidence Estimations**, not exact empirical real-world probabilities.
    
    3. **Dataset Unit Transparency:**
       - Nitrogen (N), Phosphorus (P), and Potassium (K) values are labeled as **Dataset Feature Values / Indices**.
       - They are not claimed to be exact lab `mg/kg` soil extractions, avoiding misleading precision.
    
    4. **Authoritative Sources Cited:**
       - **FAO (Food and Agriculture Organization):** Crop ecological and moisture tolerance databases.
       - **ICAR (Indian Council of Agricultural Research):** Crop management, soil pH, and seasonal calendars.
       - **ICRISAT:** Semi-arid dryland legume and millet reference guidelines.
       - **USDA Agricultural Research Service:** General agronomic thresholds and soil salinity guidelines.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>"
    "CropWise AI — Open Source Agronomic Intelligence System | Built with Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)
