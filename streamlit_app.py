"""
streamlit_app.py
CropWise.AI — Streamlit version matching the original Flask app theme exactly.
Green gradient hero, premium card layout, stat boxes, preset buttons, result cards.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import CropPredictor
from src.crop_info import FEATURE_LIMITS, get_crop_info, load_crop_information_db

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropWise.AI — ML Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject the full CropWise theme CSS ───────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* ── CSS Variables ── */
:root {
    --primary-900: #064e3b;
    --primary-800: #065f46;
    --primary-700: #047857;
    --primary-600: #059669;
    --primary-500: #10b981;
    --primary-400: #34d399;
    --primary-100: #d1fae5;
    --primary-50:  #ecfdf5;
    --accent-amber: #f59e0b;
    --accent-amber-light: #fef3c7;
    --accent-teal: #14b8a6;
    --dark-900: #0f172a;
    --dark-800: #1e293b;
    --dark-700: #334155;
    --dark-600: #475569;
    --gray-400: #94a3b8;
    --gray-300: #cbd5e1;
    --gray-200: #e2e8f0;
    --gray-100: #f1f5f9;
    --gray-50:  #f8fafc;
    --white: #ffffff;
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --shadow-sm: 0 2px 4px rgba(0,0,0,0.04);
    --shadow-md: 0 8px 24px rgba(6,78,59,0.08);
    --shadow-lg: 0 16px 36px rgba(6,78,59,0.12);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background-color: var(--gray-50) !important;
    color: var(--dark-800) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }

/* ── NAVBAR ── */
.cw-navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(226,232,240,0.8);
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 68px;
    margin: -24px -24px 0 -24px;
}
.cw-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--dark-900);
    display: flex;
    align-items: center;
    gap: 8px;
}
.cw-brand-accent { color: var(--primary-600); }
.cw-nav-links {
    display: flex;
    gap: 28px;
    list-style: none;
}
.cw-nav-links a {
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--dark-700);
    text-decoration: none;
}
.cw-nav-links a:hover { color: var(--primary-600); }

/* ── HERO SECTION ── */
.cw-hero {
    position: relative;
    padding: 80px 0 60px;
    background: radial-gradient(circle at 50% 10%, var(--primary-50) 0%, var(--gray-50) 70%);
    text-align: center;
    overflow: hidden;
    border-bottom: 1px solid var(--gray-200);
    margin-bottom: 40px;
}
.cw-hero::before {
    content: '';
    position: absolute;
    width: 400px; height: 400px;
    background: var(--primary-400);
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.25;
    top: -100px; right: -100px;
}
.cw-hero::after {
    content: '';
    position: absolute;
    width: 300px; height: 300px;
    background: var(--accent-amber);
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.2;
    bottom: -50px; left: -50px;
}
.cw-hero-inner { position: relative; z-index: 1; }

.cw-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    background: var(--white);
    border: 1px solid var(--primary-100);
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--primary-700);
    box-shadow: var(--shadow-sm);
    margin-bottom: 24px;
}
.pulse-dot {
    width: 8px; height: 8px;
    background: var(--primary-500);
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }
    70%  { box-shadow: 0 0 0 10px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}

.cw-hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: var(--dark-900);
    letter-spacing: -1px;
    margin-bottom: 16px;
    line-height: 1.2;
}
.gradient-text {
    background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-teal) 50%, var(--primary-400) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.cw-hero-subtitle {
    font-size: 1.1rem;
    color: var(--dark-600);
    max-width: 720px;
    margin: 0 auto 36px;
}

/* ── HERO STATS ── */
.cw-stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    background: var(--white);
    border-radius: var(--radius-lg);
    padding: 24px 32px;
    border: 1px solid var(--gray-200);
    box-shadow: var(--shadow-lg);
    max-width: 900px;
    margin: 0 auto;
}
.cw-stat-card {
    display: flex;
    align-items: center;
    gap: 14px;
}
.cw-stat-icon {
    width: 50px; height: 50px;
    border-radius: 14px;
    background: var(--primary-50);
    color: var(--primary-600);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
}
.cw-stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--dark-900);
    display: block;
    line-height: 1.1;
}
.cw-stat-label {
    font-size: 0.8rem;
    color: var(--gray-400);
    font-weight: 600;
}

/* ── SECTION LABELS ── */
.section-sub {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--primary-600);
    background: var(--primary-50);
    padding: 5px 14px;
    border-radius: 50px;
    margin-bottom: 10px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--dark-900);
    margin-bottom: 8px;
}
.section-desc {
    font-size: 1rem;
    color: var(--dark-600);
    max-width: 640px;
    margin: 0 auto 32px;
}

/* ── CARDS ── */
.cw-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-lg);
    padding: 32px;
    box-shadow: var(--shadow-md);
    margin-bottom: 24px;
}

/* ── PRESET BUTTONS ── */
.preset-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    background: var(--white);
    padding: 14px 20px;
    border-radius: var(--radius-md);
    border: 1px solid var(--gray-200);
    margin-bottom: 20px;
}
.preset-label {
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--dark-700);
    white-space: nowrap;
}

/* ── FORM INPUTS ── */
.stNumberInput > div > div > input,
.stSlider, .stTextInput > div > div > input {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--gray-300) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stNumberInput > div > div > input:focus {
    border-color: var(--primary-500) !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.12) !important;
}

/* ── SUBMIT BUTTON ── */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 14px 36px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.30) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(5,150,105,0.40) !important;
}

/* ── RESULT CARD ── */
.result-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-lg);
    padding: 32px;
    box-shadow: var(--shadow-lg);
    margin-top: 28px;
}
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--gray-200);
}
.badge-ml {
    background: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-700) 100%);
    color: white;
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-model {
    background: var(--gray-100);
    color: var(--dark-700);
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* ── CROP HERO BANNER ── */
.crop-banner {
    display: flex;
    align-items: center;
    gap: 24px;
    background: linear-gradient(135deg, var(--primary-50) 0%, #f0fdf4 100%);
    border: 1px solid var(--primary-100);
    border-radius: var(--radius-md);
    padding: 28px;
    margin-bottom: 24px;
}
.crop-icon-big { font-size: 4rem; line-height: 1; }
.hero-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--primary-600);
    display: block;
    margin-bottom: 4px;
}
.crop-hero-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--dark-900);
    margin: 0 0 8px;
}
.crop-explanation {
    font-size: 0.92rem;
    color: var(--dark-600);
    margin: 0;
}

/* ── CONFIDENCE BOX ── */
.conf-box {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-md);
    padding: 20px;
    text-align: center;
    min-width: 180px;
}
.conf-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--gray-400);
    display: block;
    margin-bottom: 8px;
}
.conf-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--primary-600);
}
.conf-bar-wrap {
    height: 8px;
    background: var(--gray-200);
    border-radius: 50px;
    margin: 10px 0 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary-500), var(--primary-700));
    border-radius: 50px;
    transition: width 1s ease;
}
.conf-disclaimer {
    font-size: 0.75rem;
    color: var(--gray-400);
    line-height: 1.4;
}

/* ── ALT CHIP ── */
.alt-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--gray-50);
    border: 1px solid var(--gray-200);
    border-radius: 50px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 0.9rem;
    font-weight: 600;
}
.alt-conf {
    background: var(--primary-50);
    color: var(--primary-700);
    padding: 2px 8px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* ── HEALTH GRID ── */
.health-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-top: 12px;
}
.health-card {
    background: var(--gray-50);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
}
.h-feature {
    font-weight: 700;
    font-size: 0.88rem;
    color: var(--dark-800);
    display: block;
    margin-bottom: 4px;
}
.h-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--primary-600);
    display: block;
}
.h-range {
    font-size: 0.75rem;
    color: var(--gray-400);
    margin-top: 4px;
}

/* ── AGRI GUIDANCE CARD ── */
.agri-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-md);
    padding: 24px;
    margin-top: 20px;
}
.agri-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}
.agri-header h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--dark-900);
    margin: 0;
}
.guidance-tag {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--primary-700);
    background: var(--primary-50);
    padding: 3px 12px;
    border-radius: 50px;
}
.agri-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}
.agri-item {
    background: var(--gray-50);
    border-radius: var(--radius-sm);
    padding: 14px;
    border: 1px solid var(--gray-100);
}
.agri-item-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--primary-600);
    margin-bottom: 6px;
}
.agri-item p {
    font-size: 0.92rem;
    color: var(--dark-700);
    margin: 0;
    line-height: 1.5;
}

/* ── FEATURE BOXES ── */
.feat-box {
    background: var(--gray-50);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-md);
    padding: 28px 22px;
    transition: all 0.3s ease;
}
.feat-box:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-md);
    border-color: var(--primary-400);
}
.feat-icon {
    width: 56px; height: 56px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 16px;
}
.icon-emerald { background: var(--primary-100); color: var(--primary-700); }
.icon-teal { background: #ccfbf1; color: var(--accent-teal); }
.icon-amber { background: var(--accent-amber-light); color: var(--accent-amber); }
.feat-box h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--dark-900);
    margin-bottom: 8px;
}
.feat-box p { font-size: 0.92rem; color: var(--dark-600); margin: 0; }

/* ── STEP CARDS ── */
.step-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-md);
    padding: 32px 24px;
    text-align: center;
    position: relative;
}
.step-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--primary-100);
    position: absolute;
    top: 16px; right: 20px;
    line-height: 1;
}
.step-icon {
    font-size: 2rem;
    margin-bottom: 14px;
    color: var(--primary-600);
}
.step-card h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--dark-900);
    margin-bottom: 8px;
}
.step-card p { font-size: 0.9rem; color: var(--dark-600); margin: 0; }

/* ── METRIC BOXES ── */
.metric-box {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-md);
    padding: 20px;
    text-align: center;
}
.m-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--primary-600);
    display: block;
}
.m-label {
    font-size: 0.8rem;
    color: var(--gray-400);
    font-weight: 600;
    margin-top: 4px;
    display: block;
}

/* ── MODEL TABLE ── */
.model-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.model-table th {
    background: var(--gray-50);
    font-weight: 700;
    padding: 10px 14px;
    text-align: left;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--dark-600);
    border-bottom: 2px solid var(--gray-200);
}
.model-table td {
    padding: 12px 14px;
    border-bottom: 1px solid var(--gray-100);
    color: var(--dark-700);
}
.model-table tr:last-child td { border-bottom: none; }
.badge-success {
    background: var(--primary-100);
    color: var(--primary-700);
    padding: 3px 10px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 700;
}

/* ── CROP CARDS GRID ── */
.crop-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-md);
    padding: 22px 18px;
    transition: all 0.3s ease;
    height: 100%;
}
.crop-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-md);
    border-color: var(--primary-400);
}
.crop-card-icon { font-size: 2.4rem; margin-bottom: 10px; display: block; }
.crop-card-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--dark-900);
    margin-bottom: 4px;
}
.crop-card-cat {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--primary-600);
    background: var(--primary-50);
    padding: 2px 10px;
    border-radius: 50px;
    display: inline-block;
    margin-bottom: 10px;
}
.crop-card p { font-size: 0.85rem; color: var(--dark-600); margin: 4px 0; }

/* ── DISCLAIMER ── */
.agri-disclaimer {
    background: var(--accent-amber-light);
    border-left: 4px solid var(--accent-amber);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #78350f;
    margin-top: 16px;
}

/* ── FOOTER ── */
.cw-footer {
    background: var(--dark-800);
    color: #94a3b8;
    padding: 40px 0 20px;
    margin-top: 60px;
    text-align: center;
    border-top: 1px solid var(--dark-700);
}
.cw-footer-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--white);
    margin-bottom: 6px;
}
.cw-footer-brand span { color: var(--primary-400); }
.cw-footer p { font-size: 0.88rem; max-width: 700px; margin: 0 auto 20px; }
.tech-tags { display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.tag {
    background: var(--dark-700);
    color: var(--gray-300);
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
}
.fert-item {
    background: var(--gray-50);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    margin-bottom: 10px;
}
.fert-nutrient { font-weight: 700; font-size: 0.92rem; color: var(--dark-800); margin-bottom: 6px; }
.fert-advice { font-size: 0.88rem; color: var(--dark-600); margin: 0; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--white);
    border-radius: var(--radius-md);
    padding: 6px;
    border: 1px solid var(--gray-200);
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: var(--dark-700) !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-700) 100%) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    try:
        return CropPredictor()
    except Exception as e:
        return None

predictor = load_predictor()

# Load metadata
meta_path = os.path.join(PROJECT_ROOT, "models", "metadata.json")
metadata = {}
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

db = load_crop_information_db()

# ── NAVBAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cw-navbar">
  <div class="cw-brand">🌱 CropWise<span class="cw-brand-accent">.AI</span></div>
  <ul class="cw-nav-links">
    <li><a href="#predictor">Predictor</a></li>
    <li><a href="#explorer">Crop Explorer</a></li>
    <li><a href="#ml-performance">ML Performance</a></li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cw-hero">
  <div class="cw-hero-inner">
    <div class="cw-hero-badge">
      <span class="pulse-dot"></span>
      <span>Machine Learning Decision Support</span>
    </div>
    <h1 class="cw-hero-title">
      Intelligent Crop Recommendations <br>
      Powered by <span class="gradient-text">Machine Learning</span>
    </h1>
    <p class="cw-hero-subtitle">
      Analyze soil and environmental conditions to identify crops best suited for your field.
      Receive ML predictions, parameter summaries, and agricultural reference guidance.
    </p>
    <div class="cw-stats-grid">
      <div class="cw-stat-card">
        <div class="cw-stat-icon">🌾</div>
        <div>
          <span class="cw-stat-value">22</span>
          <span class="cw-stat-label">Supported Crops</span>
        </div>
      </div>
      <div class="cw-stat-card">
        <div class="cw-stat-icon">🗄️</div>
        <div>
          <span class="cw-stat-value">2,200</span>
          <span class="cw-stat-label">Dataset Samples</span>
        </div>
      </div>
      <div class="cw-stat-card">
        <div class="cw-stat-icon">🧠</div>
        <div>
          <span class="cw-stat-value">5</span>
          <span class="cw-stat-label">ML Algorithms Evaluated</span>
        </div>
      </div>
      <div class="cw-stat-card">
        <div class="cw-stat-icon">✅</div>
        <div>
          <span class="cw-stat-value">99.55%</span>
          <span class="cw-stat-label">Benchmark Test Accuracy</span>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── HOW IT WORKS ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 0 0 20px;">
  <span class="section-sub">Scientific Feature Overview</span>
  <h2 class="section-title">How the ML Model Evaluates Inputs</h2>
  <p class="section-desc">CropWise.AI maps 7 multi-dimensional soil chemistry and climatic features to historical crop cultivation records using Random Forest classification.</p>
</div>
<div style="display:grid; grid-template-columns: repeat(3,1fr); gap: 24px; margin-bottom: 48px;">
  <div class="feat-box">
    <div class="feat-icon icon-emerald">🧪</div>
    <h3>Soil Nutrient Indices (N-P-K)</h3>
    <p>Nitrogen, Phosphorus, and Potassium represent numerical soil nutrient feature indices as captured in the benchmark dataset.</p>
  </div>
  <div class="feat-box">
    <div class="feat-icon icon-teal">🌦️</div>
    <h3>Climatic Profile</h3>
    <p>Ambient air temperature (°C) and relative humidity (%) govern plant transpiration and thermal stress windows.</p>
  </div>
  <div class="feat-box">
    <div class="feat-icon icon-amber">💧</div>
    <h3>Soil pH & Rainfall</h3>
    <p>Soil pH measures acidity/alkalinity influencing nutrient availability, while rainfall (mm) indicates seasonal water supply.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌾 Field Predictor",
    "🔍 Crop Explorer",
    "📊 ML Performance"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div id="predictor" style="text-align:center; margin-bottom:24px;">
      <span class="section-sub">Interactive Model Predictor</span>
      <h2 class="section-title">Field Feature Analysis & Crop Predictor</h2>
      <p class="section-desc">Select a preset or enter your 7 soil and climate parameter values below.</p>
    </div>
    """, unsafe_allow_html=True)

    # Presets
    PRESETS = {
        "🌾 Rice": {"N":80,"P":48,"K":40,"temperature":23.7,"humidity":82.3,"ph":6.4,"rainfall":236.0},
        "🌽 Maize": {"N":78,"P":48,"K":20,"temperature":22.4,"humidity":65.1,"ph":6.2,"rainfall":85.0},
        "☁️ Cotton": {"N":118,"P":46,"K":19,"temperature":24.0,"humidity":79.8,"ph":6.9,"rainfall":80.0},
        "☕ Coffee": {"N":101,"P":29,"K":30,"temperature":25.5,"humidity":58.9,"ph":6.8,"rainfall":158.0},
        "🍎 Apple": {"N":21,"P":134,"K":199,"temperature":22.6,"humidity":92.3,"ph":5.9,"rainfall":113.0},
        "🍉 Watermelon": {"N":100,"P":17,"K":50,"temperature":25.6,"humidity":85.0,"ph":6.5,"rainfall":50.0},
    }

    pcols = st.columns(len(PRESETS) + 1)
    pcols[0].markdown('<div style="padding-top:8px; font-weight:700; font-size:0.9rem; color:#334155;">💡 Preset:</div>', unsafe_allow_html=True)
    for i, (label, vals) in enumerate(PRESETS.items()):
        if pcols[i + 1].button(label, key=f"preset_{i}"):
            for k, v in vals.items():
                st.session_state[f"inp_{k}"] = v
            st.rerun()

    # Form
    with st.form("predictor_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🧪 Soil Nutrient Indices")
            n_val = st.number_input("Nitrogen (N) — Index", 0.0, 140.0, float(st.session_state.get("inp_N", 90.0)), 1.0, help="Dataset range: 0 – 140")
            p_val = st.number_input("Phosphorus (P) — Index", 5.0, 145.0, float(st.session_state.get("inp_P", 42.0)), 1.0, help="Dataset range: 5 – 145")
            k_val = st.number_input("Potassium (K) — Index", 5.0, 205.0, float(st.session_state.get("inp_K", 43.0)), 1.0, help="Dataset range: 5 – 205")
            ph_val = st.number_input("Soil pH", 3.5, 10.0, float(st.session_state.get("inp_ph", 6.5)), 0.1, help="pH 3.5 – 10.0")
        with col2:
            st.markdown("#### ⛅ Climate & Environment")
            temp_val = st.number_input("Temperature (°C)", 8.0, 45.0, float(st.session_state.get("inp_temperature", 23.5)), 0.1)
            hum_val  = st.number_input("Humidity (%)", 14.0, 100.0, float(st.session_state.get("inp_humidity", 82.0)), 0.1)
            rain_val = st.number_input("Rainfall (mm)", 20.0, 350.0, float(st.session_state.get("inp_rainfall", 202.9)), 0.1)

        submitted = st.form_submit_button("✨ Analyze Field", use_container_width=True)

    # ── RESULT SECTION ──────────────────────────────────────────────────────
    if submitted:
        if predictor is None:
            st.error("⚠️ Model not loaded. Please check the `models/` directory.")
        else:
            with st.spinner("Analyzing field data..."):
                result = predictor.predict({
                    "N": n_val, "P": p_val, "K": k_val,
                    "temperature": temp_val, "humidity": hum_val,
                    "ph": ph_val, "rainfall": rain_val
                })

            if not result.get("success"):
                st.error(f"❌ {result.get('error')}")
            else:
                crop_name = result.get("crop_name", "Unknown")
                crop_icon = result.get("crop_icon", "🌱")
                confidence = result.get("confidence", 0.0)
                explanation = result.get("explanation", "")
                model_used  = result.get("model_used", "Random Forest")
                crop_info   = result.get("crop_info", {})
                alts        = result.get("alternatives", [])
                field_st    = result.get("field_status", [])
                fert_guide  = result.get("fertilizer_guidance", [])

                # Result header
                st.markdown(f"""
                <div class="result-card">
                  <div class="result-header">
                    <span class="badge-ml">🤖 ML MODEL PREDICTION</span>
                    <span class="badge-model">🧠 Model: {model_used}</span>
                  </div>

                  <!-- Crop Banner -->
                  <div class="crop-banner">
                    <div class="crop-icon-big">{crop_icon}</div>
                    <div style="flex:1;">
                      <span class="hero-label">Predicted Crop Class</span>
                      <h2 class="crop-hero-name">{crop_name}</h2>
                      <p class="crop-explanation">{explanation}</p>
                    </div>
                    <div class="conf-box">
                      <span class="conf-title">Model Score (Est. Probability)</span>
                      <div class="conf-value">{confidence}%</div>
                      <div class="conf-bar-wrap">
                        <div class="conf-bar-fill" style="width:{confidence}%;"></div>
                      </div>
                      <span class="conf-disclaimer">ℹ️ Score represents model's estimated probability, not a guarantee of agricultural success.</span>
                    </div>
                  </div>
                """, unsafe_allow_html=True)

                # Alternative chips
                if alts:
                    chips_html = '<div style="margin-bottom:20px;"><h4 style="font-family:\'Space Grotesk\',sans-serif;font-size:1rem;font-weight:700;margin-bottom:10px;">🏅 Other Candidate Crops:</h4><div>'
                    for alt in alts:
                        chips_html += f"""
                        <span class="alt-chip">
                          {alt.get('icon','🌱')} {alt.get('crop_name', alt.get('crop_key','').capitalize())}
                          <span class="alt-conf">{alt.get('confidence',0)}% Score</span>
                        </span>"""
                    chips_html += "</div></div>"
                    st.markdown(chips_html, unsafe_allow_html=True)

                # Field input parameters table
                if field_st:
                    st.markdown("<h4 style=\"font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;margin:16px 0 10px;\">🎛️ Supplied Field Input Parameters & General Reference Ranges:</h4>", unsafe_allow_html=True)
                    hcards = '<div class="health-grid">'
                    for item in field_st:
                        hcards += f"""
                        <div class="health-card">
                          <span class="h-feature">{item.get('feature', item.get('parameter',''))}</span>
                          <span class="h-value">{item.get('value','')}</span>
                          <div class="h-range">Ref: {item.get('ideal_range', item.get('typical_range',''))}</div>
                        </div>"""
                    hcards += "</div>"
                    st.markdown(hcards, unsafe_allow_html=True)

                # Fertilizer guidance
                if fert_guide:
                    st.markdown("<h4 style=\"font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;margin:20px 0 10px;\">🧪 General Fertilizer & Soil Nutrient Information:</h4>", unsafe_allow_html=True)
                    for fert in fert_guide:
                        if isinstance(fert, dict):
                            st.markdown(f"""
                            <div class="fert-item">
                              <div class="fert-nutrient">🌿 <strong>{fert.get('nutrient','')}</strong> — <em>{fert.get('function','')}</em></div>
                              <p class="fert-advice">{fert.get('advice','')}</p>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="fert-item"><p class="fert-advice">• {fert}</p></div>', unsafe_allow_html=True)
                    st.markdown('<div class="agri-disclaimer">⚠️ Consult a qualified agronomist or local agricultural extension service before applying chemical fertilizers.</div>', unsafe_allow_html=True)

                # Agricultural Reference Card
                st.markdown(f"""
                  <div class="agri-card" style="margin-top:24px;">
                    <div class="agri-header">
                      <h3>📖 Crop Reference Information</h3>
                      <span class="guidance-tag">General Agronomic Reference</span>
                    </div>
                    <div class="agri-grid">
                      <div class="agri-item">
                        <div class="agri-item-title">🪨 Soil Profile</div>
                        <p>{crop_info.get('general_soil_description','—')}</p>
                      </div>
                      <div class="agri-item">
                        <div class="agri-item-title">⚗️ Typical Soil pH Range</div>
                        <p>{crop_info.get('typical_ph_range','—')}</p>
                      </div>
                      <div class="agri-item">
                        <div class="agri-item-title">📅 Growing Season</div>
                        <p>{crop_info.get('season','—')}</p>
                      </div>
                      <div class="agri-item">
                        <div class="agri-item-title">📋 Management Advice</div>
                        <p>{crop_info.get('general_management_guidance','—')}</p>
                      </div>
                    </div>
                    <div style="margin-top:14px;padding-top:12px;border-top:1px dashed var(--gray-200);font-size:0.8rem;color:var(--gray-400);">
                      ℹ️ <strong>Source / Reference:</strong> {crop_info.get('source','FAO & ICAR Extension Guidelines')}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CROP EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div id="explorer" style="text-align:center; margin-bottom:24px;">
      <span class="section-sub">Interactive Reference Database</span>
      <h2 class="section-title">Crop Explorer & Reference Search</h2>
      <p class="section-desc">Search any crop to inspect its typical soil pH, climate ranges, and cultivation guidelines.</p>
    </div>
    """, unsafe_allow_html=True)

    search_q = st.text_input("🔍 Search crops by name or category (e.g. Rice, Fruit, Pulse, Legume)...", placeholder="Type to filter...")

    filtered = [
        k for k in sorted(db.keys())
        if search_q.lower() in k.lower()
        or search_q.lower() in db[k].get("crop_name", "").lower()
        or search_q.lower() in db[k].get("category", "").lower()
    ]

    st.caption(f"Showing **{len(filtered)}** of **{len(db)}** crops")

    cols = st.columns(3)
    for i, key in enumerate(filtered):
        info = db[key]
        with cols[i % 3]:
            st.markdown(f"""
            <div class="crop-card">
              <span class="crop-card-icon">{info.get('icon','🌱')}</span>
              <div class="crop-card-name">{info.get('crop_name', key.capitalize())}</div>
              <span class="crop-card-cat">{info.get('category','Crop')} · {info.get('season','General')}</span>
              <p>🪨 {info.get('general_soil_description','Agricultural soil')}</p>
              <p>⚗️ pH: <strong>{info.get('typical_ph_range','6.0 – 7.5')}</strong></p>
              <p>🌧️ Rain: <strong>{info.get('typical_rainfall_range','100 – 200 mm')}</strong></p>
              <p>🌡️ Temp: <strong>{info.get('typical_temperature_range','20 – 30°C')}</strong></p>
              <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--gray-100);font-size:0.78rem;color:var(--gray-400);">
                {info.get('source','Agricultural Extension')}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div id="ml-performance" style="text-align:center; margin-bottom:24px;">
      <span class="section-sub">Technical Transparency</span>
      <h2 class="section-title">Machine Learning Model Performance & Dataset Specs</h2>
      <p class="section-desc">Benchmark comparison across 5 candidate algorithms evaluated on a 20% stratified held-out test set (440 samples).</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset transparency card
    st.markdown("""
    <div class="cw-card" style="margin-bottom:28px;">
      <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:16px;">🗄️ Benchmark Dataset Transparency</h3>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">
        <div class="metric-box"><span class="m-value">2,200</span><span class="m-label">Total Observations</span></div>
        <div class="metric-box"><span class="m-value">22</span><span class="m-label">Crop Categories (100 each)</span></div>
        <div class="metric-box"><span class="m-value">80 / 20</span><span class="m-label">Stratified Train / Test Split</span></div>
        <div class="metric-box"><span class="m-value">Zero</span><span class="m-label">Data / Scaler Leakage</span></div>
      </div>
      <p style="font-size:0.85rem;color:var(--dark-600);margin-top:14px;">
        ℹ️ <strong>Scope Note:</strong> Performance is measured on the available benchmark dataset (440 test samples, <code>random_state=42</code>)
        and may differ on real-world field data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="cw-card">
          <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:6px;">
            🏆 Best Model: <span style="color:var(--primary-600);">Random Forest Classifier</span>
          </h3>
          <p style="font-size:0.88rem;color:var(--dark-600);margin-bottom:16px;">
            Evaluated using 80/20 stratified train-test split on 2,200 samples across 22 classes.
          </p>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;">
            <div class="metric-box"><span class="m-value">99.55%</span><span class="m-label">Test-Set Benchmark Accuracy</span></div>
            <div class="metric-box"><span class="m-value">99.55%</span><span class="m-label">F1-Score (Weighted)</span></div>
          </div>
          <h4 style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;font-weight:700;margin-bottom:10px;">Candidate Algorithms:</h4>
          <table class="model-table">
            <thead><tr><th>Algorithm</th><th>Test Accuracy</th><th>F1-Score</th></tr></thead>
            <tbody>
              <tr><td><strong>Random Forest (Selected)</strong></td><td><span class="badge-success">99.55%</span></td><td>99.55%</td></tr>
              <tr><td>Support Vector Machine (SVM)</td><td>98.41%</td><td>98.40%</td></tr>
              <tr><td>Decision Tree</td><td>97.95%</td><td>97.94%</td></tr>
              <tr><td>K-Nearest Neighbors (KNN)</td><td>97.95%</td><td>97.93%</td></tr>
              <tr><td>Logistic Regression</td><td>97.27%</td><td>97.25%</td></tr>
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="cw-card">
          <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:6px;">
            📊 Confusion Matrix Heatmap
          </h3>
          <p style="font-size:0.88rem;color:var(--dark-600);margin-bottom:12px;">
            Evaluated on 440 test samples across 22 classes (20 test samples per class).
          </p>
        </div>
        """, unsafe_allow_html=True)
        cm_path = os.path.join(PROJECT_ROOT, "static", "images", "confusion_matrix.png")
        if not os.path.exists(cm_path):
            cm_path = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="438 out of 440 predictions correct on held-out test set", use_container_width=True)

    # How it works steps
    st.markdown("""
    <div style="text-align:center;margin:40px 0 20px;">
      <span class="section-sub">ML Pipeline</span>
      <h2 class="section-title">How CropWise.AI Operates</h2>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-bottom:40px;">
      <div class="step-card">
        <span class="step-num">01</span>
        <div class="step-icon">📋</div>
        <h3>Feature Input Vector</h3>
        <p>Enter Nitrogen, Phosphorus, Potassium indices, Temperature, Humidity, Soil pH, and Rainfall values.</p>
      </div>
      <div class="step-card">
        <span class="step-num">02</span>
        <div class="step-icon">⚙️</div>
        <h3>Standardized ML Processing</h3>
        <p>Data passes through StandardScaler (fitted ONLY on training data to prevent leakage) and is evaluated by Random Forest.</p>
      </div>
      <div class="step-card">
        <span class="step-num">03</span>
        <div class="step-icon">🍎</div>
        <h3>Prediction & Reference Output</h3>
        <p>Receive crop classification, model score, candidate probabilities, input parameter summary, and reference agronomic information.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cw-footer">
  <div class="cw-footer-brand">🌱 CropWise<span>.AI</span></div>
  <p>AI-Powered Crop Recommendation & Agricultural Decision Support System.<br>
  Model predictions are statistical recommendations based on benchmark dataset observations and do not replace certified agricultural extension advice or laboratory soil testing.</p>
  <div class="tech-tags">
    <span class="tag">Python 3</span>
    <span class="tag">Streamlit</span>
    <span class="tag">Scikit-Learn</span>
    <span class="tag">Pandas</span>
    <span class="tag">Joblib</span>
    <span class="tag">Random Forest</span>
  </div>
  <p style="margin-top:20px;font-size:0.78rem;">© 2026 CropWise.AI. Built for Educational & Decision Support Purposes.</p>
</div>
""", unsafe_allow_html=True)
