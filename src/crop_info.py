"""
crop_info.py
Loads structured crop reference information and dataset parameter boundaries.
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_PATH = os.path.join(PROJECT_ROOT, "data", "crop_information.json")


def load_crop_information_db() -> dict:
    """Load crop reference database from JSON."""
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


CROP_DATABASE = load_crop_information_db()

FEATURE_LIMITS = {
    "N": {"min": 0, "max": 140, "label": "Nitrogen (N)", "unit": "Dataset Feature Value", "desc": "Relative Soil Nitrogen Index"},
    "P": {"min": 5, "max": 145, "label": "Phosphorus (P)", "unit": "Dataset Feature Value", "desc": "Relative Soil Phosphorus Index"},
    "K": {"min": 5, "max": 205, "label": "Potassium (K)", "unit": "Dataset Feature Value", "desc": "Relative Soil Potassium Index"},
    "temperature": {"min": 8.0, "max": 45.0, "label": "Temperature", "unit": "°C", "desc": "Ambient Air Temperature"},
    "humidity": {"min": 14.0, "max": 100.0, "label": "Humidity", "unit": "%", "desc": "Relative Humidity"},
    "ph": {"min": 3.5, "max": 10.0, "label": "Soil pH", "unit": "pH", "desc": "Soil Acidic/Alkaline measure"},
    "rainfall": {"min": 20.0, "max": 350.0, "label": "Rainfall", "unit": "mm", "desc": "Seasonal Rainfall"}
}


def get_crop_info(crop_key: str) -> dict:
    """Return dictionary of agricultural details for a given crop name."""
    db = load_crop_information_db()
    key = str(crop_key).strip().lower()
    info = db.get(key, {
        "crop_name": crop_key.capitalize(),
        "category": "Agricultural Crop",
        "general_soil_description": "Fertile, well-drained agricultural soil.",
        "typical_ph_range": "6.0 - 7.5 (Slightly acidic to neutral)",
        "typical_temperature_range": "20.0°C - 30.0°C",
        "typical_rainfall_range": "50.0 - 200.0 mm",
        "typical_npk_index": "N: 50, P: 45, K: 35 (Dataset Feature Index)",
        "general_management_guidance": "Follow standard agricultural extension guidelines for optimal yield.",
        "season": "General Growing Season",
        "icon": "🌱",
        "source": "General Agricultural Extension Guidelines"
    })
    # Ensure backward compatibility for name field
    if "name" not in info and "crop_name" in info:
        info["name"] = info["crop_name"]
    return info
