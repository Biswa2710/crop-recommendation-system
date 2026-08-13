"""
data_validation.py
Input boundary validation, feature parameter summary, and qualitative fertilizer information.
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_PATH = os.path.join(PROJECT_ROOT, "data", "crop_profiles.json")
INFO_PATH = os.path.join(PROJECT_ROOT, "data", "crop_information.json")

FEATURE_LIMITS = {
    "N": {"min": 0, "max": 140, "label": "Nitrogen (N)", "unit": "Dataset Feature Value"},
    "P": {"min": 5, "max": 145, "label": "Phosphorus (P)", "unit": "Dataset Feature Value"},
    "K": {"min": 5, "max": 205, "label": "Potassium (K)", "unit": "Dataset Feature Value"},
    "temperature": {"min": 8.0, "max": 45.0, "label": "Temperature", "unit": "°C"},
    "humidity": {"min": 14.0, "max": 100.0, "label": "Humidity", "unit": "%"},
    "ph": {"min": 3.5, "max": 10.0, "label": "Soil pH", "unit": "pH"},
    "rainfall": {"min": 20.0, "max": 350.0, "label": "Rainfall", "unit": "mm"}
}


def load_json_file(file_path: str) -> dict:
    """Load JSON file cleanly if exists."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def validate_input_bounds(input_data: dict) -> tuple[bool, str, dict]:
    """
    Validate input parameters against absolute physiological/physical boundaries.
    Returns (is_valid, error_message, cleaned_dict).
    """
    cleaned = {}
    for feat, limits in FEATURE_LIMITS.items():
        if feat not in input_data or input_data[feat] is None or str(input_data[feat]).strip() == "":
            return False, f"Missing required field: {limits['label']}", {}
            
        try:
            val = float(input_data[feat])
        except (ValueError, TypeError):
            return False, f"Invalid numerical value for {limits['label']}: '{input_data[feat]}'", {}
            
        if val < limits['min']:
            return False, f"{limits['label']} ({val}) cannot be lower than {limits['min']} {limits['unit']}.", {}
        if val > limits['max']:
            return False, f"{limits['label']} ({val}) exceeds maximum realistic bound of {limits['max']} {limits['unit']}.", {}
            
        cleaned[feat] = val
        
    return True, "", cleaned


def analyze_field_health(inputs: dict, crop_key: str) -> dict:
    """
    Summarize field model inputs alongside general reference ranges.
    Returns field condition summary items and qualitative fertilizer information.
    """
    crop_info_db = load_json_file(INFO_PATH)
    crop_profiles = load_json_file(PROFILES_PATH)
    
    crop_data = crop_info_db.get(crop_key.lower().strip(), {})
    profile = crop_profiles.get(crop_key.lower().strip(), {})
    ranges = profile.get("ranges", {})
    
    field_status = [
        {
            "feature": "Nitrogen (N)",
            "value": f"{inputs['N']}",
            "unit_label": "Dataset Feature Value",
            "status": "Model Input Value",
            "badge_class": "badge-info",
            "ideal_range": f"{ranges.get('N', {}).get('min', '0')}-{ranges.get('N', {}).get('max', '140')} (Observed Range)"
        },
        {
            "feature": "Phosphorus (P)",
            "value": f"{inputs['P']}",
            "unit_label": "Dataset Feature Value",
            "status": "Model Input Value",
            "badge_class": "badge-info",
            "ideal_range": f"{ranges.get('P', {}).get('min', '5')}-{ranges.get('P', {}).get('max', '145')} (Observed Range)"
        },
        {
            "feature": "Potassium (K)",
            "value": f"{inputs['K']}",
            "unit_label": "Dataset Feature Value",
            "status": "Model Input Value",
            "badge_class": "badge-info",
            "ideal_range": f"{ranges.get('K', {}).get('min', '5')}-{ranges.get('K', {}).get('max', '205')} (Observed Range)"
        },
        {
            "feature": "Soil pH",
            "value": f"{inputs['ph']} pH",
            "unit_label": "pH Scale",
            "status": "Model Input Value",
            "badge_class": "badge-info",
            "ideal_range": crop_data.get("typical_ph_range", "5.5 - 7.5")
        },
        {
            "feature": "Temperature",
            "value": f"{inputs['temperature']} °C",
            "unit_label": "Celsius",
            "status": "Model Input Value",
            "badge_class": "badge-info",
            "ideal_range": crop_data.get("typical_temperature_range", "20.0°C - 30.0°C")
        },
        {
            "feature": "Humidity",
            "value": f"{inputs['humidity']} %",
            "unit_label": "Relative Humidity",
            "status": "Model Input Value",
            "badge_class": "badge-info",
            "ideal_range": f"{ranges.get('humidity', {}).get('min', '14')}-{ranges.get('humidity', {}).get('max', '100')} %"
        },
        {
            "feature": "Rainfall",
            "value": f"{inputs['rainfall']} mm",
            "unit_label": "Seasonal Rainfall",
            "status": "Model Input Value",
            "badge_class": "badge-info",
            "ideal_range": crop_data.get("typical_rainfall_range", "50.0 - 200.0 mm")
        }
    ]
    
    fertilizer_guidance = [
        {
            "nutrient": "Nitrogen (N)",
            "function": "Vegetative Growth",
            "advice": "Nitrogen supports leafy vegetative growth and canopy development. Fertilizer application should be based on laboratory soil test reports and local agricultural extension recommendations."
        },
        {
            "nutrient": "Phosphorus (P)",
            "function": "Root & Bloom Development",
            "advice": "Phosphorus promotes early root establishment, flower initiation, and seed development. Ensure balanced phosphorus application based on regional soil testing."
        },
        {
            "nutrient": "Potassium (K)",
            "function": "Stress Tolerance & Fruit Quality",
            "advice": "Potassium enhances plant drought resistance, water regulation, and fruit sugar quality. Apply organic compost or potash based on certified agronomic advice."
        }
    ]

    return {
        "field_status": field_status,
        "fertilizer_guidance": fertilizer_guidance
    }
