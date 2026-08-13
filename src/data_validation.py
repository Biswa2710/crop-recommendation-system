"""
data_validation.py
Input validation, field condition assessment, and fertilizer guidance calculation.
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_PATH = os.path.join(PROJECT_ROOT, "data", "crop_profiles.json")

FEATURE_LIMITS = {
    "N": {"min": 0, "max": 140, "label": "Nitrogen (N)", "unit": "mg/kg"},
    "P": {"min": 5, "max": 145, "label": "Phosphorus (P)", "unit": "mg/kg"},
    "K": {"min": 5, "max": 205, "label": "Potassium (K)", "unit": "mg/kg"},
    "temperature": {"min": 8.0, "max": 45.0, "label": "Temperature", "unit": "°C"},
    "humidity": {"min": 14.0, "max": 100.0, "label": "Humidity", "unit": "%"},
    "ph": {"min": 3.5, "max": 10.0, "label": "Soil pH", "unit": "pH"},
    "rainfall": {"min": 20.0, "max": 350.0, "label": "Rainfall", "unit": "mm"}
}


def load_crop_profiles() -> dict:
    """Load structured crop profiles from JSON."""
    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, 'r') as f:
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
            return False, f"{limits['label']} ({val} {limits['unit']}) cannot be lower than {limits['min']} {limits['unit']}.", {}
        if val > limits['max']:
            return False, f"{limits['label']} ({val} {limits['unit']}) exceeds maximum realistic bound of {limits['max']} {limits['unit']}.", {}
            
        cleaned[feat] = val
        
    return True, "", cleaned


def analyze_field_health(inputs: dict, crop_key: str) -> dict:
    """
    Compare user inputs against the target crop's actual parameter ranges.
    Returns field condition classifications and fertilizer guidance.
    """
    profiles = load_crop_profiles()
    crop_profile = profiles.get(crop_key.lower().strip(), {})
    ranges = crop_profile.get("ranges", {})
    
    if not ranges:
        return {"field_status": [], "fertilizer_guidance": []}
        
    field_status = []
    fertilizer_guidance = []
    
    # 1. Nitrogen (N)
    N_val = inputs['N']
    N_range = ranges.get('N', {})
    if N_val < N_range.get('min', 0):
        status = "Low"
        badge_class = "badge-warning"
        n_advice = f"Nitrogen level ({N_val} mg/kg) is below the recommended range ({N_range.get('min')}-{N_range.get('max')} mg/kg) for {crop_profile.get('name')}. Consider applying organic compost or nitrogen-fixing green manure."
    elif N_val > N_range.get('max', 140):
        status = "High"
        badge_class = "badge-info"
        n_advice = f"Nitrogen level ({N_val} mg/kg) is higher than usual. Avoid excessive nitrogen fertilization to prevent overly lush vegetative growth at the expense of fruiting."
    else:
        status = "Optimal"
        badge_class = "badge-success"
        n_advice = f"Nitrogen level ({N_val} mg/kg) is within the optimal target range for {crop_profile.get('name')}."
        
    field_status.append({
        "feature": "Nitrogen (N)", "value": f"{N_val} mg/kg", "status": status, "badge_class": badge_class,
        "ideal_range": f"{N_range.get('min')}-{N_range.get('max')} mg/kg"
    })
    fertilizer_guidance.append({"nutrient": "Nitrogen (N)", "status": status, "advice": n_advice})

    # 2. Phosphorus (P)
    P_val = inputs['P']
    P_range = ranges.get('P', {})
    if P_val < P_range.get('min', 0):
        status = "Low"
        badge_class = "badge-warning"
        p_advice = f"Phosphorus level ({P_val} mg/kg) is below the target range ({P_range.get('min')}-{P_range.get('max')} mg/kg). Phosphorus is essential for root establishment and bloom initialization."
    elif P_val > P_range.get('max', 145):
        status = "High"
        badge_class = "badge-info"
        p_advice = f"Phosphorus level ({P_val} mg/kg) is abundantly high. Maintain current organic soil practices."
    else:
        status = "Optimal"
        badge_class = "badge-success"
        p_advice = f"Phosphorus level ({P_val} mg/kg) is ideal for robust root development."
        
    field_status.append({
        "feature": "Phosphorus (P)", "value": f"{P_val} mg/kg", "status": status, "badge_class": badge_class,
        "ideal_range": f"{P_range.get('min')}-{P_range.get('max')} mg/kg"
    })
    fertilizer_guidance.append({"nutrient": "Phosphorus (P)", "status": status, "advice": p_advice})

    # 3. Potassium (K)
    K_val = inputs['K']
    K_range = ranges.get('K', {})
    if K_val < K_range.get('min', 0):
        status = "Low"
        badge_class = "badge-warning"
        k_advice = f"Potassium level ({K_val} mg/kg) is low relative to {crop_profile.get('name')} demand ({K_range.get('min')}-{K_range.get('max')} mg/kg). Potassium enhances disease resistance and fruit quality."
    elif K_val > K_range.get('max', 205):
        status = "High"
        badge_class = "badge-info"
        k_advice = f"Potassium level ({K_val} mg/kg) is very high, satisfying crop requirement."
    else:
        status = "Optimal"
        badge_class = "badge-success"
        k_advice = f"Potassium level ({K_val} mg/kg) is within the optimal physiological range."
        
    field_status.append({
        "feature": "Potassium (K)", "value": f"{K_val} mg/kg", "status": status, "badge_class": badge_class,
        "ideal_range": f"{K_range.get('min')}-{K_range.get('max')} mg/kg"
    })
    fertilizer_guidance.append({"nutrient": "Potassium (K)", "status": status, "advice": k_advice})

    # 4. Soil pH
    ph_val = inputs['ph']
    ph_range = ranges.get('ph', {})
    if ph_val < 5.5:
        ph_status = "Acidic"
        ph_badge = "badge-warning"
    elif ph_val > 7.5:
        ph_status = "Alkaline"
        ph_badge = "badge-info"
    else:
        ph_status = "Optimal / Neutral"
        ph_badge = "badge-success"
        
    field_status.append({
        "feature": "Soil pH", "value": f"{ph_val} pH", "status": ph_status, "badge_class": ph_badge,
        "ideal_range": f"{ph_range.get('min')}-{ph_range.get('max')} pH"
    })

    # 5. Temperature
    t_val = inputs['temperature']
    t_range = ranges.get('temperature', {})
    t_status = "Normal" if (t_range.get('min', 0) <= t_val <= t_range.get('max', 50)) else ("Cool" if t_val < t_range.get('min', 0) else "Warm")
    field_status.append({
        "feature": "Temperature", "value": f"{t_val} °C", "status": t_status, "badge_class": "badge-success" if t_status == "Normal" else "badge-warning",
        "ideal_range": f"{t_range.get('min')}-{t_range.get('max')} °C"
    })

    # 6. Humidity
    h_val = inputs['humidity']
    h_range = ranges.get('humidity', {})
    h_status = "Normal" if (h_range.get('min', 0) <= h_val <= h_range.get('max', 100)) else ("Low" if h_val < h_range.get('min', 0) else "High")
    field_status.append({
        "feature": "Humidity", "value": f"{h_val} %", "status": h_status, "badge_class": "badge-success" if h_status == "Normal" else "badge-warning",
        "ideal_range": f"{h_range.get('min')}-{h_range.get('max')} %"
    })

    # 7. Rainfall
    r_val = inputs['rainfall']
    r_range = ranges.get('rainfall', {})
    r_status = "Normal" if (r_range.get('min', 0) <= r_val <= r_range.get('max', 350)) else ("Low Water" if r_val < r_range.get('min', 0) else "High Water")
    field_status.append({
        "feature": "Rainfall", "value": f"{r_val} mm", "status": r_status, "badge_class": "badge-success" if r_status == "Normal" else "badge-warning",
        "ideal_range": f"{r_range.get('min')}-{r_range.get('max')} mm"
    })

    return {
        "field_status": field_status,
        "fertilizer_guidance": fertilizer_guidance
    }
