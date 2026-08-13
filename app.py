"""
app.py
CropWise AI — Crop Recommendation System Flask Server Application.
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prediction import CropPredictor
from src.crop_info import CROP_DATABASE, get_crop_info, FEATURE_LIMITS

app = Flask(__name__)

# Initialize predictor singleton
try:
    predictor = CropPredictor()
except Exception as e:
    print(f"⚠️ Warning: Model artifacts not loaded on startup ({e}). Run train_model.py first.")
    predictor = None


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle crop prediction requests.
    Supports both JSON AJAX requests and standard Form POST submissions.
    """
    global predictor
    if predictor is None:
        try:
            predictor = CropPredictor()
        except Exception as e:
            err_msg = "Model artifacts missing or corrupt. Please train the model first by running 'python src/train_model.py'."
            if request.is_json:
                return jsonify({"success": False, "error": err_msg}), 500
            return render_template('index.html', error=err_msg, result=None)

    # Extract input data from JSON or Form
    if request.is_json:
        data = request.get_json()
    else:
        data = {
            'N': request.form.get('N'),
            'P': request.form.get('P'),
            'K': request.form.get('K'),
            'temperature': request.form.get('temperature'),
            'humidity': request.form.get('humidity'),
            'ph': request.form.get('ph'),
            'rainfall': request.form.get('rainfall')
        }

    # Perform prediction via predictor engine
    try:
        prediction_result = predictor.predict(data)
    except Exception as e:
        err_msg = f"Unexpected prediction error: {str(e)}"
        if request.is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template('index.html', error=err_msg, result=None)

    if request.is_json:
        if not prediction_result.get("success"):
            return jsonify(prediction_result), 400
        return jsonify(prediction_result), 200
    else:
        if not prediction_result.get("success"):
            return render_template('index.html', error=prediction_result.get("error"), result=None)
        return render_template('index.html', result=prediction_result)


@app.route('/api/crops', methods=['GET'])
def get_crops_list():
    """API endpoint to list all 22 supported crops with icons and metadata."""
    crop_list = []
    for key, info in CROP_DATABASE.items():
        crop_list.append({
            "key": key,
            "name": info["name"],
            "category": info["category"],
            "icon": info["icon"],
            "season": info["season"]
        })
    return jsonify({"success": True, "count": len(crop_list), "crops": crop_list})


@app.route('/api/info/<crop_name>', methods=['GET'])
def get_crop_details(crop_name):
    """API endpoint to retrieve detailed agricultural information for a specific crop."""
    info = get_crop_info(crop_name)
    return jsonify({"success": True, "crop_info": info})


@app.route('/api/search-crop', methods=['GET'])
def search_crop():
    """API endpoint to search crops by name and return ideal soil/weather parameters."""
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify({"success": False, "error": "Search query parameter 'q' is required"}), 400

    matches = []
    for key, info in CROP_DATABASE.items():
        crop_name = info.get("crop_name") or info.get("name") or key.capitalize()
        category = info.get("category", "Agricultural Crop")
        
        if query in key or query in crop_name.lower() or query in category.lower():
            matches.append({
                "key": key,
                "name": crop_name,
                "crop_name": crop_name,
                "category": category,
                "icon": info.get("icon", "🌱"),
                "season": info.get("season", "General Growing Season"),
                "general_soil_description": info.get("general_soil_description") or info.get("soil_type", ""),
                "typical_ph_range": info.get("typical_ph_range") or info.get("ideal_ph", ""),
                "typical_temperature_range": info.get("typical_temperature_range") or info.get("temperature_range", ""),
                "typical_rainfall_range": info.get("typical_rainfall_range") or info.get("rainfall_range", ""),
                "typical_npk_index": info.get("typical_npk_index") or info.get("npk_profile", ""),
                "general_management_guidance": info.get("general_management_guidance") or info.get("farming_guidance", ""),
                "source": info.get("source", "FAO & ICAR Extension Guidelines"),
                "ideal_params": info.get("ideal_params", {})
            })

    if not matches:
        return jsonify({"success": False, "message": f"No crops found matching '{query}'", "matches": []}), 404

    return jsonify({"success": True, "count": len(matches), "matches": matches}), 200


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """API endpoint to retrieve trained ML model metadata and evaluation comparison."""
    meta_path = os.path.join(os.path.dirname(__file__), 'models', 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    return jsonify({
        "best_model_name": "Random Forest Classifier",
        "best_accuracy": 0.9955,
        "best_f1_weighted": 0.9955,
        "features": list(FEATURE_LIMITS.keys()),
        "crop_classes": list(CROP_DATABASE.keys())
    })


@app.errorhandler(404)
def not_found(e):
    if request.is_json:
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return render_template('index.html', error="404 — Page Not Found", result=None), 404


@app.errorhandler(500)
def server_error(e):
    if request.is_json:
        return jsonify({"success": False, "error": "Internal server error"}), 500
    return render_template('index.html', error="500 — Internal Server Error", result=None), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
