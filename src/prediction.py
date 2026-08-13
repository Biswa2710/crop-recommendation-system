"""
prediction.py
Inference engine and input validation for Crop Recommendation System.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.crop_info import FEATURE_LIMITS, get_crop_info, CROP_DATABASE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class CropPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load model, scaler, label encoder, and metadata from disk."""
        model_path = os.path.join(self.models_dir, "crop_model.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        le_path = os.path.join(self.models_dir, "label_encoder.pkl")
        meta_path = os.path.join(self.models_dir, "metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing at: {model_path}. Run training pipeline first.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file missing at: {scaler_path}. Run training pipeline first.")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
            
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
                
    @staticmethod
    def validate_inputs(input_data: dict) -> tuple[bool, str, dict]:
        """
        Validate input feature values against expected numerical bounds.
        Returns (is_valid, error_message, cleaned_data_dict).
        """
        cleaned = {}
        for feat, limits in FEATURE_LIMITS.items():
            if feat not in input_data or input_data[feat] is None or str(input_data[feat]).strip() == "":
                return False, f"Missing required field: {limits['label']}", {}
                
            try:
                val = float(input_data[feat])
            except (ValueError, TypeError):
                return False, f"Invalid numerical value for {limits['label']}: '{input_data[feat]}'", {}
                
            # Range checks
            if val < limits['min']:
                return False, f"{limits['label']} ({val} {limits['unit']}) cannot be lower than {limits['min']} {limits['unit']}.", {}
            if val > limits['max']:
                return False, f"{limits['label']} ({val} {limits['unit']}) exceeds maximum realistic bound of {limits['max']} {limits['unit']}.", {}
                
            cleaned[feat] = val
            
        return True, "", cleaned

    def predict(self, input_dict: dict) -> dict:
        """
        Predict crop for validated input dictionary.
        Returns detailed prediction result object.
        """
        is_valid, err_msg, cleaned_data = self.validate_inputs(input_dict)
        if not is_valid:
            return {"success": False, "error": err_msg}
            
        # Format feature vector as pandas DataFrame with feature names
        feature_df = pd.DataFrame([{
            'N': cleaned_data['N'],
            'P': cleaned_data['P'],
            'K': cleaned_data['K'],
            'temperature': cleaned_data['temperature'],
            'humidity': cleaned_data['humidity'],
            'ph': cleaned_data['ph'],
            'rainfall': cleaned_data['rainfall']
        }])
        
        # Scale features using pre-fitted scaler
        scaled_vector = self.scaler.transform(feature_df)
        
        # Predict using model
        raw_pred = self.model.predict(scaled_vector)[0]
        
        # Decode prediction if model used XGBoost with encoded labels
        if isinstance(raw_pred, (int, np.integer)) and self.label_encoder is not None:
            predicted_crop = self.label_encoder.inverse_transform([raw_pred])[0]
        else:
            predicted_crop = str(raw_pred)
            
        # Calculate probabilities if supported
        top_prob_pct = None
        alternatives = []
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled_vector)[0]
            classes = self.model.classes_
            
            if isinstance(classes[0], (int, np.integer)) and self.label_encoder is not None:
                class_names = self.label_encoder.inverse_transform(classes)
            else:
                class_names = [str(c) for c in classes]
                
            # Pair classes with probabilities
            prob_pairs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
            
            # Top crop probability
            top_prob_pct = round(float(prob_pairs[0][1]) * 100, 1)
            
            # Alternative predictions (top 2nd and 3rd)
            for crop_name, prob_val in prob_pairs[1:4]:
                if prob_val > 0.01:
                    crop_details = get_crop_info(crop_name)
                    alternatives.append({
                        "crop_key": crop_name,
                        "crop_name": crop_details["name"],
                        "icon": crop_details["icon"],
                        "confidence": round(float(prob_val) * 100, 1)
                    })
                    
        crop_info = get_crop_info(predicted_crop)
        
        # Build recommendation explanation
        explanation = self._build_explanation(predicted_crop, cleaned_data, crop_info)
        
        return {
            "success": True,
            "predicted_crop": predicted_crop,
            "crop_name": crop_info["name"],
            "crop_icon": crop_info["icon"],
            "confidence": top_prob_pct,
            "inputs": cleaned_data,
            "explanation": explanation,
            "crop_info": crop_info,
            "alternatives": alternatives,
            "model_used": self.metadata.get("best_model_name", "Random Forest Classifier") if self.metadata else "Random Forest Classifier"
        }
        
    def _build_explanation(self, crop: str, inputs: dict, crop_info: dict) -> str:
        """Construct natural language explanation of why this crop fits the parameters."""
        N, P, K = inputs['N'], inputs['P'], inputs['K']
        temp, hum, ph, rain = inputs['temperature'], inputs['humidity'], inputs['ph'], inputs['rainfall']
        
        reasons = []
        if rain > 180:
            reasons.append(f"high annual rainfall of {rain:.1f} mm")
        elif rain < 60:
            reasons.append(f"low water demand suitable for arid conditions ({rain:.1f} mm rainfall)")
        else:
            reasons.append(f"moderate rainfall of {rain:.1f} mm")
            
        if temp > 30:
            reasons.append(f"warm temperature of {temp:.1f}°C")
        elif temp < 20:
            reasons.append(f"cool climate of {temp:.1f}°C")
        else:
            reasons.append(f"optimal temperature of {temp:.1f}°C")
            
        if N > 80:
            reasons.append(f"high soil nitrogen content ({N:.0f} mg/kg)")
        if K > 100:
            reasons.append(f"rich potassium soil ({K:.0f} mg/kg)")
            
        reasons_str = ", ".join(reasons)
        return f"{crop_info['name']} is uniquely suited for your field because of your field's {reasons_str}, and soil pH of {ph:.1f}."
