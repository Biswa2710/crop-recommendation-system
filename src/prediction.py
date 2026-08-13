"""
prediction.py
Inference engine and input validation for Crop Recommendation System.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.crop_info import FEATURE_LIMITS, get_crop_info
from src.data_validation import validate_input_bounds, analyze_field_health

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
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
    @staticmethod
    def validate_inputs(input_data: dict) -> tuple[bool, str, dict]:
        """Validate input feature values against physiological bounds."""
        return validate_input_bounds(input_data)

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
        
        # Scale features using pre-fitted scaler (fitted ONLY on training data to prevent data leakage)
        scaled_vector = self.scaler.transform(feature_df)
        
        # Predict using model
        raw_pred = self.model.predict(scaled_vector)[0]
        
        # Decode prediction if model used XGBoost with encoded labels
        if isinstance(raw_pred, (int, np.integer)) and self.label_encoder is not None:
            predicted_crop = self.label_encoder.inverse_transform([raw_pred])[0]
        else:
            predicted_crop = str(raw_pred)
            
        # Calculate probabilities using model's predict_proba()
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
            
            # Top crop estimated probability score
            top_prob_pct = round(float(prob_pairs[0][1]) * 100, 1)
            
            # Alternative candidate crop predictions (top 2nd and 3rd)
            for crop_name, prob_val in prob_pairs[1:4]:
                if prob_val > 0.01:
                    crop_details = get_crop_info(crop_name)
                    alternatives.append({
                        "crop_key": crop_name,
                        "crop_name": crop_details.get("crop_name", crop_name.capitalize()),
                        "icon": crop_details.get("icon", "🌱"),
                        "confidence": round(float(prob_val) * 100, 1)
                    })
                    
        crop_info = get_crop_info(predicted_crop)
        
        # Clear scientific explanation
        model_name = self.metadata.get("best_model_name", "Random Forest Classifier") if self.metadata else "Random Forest Classifier"
        explanation = f"The {model_name} model identified {crop_info.get('crop_name', predicted_crop.capitalize())} as the target class with the highest predicted score based on the 7 supplied input feature values."
        
        # Analyze field health & fertilizer guidance
        field_analysis = analyze_field_health(cleaned_data, predicted_crop)
        
        return {
            "success": True,
            "predicted_crop": predicted_crop,
            "crop_name": crop_info.get("crop_name", predicted_crop.capitalize()),
            "crop_icon": crop_info.get("icon", "🌱"),
            "confidence": top_prob_pct,
            "inputs": cleaned_data,
            "explanation": explanation,
            "crop_info": crop_info,
            "alternatives": alternatives,
            "field_status": field_analysis.get("field_status", []),
            "fertilizer_guidance": field_analysis.get("fertilizer_guidance", []),
            "model_used": model_name
        }
