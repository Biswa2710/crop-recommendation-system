"""
test_prediction.py
Unit and integration tests for Crop Recommendation System using Python unittest.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import CropPredictor
from src.crop_info import FEATURE_LIMITS


class TestCropPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = CropPredictor()

    def test_valid_input_prediction(self):
        """Test prediction with valid input parameters."""
        valid_data = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.8,
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.9
        }
        result = self.predictor.predict(valid_data)
        self.assertTrue(result['success'])
        self.assertIn('predicted_crop', result)
        self.assertEqual(result['predicted_crop'], 'rice')
        self.assertIn('crop_name', result)
        self.assertIn('confidence', result)
        self.assertGreater(result['confidence'], 50.0)
        self.assertIn('crop_info', result)
        self.assertEqual(result['crop_info']['name'], 'Rice')

    def test_invalid_negative_input(self):
        """Test validation with negative soil nutrient input."""
        invalid_data = {
            'N': -10,
            'P': 42,
            'K': 43,
            'temperature': 20.8,
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.9
        }
        result = self.predictor.predict(invalid_data)
        self.assertFalse(result['success'])
        self.assertIn('cannot be lower than', result['error'])

    def test_invalid_out_of_bounds_ph(self):
        """Test validation with out of bounds pH input."""
        invalid_data = {
            'N': 50,
            'P': 40,
            'K': 40,
            'temperature': 25.0,
            'humidity': 60.0,
            'ph': 14.0,  # Max allowed is 10.0
            'rainfall': 100.0
        }
        result = self.predictor.predict(invalid_data)
        self.assertFalse(result['success'])
        self.assertIn('exceeds maximum', result['error'])

    def test_missing_input_field(self):
        """Test validation with missing field."""
        invalid_data = {
            'N': 50,
            'P': 40,
            'K': 40,
            'temperature': 25.0,
            'humidity': 60.0
        }
        result = self.predictor.predict(invalid_data)
        self.assertFalse(result['success'])
        self.assertIn('Missing required field', result['error'])

    def test_non_numeric_input(self):
        """Test validation with string/non-numeric input."""
        invalid_data = {
            'N': "abc",
            'P': 40,
            'K': 40,
            'temperature': 25.0,
            'humidity': 60.0,
            'ph': 6.5,
            'rainfall': 100.0
        }
        result = self.predictor.predict(invalid_data)
        self.assertFalse(result['success'])
        self.assertIn('Invalid numerical value', result['error'])


if __name__ == '__main__':
    unittest.main()
