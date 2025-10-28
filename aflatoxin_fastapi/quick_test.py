#!/usr/bin/env python3
"""
Quick test to verify the model and API work
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.inference import AflatoxinPredictor

def test_predictor():
    """Test the predictor class directly"""
    print("Testing AflatoxinPredictor...")
    
    try:
        # Initialize predictor
        predictor = AflatoxinPredictor()
        
        # Load model
        print("Loading model...")
        predictor.load_model()
        
        if predictor.is_model_loaded():
            print("✓ Model loaded successfully")
        else:
            print("✗ Model failed to load")
            return False
        
        # Test prediction
        print("Testing prediction...")
        result = predictor.predict("Lagos", "2024-08-01")
        
        if result:
            print("✓ Prediction successful:")
            print(f"  Location: {result.location}")
            print(f"  Representative City: {result.representative_city}")
            print(f"  Weather Period: {result.weather_period}")
            print(f"  Predicted Risk: {result.predicted_risk}")
            print(f"  Risk Level: {result.risk_level}")
            return True
        else:
            print("✗ Prediction failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_predictor()
    if success:
        print("\n🎉 All tests passed! Your model is ready for deployment.")
    else:
        print("\n❌ Tests failed. Check the errors above.")