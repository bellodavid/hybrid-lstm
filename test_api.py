#!/usr/bin/env python3
"""
Simple test script for the Aflatoxin Risk Prediction API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(BASE_URL)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint test failed: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\nTesting prediction endpoint...")
    
    test_cases = [
        {"location": "Lagos"},
        {"location": "Kaduna", "date": "2024-08-01"},
        {"location": "Rivers", "date": "2024-07-15"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_case}")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Prediction test {i} failed: {e}")

def main():
    print("Aflatoxin Risk Prediction API Test")
    print("=" * 40)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Run tests
    health_ok = test_health_check()
    root_ok = test_root_endpoint()
    
    if health_ok:
        test_prediction()
    else:
        print("\nSkipping prediction tests - health check failed")
    
    print("\n" + "=" * 40)
    print("Test completed")

if __name__ == "__main__":
    main()