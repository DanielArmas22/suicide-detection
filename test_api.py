import requests
import json

def test_api():
    """Test the FastAPI suicide detection API"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Suicide Detection API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: API Info
    print("\n2. Testing API info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/info")
        print(f"Status: {response.status_code}")
        print(f"API Info: {response.json()['name']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Prediction with normal text
    print("\n3. Testing prediction with normal text...")
    try:
        test_data = {
            "text": "I had a great day today! Looking forward to the weekend."
        }
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Input: {test_data['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Suicide probability: {result['suicide_probability']:.2%}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Prediction with concerning text
    print("\n4. Testing prediction with concerning text...")
    try:
        test_data = {
            "text": "I feel so hopeless and don't know if I can keep going anymore."
        }
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Input: {test_data['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Suicide probability: {result['suicide_probability']:.2%}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Error handling - empty text
    print("\n5. Testing error handling with empty text...")
    try:
        test_data = {
            "text": ""
        }
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API testing completed!")
    print("\n📖 Documentation available at: http://localhost:8000/docs")
    print("📖 Alternative docs at: http://localhost:8000/redoc")

if __name__ == "__main__":
    test_api()
