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
        response = requests.get(f"{base_url}/health", timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: API Info
    print("\n2. Testing API info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/info", timeout=30)
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
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Input: {test_data['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Suicidal probability: {result['suicidal_probability']:.2%}")
        print(f"Risk level: {result['risk_level']}")
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
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Input: {test_data['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Suicidal probability: {result['suicidal_probability']:.2%}")
        print(f"Risk level: {result['risk_level']}")
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
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"Status: {response.status_code}")
        f"Response: {response.json()}"
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 6: Detailed prediction
    print("\n6. Testing detailed prediction endpoint...")
    try:
        test_data = {
            "text": "I feel so hopeless and tired. I don't want to be alive anymore. Nobody would miss me."
        }
        response = requests.post(
            f"{base_url}/predict/detailed",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Input: {test_data['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Suicidal probability: {result['suicidal_probability']:.2%}")
        print(f"Risk level: {result['risk_level']}")
        print(f"Indicators found: {result['analysis']['indicators_found']}")
        print(f"First person pronouns: {result['analysis']['first_person_count']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 7: Get all user messages
    print("\n7. Testing get all user messages endpoint...")
    try:
        response = requests.get(f"{base_url}/users/messages", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Total messages: {result['total']}")
            print(f"Users count: {result['users_count']}")
            print(f"First message preview: {result['messages'][0]['content'][:50]}..." if result['messages'] else "No messages")
        else:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 8: Test message prediction with translation
    print("\n8. Testing message prediction with translation...")
    try:
        test_data = {
            "message_id": "test_msg_001",
            "text": "Me siento muy triste y sin esperanza. A veces pienso que sería mejor no estar aquí."
        }
        response = requests.post(
            f"{base_url}/predict/message",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Original text: {result['original_text']}")
            print(f"Translated text: {result['translated_text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Risk level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Indicators found: {result['analysis']['indicator_count']}")
        else:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 9: Test with English text
    print("\n9. Testing message prediction with English text...")
    try:
        test_data = {
            "message_id": "test_msg_002",
            "text": "I feel hopeless and I don't want to be alive anymore."
        }
        response = requests.post(
            f"{base_url}/predict/message",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Was translation needed: {result['original_text'] != result['translated_text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Risk level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.2%}")
        else:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)
    print("✅ API testing completed!")
    print(f"\n📖 Documentation available at: {base_url}/docs")
    print(f"📖 Alternative docs at: {base_url}/redoc")

if __name__ == "__main__":
    test_api()
