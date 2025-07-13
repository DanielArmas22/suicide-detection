#!/usr/bin/env python3
"""
Validation script to test the model functionality directly
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import sys

def validate_model_loading():
    """Test if the model can be loaded correctly"""
    print("🔍 Validating model loading...")
    
    # Try to load from output directory
    model_path = "output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            print(f"✅ Found trained model in {model_path}")
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()
            print(f"✅ Model loaded successfully on {device}")
            
            # Test tokenization
            test_text = "I feel sad today"
            encoding = tokenizer.encode_plus(
                test_text,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            print(f"✅ Tokenization works correctly")
            print(f"   Input shape: {encoding['input_ids'].shape}")
            print(f"   Attention mask shape: {encoding['attention_mask'].shape}")
            
            # Test prediction
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            print(f"✅ Prediction works correctly")
            print(f"   Output shape: {outputs.logits.shape}")
            print(f"   Probabilities: {probs[0].tolist()}")
            
            return True
            
        else:
            print(f"❌ No trained model found in {model_path}")
            print("   Make sure you have trained the model first")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def validate_preprocessing():
    """Test the preprocessing function"""
    print("\n🧹 Validating text preprocessing...")
    
    import re
    
    def preprocess_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        else:
            return ""
    
    test_cases = [
        "I'm feeling really sad today! :( #depression",
        "Check this out: https://example.com @user123",
        "I have 5 problems and 10 solutions...",
        ""
    ]
    
    expected_results = [
        "im feeling really sad today depression",
        "check this out",
        "i have problems and solutions",
        ""
    ]
    
    for i, (input_text, expected) in enumerate(zip(test_cases, expected_results)):
        result = preprocess_text(input_text)
        if result == expected:
            print(f"✅ Test case {i+1}: '{input_text}' -> '{result}'")
        else:
            print(f"❌ Test case {i+1}: '{input_text}' -> '{result}' (expected: '{expected}')")
            return False
    
    return True

def main():
    """Main validation function"""
    print("🧪 Model Validation Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("fastapi_app.py"):
        print("❌ Please run this script from the sucidalback directory")
        sys.exit(1)
    
    success = True
    
    # Validate preprocessing
    if not validate_preprocessing():
        success = False
    
    # Validate model loading
    if not validate_model_loading():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All validations passed! The model is ready to use.")
    else:
        print("❌ Some validations failed. Please check the issues above.")
    
    return success

if __name__ == "__main__":
    main()
