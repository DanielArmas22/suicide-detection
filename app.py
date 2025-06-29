from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Text preprocessing function (same as in training)
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    else:
        return ""

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer
    
    try:
        model_path = "models/suicide_detection_model"
        
        if os.path.exists(model_path):
            logger.info("Loading trained model...")
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()
            logger.info("Model loaded successfully!")
        else:
            logger.warning("No trained model found. Using pre-trained BERT model.")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            model.to(device)
            model.eval()
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False
    
    return True

def predict_text(text):
    """Make prediction on input text"""
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        if not processed_text:
            return {"error": "Empty text after preprocessing"}
        
        # Tokenize
        inputs = tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get probabilities
        suicide_prob = predictions[0][1].item()
        non_suicide_prob = predictions[0][0].item()
        
        # Determine prediction
        prediction = "suicide" if suicide_prob > 0.5 else "non-suicide"
        confidence = max(suicide_prob, non_suicide_prob)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "suicide_probability": round(suicide_prob, 4),
            "non_suicide_probability": round(non_suicide_prob, 4),
            "processed_text": processed_text
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {"error": str(e)}

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detección de Suicidio - AI Model</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5rem;
        }
        
        .warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #555;
        }
        
        textarea {
            width: 100%;
            min-height: 150px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        
        .result.suicide {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .result.non-suicide {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .result.error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .confidence {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Detección de Suicidio</h1>
        
        <div class="warning">
            <strong>⚠️ Advertencia:</strong> Esta herramienta es solo para fines educativos y de investigación. 
            Si tú o alguien que conoces está en crisis, busca ayuda profesional inmediatamente.
        </div>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="text">Ingresa el texto para analizar:</label>
                <textarea id="text" name="text" placeholder="Escribe o pega el texto aquí..." required></textarea>
            </div>
            
            <button type="submit" class="btn">Analizar Texto</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analizando texto...</p>
        </div>
        
        <div class="result" id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = document.getElementById('text').value;
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            // Show loading
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({text: text})
                });
                
                const data = await response.json();
                
                // Hide loading
                loading.style.display = 'none';
                
                if (data.error) {
                    result.className = 'result error';
                    result.innerHTML = `<strong>Error:</strong> ${data.error}`;
                } else {
                    result.className = `result ${data.prediction}`;
                    result.innerHTML = `
                        <h3>Resultado del Análisis</h3>
                        <p><strong>Predicción:</strong> ${data.prediction === 'suicide' ? 'Contenido de riesgo suicida' : 'Contenido normal'}</p>
                        <div class="confidence">Confianza: ${(data.confidence * 100).toFixed(2)}%</div>
                        <p><strong>Probabilidad de riesgo suicida:</strong> ${(data.suicide_probability * 100).toFixed(2)}%</p>
                        <p><strong>Probabilidad de contenido normal:</strong> ${(data.non_suicide_probability * 100).toFixed(2)}%</p>
                    `;
                }
                
                result.style.display = 'block';
                
            } catch (error) {
                loading.style.display = 'none';
                result.className = 'result error';
                result.innerHTML = `<strong>Error:</strong> ${error.message}`;
                result.style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the main interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Make prediction
        result = predict_text(text)
        
        # Log prediction (without sensitive data)
        logger.info(f"Prediction made: {result.get('prediction', 'error')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Suicide Detection API",
        "version": "1.0.0",
        "description": "AI model for detecting suicide risk in text",
        "endpoints": {
            "/": "Web interface",
            "/health": "Health check",
            "/predict": "Text prediction (POST)",
            "/api/info": "API information"
        }
    })

if __name__ == '__main__':
    logger.info("Starting Suicide Detection API...")
    
    # Load model
    if load_model():
        logger.info("Model loaded successfully!")
    else:
        logger.error("Failed to load model!")
    
    # Start Flask app
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
