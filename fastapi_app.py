from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os
import logging
from datetime import datetime
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Suicide Detection API...")
    
    if load_model():
        logger.info("Model loaded successfully!")
    else:
        logger.error("Failed to load model!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Suicide Detection API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Suicide Detection API",
    description="AI model for detecting suicide risk in text using BERT",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pydantic models for request/response
class TextRequest(BaseModel):
    text: str
    
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    suicidal_probability: float
    non_suicidal_probability: float
    processed_text: str
    risk_level: str

class DetailedPredictionResponse(BaseModel):
    prediction: str
    confidence: float
    suicidal_probability: float
    non_suicidal_probability: float
    processed_text: str
    risk_level: str
    analysis: dict
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Text preprocessing function (same as in training)
def preprocess_text(text: str) -> str:
    """Preprocess input text for model prediction"""
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

def load_model() -> bool:
    """Load the trained model and tokenizer"""
    global model, tokenizer
    
    try:
        # Try to load from output directory first
        model_path = "output"
        
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            logger.info(f"Loading trained model from {model_path}...")
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()
            logger.info("Trained model loaded successfully!")
        else:
            # Fallback to models directory
            model_path = "models/suicide_detection_model"
            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}...")
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
                logger.warning("Using pre-trained model - predictions may not be accurate!")
            
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Set model to None to prevent crashes
        model = None
        tokenizer = None
        return False

def predict_text(text: str) -> dict:
    """Make prediction on input text using the same method as training"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        if not processed_text:
            raise HTTPException(status_code=400, detail="Empty text after preprocessing")
        
        # Tokenize using the same method as training
        encoding = tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # Get prediction probabilities using softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get probabilities for each class
        non_suicidal_prob = probs[0][0].item()  # Class 0: non-suicidal
        suicidal_prob = probs[0][1].item()      # Class 1: suicidal
        
        # Get class with highest probability
        _, prediction_idx = torch.max(probs, dim=1)
        prediction = "suicidal" if prediction_idx.item() == 1 else "non-suicidal"
        confidence = probs[0][prediction_idx.item()].item()
        
        # Determine risk level based on probability
        risk_level = "low"
        if suicidal_prob > 0.8:
            risk_level = "very high"
        elif suicidal_prob > 0.6:
            risk_level = "high"
        elif suicidal_prob > 0.5:
            risk_level = "medium"
        elif suicidal_prob > 0.3:
            risk_level = "low"
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "suicidal_probability": round(suicidal_prob, 4),
            "non_suicidal_probability": round(non_suicidal_prob, 4),
            "processed_text": processed_text,
            "risk_level": risk_level
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Suicide risk indicators from the reference script
SUICIDE_INDICATORS = [
    'kill', 'die', 'suicide', 'end', 'pain', 'life', 'anymore', 'want', 'hope', 
    'help', 'death', 'dead', 'hate', 'tired', 'pills', 'hurt', 'alone', 'sad', 
    'depression', 'anxiety', 'lost', 'cut', 'empty', 'worthless'
]

FIRST_PERSON_PRONOUNS = ['i', 'me', 'my', 'mine', 'myself']

def analyze_text_indicators(text: str) -> dict:
    """Analyze text for suicide risk indicators (similar to reference script)"""
    processed_text = preprocess_text(text)
    words = processed_text.lower().split()
    
    # Check for suicide indicators
    indicators_present = [word for word in SUICIDE_INDICATORS if word in words]
    
    # Count first-person pronouns
    first_person_count = sum(1 for word in words if word in FIRST_PERSON_PRONOUNS)
    
    # Text statistics
    text_length = len(text)
    word_count = len(words)
    
    return {
        "indicators_found": indicators_present,
        "indicator_count": len(indicators_present),
        "first_person_count": first_person_count,
        "text_length": text_length,
        "word_count": word_count
    }

# API Routes

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Suicide Detection API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_suicide_risk(request: TextRequest):
    """
    Predict suicide risk in the provided text
    
    - **text**: The text to analyze for suicide risk indicators
    
    Returns prediction with confidence scores and probabilities
    """
    try:
        # Check if model is loaded
        if model is None or tokenizer is None:
            raise HTTPException(
                status_code=503, 
                detail="Model not available. Please try again later."
            )
        
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        # Make prediction
        result = predict_text(request.text.strip())
        
        # Log prediction (without sensitive data)
        logger.info(f"Prediction made: {result['prediction']} (confidence: {result['confidence']}, risk_level: {result['risk_level']})")
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/detailed", response_model=DetailedPredictionResponse, tags=["Prediction"])
async def predict_suicide_risk_detailed(request: TextRequest):
    """
    Predict suicide risk with detailed analysis
    
    - **text**: The text to analyze for suicide risk indicators
    
    Returns prediction with confidence scores, probabilities, and detailed text analysis
    """
    try:
        # Check if model is loaded
        if model is None or tokenizer is None:
            raise HTTPException(
                status_code=503, 
                detail="Model not available. Please try again later."
            )
        
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        # Make prediction
        result = predict_text(request.text.strip())
        
        # Get detailed analysis
        analysis = analyze_text_indicators(request.text.strip())
        
        # Log prediction (without sensitive data)
        logger.info(f"Detailed prediction made: {result['prediction']} (confidence: {result['confidence']}, risk_level: {result['risk_level']})")
        
        return DetailedPredictionResponse(**result, analysis=analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detailed prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/info", tags=["General"])
async def api_info():
    """Get detailed API information"""
    return {
        "name": "Suicide Detection API",
        "version": "1.0.0",
        "description": "AI model for detecting suicide risk in text using BERT",
        "model_info": {
            "architecture": "BERT-base-uncased",
            "task": "binary classification",
            "classes": ["non-suicidal", "suicidal"],
            "max_length": 128,
            "risk_levels": ["low", "medium", "high"]
        },
        "endpoints": {
            "/": "Root endpoint",
            "/health": "Health check",
            "/predict": "Text prediction (POST)",
            "/predict/detailed": "Text prediction with detailed analysis (POST)",
            "/api/info": "API information",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        },
        "usage": {
            "content_type": "application/json",
            "example_request": {
                "text": "I'm feeling really down today but I'll be okay"
            }
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": "An unexpected error occurred"
    }

if __name__ == "__main__":
    # Configuration
    HOST = "0.0.0.0"
    PORT = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    
    # Run the application
    uvicorn.run(
        "fastapi_app:app",
        host=HOST,
        port=PORT,
        reload=False,  # Set to True for development
        log_level="info"
    )
