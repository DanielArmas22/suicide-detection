from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os
import logging
from datetime import datetime
from typing import Optional, List
import uvicorn
from contextlib import asynccontextmanager
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize translator
translator = Translator()

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
    original_text: str
    translated_text: str

class DetailedPredictionResponse(BaseModel):
    prediction: str
    confidence: float
    suicidal_probability: float
    non_suicidal_probability: float
    processed_text: str
    risk_level: str
    analysis: dict
    original_text: str
    translated_text: str
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class UserMessage(BaseModel):
    id: str
    user_id: str
    user_name: str
    content: str
    timestamp: datetime
    type: str  # "post", "diary", "comment"
    category: Optional[str] = None

class UserMessagesResponse(BaseModel):
    messages: List[UserMessage]
    total: int
    user_name: str

class PredictMessageRequest(BaseModel):
    message_id: str
    text: str

class PredictMessageResponse(BaseModel):
    message_id: str
    original_text: str
    translated_text: str
    prediction: str
    confidence: float
    suicidal_probability: float
    non_suicidal_probability: float
    risk_level: str
    analysis: dict

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

def predict_text_with_translation(text: str) -> dict:
    """Make prediction on input text with translation support"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # First, translate to English if needed
        translated_text, was_translated = translate_to_english(text)
        
        # Preprocess the translated text
        processed_text = preprocess_text(translated_text)
        
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
            risk_level = "high"
        elif suicidal_prob > 0.5:
            risk_level = "medium"
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "suicidal_probability": round(suicidal_prob, 4),
            "non_suicidal_probability": round(non_suicidal_prob, 4),
            "processed_text": processed_text,
            "risk_level": risk_level,
            "original_text": text,
            "translated_text": translated_text,
            "was_translated": was_translated
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def predict_text(text: str) -> dict:
    """Make prediction on input text using the same method as training"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Translate Spanish text to English first
        english_text, was_translated = translate_to_english(text)
        logger.info(f"Processing text: '{english_text}'")
        
        # Preprocess text
        processed_text = preprocess_text(english_text)
        
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
            "risk_level": risk_level,
            "original_text": text,
            "translated_text": english_text
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

def translate_to_english(text: str) -> tuple[str, bool]:
    """Translate Spanish text to English"""
    try:
        # Detect language
        detection = translator.detect(text)
        
        if detection.lang == 'en':
            logger.info("Text is already in English")
            return text, False
        elif detection.lang == 'es':
            logger.info("Translating Spanish text to English")
            translated = translator.translate(text, src='es', dest='en')
            print(translated)
            return translated.text, True
        else:
            logger.info(f"Detected language: {detection.lang}, translating to English")
            translated = translator.translate(text, dest='en')
            return translated.text, True
            
    except Exception as e:
        logger.warning(f"Translation failed: {str(e)}, using original text")
        return text, False

# Mock data for demonstration
MOCK_USERS_MESSAGES = {
    "user1": {
        "name": "Ana García",
        "messages": [
            {
                "id": "msg1",
                "user_id": "user1",
                "content": "Me siento muy triste últimamente. No sé qué hacer con mi vida.",
                "timestamp": datetime(2025, 1, 15, 14, 30),
                "type": "diary",
                "category": "personal"
            },
            {
                "id": "msg2",
                "user_id": "user1",
                "content": "A veces pienso que sería mejor no estar aquí. Todo es tan difícil.",
                "timestamp": datetime(2025, 1, 16, 20, 15),
                "type": "diary",
                "category": "personal"
            },
            {
                "id": "msg3",
                "user_id": "user1",
                "content": "Hoy fue un mejor día. Hablé con mi familia y me siento un poco mejor.",
                "timestamp": datetime(2025, 1, 17, 10, 45),
                "type": "post",
                "category": "general"
            }
        ]
    },
    "user2": {
        "name": "Carlos Mendoza",
        "messages": [
            {
                "id": "msg4",
                "user_id": "user2",
                "content": "La universidad es muy estresante. No puedo manejar toda la presión.",
                "timestamp": datetime(2025, 1, 15, 16, 20),
                "type": "post",
                "category": "academic"
            },
            {
                "id": "msg5",
                "user_id": "user2",
                "content": "Me duele todo el cuerpo y no puedo dormir. Creo que necesito ayuda profesional.",
                "timestamp": datetime(2025, 1, 16, 23, 10),
                "type": "diary",
                "category": "health"
            }
        ]
    },
    "user3": {
        "name": "María Rodriguez",
        "messages": [
            {
                "id": "msg6",
                "user_id": "user3",
                "content": "Estoy muy feliz con mis resultados académicos este semestre.",
                "timestamp": datetime(2025, 1, 14, 12, 30),
                "type": "post",
                "category": "academic"
            },
            {
                "id": "msg7",
                "user_id": "user3",
                "content": "Algunas veces me siento abrumada, pero tengo buenos amigos que me apoyan.",
                "timestamp": datetime(2025, 1, 16, 14, 45),
                "type": "diary",
                "category": "social"
            }
        ]
    }
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
        print("prueba endpoint")
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
            "/predict/message": "Message prediction with translation (POST)",
            "/users/messages": "Get all user messages (GET)",
            "/users/{user_id}/messages": "Get messages for specific user (GET)",
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

@app.get("/users/messages", tags=["Users"])
async def get_all_users_messages():
    """Get all users and their messages for admin dashboard"""
    try:
        all_messages = []
        
        for user_id, user_data in MOCK_USERS_MESSAGES.items():
            for msg_data in user_data["messages"]:
                message = UserMessage(
                    id=msg_data["id"],
                    user_id=msg_data["user_id"],
                    user_name=user_data["name"],
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    type=msg_data["type"],
                    category=msg_data.get("category")
                )
                all_messages.append(message)
        
        # Sort by timestamp descending
        all_messages.sort(key=lambda x: x.timestamp, reverse=True)
        
        logger.info(f"Retrieved {len(all_messages)} messages from {len(MOCK_USERS_MESSAGES)} users")
        
        return {
            "messages": all_messages,
            "total": len(all_messages),
            "users_count": len(MOCK_USERS_MESSAGES)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving messages")

@app.get("/users/{user_id}/messages", response_model=UserMessagesResponse, tags=["Users"])
async def get_user_messages(user_id: str):
    """Get messages for a specific user"""
    try:
        if user_id not in MOCK_USERS_MESSAGES:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = MOCK_USERS_MESSAGES[user_id]
        messages = []
        
        for msg_data in user_data["messages"]:
            message = UserMessage(
                id=msg_data["id"],
                user_id=msg_data["user_id"],
                user_name=user_data["name"],
                content=msg_data["content"],
                timestamp=msg_data["timestamp"],
                type=msg_data["type"],
                category=msg_data.get("category")
            )
            messages.append(message)
        
        # Sort by timestamp descending
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        
        return UserMessagesResponse(
            messages=messages,
            total=len(messages),
            user_name=user_data["name"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving user messages")

@app.post("/predict/message", response_model=PredictMessageResponse, tags=["Prediction"])
async def predict_message_risk(request: PredictMessageRequest):
    """
    Predict suicide risk for a specific message with translation support
    
    - **message_id**: ID of the message being analyzed
    - **text**: The text content to analyze (in Spanish or English)
    
    Returns prediction with translation info and detailed analysis
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
        
        # Make prediction with translation
        result = predict_text_with_translation(request.text.strip())
        
        # Get detailed analysis (using translated text)
        analysis = analyze_text_indicators(result["translated_text"])
        
        # Log prediction (without sensitive data)
        logger.info(f"Message prediction made: {result['prediction']} (confidence: {result['confidence']}, risk_level: {result['risk_level']}, translated: {result['was_translated']})")
        
        return PredictMessageResponse(
            message_id=request.message_id,
            original_text=result["original_text"],
            translated_text=result["translated_text"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            suicidal_probability=result["suicidal_probability"],
            non_suicidal_probability=result["non_suicidal_probability"],
            risk_level=result["risk_level"],
            analysis=analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in message prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )

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
