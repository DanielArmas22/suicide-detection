#!/bin/bash

echo "🚀 Iniciando aplicación de detección de suicidio..."

# Create models directory if it doesn't exist
mkdir -p models
mkdir -p logs

# Check if model exists
if [ ! -d "models/suicide_detection_model" ]; then
    echo "📚 Modelo no encontrado. Entrenando modelo..."
    python training.py
else
    echo "✅ Modelo encontrado. Saltando entrenamiento."
fi

# Start the web application
echo "🌐 Iniciando servidor web..."
if [ "$ENVIRONMENT" = "production" ]; then
    # Production mode with gunicorn
    gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 300 --keep-alive 2 --max-requests 1000 --max-requests-jitter 50 app:app
else
    # Development mode
    python app.py
fi
