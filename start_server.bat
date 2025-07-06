@echo off
echo 🚀 Starting Suicide Detection FastAPI Server
echo ==============================================

REM Check if virtual environment exists
if exist "venv" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo ⚠️  No virtual environment found. Consider creating one:
    echo    python -m venv venv
    echo    venv\Scripts\activate
    echo    pip install -r requirements.txt
    echo.
)

REM Install dependencies if needed
echo 📋 Checking dependencies...
pip install -q -r requirements.txt

echo.
echo 🔧 Server Configuration:
echo    Host: 0.0.0.0
echo    Port: 8000
echo    Model Path: output/ or models/suicide_detection_model/
echo.
echo 🌐 Access points:
echo    API: http://localhost:8000
echo    Docs: http://localhost:8000/docs
echo    Health: http://localhost:8000/health
echo.
echo 🔄 Starting server...
echo    Press Ctrl+C to stop
echo.

REM Start the FastAPI server
python fastapi_app.py
