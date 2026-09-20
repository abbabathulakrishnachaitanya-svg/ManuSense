@echo off
echo ========================================
echo ManuSense - Smart Manufacturing Platform
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install/upgrade dependencies
echo Checking dependencies...
pip install -r requirements.txt --quiet
echo Dependencies installed!
echo.

REM Run Streamlit app
echo Starting ManuSense application...
echo.
echo The application will open in your browser automatically.
echo If not, navigate to: http://localhost:8501
echo.
echo Login credentials:
echo   Username: demo
echo   Password: demo
echo.
echo Press Ctrl+C to stop the application
echo ========================================
echo.

streamlit run streamlit_app.py
