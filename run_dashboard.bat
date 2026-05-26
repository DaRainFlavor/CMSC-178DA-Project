@echo off
title Streamlit Dashboard - Inverters
echo [INVERTERS DASHBOARD] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment venv not found! Please make sure python environment is setup.
    pause
    exit /b
)
call venv\Scripts\activate.bat
echo [INVERTERS DASHBOARD] Running Streamlit application (app.py)...
streamlit run app.py
pause
