@echo off
REM One-click setup: creates venv, installs packages, downloads data, trains, starts app.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    python -m venv .venv || (echo Python not found. Install Python 3.10-3.12 from python.org and tick "Add to PATH". & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Package installation failed. & pause & exit /b 1)
python download_datasets.py
python train_models.py
python smoke_test.py
echo.
echo Starting the web app - your browser should open at http://localhost:8501
streamlit run app.py
pause
