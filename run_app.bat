@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run setup_beta.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
