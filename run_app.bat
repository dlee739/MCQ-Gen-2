@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run: py -3.13 -m venv .venv
  echo Then: .venv\Scripts\python -m pip install -e .
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
