@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher was not found.
  echo Install Python 3.13 from https://www.python.org/downloads/windows/
  pause
  exit /b 1
)

py -3.13 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python 3.13 was not found.
  echo Install Python 3.13 and enable the Python Launcher, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the virtual environment...
  py -3.13 -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing the tested beta dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -e ".[dev]"
if errorlevel 1 goto :failed

if not exist ".env" (
  copy /Y ".env.example" ".env" >nul
  echo Created .env. Add your OpenAI project API key before generating questions.
) else (
  echo Existing .env was preserved.
)

echo.
echo Setup complete. Edit .env if needed, then double-click run_app.bat.
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the error above and try again.
pause
exit /b 1
