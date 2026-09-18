@echo off
REM Double-click this to open Teacher. It sets everything up the first time.
setlocal
cd /d "%~dp0"

set PY=python
if exist "ai_studio\.venv\Scripts\python.exe" set PY=ai_studio\.venv\Scripts\python.exe

echo Checking Teacher is ready...
%PY% -c "import gradio, torch, transformers" 2>nul
if errorlevel 1 (
  echo.
  echo First run - installing what Teacher needs. This takes a few minutes.
  echo.
  %PY% -m pip install -r ai_studio\requirements.txt
  if errorlevel 1 (
    echo.
    echo Install failed. Is Python 3.11 or newer installed and on your PATH?
    pause
    exit /b 1
  )
)

echo.
echo Opening Teacher in your browser...
%PY% -m teacher ui
if errorlevel 1 pause
