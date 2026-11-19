@echo off
setlocal
cd /d "%~dp0"
title RAG Studio 1.0

set "PY_CMD="
where py >nul 2>nul
if %errorlevel%==0 set "PY_CMD=py -3"
if not defined PY_CMD (
    where python >nul 2>nul
    if %errorlevel%==0 set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Python 3.11-3.13 is required. Download it from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [RAG Studio 1.0] Creating local virtual environment - first run only...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set "REQ_HASH="
for /f "skip=1 delims=" %%H in ('certutil -hashfile requirements.txt SHA256') do (
    if not defined REQ_HASH set "REQ_HASH=%%H"
)
set "STAMP_FILE=.venv\.deps_stamp"
set "OLD_HASH="
if exist "%STAMP_FILE%" set /p OLD_HASH=<"%STAMP_FILE%"

if /i not "%REQ_HASH%"=="%OLD_HASH%" (
    echo [RAG Studio 1.0] Installing dependencies - first run only, cached for future launches...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --prefer-binary -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check your internet connection and retry.
        pause
        exit /b 1
    )
    >"%STAMP_FILE%" echo %REQ_HASH%
)

echo [RAG Studio 1.0] Starting local workspace at http://127.0.0.1:8765 ...
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8765
if errorlevel 1 (
    echo.
    echo [ERROR] RAG Studio server stopped unexpectedly.
    pause
)
endlocal
