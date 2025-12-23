@echo off
echo ========================================
echo Setting up Python 3.10 Environment
echo ========================================
echo.

REM Try different Python 3.10 commands
set PYTHON_CMD=

REM Try py -3.10 (Python Launcher for Windows)
py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.10
    echo Using: py -3.10
    goto :found_python
)

REM Try python3.10
python3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3.10
    echo Using: python3.10
    goto :found_python
)

REM Try python and check if it's 3.10
python --version 2>nul | findstr /R "3\.10" >nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    echo Using: python (detected 3.10)
    goto :found_python
)

REM If none found, show error
echo ERROR: Python 3.10 is not found!
echo.
echo Current Python version:
python --version 2>nul || echo Python not found in PATH
echo.
echo Please do one of the following:
echo 1. Install Python 3.10 from https://www.python.org/downloads/
echo 2. Use Python Launcher: py -3.10 -m venv venv
echo 3. Add Python 3.10 to PATH
echo.
pause
exit /b 1

:found_python
%PYTHON_CMD% --version
echo Python 3.10 detected!
echo.

REM Create virtual environment
echo Creating virtual environment...
%PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created!
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the Streamlit app:
echo   streamlit run app.py
echo.
pause

