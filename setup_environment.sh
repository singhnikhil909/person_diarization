#!/bin/bash

echo "========================================"
echo "Setting up Python 3.10 Environment"
echo "========================================"
echo ""

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "ERROR: Python 3.10 is not found!"
    echo "Please install Python 3.10 from https://www.python.org/downloads/"
    exit 1
fi

echo "Python 3.10 detected!"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3.10 -m venv venv

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created!"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the Streamlit app:"
echo "  streamlit run app.py"
echo ""

