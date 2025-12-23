# Quick Start Guide

## Step 1: Install Python 3.10

If you don't have Python 3.10:
1. Download from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Verify installation: `python --version` (should show 3.10.x)

## Step 2: Set Up Environment

**Windows:**
```bash
setup_environment.bat
```

**Linux/Mac:**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

This will:
- Create a virtual environment with Python 3.10
- Install all required packages (PyTorch, Pyannote, Streamlit, etc.)
- Take about 5-10 minutes (first time)

## Step 3: Accept Model Terms

Before first use, visit these links and accept the terms:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/embedding

## Step 4: Run the App

**Windows:**
```bash
venv\Scripts\activate
streamlit run app.py
```

**Linux/Mac:**
```bash
source venv/bin/activate
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Step 5: Use the App

1. Upload an audio file (WAV, MP3, FLAC, M4A, OGG)
2. Click "🚀 Start Diarization"
3. Wait for processing (first time may take a few minutes to download models)
4. View results and download if needed

## Troubleshooting

**"Python 3.10 not found"**
- Make sure Python 3.10 is installed and in PATH
- Try: `python3.10 --version`

**"Module not found"**
- Make sure virtual environment is activated
- Re-run setup script

**"Authentication error"**
- Make sure you accepted terms on Hugging Face
- Check that `.env` file exists with your token

**"Models not loading"**
- First run downloads models (2-5 minutes)
- Check internet connection
- Verify HF_TOKEN in `.env` file

