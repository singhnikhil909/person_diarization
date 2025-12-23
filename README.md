# Speaker Diarization System

A Python-based speaker diarization system that identifies who is speaking in an audio file. Speakers are automatically labeled as "Person 1", "Person 2", etc.

## Features

- **Speaker Diarization**: Automatically segments audio by speaker using Pyannote models
- **Web UI**: Beautiful Streamlit interface for easy audio upload and visualization
- **Interactive Visualizations**: Timeline charts and pie charts showing speaking time
- **Export Options**: Download results as JSON or CSV
- **Python 3.10 Compatible**: Optimized for PyTorch and Pyannote compatibility

## Requirements

- Python 3.10
- CUDA-capable GPU (optional, but recommended for faster processing)

## Installation

### 1. Install Python 3.10

If you don't have Python 3.10 installed:
- Download from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation

### 2. Set Up Environment

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
- Create a Python 3.10 virtual environment
- Install all required packages
- Set up the environment for you

### 3. Configure Hugging Face Token

The `.env` file is already created with your token. If you need to update it:

```env
HF_TOKEN=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
```

### 4. Accept Model Terms

Before first use, accept the terms of use for the pretrained models:
- Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the terms
- Visit https://huggingface.co/pyannote/embedding and accept the terms

## Usage

### Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Run the Web UI

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## How to Use the UI

1. **Upload Audio**: Click "Browse files" and select an audio file (WAV, MP3, FLAC, M4A, OGG)
2. **Configure Settings** (Optional): Set minimum/maximum speaker counts in the sidebar
3. **Start Diarization**: Click "🚀 Start Diarization" button
4. **View Results**: 
   - See timeline visualization
   - Check speaking time distribution
   - Review detailed segment information
   - Download results as JSON or CSV

## Supported Audio Formats

- WAV
- MP3
- FLAC
- M4A
- OGG

## Troubleshooting

### Model Loading Issues

If you encounter authentication errors:
1. Make sure you've accepted the terms of use on Hugging Face
2. Verify your Hugging Face token is correct in the `.env` file
3. Check that the `.env` file is in the same directory as the scripts

### GPU Not Detected

The system will automatically use CPU if GPU is not available. Processing will be slower but will still work.

### Python Version Issues

Make sure you're using Python 3.10. Check your version:
```bash
python --version
```

If it's not 3.10, reinstall Python 3.10 and recreate the virtual environment.

## Project Structure

```
speech diarization/
├── .env                    # Environment variables (HF token)
├── app.py                 # Streamlit web UI
├── diarization_utils.py   # Diarization utilities
├── speaker_db.py          # Speaker database manager
├── requirements.txt       # Python dependencies
├── setup_environment.bat  # Windows setup script
├── setup_environment.sh   # Linux/Mac setup script
└── README.md             # This file
```

## License

This project uses pretrained models from Hugging Face. Please review and comply with their respective licenses:
- pyannote/speaker-diarization-3.1
- pyannote/embedding

