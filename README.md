# Speaker Identification System

A deep learning-based speaker identification system that can identify speakers in audio files by comparing them against reference voice samples. Built with Resemblyzer and Streamlit.

## Features

- 🎤 **Speaker Identification**: Identify speakers in audio files using pretrained deep learning models
- 📊 **Similarity Scores**: Get matching percentages for each detected speaker
- 🎯 **Multiple Format Support**: Supports MP3, WAV, M4A, MP4, FLAC, OGG audio formats
- 📈 **Visual Analytics**: Interactive charts and progress bars showing similarity scores
- ⚙️ **Configurable**: Adjustable segment duration, overlap, and similarity thresholds

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have FFmpeg installed** (required for audio format conversion):
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) or use `choco install ffmpeg`
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

## Project Structure

```
speaker_identification/
├── voice_samples/          # Reference voice samples (organized by speaker name)
│   ├── aman/
│   ├── harsh/
│   ├── jony/
│   └── ...
├── speaker_identifier.py   # Core speaker identification module
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage

### 1. Prepare Reference Voice Samples

Organize your reference voice samples in the `voice_samples` directory:
- Each speaker should have their own folder
- Place multiple audio samples per speaker for better accuracy
- Supported formats: MP3, WAV, M4A, MP4, FLAC, OGG

Example structure:
```
voice_samples/
├── aman/
│   ├── sample1.mp3
│   ├── sample2.mp3
│   └── ...
├── harsh/
│   └── sample1.mp3
└── ...
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Using the App

1. **Load Reference Voices**: Click "Load Reference Voices" in the sidebar to load all voice samples
2. **Upload Audio**: Upload an audio file containing voices to identify
3. **Adjust Settings** (optional):
   - Segment Duration: How long each analysis segment should be (2-10 seconds)
   - Segment Overlap: Overlap between segments for better coverage
   - Similarity Threshold: Minimum percentage to consider a match
4. **View Results**: See detected speakers with similarity percentages

## How It Works

1. **Reference Voice Loading**: 
   - Loads all audio files from each speaker's folder
   - Extracts speaker embeddings using Resemblyzer's pretrained model
   - Averages embeddings from multiple samples per speaker for better representation

2. **Audio Analysis**:
   - Segments the uploaded audio into smaller chunks
   - Extracts embeddings from each segment
   - Compares each segment's embedding with all reference speaker embeddings

3. **Similarity Calculation**:
   - Uses cosine similarity to compare embeddings
   - Normalizes scores to 0-100% range
   - Aggregates results across all segments

4. **Results Display**:
   - Shows average and maximum similarity for each speaker
   - Displays visualizations and progress bars
   - Filters results based on similarity threshold

## Technical Details

- **Model**: Resemblyzer (pretrained speaker verification model)
- **Embedding Dimension**: 256-dimensional vectors
- **Sample Rate**: 16kHz (automatically resampled)
- **Minimum Segment Length**: 1 second

## Troubleshooting

### Audio Loading Issues
- Ensure FFmpeg is installed for format conversion
- Check that audio files are not corrupted
- Try converting audio to WAV format manually

### Low Similarity Scores
- Add more reference samples per speaker (5+ samples recommended)
- Ensure reference samples are clear and contain speech
- Try adjusting the similarity threshold

### Performance Issues
- Reduce segment duration for faster processing
- Use shorter audio files for testing
- Close other applications to free up memory

## Requirements

- Python 3.8+
- FFmpeg (for audio format conversion)
- See `requirements.txt` for Python packages

## License

This project uses Resemblyzer, which is open source. Please refer to their license for usage terms.

