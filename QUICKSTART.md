# Quick Start Guide

Get started with the Speaker Identification System in minutes!

## Option 1: Using Resemblyzer (Easiest - No Setup Required)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
streamlit run app.py
```

### Step 3: Use the App
1. In the sidebar, keep "resemblyzer" selected (default)
2. Click "🔄 Load Reference Voices"
3. Upload an audio file
4. View results!

✅ **No authentication needed** - Works out of the box!

---

## Option 2: Using Pyannote Community 3.1 (Better Accuracy)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Hugging Face Authentication

**Method A: Environment Variable (Recommended)**

1. Get your token from: https://huggingface.co/settings/tokens
2. Accept agreement at: https://huggingface.co/pyannote/embedding

**Windows PowerShell:**
```powershell
$env:HF_TOKEN="your_token_here"
streamlit run app.py
```

**Windows CMD:**
```cmd
set HF_TOKEN=your_token_here
streamlit run app.py
```

**Linux/Mac:**
```bash
export HF_TOKEN="your_token_here"
streamlit run app.py
```

**Method B: CLI Login**
```bash
huggingface-cli login
streamlit run app.py
```

### Step 3: Use the App
1. In the sidebar, select "pyannote" from the dropdown
2. Click "🔄 Load Reference Voices"
3. Upload an audio file
4. View results!

✅ **Better accuracy** - State-of-the-art model!

---

## Troubleshooting

### "Weights only load failed" error (PyTorch 2.6+)
✅ **Automatically fixed!** The code handles this for you.

If you still see this error:
```bash
pip install -r requirements.txt
```

### "Reference voices not loaded" error
- Click the "🔄 Load Reference Voices" button in the sidebar first
- Make sure the `voice_samples` folder exists and contains speaker folders

### Pyannote authentication errors
```bash
# Check if token is set
echo $HF_TOKEN  # Linux/Mac
echo %HF_TOKEN%  # Windows CMD
$env:HF_TOKEN   # Windows PowerShell

# If not set, login again
huggingface-cli login
```

### Audio format not supported
- Install FFmpeg (see README.md)
- Or convert your audio to WAV format

### Slow processing
- Use Resemblyzer model for faster processing
- Reduce segment duration in sidebar
- Use shorter audio files for testing

---

## Tips for Best Results

1. **Reference Samples**:
   - Use 3-5 samples per speaker
   - Each sample should be 3-10 seconds long
   - Clear audio with minimal background noise

2. **Settings**:
   - **Resemblyzer**: 
     - Segment Duration: 3 seconds
     - Threshold: 40-50%
   - **Pyannote**:
     - Segment Duration: 5 seconds
     - Threshold: 60-70%

3. **Upload Audio**:
   - Clear audio works best
   - 10-60 seconds is optimal for testing
   - Multiple speakers can be detected

---

## Next Steps

- Read [PYANNOTE_SETUP.md](PYANNOTE_SETUP.md) for detailed Pyannote setup
- Check [example_usage.py](example_usage.py) for programmatic usage
- See [README.md](README.md) for full documentation

---

## Still Having Issues?

Check the error message in the app:
- Red error boxes show what went wrong
- Click "Show error details" for more information
- Common issues are usually authentication or missing dependencies

Need help? Check:
- Pyannote docs: https://github.com/pyannote/pyannote-audio
- Hugging Face docs: https://huggingface.co/docs

