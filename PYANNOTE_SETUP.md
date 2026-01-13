# Pyannote Community 3.1 Setup Guide

This guide will help you set up and use the Pyannote Community 3.1 model for speaker identification.

## Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a Hugging Face Account**
   - Go to [huggingface.co](https://huggingface.co) and create an account if you don't have one

3. **Accept User Agreement**
   - Visit [pyannote/embedding](https://huggingface.co/pyannote/embedding)
   - Click on "Agree and access repository"
   - Accept the user agreement

## Authentication Options

### Option 1: Using Environment Variable (Recommended)

1. Get your Hugging Face token:
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "read" permission
   - Copy the token

2. Set the environment variable:

   **Windows (PowerShell):**
   ```powershell
   $env:HF_TOKEN="your_token_here"
   ```

   **Windows (Command Prompt):**
   ```cmd
   set HF_TOKEN=your_token_here
   ```

   **Linux/Mac:**
   ```bash
   export HF_TOKEN="your_token_here"
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Option 2: Using Hugging Face CLI

1. Install the CLI (already included in requirements):
   ```bash
   pip install huggingface-hub
   ```

2. Login via CLI:
   ```bash
   huggingface-cli login
   ```
   
3. Enter your token when prompted

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Using the Pyannote Model

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Select Model in Sidebar**
   - In the sidebar, select "pyannote" from the "Model Type" dropdown
   - The default is "resemblyzer"

3. **Load Reference Voices**
   - Click "🔄 Load Reference Voices" button
   - The app will load voice samples using the pyannote embedding model

4. **Upload and Analyze**
   - Upload your audio file
   - The system will identify speakers using the pyannote model

## Model Comparison

### Resemblyzer (Default)
- ✅ No authentication required
- ✅ Fast and lightweight
- ✅ Good for general use cases
- ❌ May have lower accuracy on complex audio

### Pyannote Community 3.1
- ✅ State-of-the-art accuracy
- ✅ Better handling of noisy audio
- ✅ More robust speaker embeddings
- ❌ Requires Hugging Face authentication
- ❌ Slightly slower processing

## Troubleshooting

### Error: "Weights only load failed" or PyTorch 2.6+ pickle error
✅ **Automatically fixed with direct torch.load patching!**

The code now handles PyTorch 2.6+ compatibility by temporarily patching `torch.load` to use `weights_only=False` specifically for trusted pyannote models.

**Safety Notes:**
- ✅ Only applies to official pyannote models from Hugging Face
- ✅ Patch is temporary and restored immediately after loading
- ✅ Safe because pyannote is an established, peer-reviewed academic project

**If you still see this error:**
```bash
# Update all dependencies
pip install -r requirements.txt

# Try clearing cache and reinstalling
pip uninstall torch torchaudio lightning
pip install torch torchaudio lightning>=2.0.0
```

**What's happening:**
- PyTorch 2.6+ changed security defaults to prevent arbitrary code execution
- Pyannote models contain OmegaConf and PyTorch Lightning configuration classes
- The code temporarily bypasses `weights_only=True` for trusted official models only

### Error: "pyannote.audio is not installed"
```bash
pip install pyannote.audio>=3.1.0
```

### Error: "Access denied" or "Authentication required"
- Make sure you've accepted the user agreement at https://huggingface.co/pyannote/embedding
- Verify your HF_TOKEN is set correctly
- Try logging in via `huggingface-cli login`

### Error: "CUDA out of memory"
- The model will automatically use CPU if CUDA is not available
- Close other applications using GPU memory

### Model takes too long to load
- First-time loading downloads the model (~100MB)
- Subsequent loads will use cached model
- Consider using Resemblyzer for faster processing if speed is critical

## Performance Tips

1. **Segment Duration**: 
   - Longer segments (5-7 seconds) work better with pyannote
   - Shorter segments (2-3 seconds) work better with resemblyzer

2. **Overlap**:
   - Use 1-2 seconds overlap for better continuity

3. **Similarity Threshold**:
   - Pyannote typically produces higher similarity scores
   - Try threshold around 60-70% for pyannote
   - Keep threshold around 40-50% for resemblyzer

## Additional Resources

- [Pyannote Documentation](https://github.com/pyannote/pyannote-audio)
- [Pyannote Paper](https://arxiv.org/abs/2104.04045)
- [Hugging Face Documentation](https://huggingface.co/docs)

