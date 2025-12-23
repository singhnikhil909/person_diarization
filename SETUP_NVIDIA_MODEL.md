# Setting Up NVIDIA Diarization Model

The NVIDIA `diar_sortformer_4spk-v1` model is now supported and **highly recommended** for better speaker identification!

## Why Use NVIDIA Model?

✅ **Built-in speaker clustering** - Automatically groups similar voices  
✅ **Better accuracy** - Lower Diarization Error Rate (DER)  
✅ **Simpler processing** - No manual chunking/clustering needed  
✅ **Designed for diarization** - End-to-end neural model  

## Installation Steps

### 1. Upgrade PyTorch (REQUIRED)

NeMo requires PyTorch 2.2.0 or higher. First upgrade PyTorch:

```bash
# Upgrade PyTorch to 2.2.0+ (required for NeMo)
pip install torch>=2.2.0 torchaudio>=2.2.0 --upgrade
```

**Important:** If you're using CPU-only PyTorch, upgrade with:
```bash
pip install torch>=2.2.0+cpu torchaudio>=2.2.0+cpu --upgrade --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install Prerequisites

```bash
# Install Cython and packaging (required for NeMo)
pip install Cython packaging

# Install NVIDIA NeMo
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

**Note:** NeMo installation may take a few minutes as it's a large framework.

### 2. The Model is Already Configured!

The code will automatically use the NVIDIA model by default. No code changes needed!

### 3. Model Limitations

- **Maximum 4 speakers** - The model handles up to 4 speakers per audio file
- If you have more than 4 speakers, the model will still work but may group some speakers together

## Usage

Just run the app as usual:

```bash
streamlit run app.py
```

The NVIDIA model will be used automatically. If NeMo is not installed, it will fall back to the transformers-based model.

## Switching Models

If you want to use the old model instead, you can modify `app.py`:

```python
# In app.py, change the load_pipeline() call to:
pipeline = load_diarization_pipeline(model_name="BUT-FIT/diarizen-wavlm-large-s80-mlc")
```

## Troubleshooting

### "NVIDIA NeMo not installed" or "device_mesh" error
- **PyTorch version issue**: NeMo requires PyTorch 2.2.0+. Upgrade with:
  ```bash
  pip install torch>=2.2.0 torchaudio>=2.2.0 --upgrade
  ```
- Make sure you've installed NeMo: `pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]`
- Check that Cython and packaging are installed: `pip install Cython packaging`

### Model loading fails
- The code will automatically fall back to the transformers model
- Check your internet connection (model downloads from Hugging Face)
- If you see warnings about "Megatron" or "ffmpeg", these are harmless and can be ignored

