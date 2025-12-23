# Testing Model Output

This guide helps you debug why the model might only be detecting one speaker.

## Quick Test

Run the test script with your audio file:

```bash
python test_model_output.py your_audio_file.wav
```

Or specify a different model:

```bash
python test_model_output.py your_audio_file.wav BUT-FIT/diarizen-wavlm-large-s80-mlc
```

## What the Test Shows

The script will display:

1. **Model Input Info**: Shape and format of input to the model
2. **Model Output Info**: Structure of model outputs
3. **Prediction Analysis**:
   - Number of speaker classes the model can predict
   - Unique speaker IDs predicted
   - Distribution of predictions (how many frames for each speaker)
   - Probability statistics
   - Sample predictions with probabilities

## Understanding the Results

### If you see only one speaker ID:
- **Problem**: The model is only predicting one speaker class
- **Possible causes**:
  - Model architecture limitation
  - Model not trained for multi-speaker scenarios
  - Audio preprocessing issue

### If you see multiple speaker IDs but clustering fails:
- **Problem**: Model predicts multiple speakers but clustering groups them together
- **Solution**: Check embedding extraction and clustering parameters

### If probabilities are very high (>0.95):
- **Problem**: Model is too confident in single speaker
- **Solution**: Model might need different parameters or preprocessing

## Next Steps

1. Run the test script
2. Share the output (especially the "PREDICTION ANALYSIS" section)
3. We can then adjust the code based on what the model is actually outputting

