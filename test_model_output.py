"""
Test script to check what the diarization model is actually outputting
This helps debug why only one speaker is being detected
Outputs results to a text file with cluster information
"""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel, AutoProcessor
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv(encoding='utf-8')

class OutputLogger:
    """Helper class to log output to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file = open(log_file, 'w', encoding='utf-8')
    
    def write(self, text):
        """Write to both console and file"""
        print(text, end='')
        self.file.write(text)
    
    def writeln(self, text=''):
        """Write line to both console and file"""
        print(text)
        self.file.write(text + '\n')
    
    def close(self):
        self.file.close()

def test_model_output(audio_path: str, model_name: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2", output_file: str = None):
    """
    Test the model and show its raw outputs
    
    Args:
        audio_path: Path to audio file to test
        model_name: Model name to use
        output_file: Path to output text file (optional, auto-generated if None)
    """
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"model_output_{base_name}_{timestamp}.txt"
    
    # Create logger
    log = OutputLogger(output_file)
    
    log.writeln("=" * 80)
    log.writeln("MODEL OUTPUT TEST")
    log.writeln("=" * 80)
    log.writeln(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.writeln(f"Model: {model_name}")
    log.writeln(f"Audio file: {audio_path}")
    log.writeln(f"Output file: {output_file}")
    log.writeln()
    
    # Get token
    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except:
            token = None
    
    # Load model
    log.writeln("Loading model...")
    model_kwargs = {}
    if token:
        model_kwargs["use_auth_token"] = token
    
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    log.writeln(f"Model loaded on: {device}")
    log.writeln()
    
    # Try to load processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            use_auth_token=token if token else None
        )
        log.writeln("Processor loaded successfully")
    except Exception as e:
        processor = None
        log.writeln(f"Processor not available: {e}")
    log.writeln()
    
    # Load audio
    log.writeln("Loading audio...")
    audio, sample_rate = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        log.writeln("Converted stereo to mono")
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
        log.writeln(f"Resampled to 16kHz")
    
    duration = len(audio) / sample_rate
    log.writeln(f"Audio duration: {duration:.2f} seconds")
    log.writeln(f"Audio samples: {len(audio)}")
    log.writeln()
    
    # Process a small chunk first (first 10 seconds)
    chunk_duration = min(10.0, duration)
    chunk_samples = int(chunk_duration * sample_rate)
    audio_chunk = audio[:chunk_samples]
    
    log.writeln(f"Processing first {chunk_duration:.1f} seconds...")
    log.writeln()
    
    # Prepare input
    with torch.no_grad():
        if processor:
            try:
                inputs = processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt")
                log.writeln("Using processor for input preparation")
            except Exception as e:
                log.writeln(f"Processor failed: {e}")
                processor = None
        
        if not processor:
            # Manual processing
            audio_tensor = torch.from_numpy(audio_chunk).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            inputs = {"input_values": audio_tensor}
            log.writeln("Using manual input preparation")
        
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        log.writeln("\n" + "=" * 80)
        log.writeln("MODEL INPUT INFO")
        log.writeln("=" * 80)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                log.writeln(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            else:
                log.writeln(f"{key}: {type(value)}")
        log.writeln()
        
        # Run inference
        log.writeln("Running inference...")
        outputs = model(**inputs)
        
        log.writeln("\n" + "=" * 80)
        log.writeln("MODEL OUTPUT INFO")
        log.writeln("=" * 80)
        
        # Check output structure
        if isinstance(outputs, dict):
            print("Output is a dictionary")
            print(f"Keys: {list(outputs.keys())}")
            
            if 'logits' in outputs:
                logits = outputs['logits']
                print(f"\nLogits shape: {logits.shape}")
                print(f"Logits dtype: {logits.dtype}")
                print(f"Logits device: {logits.device}")
                
                # Move to CPU for analysis
                logits_cpu = logits.cpu()
                
                # Get probabilities
                probs = torch.softmax(logits_cpu, dim=-1)
                print(f"\nProbabilities shape: {probs.shape}")
                
                # Get predictions
                speaker_preds = torch.argmax(probs, dim=-1)
                speaker_preds_np = speaker_preds.numpy().flatten()
                
                print("\n" + "=" * 80)
                print("PREDICTION ANALYSIS")
                print("=" * 80)
                
                # Number of speaker classes
                num_classes = logits.shape[-1]
                print(f"Number of speaker classes in model: {num_classes}")
                print(f"Number of predictions: {len(speaker_preds_np)}")
                
                # Unique predictions
                unique_preds = np.unique(speaker_preds_np)
                print(f"\nUnique speaker IDs predicted: {unique_preds}")
                
                # Distribution
                unique, counts = np.unique(speaker_preds_np, return_counts=True)
                distribution = dict(zip(unique, counts))
                print(f"\nSpeaker ID distribution:")
                for speaker_id, count in sorted(distribution.items()):
                    percentage = (count / len(speaker_preds_np)) * 100
                    print(f"  Speaker {speaker_id}: {count} frames ({percentage:.1f}%)")
                
                # Probability analysis
                print(f"\nProbability analysis:")
                max_probs = torch.max(probs, dim=-1)[0]
                print(f"  Average max probability: {max_probs.mean().item():.4f}")
                print(f"  Min max probability: {max_probs.min().item():.4f}")
                print(f"  Max max probability: {max_probs.max().item():.4f}")
                
                # Check if probabilities are diverse
                print(f"\nProbability diversity:")
                for class_idx in range(min(5, num_classes)):  # Check first 5 classes
                    class_probs = probs[:, :, class_idx].flatten()
                    avg_prob = class_probs.mean().item()
                    max_prob = class_probs.max().item()
                    print(f"  Class {class_idx}: avg={avg_prob:.4f}, max={max_prob:.4f}")
                
                # Sample some predictions with probabilities
                print(f"\nSample predictions (first 20 frames):")
                for i in range(min(20, len(speaker_preds_np))):
                    pred_id = speaker_preds_np[i]
                    prob = probs.flatten()[i * num_classes + pred_id].item()
                    top2_probs, top2_indices = torch.topk(probs.flatten()[i*num_classes:(i+1)*num_classes], k=min(2, num_classes))
                    print(f"  Frame {i}: Speaker {pred_id} (prob={prob:.4f}), top2={[(int(idx), p.item()) for idx, p in zip(top2_indices, top2_probs)]}")
                
        elif hasattr(outputs, 'logits'):
            log.writeln("Output has logits attribute")
            logits = outputs.logits
            log.writeln(f"Logits shape: {logits.shape}")
            # Similar analysis as above
            logits_cpu = logits.cpu()
            probs = torch.softmax(logits_cpu, dim=-1)
            speaker_preds = torch.argmax(probs, dim=-1)
            speaker_preds_np = speaker_preds.numpy().flatten()
            
            log.writeln(f"\nNumber of speaker classes: {logits.shape[-1]}")
            log.writeln(f"Unique speaker IDs: {np.unique(speaker_preds_np)}")
            unique, counts = np.unique(speaker_preds_np, return_counts=True)
            log.writeln(f"Distribution: {dict(zip(unique, counts))}")
        else:
            log.writeln("Unknown output format!")
            log.writeln(f"Output type: {type(outputs)}")
            log.writeln(f"Output attributes: {dir(outputs)}")
    
    log.writeln("\n" + "=" * 80)
    log.writeln("TEST COMPLETE")
    log.writeln("=" * 80)
    log.writeln(f"\nResults saved to: {output_file}")
    log.close()
    
    return output_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_model_output.py <audio_file_path> [model_name] [output_file]")
        print("\nExample:")
        print("  python test_model_output.py test_audio.wav")
        print("  python test_model_output.py test_audio.wav BUT-FIT/diarizen-wavlm-large-s80-mlc")
        print("  python test_model_output.py test_audio.wav BUT-FIT/diarizen-wavlm-large-s80-md-v2 output.txt")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Optional: specify model name as second argument
    model_name = sys.argv[2] if len(sys.argv) > 2 else "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    
    # Optional: specify output file as third argument
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    result_file = test_model_output(audio_file, model_name, output_file)
    print(f"\n✓ Results saved to: {result_file}")

