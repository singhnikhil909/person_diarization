"""
Utility functions for speaker diarization and embedding extraction
"""

import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from pyannote.core import Segment, Annotation
import soundfile as sf
from typing import List, Tuple, Optional, Dict
import warnings
import os
import tempfile
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Suppress torchaudio deprecation warning
warnings.filterwarnings("ignore", message=".*torchaudio._backend.set_audio_backend.*")

# Load environment variables with explicit encoding and file path
try:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=env_path, encoding='utf-8')
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Make sure .env file exists and is UTF-8 encoded")


def load_diarization_pipeline(hf_token: str = None, model_name: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"):
    """
    Load the speaker diarization model from Hugging Face
    
    Args:
        hf_token: Hugging Face authentication token (optional, uses .env or CLI login if not provided)
        model_name: Model name to use. Default: "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
        
    Returns:
        Diarization model and processor
    """
    # Get token from parameter, environment, or Hugging Face CLI
    token = hf_token
    if not token:
        token = os.getenv("HF_TOKEN")
    if not token:
        # Try to get token from Hugging Face CLI login
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    
    print(f"Loading diarization model from Hugging Face...")
    print(f"Model: {model_name}")
    print("Loading transformers-based diarization model...")
    
    # Load model and processor
    model_kwargs = {}
    if token:
        model_kwargs["use_auth_token"] = token
    
    model = AutoModel.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Try to load processor if available
    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            use_auth_token=token if token else None
        )
    except Exception:
        processor = None
        print("Note: Processor not available, will process audio manually")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    if torch.cuda.is_available():
        print("Using GPU for diarization")
    else:
        print("Using CPU for diarization")
    
    return {"model": model, "processor": processor, "model_type": "transformers", "device": device}


def load_embedding_model(hf_token: str = None):
    """
    Load speaker embedding model for identification
    
    Args:
        hf_token: Hugging Face authentication token (optional, uses .env or CLI login if not provided)
        
    Returns:
        Speaker embedding model
    """
    # Get token from parameter, environment, or Hugging Face CLI
    token = hf_token
    if not token:
        token = os.getenv("HF_TOKEN")
    if not token:
        # Try to get token from Hugging Face CLI login
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    
    print("Loading speaker embedding model...")
    from pyannote.audio import Model
    
    # Use use_auth_token if token is available
    if token:
        model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=token
        )
    else:
        # Try without token (will use CLI login if available)
        model = Model.from_pretrained(
            "pyannote/embedding"
        )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        print("Using GPU for embeddings")
    else:
        print("Using CPU for embeddings")
    
    return model


def extract_embedding_from_audio(audio_path: str, embedding_model, segment: Optional[Segment] = None) -> Optional[np.ndarray]:
    """
    Extract speaker embedding from audio file
    
    Args:
        audio_path: Path to audio file
        embedding_model: Pyannote embedding model
        segment: Optional time segment (start, end) in seconds
        
    Returns:
        Speaker embedding vector or None if extraction fails
    """
    try:
        from pyannote.audio import Inference
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(embedding_model, 'device'):
            device = embedding_model.device
        
        inference = Inference(embedding_model, device=device)
        
        if segment is not None:
            # Extract embedding from specific segment
            embedding = inference({
                "audio": audio_path,
                "segment": segment
            })
        else:
            # Extract embedding from entire audio
            embedding = inference(audio_path)
        
        # Convert to numpy if needed
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        # Ensure it's a 1D array
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


def perform_diarization(audio_path: str, pipeline, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None):
    """
    Perform speaker diarization on audio file
    
    Args:
        audio_path: Path to audio file
        pipeline: Diarization model dict (from load_diarization_pipeline)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        
    Returns:
        Diarization results (pyannote.core.Annotation)
    """
    print(f"Performing diarization on {audio_path}...")
    
    model = pipeline["model"]
    model_type = pipeline.get("model_type", "transformers")
    device = pipeline["device"]
    
    # Check if using NVIDIA NeMo model (disabled - using transformers)
    if False and model_type == "nemo":  # Always use transformers pipeline
        print("Using NVIDIA NeMo diarization model...")
        try:
            # NeMo model has built-in diarize() method
            # It handles everything: embeddings, clustering, and speaker identification
            print("Running diarization (this may take a few moments)...")
            
            # NeMo diarize() can return different formats depending on version
            # Try the diarize method
            try:
                predicted_segments = model.diarize(audio=audio_path, batch_size=1)
            except Exception as e:
                print(f"Error calling diarize(): {e}")
                # Try alternative method
                try:
                    predicted_segments = model.transcribe(paths2audio_files=[audio_path], batch_size=1)
                except:
                    raise e
            
            # Debug: Print output type and structure
            print(f"NeMo output type: {type(predicted_segments)}")
            if isinstance(predicted_segments, (list, tuple)) and len(predicted_segments) > 0:
                print(f"First element type: {type(predicted_segments[0])}")
                print(f"First element: {predicted_segments[0]}")
            elif isinstance(predicted_segments, dict):
                print(f"Dictionary keys: {predicted_segments.keys()}")
                print(f"Dictionary sample: {list(predicted_segments.items())[:3]}")
            
            # Convert NeMo output to pyannote Annotation format
            annotation = Annotation()
            
            # NeMo can return different formats - handle all possibilities
            segments_to_process = []
            
            if isinstance(predicted_segments, dict):
                # Check if it's a dictionary with segments
                if 'segments' in predicted_segments:
                    segments_to_process = predicted_segments['segments']
                elif 'diarization' in predicted_segments:
                    segments_to_process = predicted_segments['diarization']
                else:
                    # Try to extract segments from dictionary values
                    for key, value in predicted_segments.items():
                        if isinstance(value, (list, tuple)):
                            segments_to_process.extend(value)
                        elif isinstance(value, dict) and ('start' in value or 'start_time' in value):
                            segments_to_process.append(value)
            elif isinstance(predicted_segments, (list, tuple)):
                segments_to_process = predicted_segments
            elif hasattr(predicted_segments, '__iter__'):
                # Try to iterate over it
                try:
                    segments_to_process = list(predicted_segments)
                except:
                    pass
            
            # Process segments
            for seg in segments_to_process:
                start = None
                end = None
                speaker = None
                
                if isinstance(seg, dict):
                    start = seg.get('start', seg.get('start_time', seg.get('start_sec', None)))
                    end = seg.get('end', seg.get('end_time', seg.get('end_sec', None)))
                    speaker = seg.get('speaker', seg.get('label', seg.get('speaker_label', seg.get('speaker_id', None))))
                elif hasattr(seg, 'start') or hasattr(seg, 'start_time'):
                    start = getattr(seg, 'start', getattr(seg, 'start_time', None))
                    end = getattr(seg, 'end', getattr(seg, 'end_time', None))
                    speaker = getattr(seg, 'speaker', getattr(seg, 'label', getattr(seg, 'speaker_label', None)))
                
                if start is not None and end is not None:
                    # Ensure speaker label format
                    if speaker is None:
                        speaker = 'SPEAKER_00'
                    elif isinstance(speaker, (int, float)):
                        speaker = f"SPEAKER_{int(speaker):02d}"
                    elif not isinstance(speaker, str):
                        speaker = str(speaker)
                    
                    if not speaker.startswith('SPEAKER_'):
                        # Try to extract speaker number
                        if isinstance(speaker, str) and any(char.isdigit() for char in speaker):
                            speaker = f"SPEAKER_{speaker}"
                        else:
                            speaker = f"SPEAKER_{speaker}"
                    
                    annotation[Segment(float(start), float(end))] = speaker
            
            print(f"Diarization complete! Found {len(annotation.labels())} speakers with {len(annotation)} segments.")
            if len(annotation) == 0:
                print("WARNING: No segments found. NeMo output format may be different than expected.")
                print(f"Raw output: {predicted_segments}")
            return annotation
            
        except Exception as e:
            print(f"Error with NeMo diarization: {e}")
            print("Falling back to manual processing...")
            # Fall through to transformers-based processing
    
    # Transformers-based model processing (original code)
    processor = pipeline.get("processor")
    
    # Load audio
    audio, sample_rate = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Prepare input - most diarization models expect 16kHz
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    duration = len(audio) / sample_rate
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Process in chunks to avoid memory issues
    # Start with smaller chunks (10 seconds) to avoid memory issues
    chunk_duration = 10.0  # seconds - start small to avoid OOM
    overlap_duration = 2.0  # seconds
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    
    # Store predictions directly (not full outputs to save memory)
    all_speaker_preds = []
    all_frame_times = []
    
    # Process audio in chunks
    start_sample = 0
    chunk_index = 0
    total_chunks = int(np.ceil((len(audio) - overlap_samples) / (chunk_samples - overlap_samples)))
    
    print(f"Total chunks to process: {total_chunks}")
    
    import time
    start_time = time.time()
    
    max_iterations = len(audio) // max(chunk_samples - overlap_samples, 1) + 20  # Safety limit
    
    while start_sample < len(audio) and chunk_index < max_iterations:
        end_sample = min(start_sample + chunk_samples, len(audio))
        
        # Safety check - if chunk is too small, break
        if end_sample <= start_sample:
            print(f"Warning: Chunk size became too small. Breaking loop.")
            break
            
        audio_chunk = audio[start_sample:end_sample]
        chunk_start_time = start_sample / sample_rate
        
        chunk_start_processing = time.time()
        print(f"Processing chunk {chunk_index + 1}/{total_chunks}: {chunk_start_time:.2f}s - {end_sample/sample_rate:.2f}s (chunk size: {len(audio_chunk)/sample_rate:.1f}s)")
        
        chunk_success = False
        try:
            with torch.no_grad():
                # Prepare input tensor for chunk
                if processor:
                    try:
                        inputs = processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt")
                    except Exception as e:
                        print(f"Processor failed for chunk, using manual processing: {e}")
                        processor_chunk = None
                else:
                    processor_chunk = None
                
                if not processor_chunk:
                    # Manual processing - convert to tensor
                    audio_tensor = torch.from_numpy(audio_chunk).float()
                    # Add batch dimension if needed
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    inputs = {"input_values": audio_tensor}
                
                # Move inputs to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Run inference on chunk
                chunk_inference_start = time.time()
                outputs = model(**inputs)
                inference_time = time.time() - chunk_inference_start
                
                # Immediately process outputs to save memory
                speaker_preds_np = None
                
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                        probs = torch.softmax(logits, dim=-1)
                        speaker_preds = torch.argmax(probs, dim=-1)
                        speaker_preds_np = speaker_preds.cpu().numpy().flatten()
                        # Clear GPU memory
                        del logits, probs, speaker_preds
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    speaker_preds = torch.argmax(probs, dim=-1)
                    speaker_preds_np = speaker_preds.cpu().numpy().flatten()
                    del logits, probs, speaker_preds
                
                # Clear outputs from GPU
                del outputs
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if speaker_preds_np is not None:
                    chunk_dur = len(audio_chunk) / sample_rate
                    num_frames = len(speaker_preds_np)
                    frame_duration = chunk_dur / num_frames if num_frames > 0 else 0.5
                    
                    # Store predictions directly (not the full outputs)
                    for i, pred in enumerate(speaker_preds_np):
                        frame_time = chunk_start_time + (i * frame_duration)
                        all_speaker_preds.append(int(pred))
                        all_frame_times.append(frame_time)
                
                chunk_time = time.time() - chunk_start_processing
                elapsed_time = time.time() - start_time
                avg_time_per_chunk = elapsed_time / (chunk_index + 1)
                remaining_chunks = total_chunks - (chunk_index + 1)
                estimated_remaining = avg_time_per_chunk * remaining_chunks
                
                print(f"  Chunk completed in {chunk_time:.1f}s (inference: {inference_time:.1f}s)")
                print(f"  Progress: {((chunk_index + 1) / total_chunks * 100):.1f}% | "
                      f"Elapsed: {elapsed_time/60:.1f}min | "
                      f"Est. remaining: {estimated_remaining/60:.1f}min")
                
                chunk_success = True
                
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "not enough memory" in error_msg:
                print(f"Memory error on chunk {chunk_index + 1}. Trying smaller chunk size...")
                # Try smaller chunk - but advance position to avoid infinite loop
                old_chunk_samples = chunk_samples
                chunk_samples = max(chunk_samples // 2, int(sample_rate * 2))  # At least 2 seconds
                overlap_samples = max(overlap_samples // 2, int(sample_rate * 0.5))  # At least 0.5 seconds
                
                if chunk_samples == old_chunk_samples or chunk_samples < sample_rate:
                    print(f"ERROR: Cannot reduce chunk size further. Audio too large for available memory.")
                    raise RuntimeError("Audio file too large or model too memory-intensive. Try a shorter audio file or use a machine with more RAM.")
                
                # Recalculate total chunks
                total_chunks = int(np.ceil((len(audio) - overlap_samples) / max(chunk_samples - overlap_samples, 1)))
                print(f"Reduced chunk size to {chunk_samples/sample_rate:.1f}s. New total chunks: {total_chunks}")
                
                # Advance start_sample slightly to avoid infinite loop, then retry
                start_sample = min(start_sample + int(sample_rate * 1), len(audio) - 1)  # Skip 1 second
                if start_sample >= len(audio):
                    break
                continue
            else:
                raise
        
        # Only advance if chunk processing succeeded
        if chunk_success:
            # Move to next chunk (with overlap)
            start_sample = end_sample - overlap_samples
            chunk_index += 1
            
            # Safety check
            if start_sample >= len(audio):
                break
        
        # Clear cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Process collected predictions
    print(f"Processing {len(all_speaker_preds)} frame predictions...")
    
    # Convert outputs to diarization annotation
    annotation = Annotation()
    
    # Process frame-level predictions to create segments
    if all_speaker_preds:
        speaker_preds_np = np.array(all_speaker_preds)
        all_frame_times = np.array(all_frame_times)
        
        # Debug: Check what speaker IDs the model is predicting
        unique_speakers = np.unique(speaker_preds_np)
        print(f"Model predicted speaker IDs: {unique_speakers}")
        print(f"Speaker ID distribution: {dict(zip(*np.unique(speaker_preds_np, return_counts=True)))}")
        
        # Calculate average frame duration
        if len(all_frame_times) > 1:
            frame_duration = np.mean(np.diff(all_frame_times))
        else:
            frame_duration = duration / len(all_speaker_preds) if len(all_speaker_preds) > 0 else 0.5
        
        # Apply smoothing to reduce rapid speaker changes
        # Use majority vote in a small window (reduced window size to preserve speaker changes)
        window_size = max(3, int(0.3 / frame_duration))  # 0.3 second window (reduced from 0.5)
        smoothed_preds = []
        
        for i in range(len(speaker_preds_np)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(speaker_preds_np), i + window_size // 2 + 1)
            window = speaker_preds_np[start_idx:end_idx]
            # Get most common speaker in window
            smoothed_pred = np.bincount(window).argmax()
            smoothed_preds.append(smoothed_pred)
        
        smoothed_preds = np.array(smoothed_preds)
        
        # Debug: Check smoothed predictions
        unique_smoothed = np.unique(smoothed_preds)
        print(f"After smoothing, unique speaker IDs: {unique_smoothed}")
        
        # Create initial segments from smoothed predictions
        initial_segments = []
        current_speaker = None
        segment_start_idx = 0
        
        for i, speaker_id in enumerate(smoothed_preds):
            speaker_label = int(speaker_id)
            if speaker_label != current_speaker:
                if current_speaker is not None:
                    segment_start_time = all_frame_times[segment_start_idx]
                    segment_end_time = all_frame_times[i] if i < len(all_frame_times) else duration
                    segment_duration = segment_end_time - segment_start_time
                    
                    if segment_duration >= 0.3:  # Only segments >= 0.3 seconds
                        initial_segments.append({
                            'start': segment_start_time,
                            'end': segment_end_time,
                            'speaker_id': current_speaker
                        })
                current_speaker = speaker_label
                segment_start_idx = i
        
        # Add final segment
        if current_speaker is not None:
            segment_start_time = all_frame_times[segment_start_idx]
            segment_end_time = duration
            segment_duration = segment_end_time - segment_start_time
            if segment_duration >= 0.3:
                initial_segments.append({
                    'start': segment_start_time,
                    'end': segment_end_time,
                    'speaker_id': current_speaker
                })
        
        print(f"Created {len(initial_segments)} initial segments")
        
        # IMPORTANT: If model only predicted one speaker, create time-based segments instead
        # This ensures we can still separate speakers using embeddings
        if len(unique_speakers) == 1 and len(initial_segments) == 1:
            print("WARNING: Model only predicted one speaker class. Creating time-based segments for embedding clustering...")
            # Create segments based on time windows (e.g., every 3-5 seconds)
            segment_duration = 5.0  # seconds per segment
            time_based_segments = []
            current_time = 0.0
            segment_idx = 0
            
            while current_time < duration:
                seg_end = min(current_time + segment_duration, duration)
                if seg_end - current_time >= 1.0:  # At least 1 second
                    time_based_segments.append({
                        'start': current_time,
                        'end': seg_end,
                        'speaker_id': 0  # Placeholder, will be replaced by clustering
                    })
                current_time = seg_end
                segment_idx += 1
            
            if len(time_based_segments) > 1:
                print(f"Created {len(time_based_segments)} time-based segments for clustering")
                initial_segments = time_based_segments
        
        # Now cluster segments by speaker similarity using embeddings
        # Load embedding model for speaker clustering
        try:
            print("Loading embedding model for speaker clustering...")
            embedding_model = load_embedding_model()
            from pyannote.audio import Inference
            embedding_inference = Inference(embedding_model, device=device)
            
            # Extract embeddings for each segment
            print("Extracting speaker embeddings from segments...")
            segment_embeddings = []
            valid_segments = []
            
            for seg in initial_segments:
                try:
                    # Extract audio segment
                    start_sample = int(seg['start'] * sample_rate)
                    end_sample = int(seg['end'] * sample_rate)
                    seg_audio = audio[start_sample:end_sample]
                    
                    # Skip if segment is too short (need at least 1 second for good embedding)
                    if len(seg_audio) < sample_rate * 1.0:  # Less than 1 second
                        continue
                    
                    # Save segment to temporary file for embedding extraction
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        sf.write(tmp_file.name, seg_audio, sample_rate)
                        tmp_path = tmp_file.name
                    
                    # Extract embedding
                    embedding = embedding_inference(tmp_path)
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    segment_embeddings.append(embedding)
                    valid_segments.append(seg)
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    print(f"Warning: Could not extract embedding for segment {seg['start']:.2f}-{seg['end']:.2f}: {e}")
                    continue
            
            if len(segment_embeddings) >= 2:  # Need at least 2 segments to cluster
                print(f"Extracted {len(segment_embeddings)} embeddings. Clustering speakers...")
                
                # Cluster embeddings using cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                from sklearn.cluster import AgglomerativeClustering
                
                embeddings_matrix = np.array(segment_embeddings)
                
                # Determine number of clusters (speakers)
                # ALWAYS force at least 2 clusters if we have multiple segments and user specified speakers
                if min_speakers:
                    n_clusters = max(min_speakers, 2) if len(segment_embeddings) >= 2 else min_speakers
                elif max_speakers:
                    n_clusters = min(max_speakers, max(2, len(segment_embeddings)))
                else:
                    # Default: try to detect 2 speakers if we have enough segments
                    # Use silhouette score to determine optimal number if we have enough segments
                    if len(segment_embeddings) >= 4:
                        n_clusters = 2  # Default to 2 speakers
                    else:
                        n_clusters = min(2, len(segment_embeddings))
                
                n_clusters = max(1, min(n_clusters, len(segment_embeddings)))
                print(f"Clustering {len(segment_embeddings)} segments into {n_clusters} speakers")
                
                # Use hierarchical clustering with cosine distance
                # Convert cosine similarity to distance: distance = 1 - similarity
                distance_matrix = 1 - cosine_similarity(embeddings_matrix)
                
                # Use 'ward' linkage for better separation when we know there are 2 speakers
                linkage_method = 'ward' if n_clusters == 2 else 'average'
                
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage=linkage_method
                )
                cluster_labels = clustering.fit_predict(distance_matrix)
                
                # Debug: Show cluster distribution
                unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
                print(f"Cluster distribution: {dict(zip(unique_clusters, cluster_counts))}")
                
                # Assign clustered speaker IDs
                for i, seg in enumerate(valid_segments):
                    seg['clustered_speaker'] = int(cluster_labels[i])
                
                print(f"Clustered into {n_clusters} speakers")
                
                # Create final annotation with clustered speaker labels
                for seg in valid_segments:
                    speaker_label = f"SPEAKER_{seg['clustered_speaker']:02d}"
                    annotation[Segment(seg['start'], seg['end'])] = speaker_label
            elif len(segment_embeddings) == 1:
                # Only one segment - can't cluster
                print("Warning: Only one segment extracted. Cannot cluster speakers.")
                seg = valid_segments[0]
                annotation[Segment(seg['start'], seg['end'])] = "SPEAKER_00"
            else:
                # Fallback: use initial segments without clustering
                print("Warning: Could not extract enough embeddings. Using initial segments without clustering.")
                for seg in initial_segments:
                    speaker_label = f"SPEAKER_{seg['speaker_id']:02d}"
                    annotation[Segment(seg['start'], seg['end'])] = speaker_label
                    
        except Exception as e:
            print(f"Warning: Could not perform speaker clustering: {e}")
            import traceback
            traceback.print_exc()
            print("Using initial segments without clustering...")
            # Fallback: use initial segments without clustering
            for seg in initial_segments:
                speaker_label = f"SPEAKER_{seg['speaker_id']:02d}"
                annotation[Segment(seg['start'], seg['end'])] = speaker_label
    else:
        # Fallback: create a simple annotation covering entire audio
        print("Warning: Model output format not recognized. Creating placeholder annotation.")
        annotation[Segment(0, duration)] = "SPEAKER_00"
    
    return annotation


def format_time(seconds: float) -> str:
    """Format time in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

