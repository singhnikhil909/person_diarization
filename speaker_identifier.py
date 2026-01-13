"""
Speaker Identification Module using Resemblyzer or Pyannote
Extracts speaker embeddings and compares them to identify speakers
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import torch
warnings.filterwarnings('ignore')

# Import Resemblyzer
from resemblyzer import VoiceEncoder, preprocess_wav

# Import Pyannote
omegaconf_available = False
try:
    from pyannote.audio import Model
    from pyannote.audio import Inference
    PYANNOTE_AVAILABLE = True

    # Set up safe globals for PyTorch 2.6+ compatibility
    # This must be done before loading any pyannote models
    try:
        if hasattr(torch.serialization, 'add_safe_globals'):
            from collections import OrderedDict

            # Start with basic classes
            safe_globals = [OrderedDict]

            # Add PyTorch Lightning callbacks and utilities
            try:
                import pytorch_lightning.callbacks.early_stopping
                import pytorch_lightning.callbacks.model_checkpoint
                import pytorch_lightning.callbacks.lr_monitor
                safe_globals.extend([
                    pytorch_lightning.callbacks.early_stopping.EarlyStopping,
                    pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint,
                    pytorch_lightning.callbacks.lr_monitor.LearningRateMonitor,
                ])
            except ImportError:
                pass

            # Add OmegaConf classes (used by pyannote for configuration)
            try:
                import omegaconf
                safe_globals.extend([
                    omegaconf.dictconfig.DictConfig,
                    omegaconf.listconfig.ListConfig,
                    omegaconf.DictConfig,
                    omegaconf.ListConfig,
                ])
                # Store for later use in context manager
                omegaconf_available = True
            except ImportError:
                omegaconf_available = False

            # Add other common classes that might be in checkpoints
            try:
                import argparse
                safe_globals.append(argparse.Namespace)
            except:
                pass

            # Register all safe globals
            if safe_globals:
                torch.serialization.add_safe_globals(safe_globals)
                print(f"Configured {len(safe_globals)} safe globals for PyTorch 2.6+")
    except Exception as e:
        print(f"Note: Could not configure PyTorch safe globals: {e}")

except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Pyannote not available. Install with: pip install pyannote.audio")


class SpeakerIdentifier:
    def __init__(self, voice_samples_dir: str = "voice_samples", model_type: str = "resemblyzer"):
        """
        Initialize the Speaker Identifier
        
        Args:
            voice_samples_dir: Directory containing voice samples organized by speaker name
            model_type: Type of model to use - "resemblyzer" or "pyannote"
        """
        self.voice_samples_dir = voice_samples_dir
        self.model_type = model_type.lower()
        self.speaker_embeddings = {}
        self.speaker_names = []
        
        # Initialize encoder based on model type
        if self.model_type == "resemblyzer":
            self.encoder = VoiceEncoder()
            self.inference = None
        elif self.model_type == "pyannote":
            if not PYANNOTE_AVAILABLE:
                raise ImportError("Pyannote.audio is not installed. Install with: pip install pyannote.audio")
            # Load pyannote embedding model
            # Using the community model for speaker embedding
            self.encoder = None
            try:
                # Use the embedding model from pyannote
                # Safe globals are configured at module level for PyTorch 2.6+ compatibility
                
                # For PyTorch 2.6+, we need to handle the weights_only security restriction
                if hasattr(torch, '__version__') and tuple(map(int, torch.__version__.split('.')[:2])) >= (2, 6):
                    try:
                        # First try: Use safe_globals context manager if available
                        if omegaconf_available:
                            import omegaconf
                            with torch.serialization.safe_globals([omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig]):
                                model = Model.from_pretrained("pyannote/embedding",
                                                             use_auth_token=os.environ.get("HF_TOKEN"))
                        else:
                            raise AttributeError("OmegaConf not available")
                    except (AttributeError, Exception):
                        # Second try: Direct torch.load patching
                        original_torch_load = torch.load

                        def patched_torch_load(*args, **kwargs):
                            # Force weights_only=False for pyannote models
                            kwargs['weights_only'] = False
                            return original_torch_load(*args, **kwargs)

                        torch.load = patched_torch_load
                        try:
                            model = Model.from_pretrained("pyannote/embedding",
                                                         use_auth_token=os.environ.get("HF_TOKEN"))
                        finally:
                            torch.load = original_torch_load
                else:
                    model = Model.from_pretrained("pyannote/embedding",
                                                 use_auth_token=os.environ.get("HF_TOKEN"))
                
                self.inference = Inference(model, window="whole")
                print("Loaded pyannote embedding model (community 3.1)")
            except Exception as e:
                print(f"Error loading pyannote model: {e}")
                print("Note: You may need to accept the user agreement at https://huggingface.co/pyannote/embedding")
                print("and set HF_TOKEN environment variable with your Hugging Face token")
                raise
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'resemblyzer' or 'pyannote'")
        
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Load audio file and convert to mono, resample if needed
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (Resemblyzer uses 16kHz)
            
        Returns:
            Audio array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for the selected model
        
        Args:
            audio: Audio array
            
        Returns:
            Preprocessed audio
        """
        if audio is None:
            return None
        
        if self.model_type == "resemblyzer":
            # Resemblyzer preprocessing
            wav = preprocess_wav(audio)
            return wav
        elif self.model_type == "pyannote":
            # Pyannote expects normalized audio
            # Normalize to [-1, 1] range if not already
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            return audio
        
        return audio
    
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract speaker embedding from audio
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Speaker embedding vector
        """
        if audio is None or len(audio) == 0:
            return None
        
        # Ensure minimum length (at least 1 second)
        min_length = 16000  # 1 second at 16kHz
        if len(audio) < min_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        try:
            if self.model_type == "resemblyzer":
                embedding = self.encoder.embed_utterance(audio)
                return embedding
            elif self.model_type == "pyannote":
                # Convert audio to the format expected by pyannote
                # Pyannote expects (channel, samples) format
                if audio.ndim == 1:
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
                else:
                    audio_tensor = torch.from_numpy(audio).float()
                
                # Get embedding from pyannote
                # The inference expects a dictionary with 'waveform' and 'sample_rate'
                embedding = self.inference({
                    "waveform": audio_tensor,
                    "sample_rate": 16000
                })
                
                # Convert to numpy if it's a tensor
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                
                # Flatten if needed
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                return embedding
        except Exception as e:
            print(f"Error extracting embedding with {self.model_type}: {e}")
            return None
    
    def load_reference_voices(self) -> Dict[str, np.ndarray]:
        """
        Load all reference voice samples and create embeddings
        
        Returns:
            Dictionary mapping speaker names to their average embeddings
        """
        speaker_embeddings = {}
        
        if not os.path.exists(self.voice_samples_dir):
            print(f"Voice samples directory not found: {self.voice_samples_dir}")
            return speaker_embeddings
        
        # Iterate through each speaker folder
        for speaker_name in os.listdir(self.voice_samples_dir):
            speaker_path = os.path.join(self.voice_samples_dir, speaker_name)
            
            if not os.path.isdir(speaker_path):
                continue
            
            # Skip hidden directories
            if speaker_name.startswith('.'):
                continue
            
            embeddings_list = []
            
            # Process all audio files for this speaker
            for audio_file in os.listdir(speaker_path):
                if audio_file.startswith('.'):
                    continue
                    
                audio_path = os.path.join(speaker_path, audio_file)
                
                # Skip directories (like __MACOSX)
                if os.path.isdir(audio_path):
                    continue
                
                print(f"Processing {speaker_name}/{audio_file}...")
                
                # Load and preprocess audio
                audio = self.load_audio(audio_path)
                if audio is None:
                    continue
                
                wav = self.preprocess_audio(audio)
                if wav is None:
                    continue
                
                # Extract embedding
                embedding = self.extract_embedding(wav)
                if embedding is not None:
                    embeddings_list.append(embedding)
            
            if embeddings_list:
                # Average embeddings for this speaker (better representation)
                avg_embedding = np.mean(embeddings_list, axis=0)
                speaker_embeddings[speaker_name] = avg_embedding
                print(f"Loaded {len(embeddings_list)} samples for {speaker_name}")
        
        self.speaker_embeddings = speaker_embeddings
        self.speaker_names = list(speaker_embeddings.keys())
        return speaker_embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Normalize to 0-1 range (cosine similarity is -1 to 1, but embeddings are usually positive)
        similarity = (similarity + 1) / 2
        return float(similarity)
    
    def identify_speakers_in_audio(self, audio_path: str, 
                                   segment_duration: float = 3.0,
                                   overlap: float = 1.0) -> List[Dict]:
        """
        Identify speakers in an audio file by segmenting and comparing
        
        Args:
            audio_path: Path to audio file to analyze
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            List of dictionaries with speaker matches and confidence scores
        """
        if not self.speaker_embeddings:
            print("No reference voices loaded. Loading reference voices...")
            self.load_reference_voices()
        
        if not self.speaker_embeddings:
            return []
        
        # Load the audio file
        print(f"Loading audio: {audio_path}")
        audio = self.load_audio(audio_path)
        if audio is None:
            return []
        
        sr = 16000  # Resemblyzer uses 16kHz
        segment_samples = int(segment_duration * sr)
        overlap_samples = int(overlap * sr)
        step_samples = segment_samples - overlap_samples
        
        results = []
        
        # Process audio in segments
        for start_idx in range(0, len(audio), step_samples):
            end_idx = min(start_idx + segment_samples, len(audio))
            segment = audio[start_idx:end_idx]
            
            if len(segment) < 16000:  # Skip segments shorter than 1 second
                continue
            
            # Preprocess and extract embedding
            wav = self.preprocess_audio(segment)
            if wav is None:
                continue
            
            embedding = self.extract_embedding(wav)
            if embedding is None:
                continue
            
            # Compare with all reference speakers
            segment_results = {
                'start_time': start_idx / sr,
                'end_time': end_idx / sr,
                'matches': []
            }
            
            for speaker_name, ref_embedding in self.speaker_embeddings.items():
                similarity = self.cosine_similarity(embedding, ref_embedding)
                segment_results['matches'].append({
                    'speaker': speaker_name,
                    'similarity': similarity,
                    'percentage': similarity * 100
                })
            
            # Sort by similarity
            segment_results['matches'].sort(key=lambda x: x['similarity'], reverse=True)
            results.append(segment_results)
        
        return results
    
    def get_overall_speaker_matches(self, audio_path: str) -> List[Dict]:
        """
        Get overall speaker matches for the entire audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speaker matches with average similarity percentages
        """
        segment_results = self.identify_speakers_in_audio(audio_path)
        
        if not segment_results:
            return []
        
        # Aggregate results across all segments
        speaker_scores = {}
        
        for segment in segment_results:
            for match in segment['matches']:
                speaker = match['speaker']
                similarity = match['similarity']
                
                if speaker not in speaker_scores:
                    speaker_scores[speaker] = []
                speaker_scores[speaker].append(similarity)
        
        # Calculate average similarity for each speaker
        overall_matches = []
        for speaker, scores in speaker_scores.items():
            avg_similarity = np.mean(scores)
            max_similarity = np.max(scores)
            overall_matches.append({
                'speaker': speaker,
                'average_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'average_percentage': avg_similarity * 100,
                'max_percentage': max_similarity * 100,
                'segments_found': len(scores)
            })
        
        # Sort by average similarity
        overall_matches.sort(key=lambda x: x['average_similarity'], reverse=True)
        
        return overall_matches

