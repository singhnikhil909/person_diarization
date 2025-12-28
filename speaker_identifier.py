"""
Speaker Identification Module using Resemblyzer
Extracts speaker embeddings and compares them to identify speakers
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SpeakerIdentifier:
    def __init__(self, voice_samples_dir: str = "voice_samples"):
        """
        Initialize the Speaker Identifier
        
        Args:
            voice_samples_dir: Directory containing voice samples organized by speaker name
        """
        self.voice_samples_dir = voice_samples_dir
        self.encoder = VoiceEncoder()
        self.speaker_embeddings = {}
        self.speaker_names = []
        
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
        Preprocess audio for Resemblyzer
        
        Args:
            audio: Audio array
            
        Returns:
            Preprocessed audio
        """
        if audio is None:
            return None
        # Resemblyzer preprocessing
        wav = preprocess_wav(audio)
        return wav
    
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
            embedding = self.encoder.embed_utterance(audio)
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
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

