"""
Speaker Database Manager
Stores and manages speaker embeddings for identification
"""

import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import json


class SpeakerDatabase:
    """Manages speaker embeddings and identification"""
    
    def __init__(self, db_path: str = "speaker_db.pkl"):
        self.db_path = db_path
        self.speakers: Dict[str, np.ndarray] = {}  # name -> embedding
        self.speaker_metadata: Dict[str, Dict] = {}  # name -> metadata
        self.load_database()
    
    def load_database(self):
        """Load speaker database from file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.speakers = data.get('speakers', {})
                    self.speaker_metadata = data.get('metadata', {})
                print(f"Loaded {len(self.speakers)} speakers from database")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.speakers = {}
                self.speaker_metadata = {}
    
    def save_database(self):
        """Save speaker database to file"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'speakers': self.speakers,
                    'metadata': self.speaker_metadata
                }, f)
            print(f"Saved {len(self.speakers)} speakers to database")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def add_speaker(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """
        Add a speaker to the database
        
        Args:
            name: Speaker name/identifier
            embedding: Speaker embedding vector
            metadata: Optional metadata (e.g., source file, date)
        """
        if embedding.ndim > 1:
            # Average if multiple embeddings provided
            embedding = np.mean(embedding, axis=0)
        
        self.speakers[name] = embedding
        self.speaker_metadata[name] = metadata or {}
        self.save_database()
        print(f"Added speaker '{name}' to database")
    
    def add_speaker_from_audio(self, name: str, audio_path: str, embedding_model, metadata: Optional[Dict] = None):
        """
        Add a speaker from an audio file
        
        Args:
            name: Speaker name/identifier
            audio_path: Path to audio file with speaker's voice
            embedding_model: Model to extract embeddings
            metadata: Optional metadata
        """
        from diarization_utils import extract_embedding_from_audio
        
        embedding = extract_embedding_from_audio(audio_path, embedding_model)
        if embedding is not None:
            self.add_speaker(name, embedding, metadata)
            return True
        return False
    
    def identify_speaker(self, embedding: np.ndarray, threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """
        Identify a speaker from an embedding
        
        Args:
            embedding: Speaker embedding vector
            threshold: Similarity threshold for identification
            
        Returns:
            Tuple of (speaker_name, similarity_score) or (None, score) if unknown
        """
        if len(self.speakers) == 0:
            return None, 0.0
        
        if embedding.ndim > 1:
            embedding = np.mean(embedding, axis=0)
        
        # Calculate cosine similarity with all speakers
        similarities = {}
        for name, db_embedding in self.speakers.items():
            # Ensure embeddings are 2D for cosine_similarity
            emb1 = embedding.reshape(1, -1)
            emb2 = db_embedding.reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            similarities[name] = similarity
        
        # Find best match
        best_match = max(similarities.items(), key=lambda x: x[1])
        name, score = best_match
        
        if score >= threshold:
            return name, score
        else:
            return None, score
    
    def list_speakers(self) -> List[str]:
        """List all registered speakers"""
        return list(self.speakers.keys())
    
    def remove_speaker(self, name: str):
        """Remove a speaker from the database"""
        if name in self.speakers:
            del self.speakers[name]
            if name in self.speaker_metadata:
                del self.speaker_metadata[name]
            self.save_database()
            print(f"Removed speaker '{name}' from database")
        else:
            print(f"Speaker '{name}' not found in database")
    
    def get_speaker_count(self) -> int:
        """Get number of registered speakers"""
        return len(self.speakers)

