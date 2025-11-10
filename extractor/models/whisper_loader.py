"""
Whisper model loader for audio transcription.
"""

import time
import whisper
from typing import Optional
from extractor.config import WHISPER_MODEL


class WhisperLoader:
    """Manages Whisper model loading and inference."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Whisper loader.
        
        Args:
            model_name: Optional model name override
        """
        self.model_name = model_name or WHISPER_MODEL
        self.model = None
        self._loaded = False
    
    def load(self):
        """Load Whisper model."""
        if self._loaded:
            return
        
        print("Loading Whisper model...")
        start = time.time()
        
        self.model = whisper.load_model(self.model_name)
        
        elapsed = time.time() - start
        print(f"Whisper model loaded in {elapsed:.2f}s")
        self._loaded = True
    
    def transcribe(self, audio_path: str, language: str = "en", task: str = "transcribe") -> dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")
            task: Task type - "transcribe" or "translate"
            
        Returns:
            Transcription result dictionary
        """
        if not self._loaded:
            self.load()
        
        return self.model.transcribe(audio_path, language=language, task=task)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

