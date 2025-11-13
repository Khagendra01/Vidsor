"""
BakLLaVA model loader for image captioning via Ollama.
"""

import io
import time
import base64
import requests
from typing import Optional, List, Tuple
from PIL import Image
from extractor.config import OLLAMA_URL, OLLAMA_MODEL


# Optimized prompt for BakLLaVA to prevent hallucinations about multiple images
VISION_MODEL_PROMPT = """You are analyzing a single video frame. Describe what you see in this ONE image.

Focus on:
- Objects, people, and actions visible in the scene
- The setting and environment
- Spatial relationships and positions
- Any notable visual details

Important: This is ONE single frame from a video. Do not mention multiple images or batches. Describe only what is visible in this single frame."""


class BakLLaVALoader:
    """Manages BakLLaVA model loading and inference via Ollama."""
    
    def __init__(self, ollama_url: Optional[str] = None, ollama_model: Optional[str] = None):
        """
        Initialize BakLLaVA loader.
        
        Args:
            ollama_url: Optional Ollama URL override
            ollama_model: Optional Ollama model override
        """
        self.ollama_url = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self._loaded = True  # Always "loaded" since it's via API
    
    def load(self):
        """No-op for API-based model (always available)."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is available."""
        return True
    
    def caption(self, image: Image.Image, max_new_tokens: int = 200) -> str:
        """
        Generate caption for a single image via Ollama.
        
        Args:
            image: PIL Image
            max_new_tokens: Maximum tokens to generate (not used for Ollama, kept for compatibility)
            
        Returns:
            Caption string
        """
        caption, _ = self._process_with_bakllava(image)
        return caption
    
    def _process_with_bakllava(self, image: Image.Image) -> Tuple[str, float]:
        """
        Process a single image with BakLLaVA via Ollama.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (caption, processing_time_in_seconds)
        """
        # Encode image to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Prepare request payload
        payload = {
            'model': self.ollama_model,
            'prompt': VISION_MODEL_PROMPT,
            'images': [img_base64],  # Single image in list (Ollama API format)
            'stream': False,
            'options': {
                'temperature': 0.2,  # Lower temperature = more deterministic
                'top_p': 0.9,  # Nucleus sampling
                'num_predict': 200,  # Max tokens in response
            }
        }
        
        start = time.time()
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            processing_time = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                caption = result.get('response', '').strip()
                return caption, processing_time
            else:
                return f"Error: {response.status_code}", processing_time
        except Exception as e:
            processing_time = time.time() - start
            return f"Error: {str(e)}", processing_time
    
    def caption_batch(self, images: List[Image.Image], max_new_tokens: int = 200) -> List[str]:
        """
        Generate captions for multiple images.
        
        Note: Ollama doesn't support true batch processing, so this processes
        images sequentially. For parallel processing, use the parallel captioning
        in the segment tree generator.
        
        Args:
            images: List of PIL Images
            max_new_tokens: Maximum tokens to generate per image
            
        Returns:
            List of caption strings (same order as input images)
        """
        captions = []
        for image in images:
            caption, _ = self._process_with_bakllava(image)
            captions.append(caption)
        return captions

