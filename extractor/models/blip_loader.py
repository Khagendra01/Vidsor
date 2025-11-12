"""
BLIP model loader for image captioning.
"""

import time
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from typing import Optional, List
from extractor.config import BLIP_MODEL


class BLIPLoader:
    """Manages BLIP model loading and inference."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize BLIP loader.
        
        Args:
            model_name: Optional model name override
        """
        self.model_name = model_name or BLIP_MODEL
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load(self):
        """Load BLIP model and processor."""
        if self._loaded:
            return
        
        print("Loading BLIP model...")
        start = time.time()
        
        self.processor = BlipProcessor.from_pretrained(self.model_name, use_fast=True)
        
        # Configure 8-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            use_safetensors=True
        )
        
        elapsed = time.time() - start
        print(f"BLIP model loaded in {elapsed:.2f}s")
        self._loaded = True
    
    def caption(self, image, max_new_tokens: int = 50) -> str:
        """
        Generate caption for an image.
        
        Args:
            image: PIL Image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Caption string
        """
        if not self._loaded:
            self.load()
        
        inputs = self.processor(image, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
    
    def caption_batch(self, images: List, max_new_tokens: int = 50) -> List[str]:
        """
        Generate captions for multiple images in a batch (much faster on GPU).
        
        Args:
            images: List of PIL Images
            max_new_tokens: Maximum tokens to generate per image
            
        Returns:
            List of caption strings (same order as input images)
        """
        if not self._loaded:
            self.load()
        
        if not images:
            return []
        
        # Process images in batch
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode all captions
        captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
        return captions
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

