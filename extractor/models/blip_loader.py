"""
BLIP model loader for image captioning.
"""

import os
import time
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    BitsAndBytesConfig
)
from typing import Optional, List
from extractor.config import BLIP_MODEL

# Enable HuggingFace Rust-based fast downloader (hf_transfer) for faster model downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


class BLIPLoader:
    """Manages BLIP model loading and inference."""
    
    def __init__(self, model_name: Optional[str] = None, load_in_4bit: bool = False, load_in_8bit: bool = False):
        """
        Initialize BLIP loader.
        
        Args:
            model_name: Optional model name override
            load_in_4bit: Load model in 4-bit quantization (more memory efficient)
            load_in_8bit: Load model in 8-bit quantization
        """
        self.model_name = model_name or BLIP_MODEL
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load(self):
        """Load BLIP or BLIP2 model and processor."""
        if self._loaded:
            return
        
        # Detect if this is a BLIP2 model
        is_blip2 = "blip2" in self.model_name.lower()
        
        print(f"Loading {'BLIP2' if is_blip2 else 'BLIP'} model...")
        start = time.time()
        
        if is_blip2:
            self.processor = Blip2Processor.from_pretrained(self.model_name, use_fast=True)
        else:
            self.processor = BlipProcessor.from_pretrained(self.model_name, use_fast=True)
        
        # Configure quantization
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            print("Using 4-bit quantization")
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Using 8-bit quantization")
        else:
            print("Using 16-bit (float16) precision without quantization")
        
        # Prepare model loading kwargs - default to float16 (16-bit)
        model_kwargs = {
            "dtype": torch.float16,  # 16-bit half precision
            "device_map": "auto",
            "use_safetensors": True
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if is_blip2:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        else:
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        
        elapsed = time.time() - start
        print(f"{'BLIP2' if is_blip2 else 'BLIP'} model loaded in {elapsed:.2f}s")
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

