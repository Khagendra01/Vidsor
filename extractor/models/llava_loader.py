"""
LLaVA model loader for image captioning with batch processing support.
"""

import os
import time
import torch
from transformers import (
    LlavaProcessor, LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
from typing import Optional, List
from PIL import Image
from extractor.config import LLAVA_MODEL

# Enable HuggingFace Rust-based fast downloader (hf_transfer) for faster model downloads
# This speeds up downloads 10-50x for large models like LLaVA
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


class LLaVALoader:
    """Manages LLaVA model loading and inference with batch processing."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        load_in_4bit: bool = False, 
        load_in_8bit: bool = False,
        prompt: str = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    ):
        """
        Initialize LLaVA loader.
        
        Args:
            model_name: Optional model name override (default: llava-hf/llava-1.5-7b-hf)
            load_in_4bit: Load model in 4-bit quantization (more memory efficient)
            load_in_8bit: Load model in 8-bit quantization
            prompt: Prompt template for image description
        """
        self.model_name = model_name or LLAVA_MODEL
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.prompt = prompt
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load(self):
        """Load LLaVA model and processor."""
        if self._loaded:
            return
        
        print(f"Loading LLaVA model: {self.model_name}...")
        start = time.time()
        
        # Load processor
        self.processor = LlavaProcessor.from_pretrained(self.model_name)
        
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
        
        # Load model
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        elapsed = time.time() - start
        print(f"LLaVA model loaded in {elapsed:.2f}s")
        self._loaded = True
    
    def caption(self, image: Image.Image, max_new_tokens: int = 200, prompt: Optional[str] = None) -> str:
        """
        Generate caption for an image.
        
        Args:
            image: PIL Image
            max_new_tokens: Maximum tokens to generate
            prompt: Optional prompt override
            
        Returns:
            Caption string
        """
        if not self._loaded:
            self.load()
        
        # Use provided prompt or default
        prompt_text = prompt or self.prompt
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # Deterministic output
            )
        
        # Decode response (skip the prompt part)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        # Remove the prompt from the response
        if prompt_text in caption:
            caption = caption.split(prompt_text)[-1].strip()
        
        return caption
    
    def caption_batch(self, images: List[Image.Image], max_new_tokens: int = 200, prompt: Optional[str] = None) -> List[str]:
        """
        Generate captions for multiple images in a batch (much faster on GPU).
        
        Args:
            images: List of PIL Images
            max_new_tokens: Maximum tokens to generate per image
            prompt: Optional prompt override
            
        Returns:
            List of caption strings (same order as input images)
        """
        if not self._loaded:
            self.load()
        
        if not images:
            return []
        
        # Use provided prompt or default
        prompt_text = prompt or self.prompt
        
        # Prepare batch inputs
        # For batch processing, we need to create a list of prompts (one per image)
        prompts = [prompt_text] * len(images)
        
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # Deterministic output
            )
        
        # Decode all captions
        captions = []
        for i, output in enumerate(outputs):
            caption = self.processor.decode(output, skip_special_tokens=True)
            # Remove the prompt from the response
            if prompt_text in caption:
                caption = caption.split(prompt_text)[-1].strip()
            captions.append(caption)
        
        return captions
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

