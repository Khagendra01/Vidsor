"""
LLaVA processing module for unified descriptions via Ollama.
"""

import io
import time
import base64
import requests
from typing import Dict, List, Optional
from PIL import Image

from agent.prompts.llava_prompts import get_llava_prompt
from extractor.config import OLLAMA_URL, OLLAMA_MODEL


class LLaVAProcessor:
    """Processes descriptions with LLaVA via Ollama."""
    
    def __init__(self, ollama_url: Optional[str] = None, ollama_model: Optional[str] = None, use_images: bool = False):
        """
        Initialize LLaVA processor.
        
        Args:
            ollama_url: Optional Ollama URL override
            ollama_model: Optional Ollama model override
            use_images: Whether to send images to LLaVA
        """
        self.ollama_url = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.use_images = use_images
    
    def process(
        self,
        blip_descriptions: List[str],
        detection_summary: str,
        images: Optional[List[Image.Image]] = None
    ) -> Dict:
        """
        Process with LLaVA via Ollama.
        
        Args:
            blip_descriptions: List of BLIP text descriptions
            detection_summary: Summary of object detections
            images: Optional list of PIL Images to send to LLaVA
            
        Returns:
            Dictionary with unified description and metadata
        """
        # Build image descriptions list
        descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(blip_descriptions)])
        
        # Use simple prompt that only asks for what can be inferred from the data
        prompt = get_llava_prompt(
            descriptions_text=descriptions_text,
            detection_summary=detection_summary,
            include_technical=False,  # Use new simple prompt (default)
            short=False
        )
        
        # Prepare request payload
        payload = {
            'model': self.ollama_model,
            'prompt': prompt,
            'stream': False
        }
        
        # Add images if provided and use_images is enabled
        if images and self.use_images:
            encoded_images = []
            for img in images:
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                encoded_images.append(img_base64)
            payload['images'] = encoded_images
        
        start = time.time()
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60
            )
            processing_time = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                metadata = {
                    "model": self.ollama_model,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "processing_time": round(processing_time, 2)
                }
                
                if self.use_images and images:
                    metadata["images_sent"] = len(images)
                    metadata["note"] = "LLaVA processed with both text and images"
                else:
                    metadata["note"] = "LLaVA processed with text only"
                
                return {
                    "unified_description": result.get('response', ''),
                    "llava_metadata": metadata
                }
            else:
                return {
                    "unified_description": f"Error: {response.status_code}",
                    "llava_metadata": {
                        "model": self.ollama_model,
                        "processing_time": round(processing_time, 2)
                    }
                }
        except Exception as e:
            processing_time = time.time() - start
            return {
                "unified_description": f"Error: {str(e)}",
                "llava_metadata": {
                    "model": self.ollama_model,
                    "processing_time": round(processing_time, 2),
                    "error": str(e)
                }
            }

