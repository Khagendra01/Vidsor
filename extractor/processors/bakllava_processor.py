"""
BakLLaVA processing module for unified descriptions via Ollama.
This replaces both BLIP and GPT-4o-mini - BakLLaVA's response is used directly.
"""

import io
import time
import base64
import requests
from typing import Dict, List, Optional
from PIL import Image
from extractor.config import OLLAMA_URL, OLLAMA_MODEL


class BakLLaVAProcessor:
    """Processes frames with BakLLaVA via Ollama and uses response directly as unified description."""
    
    def __init__(self, ollama_url: Optional[str] = None, ollama_model: Optional[str] = None, use_images: bool = False):
        """
        Initialize BakLLaVA processor.
        
        Args:
            ollama_url: Optional Ollama URL override
            ollama_model: Optional Ollama model override
            use_images: Whether to send images (always True for BakLLaVA)
        """
        self.ollama_url = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.use_images = True  # BakLLaVA always uses images
    
    def process(
        self,
        frame_images: List[Image.Image],
        detection_summary: str
    ) -> Dict:
        """
        Process frames with BakLLaVA via Ollama.
        
        Args:
            frame_images: List of PIL Images to process
            detection_summary: Summary of object detections (for context in prompt)
            
        Returns:
            Dictionary with unified description and metadata
        """
        if not frame_images:
            return {
                "unified_description": detection_summary or "No images available",
                "bakllava_metadata": {
                    "model": self.ollama_model,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "processing_time": 0,
                    "note": "No images provided"
                }
            }
        
        # For multiple frames, we'll process the middle frame or combine them
        # Since BakLLaVA processes one image at a time, we'll use the middle frame
        # or the first frame if only one is provided
        image_to_process = frame_images[len(frame_images) // 2] if len(frame_images) > 1 else frame_images[0]
        
        # Build prompt that includes detection context
        prompt = f"""You are analyzing a video frame. Describe what you see in this image.

Context from object detection: {detection_summary}

Focus on:
- Objects, people, and actions visible in the scene
- The setting and environment
- Spatial relationships and positions
- Any notable visual details
- How the detected objects relate to the scene

Important: This is ONE single frame from a video. Describe only what is visible in this single frame."""
        
        # Encode image to base64
        img_buffer = io.BytesIO()
        image_to_process.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Prepare request payload
        payload = {
            'model': self.ollama_model,
            'prompt': prompt,
            'images': [img_base64],
            'stream': False,
            'options': {
                'temperature': 0.2,
                'top_p': 0.9,
                'num_predict': 300,  # More tokens for richer descriptions
            }
        }
        
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
                unified_description = result.get('response', '').strip()
                
                metadata = {
                    "model": self.ollama_model,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "processing_time": round(processing_time, 2),
                    "images_processed": len(frame_images),
                    "note": "BakLLaVA processed with image and detection context"
                }
                
                return {
                    "unified_description": unified_description,
                    "bakllava_metadata": metadata
                }
            else:
                return {
                    "unified_description": f"Error: {response.status_code}",
                    "bakllava_metadata": {
                        "model": self.ollama_model,
                        "processing_time": round(processing_time, 2),
                        "error": f"HTTP {response.status_code}"
                    }
                }
        except Exception as e:
            processing_time = time.time() - start
            return {
                "unified_description": f"Error: {str(e)}",
                "bakllava_metadata": {
                    "model": self.ollama_model,
                    "processing_time": round(processing_time, 2),
                    "error": str(e)
                }
            }

