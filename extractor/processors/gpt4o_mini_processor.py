"""
GPT-4o-mini processing module for unified descriptions via OpenAI API.
"""

import io
import time
import base64
import os
from typing import Dict, List, Optional
from PIL import Image
from openai import OpenAI
from openai import RateLimitError, APIError

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not installed, will use environment variables only

from agent.prompts.llava_prompts import get_llava_prompt


class GPT4oMiniProcessor:
    """Processes descriptions with GPT-4o-mini via OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, use_images: bool = False):
        """
        Initialize GPT-4o-mini processor.
        
        Args:
            api_key: Optional OpenAI API key override (defaults to OPENAI_API_KEY env var)
            use_images: Whether to send images to GPT-4o-mini
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.use_images = use_images
        self.model = "gpt-4o-mini"
    
    def process(
        self,
        blip_descriptions: List[str],
        detection_summary: str,
        images: Optional[List[Image.Image]] = None
    ) -> Dict:
        """
        Process with GPT-4o-mini via OpenAI API.
        
        Args:
            blip_descriptions: List of BLIP text descriptions
            detection_summary: Summary of object detections
            images: Optional list of PIL Images to send to GPT-4o-mini
            
        Returns:
            Dictionary with unified description and metadata
        """
        # Build image descriptions list
        descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(blip_descriptions)])
        
        # Use enhanced prompt from prompt.py
        prompt = get_llava_prompt(
            descriptions_text=descriptions_text,
            detection_summary=detection_summary,
            include_technical=True,
            short=False
        )
        
        # Prepare messages
        messages = [{"role": "user", "content": []}]
        
        # Add text content
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        # Add images if provided and use_images is enabled
        if images and self.use_images:
            for img in images:
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
        
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
            processing_time = time.time() - start
            
            unified_description = response.choices[0].message.content
            
            metadata = {
                "model": self.model,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "processing_time": round(processing_time, 2),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            if self.use_images and images:
                metadata["images_sent"] = len(images)
                metadata["note"] = "GPT-4o-mini processed with both text and images"
            else:
                metadata["note"] = "GPT-4o-mini processed with text only"
            
            return {
                "unified_description": unified_description,
                "llava_metadata": metadata  # Keep same key for compatibility
            }
        except RateLimitError as e:
            processing_time = time.time() - start
            return {
                "unified_description": f"Rate limit error: {str(e)}",
                "llava_metadata": {
                    "model": self.model,
                    "processing_time": round(processing_time, 2),
                    "error": f"RateLimitError: {str(e)}"
                }
            }
        except APIError as e:
            processing_time = time.time() - start
            return {
                "unified_description": f"API error: {str(e)}",
                "llava_metadata": {
                    "model": self.model,
                    "processing_time": round(processing_time, 2),
                    "error": f"APIError: {str(e)}"
                }
            }
        except Exception as e:
            processing_time = time.time() - start
            return {
                "unified_description": f"Error: {str(e)}",
                "llava_metadata": {
                    "model": self.model,
                    "processing_time": round(processing_time, 2),
                    "error": str(e)
                }
            }

