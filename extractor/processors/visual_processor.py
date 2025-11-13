"""
Visual processing module for BakLLaVA image captioning.
"""

import time
from typing import Dict, List
from PIL import Image
from extractor.models.bakllava_loader import BakLLaVALoader
from extractor.config import OLLAMA_MODEL


class VisualProcessor:
    """Processes visual frames with BakLLaVA."""
    
    def __init__(self, bakllava_loader: BakLLaVALoader):
        """
        Initialize visual processor.
        
        Args:
            bakllava_loader: BakLLaVA model loader instance
        """
        self.bakllava_loader = bakllava_loader
    
    def process_frame(self, frame_number: int, frame_image: Image.Image) -> Dict:
        """
        Process single frame with BakLLaVA.
        
        Args:
            frame_number: Frame number
            frame_image: PIL Image
            
        Returns:
            Dictionary with description and metadata
        """
        start = time.time()
        caption = self.bakllava_loader.caption(frame_image)
        processing_time = time.time() - start
        
        return {
            "description": caption,
            "bakllava_metadata": {
                "model": OLLAMA_MODEL,
                "processing_time": round(processing_time, 2)
            }
        }
    
    def process_frames(self, frame_data: List[tuple]) -> List[Dict]:
        """
        Process multiple frames in parallel.
        
        Args:
            frame_data: List of (frame_number, frame_image) tuples
            
        Returns:
            List of processing results
        """
        results = []
        for frame_number, frame_image in frame_data:
            if frame_image:
                result = self.process_frame(frame_number, frame_image)
                results.append((frame_number, result))
        return results

