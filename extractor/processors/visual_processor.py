"""
Visual processing module for BLIP image captioning.
"""

import time
from typing import Dict, List
from PIL import Image
from extractor.models.blip_loader import BLIPLoader
from extractor.config import BLIP_MODEL


class VisualProcessor:
    """Processes visual frames with BLIP."""
    
    def __init__(self, blip_loader: BLIPLoader):
        """
        Initialize visual processor.
        
        Args:
            blip_loader: BLIP model loader instance
        """
        self.blip_loader = blip_loader
    
    def process_frame(self, frame_number: int, frame_image: Image.Image) -> Dict:
        """
        Process single frame with BLIP.
        
        Args:
            frame_number: Frame number
            frame_image: PIL Image
            
        Returns:
            Dictionary with description and metadata
        """
        start = time.time()
        caption = self.blip_loader.caption(frame_image)
        processing_time = time.time() - start
        
        return {
            "description": caption,
            "blip_metadata": {
                "model": BLIP_MODEL,
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

