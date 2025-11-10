"""
Video utility functions for frame extraction and metadata.
"""

import cv2
import subprocess
import threading
from typing import Optional
from PIL import Image
from extractor.config import FPS


def get_video_duration(video_path: str) -> Optional[float]:
    """
    Get video duration using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds, or None if unavailable
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


class VideoUtils:
    """Thread-safe video frame extraction utilities."""
    
    def __init__(self, video_path: str):
        """
        Initialize video utilities.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = None
        self.cap_lock = threading.Lock()
        self._duration = None
    
    def get_frame(self, frame_number: int) -> Optional[Image.Image]:
        """
        Extract frame from video (thread-safe).
        
        Args:
            frame_number: Frame number (1-indexed)
            
        Returns:
            PIL Image or None if frame not found
        """
        with self.cap_lock:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.video_path)
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
    
    def get_duration(self) -> Optional[float]:
        """
        Get video duration.
        
        Returns:
            Duration in seconds
        """
        if self._duration is None:
            self._duration = get_video_duration(self.video_path)
        return self._duration
    
    def get_fps(self) -> float:
        """
        Get video FPS (defaults to config FPS if unavailable).
        
        Returns:
            Frames per second
        """
        with self.cap_lock:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.video_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else FPS
    
    def release(self):
        """Release video capture resources."""
        with self.cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

