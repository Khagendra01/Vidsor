"""
Configuration constants for the extractor package.
"""

from dataclasses import dataclass
from typing import Optional

# Video processing constants
FPS = 30
FRAMES_PER_GROUP = 5
GROUPS_PER_SECOND = 6  # 30 frames / 5 = 6 groups

# Model configuration
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
WHISPER_MODEL = "base"  # Fast and accurate balance
YOLO_MODEL = "old-test-utils/yolo11s.pt"  # YOLO model in old-test-utils folder

# LLaVA/Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "bakllava"

# Audio processing
AUDIO_SEGMENT_DURATION = 5  # Transcribe 5-second segments for better accuracy
AUDIO_OVERLAP = 1  # 1 second overlap between segments

# Parallel processing
DEFAULT_MAX_WORKERS = 3

# Tracking configuration
DEFAULT_TRACKER = "bytetrack"  # or "deepsort"
DEFAULT_YOLO_STRIDE = 10  # Process every Nth frame

# BLIP configuration
DEFAULT_BLIP_SPLIT = 1  # 1=middle, 2=first+last, 3=first+middle+last

# Hierarchical tree configuration
DEFAULT_LEAF_DURATION = 5.0
DEFAULT_BRANCHING_FACTOR = 2


@dataclass
class ExtractorConfig:
    """Configuration class for the extractor pipeline."""
    
    # Video paths
    video_path: str
    tracking_json_path: str
    output_path: str
    
    # Model settings
    tracker: str = DEFAULT_TRACKER
    yolo_stride: int = DEFAULT_YOLO_STRIDE
    blip_split: int = DEFAULT_BLIP_SPLIT
    
    # Processing options
    use_llava: bool = True
    use_images: bool = False
    
    # Parallel processing
    max_workers: int = DEFAULT_MAX_WORKERS
    
    # Hierarchical tree
    generate_hierarchical: bool = True
    leaf_duration: float = DEFAULT_LEAF_DURATION
    branching_factor: int = DEFAULT_BRANCHING_FACTOR
    
    # Embeddings
    generate_embeddings: bool = True
    
    # Model paths (optional overrides)
    blip_model: Optional[str] = None
    whisper_model: Optional[str] = None
    yolo_model: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None

