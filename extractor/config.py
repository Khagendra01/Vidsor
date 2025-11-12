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
# BLIP_MODEL"Salesforce/blip-image-captioning-base" 
BLIP_MODEL = "Salesforce/blip2-opt-2.7b" 
WHISPER_MODEL = "base"  # Fast and accurate balance
YOLO_MODEL = "old-test-utils/yolo11s.pt"  # YOLO model in old-test-utils folder

# LLaVA/Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "bakllava"

# Audio processing
AUDIO_SEGMENT_DURATION = 5  # Transcribe 5-second segments for better accuracy
AUDIO_OVERLAP = 1  # 1 second overlap between segments

# Parallel processing
DEFAULT_MAX_WORKERS = 10  # 10 concurrent requests for GPT-4o-mini (safe for Tier 1, 500 RPM)

# Tracking configuration
DEFAULT_TRACKER = "bytetrack"  # or "deepsort"
DEFAULT_YOLO_STRIDE = 10  # Process every Nth frame

# BLIP configuration
DEFAULT_BLIP_SPLIT = 1  # 1=middle, 2=first+last, 3=first+middle+last
DEFAULT_BLIP_BATCH_SIZE = 48  # Batch size for BLIP processing (GPU memory dependent) - moderate setting for 8GB GPU

# Hierarchical tree configuration
DEFAULT_LEAF_DURATION = 5.0
DEFAULT_BRANCHING_FACTOR = 2

# Embedding model configuration
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # High-quality embeddings for visual descriptions


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
    blip_batch_size: int = DEFAULT_BLIP_BATCH_SIZE
    
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
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    
    # Model paths (optional overrides)
    blip_model: Optional[str] = None
    whisper_model: Optional[str] = None
    yolo_model: Optional[str] = None
    openai_api_key: Optional[str] = None

