# Video Segment Tree Extractor

A modular, parallel-processing package for extracting comprehensive segment trees from videos with object tracking, visual descriptions, audio transcription, hierarchical tree generation, and semantic embeddings.

## Structure

```
extractor/
├── __init__.py              # Package initialization
├── config.py                # Configuration constants and dataclass
├── pipeline.py              # Main pipeline orchestrator
├── main.py                  # CLI entry point
├── models/                  # Model loaders
│   ├── __init__.py
│   ├── blip_loader.py       # BLIP image captioning model
│   ├── whisper_loader.py    # Whisper audio transcription model
│   └── yolo_loader.py       # YOLO object detection/tracking model
├── processors/              # Processing modules
│   ├── __init__.py
│   ├── visual_processor.py  # BLIP visual processing
│   ├── audio_processor.py   # Whisper audio processing
│   ├── llava_processor.py   # LLaVA unified description processing
│   └── tracking_processor.py # Tracking data processing and grouping
├── generators/              # Generation modules
│   ├── __init__.py
│   ├── segment_tree_generator.py    # Main segment tree generation
│   ├── hierarchical_generator.py    # Hierarchical tree generation
│   └── embeddings_generator.py      # Embeddings generation
└── utils/                   # Utility functions
    ├── __init__.py
    └── video_utils.py        # Video frame extraction and metadata
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules for models, processors, and generators
- **Parallel Processing**: Supports parallel processing of video seconds, frames, and audio segments
- **Thread-Safe**: Thread-safe video frame extraction for concurrent processing
- **Configurable**: Comprehensive configuration via `ExtractorConfig` dataclass
- **Extensible**: Easy to add new models, processors, or generators

## Usage

### Basic Usage

```python
from extractor import SegmentTreePipeline, ExtractorConfig

config = ExtractorConfig(
    video_path="video.mp4",
    tracking_json_path="tracking.json",
    output_path="segment_tree.json"
)

pipeline = SegmentTreePipeline(config)
output_path = pipeline.run()
```

### CLI Usage

```bash
# Basic usage
python -m extractor.main --video camp.mp4 --output camp_segment_tree.json

# With custom options
python -m extractor.main \
    --video camp.mp4 \
    --output camp_segment_tree.json \
    --tracker bytetrack \
    --yolo-stride 10 \
    --blip-split 3 \
    --max-workers 4 \
    --use-images

# Skip hierarchical tree or embeddings
python -m extractor.main \
    --video camp.mp4 \
    --no-hierarchical \
    --no-embeddings
```

### Configuration Options

- `video_path`: Path to input video file
- `tracking_json_path`: Path to tracking JSON (generated if missing)
- `output_path`: Path to output segment tree JSON
- `tracker`: Tracker type ("bytetrack" or "deepsort")
- `yolo_stride`: Process every Nth frame with YOLO
- `blip_split`: Number of frames per second for BLIP (1, 2, or 3)
- `use_llava`: Enable/disable LLaVA processing
- `use_images`: Send images to LLaVA (requires `use_llava=True`)
- `max_workers`: Number of parallel workers
- `generate_hierarchical`: Generate hierarchical tree
- `leaf_duration`: Duration of hierarchical tree leaf nodes
- `branching_factor`: Hierarchical tree branching factor
- `generate_embeddings`: Generate semantic embeddings

## Architecture

### Models (`models/`)
- **BLIPLoader**: Loads and manages BLIP model for image captioning
- **WhisperLoader**: Loads and manages Whisper model for audio transcription
- **YOLOLoader**: Loads and manages YOLO model for object detection/tracking

### Processors (`processors/`)
- **VisualProcessor**: Processes frames with BLIP
- **AudioProcessor**: Processes audio segments with Whisper
- **LLaVAProcessor**: Processes descriptions with LLaVA via Ollama
- **TrackingProcessor**: Processes and groups tracking data

### Generators (`generators/`)
- **SegmentTreeGenerator**: Generates main segment tree with all metadata
- **HierarchicalGenerator**: Generates hierarchical keyword tree
- **EmbeddingsGenerator**: Generates semantic embeddings for search

### Pipeline (`pipeline.py`)
- **SegmentTreePipeline**: Main orchestrator that coordinates all components

## Parallel Processing

The pipeline supports parallel processing at multiple levels:

1. **Video Seconds**: Multiple seconds processed in parallel
2. **BLIP Frames**: Multiple frames within a second processed in parallel
3. **Audio Segments**: Audio transcription segments processed sequentially (can be parallelized)

All parallel processing uses `ThreadPoolExecutor` with configurable worker count.

## Thread Safety

- Video frame extraction is thread-safe using locks
- Model loaders are designed to be shared across threads
- Processors are stateless and thread-safe

## Dependencies

- `transformers` (BLIP)
- `whisper` (audio transcription)
- `ultralytics` (YOLO)
- `opencv-python` (video processing)
- `PIL` (image processing)
- `requests` (Ollama API)
- `sentence-transformers` (embeddings, optional)

## Best Practices

1. **Model Loading**: Models are loaded once and reused across threads
2. **Resource Cleanup**: Video captures are properly released
3. **Error Handling**: Errors are caught and logged without stopping the pipeline
4. **Configuration**: All settings are centralized in `ExtractorConfig`
5. **Modularity**: Each component can be used independently or as part of the pipeline

