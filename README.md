# Vidsor

**Vidsor** is an intelligent video editing system that combines AI-powered video analysis with natural language editing capabilities. It enables users to edit videos using conversational queries, automatically extract highlights, find specific moments, and perform complex editing operations through an intuitive interface.

## Overview

Vidsor consists of three main components:

1. **Extractor** - Analyzes videos to create rich segment trees with visual descriptions, audio transcriptions, object tracking, and semantic embeddings
2. **Agent** - LangGraph-based AI agents that understand natural language queries and execute video editing operations
3. **Vidsor UI** - Interactive video editor with timeline management, preview, and chat-based editing interface

## Features

### 🎬 Video Analysis & Extraction
- **Multi-modal Analysis**: Combines visual descriptions (BLIP, LLaVA, BakLLaVA), audio transcription (Whisper), and object tracking (YOLO + ByteTrack/DeepSORT)
- **Hierarchical Segment Trees**: Organizes video content into searchable, hierarchical structures
- **Semantic Embeddings**: Generates embeddings for fast semantic search across video content
- **Parallel Processing**: Efficient multi-threaded processing for faster analysis

### 🤖 AI-Powered Editing Agents
- **Natural Language Queries**: Edit videos using conversational commands
- **Planner Agent**: Understands user intent and plans editing operations
- **Executor Agent**: Performs video editing operations (cut, trim, insert, replace, etc.)
- **Orchestrator Agent**: Handles complex multi-step editing workflows
- **Self-Correction**: Validates and corrects operations automatically

### ✂️ Editing Operations
- **Cut**: Remove segments from timeline
- **Trim**: Trim segments to specific time ranges
- **Insert**: Add segments at specific positions
- **Replace**: Replace segments with new content
- **Find Highlights**: Automatically detect and extract highlight moments
- **Find B-roll**: Locate relevant B-roll footage
- **Apply Effects**: Apply video effects and transformations

### 🎨 User Interface
- **Timeline Editor**: Visual timeline with drag-and-drop editing
- **Video Preview**: Real-time preview of edited content
- **Chat Interface**: Conversational editing through natural language
- **Project Management**: Save and load editing projects
- **Export**: Export final edited videos

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Vidsor
```

2. **Activate conda environment**:
```bash
conda activate p310
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Key Dependencies

- **Video Processing**: `moviepy`, `opencv-python`
- **AI Models**: `transformers`, `whisper`, `ultralytics`
- **LLM Integration**: `langchain`, `langgraph`, `openai`
- **UI**: `tkinter`, `pygame`, `PIL`
- **Embeddings**: `sentence-transformers`
- **Utilities**: `numpy`, `requests`

## Tech Stack

Vidsor is built on a modern Python stack combining AI/ML libraries, video processing tools, and agent frameworks.

### Core Frameworks

- **LangGraph** (`langgraph>=0.2.0`) - Agent orchestration and workflow management
  - Used for building the planner, executor, and orchestrator agents
  - Enables stateful, multi-step agent workflows with conditional routing
  
- **LangChain** (`langchain-core>=0.3.0`, `langchain-openai>=0.2.0`) - LLM integration and abstractions
  - Provides standardized interfaces for LLM interactions
  - Handles prompt management and message formatting
  - Supports multiple LLM providers (OpenAI, Anthropic)

- **MoviePy** (`moviepy>=1.0.3`) - Video editing and manipulation
  - Core video processing engine for all editing operations
  - Handles video loading, cutting, trimming, concatenation
  - Supports video effects and transformations

### AI/ML Models & Libraries

- **Transformers** (`transformers`) - Hugging Face model ecosystem
  - **BLIP/BLIP2**: Image captioning and visual understanding
  - **BakLLaVA**: Multimodal vision-language model via Ollama
  - Model loading, quantization, and inference management

- **Whisper** (`whisper`) - Audio transcription
  - Speech-to-text for video audio tracks
  - Supports multiple languages and model sizes

- **Ultralytics YOLO** (`ultralytics`) - Object detection and tracking
  - Real-time object detection in video frames
  - Integration with ByteTrack and DeepSORT for multi-object tracking
  - Person, vehicle, and general object detection

- **Sentence Transformers** (`sentence-transformers>=2.2.0`) - Semantic embeddings
  - Generates embeddings for semantic search
  - Default model: `BAAI/bge-large-en-v1.5`
  - Enables meaning-based video segment retrieval

- **PyTorch** (`torch>=2.0.0`) - Deep learning framework
  - Backend for all transformer-based models
  - GPU acceleration support
  - Model inference and tensor operations

### LLM Providers

- **OpenAI** (`openai>=1.0.0`) - Primary LLM provider
  - GPT-4o-mini (default) for agent reasoning
  - GPT-4 for complex planning tasks
  - Chat completions API for natural language understanding

- **Anthropic** (`anthropic>=0.25.0`, `langchain-anthropic>=0.2.0`) - Alternative LLM provider
  - Claude models support
  - Fallback option for LLM operations

### Video Processing

- **OpenCV** (`opencv-python`) - Computer vision and video I/O
  - Video frame extraction
  - Image processing and manipulation
  - Video metadata extraction (FPS, duration, resolution)

- **Pillow/PIL** (`PIL`) - Image processing
  - Image loading, resizing, and format conversion
  - Frame-to-image conversion for model inference
  - UI image rendering

- **NumPy** (`numpy>=1.20.0`) - Numerical computing
  - Array operations for video frames
  - Mathematical computations
  - Data structure for embeddings and feature vectors

### User Interface

- **Tkinter** - Native Python GUI framework
  - Main application window and widgets
  - Timeline visualization
  - Chat interface
  - File dialogs and menus

- **Pygame** (`pygame`) - Audio playback
  - Audio playback for video preview
  - Sound effects and audio feedback

### Utilities

- **Requests** (`requests`) - HTTP client
  - Ollama API communication for LLaVA/BakLLaVA
  - External API calls

- **Python-dotenv** (`python-dotenv`) - Environment variable management
  - API key management
  - Configuration loading

- **Pathlib** - Modern path handling
  - Cross-platform file path operations
  - File system navigation

### Architecture Patterns

- **State Management**: Custom state classes for agent workflows
- **Parallel Processing**: `ThreadPoolExecutor` for concurrent video processing
- **Modular Design**: Clean separation between extractor, agent, and UI components
- **Plugin Architecture**: Extensible handlers for new editing operations

### Development Tools

- **Type Hints**: Full type annotation support for better IDE support
- **Dataclasses**: Structured data models for configuration and state
- **Logging**: Custom dual logger (console + file) for debugging
- **JSON**: Data serialization for segment trees and timelines

## Quick Start

### 1. Extract Segment Tree from Video

First, analyze your video to create a segment tree:

```bash
python -m extractor.main \
    --video your_video.mp4 \
    --output segment_tree.json \
    --tracker bytetrack \
    --max-workers 4
```

This will generate a `segment_tree.json` file containing:
- Visual descriptions for each second
- Audio transcriptions
- Object tracking data
- Hierarchical organization
- Semantic embeddings

### 2. Start the Video Editor

Launch the Vidsor UI:

```bash
python -m vidsor.core.main your_video.mp4 --segment-tree segment_tree.json
```

Or start with an empty project and load a video through the UI:

```bash
python -m vidsor.core.main
```

### 3. Edit Using Natural Language

In the chat interface, you can use commands like:

- `"Find all moments where someone catches a fish"`
- `"Cut the first 30 seconds"`
- `"Trim timeline index 0 to 5-10 seconds"`
- `"Insert the highlight at position 2"`
- `"Find B-roll of nature scenes"`
- `"Replace segment 1 with the highlight"`

## Project Structure

```
Vidsor/
├── extractor/              # Video analysis and segment tree extraction
│   ├── models/            # Model loaders (BLIP, Whisper, YOLO)
│   ├── processors/        # Processing modules (visual, audio, tracking)
│   ├── generators/        # Segment tree, hierarchical, embeddings generators
│   ├── pipeline.py        # Main extraction pipeline
│   └── main.py            # CLI entry point
│
├── agent/                  # AI editing agents
│   ├── nodes/             # LangGraph nodes (planner, executor, orchestrator)
│   ├── handlers/          # Operation handlers (cut, trim, insert, etc.)
│   ├── state/             # Agent state management
│   ├── prompts/           # LLM prompts
│   ├── utils/             # Utilities (search, processing, logging)
│   └── orchestrator_runner.py  # Orchestrator execution
│
├── vidsor/                 # Main application
│   ├── core/              # Core application logic
│   ├── ui/                # User interface components
│   ├── managers/          # Timeline, project, chat managers
│   ├── analyzers/         # Video analysis
│   ├── export/            # Video export functionality
│   └── integrations/      # Agent integration
│
└── old-test-utils/        # Legacy test utilities and scripts
```

## Usage Examples

### Extract Segment Tree with Custom Options

```bash
python -m extractor.main \
    --video camp.mp4 \
    --output camp_segment_tree.json \
    --tracker bytetrack \
    --yolo-stride 10 \
    --bakllava-split 2 \
    --max-workers 8 \
    --leaf-duration 5.0 \
    --branching-factor 2
```

### Run Agent Directly (CLI)

```bash
python -m agent \
    "find moments where they catch fish" \
    --json camp_segment_tree.json \
    --video camp.mp4 \
    --model gpt-4o-mini
```

### Analyze Video Only (No UI)

```bash
python -m vidsor.core.main your_video.mp4 --analyze-only
```

## Configuration

### Extractor Configuration

The extractor supports various configuration options:

- **Tracking**: Choose between `bytetrack` or `deepsort`
- **Frame Processing**: Configure YOLO stride and frame selection
- **Models**: Override default model paths and settings
- **Parallel Processing**: Adjust worker count for performance
- **Hierarchical Tree**: Configure leaf duration and branching factor
- **Embeddings**: Choose embedding model

### Agent Configuration

- **LLM Model**: Select OpenAI model (default: `gpt-4o-mini`)
- **Verbose Logging**: Enable detailed operation logs
- **Self-Correction**: Automatic validation and correction
- **Memory**: Context retention across operations

## Architecture

### Extractor Pipeline

1. **Video Loading**: Loads video and extracts metadata
2. **Parallel Processing**: Processes frames, audio, and tracking concurrently
3. **Model Inference**: Runs BLIP/LLaVA, Whisper, and YOLO models
4. **Tree Generation**: Builds hierarchical segment tree
5. **Embedding Generation**: Creates semantic embeddings for search

### Agent Workflow

1. **Query Input**: User provides natural language query
2. **Clarification**: Agent asks for clarification if needed
3. **Planning**: Planner agent analyzes query and creates operation plan
4. **Search**: Semantic search finds relevant video segments
5. **Execution**: Executor agent performs editing operations
6. **Validation**: Self-correction validates and fixes operations

### Orchestrator

The orchestrator agent handles complex multi-step operations:
- Coordinates multiple editing operations
- Manages timeline state
- Handles dependencies between operations
- Provides operation history and rollback

## Advanced Features

### Semantic Search

The system uses semantic embeddings to find video segments based on meaning rather than exact text matches. This enables queries like:
- `"moments of excitement"`
- `"scenes with water"`
- `"people talking"`

### Hierarchical Organization

Videos are organized into hierarchical trees that enable:
- Fast navigation at different granularities
- Efficient search across time scales
- Natural grouping of related content

### Timeline Management

The timeline manager tracks:
- All segments and their properties
- Operation history
- Undo/redo capabilities
- Export state

## Development

### Running Tests

```bash
# Run extractor tests
python -m pytest extractor/tests/

# Run agent tests
python -m pytest agent/tests/
```

### Adding New Operations

1. Create handler in `agent/handlers/`
2. Add operation to orchestrator in `agent/nodes/orchestrator.py`
3. Update prompts in `agent/prompts/orchestrator_prompts.py`

### Extending Models

1. Add model loader in `extractor/models/`
2. Create processor in `extractor/processors/`
3. Integrate into pipeline in `extractor/pipeline.py`

## Troubleshooting

### Common Issues

**Model Loading Errors**:
- Ensure models are downloaded and accessible
- Check model paths in configuration
- Verify sufficient GPU/CPU memory

**Processing Slow**:
- Reduce `--max-workers` if memory constrained
- Increase `--yolo-stride` to process fewer frames
- Disable hierarchical tree or embeddings if not needed

**Agent Errors**:
- Check OpenAI API key is set
- Verify segment tree JSON is valid
- Review logs for detailed error messages

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add license information here]

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Uses [MoviePy](https://github.com/Zulko/moviepy) for video processing
- Integrates multiple AI models: BLIP, LLaVA, Whisper, YOLO

## Roadmap

See `agent/AUTONOMY_ROADMAP.md` for planned features and improvements.

---

For more detailed documentation:
- **Extractor**: See `extractor/README.md`
- **Agent Architecture**: See `agent/AUTONOMY_ROADMAP.md`
- **Early Stopping**: See `agent/EARLY_STOPPING_IMPLEMENTATION.md`

