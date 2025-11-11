# Orchestrator Agent Usage Guide

## Overview

The Orchestrator Agent is a movie director-like system that manages video timeline editing operations. It coordinates between user queries, timeline state, and the planner/executor agents to create cohesive video narratives.

## Quick Start

### Basic Usage

```python
from agent.orchestrator_runner import run_orchestrator

result = run_orchestrator(
    query="find the highlights of the video",
    timeline_path="projects/asdf/timeline.json",
    json_path="projects/asdf/segment_tree.json",
    video_path="projects/asdf/video/camp_5min.mp4",
    model_name="gpt-4o-mini",
    verbose=True
)
```

### Command Line Usage

```bash
# Single query
python -m agent.orchestrator_runner "find highlights" \
    --timeline projects/asdf/timeline.json \
    --json projects/asdf/segment_tree.json \
    --video projects/asdf/video/camp_5min.mp4

# Interactive mode
python -m agent.orchestrator_runner \
    --timeline projects/asdf/timeline.json \
    --json projects/asdf/segment_tree.json \
    --video projects/asdf/video/camp_5min.mp4
```

## Supported Operations

### 1. FIND_HIGHLIGHTS
Find and add highlights to the timeline.

```python
query = "find the highlights of the video"
# or
query = "show me the best moments"
```

### 2. CUT
Remove chunks from the timeline.

```python
query = "cut timeline index 0"
query = "cut timeline index 0 and 1"
query = "remove the first two clips"
```

### 3. REPLACE
Replace chunks with new content.

```python
query = "replace timeline index 0-2 with cooking clips"
query = "replace the first clip with people fishing"
```

### 4. INSERT
Add clips between existing chunks.

```python
query = "add a clip between timeline index 1 and 2"
query = "insert a transition clip after index 0"
```

### 5. FIND_BROLL
Find complementary B-roll for selected segments.

```python
query = "find B-roll for timeline 0 to 2"
query = "find B-roll for the first three clips"
```

## Interactive Mode

Run in interactive mode to perform multiple operations:

```bash
python -m agent.orchestrator_runner \
    --timeline projects/asdf/timeline.json \
    --json projects/asdf/segment_tree.json \
    --video projects/asdf/video/camp_5min.mp4
```

Commands:
- Enter any query to perform an operation
- Type `show` to display current timeline
- Type `quit` or `exit` to stop

## Timeline Structure

The timeline.json file contains:

```json
{
  "version": "1.0",
  "created_at": "2025-11-10T15:27:00.000000",
  "updated_at": "2025-11-10T15:27:00.000000",
  "chunks": [
    {
      "start_time": 0.0,           // Timeline position
      "end_time": 6.0,
      "original_start_time": 22.0, // Source video time
      "original_end_time": 28.0,
      "chunk_type": "highlight",
      "description": "...",
      "unified_description": "...",
      "audio_description": "...",
      "score": 1.0,
      "clip_path": "extracted_clips/..."
    }
  ]
}
```

## How It Works

1. **Query Classification**: The orchestrator analyzes your query and determines the operation type
2. **Parameter Extraction**: Extracts timeline indices, search queries, and other parameters
3. **Operation Execution**: 
   - For content discovery (FIND_HIGHLIGHTS, REPLACE, INSERT, FIND_BROLL): Calls planner agent
   - For timeline modifications (CUT): Directly modifies timeline
4. **Timeline Update**: Recalculates all timeline positions to maintain continuity
5. **Save**: Automatically saves updated timeline.json

## Examples

### Example 1: Create Highlights Timeline

```python
# Start with empty timeline
result = run_orchestrator(
    query="find the highlights of the video",
    timeline_path="projects/asdf/timeline.json",
    json_path="projects/asdf/segment_tree.json",
    video_path="projects/asdf/video/camp_5min.mp4"
)
```

### Example 2: Edit Existing Timeline

```python
# Cut first two clips
result = run_orchestrator(
    query="cut timeline index 0 and 1",
    timeline_path="projects/asdf/timeline.json",
    json_path="projects/asdf/segment_tree.json",
    video_path="projects/asdf/video/camp_5min.mp4"
)

# Replace with new content
result = run_orchestrator(
    query="replace timeline index 0 with cooking clips",
    timeline_path="projects/asdf/timeline.json",
    json_path="projects/asdf/segment_tree.json",
    video_path="projects/asdf/video/camp_5min.mp4"
)
```

### Example 3: Add B-Roll

```python
# Find B-roll for selected segments
result = run_orchestrator(
    query="find B-roll for timeline 0 to 2",
    timeline_path="projects/asdf/timeline.json",
    json_path="projects/asdf/segment_tree.json",
    video_path="projects/asdf/video/camp_5min.mp4"
)
```

## API Reference

### `run_orchestrator()`

Main function to run the orchestrator agent.

**Parameters:**
- `query` (str): User query string
- `timeline_path` (str): Path to timeline.json file
- `json_path` (str): Path to segment tree JSON file
- `video_path` (str): Path to source video file
- `model_name` (str): LLM model name (default: "gpt-4o-mini")
- `verbose` (bool): Print verbose output (default: True)

**Returns:**
- Dictionary with:
  - `success` (bool): Whether operation succeeded
  - `operation` (str): Operation type executed
  - `operation_result` (dict): Detailed operation results
  - `timeline_chunks` (list): Updated timeline chunks
  - `timeline_path` (str): Path to timeline file

### `run_orchestrator_interactive()`

Run orchestrator in interactive mode.

**Parameters:**
- Same as `run_orchestrator()` except no `query` parameter

## Tips

1. **Timeline Indices**: Start at 0. Use "timeline index 0", "the first two", "0-2", etc.
2. **Search Queries**: Be specific for REPLACE and INSERT operations
3. **B-Roll**: Automatically excludes main action keywords to find complementary content
4. **Timeline Continuity**: All operations automatically maintain timeline continuity
5. **Backup**: Consider backing up timeline.json before major edits

## Troubleshooting

### "No timeline indices provided"
- Make sure your query includes timeline indices (e.g., "timeline index 0")

### "No search query provided"
- For REPLACE operations, include what to replace with (e.g., "with cooking clips")

### "Invalid timeline index"
- Check that the index exists in your timeline (use `show` command in interactive mode)

### "No highlights found"
- The planner couldn't find matching content. Try a more specific query.

## Next Steps

- See `ORCHESTRATOR_DESIGN.md` for architecture details
- See `test_orchestrator_operations.py` for test examples
- See `example_orchestrator_usage.py` for code examples

