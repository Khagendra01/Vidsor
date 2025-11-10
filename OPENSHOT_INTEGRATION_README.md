# OpenShot Video Editor Integration

This integration allows you to automatically create OpenShot Video Editor project files from clips extracted by the video clip agent.

## Features

- **Automatic Project Creation**: Creates OpenShot project files (.osp) from agent-extracted clips
- **Timeline Arrangement**: Automatically arranges clips on the timeline in sequence
- **Auto-Open**: Optionally opens the project in OpenShot GUI automatically
- **Standalone Usage**: Can also be used independently to create projects from any clip files

## Installation

### Option 1: Use OpenShot Integration (Recommended - No Extra Installation)

The integration works by creating OpenShot project files that can be opened in the OpenShot GUI. You just need to:

1. **Install OpenShot Video Editor** (if not already installed):
   - Download from: https://www.openshot.org/download/
   - Available for Windows, macOS, and Linux

2. **Use the integration** - No additional Python packages needed!

### Option 2: Use libopenshot Python API (Advanced)

For programmatic control, you can install libopenshot:

```bash
# Linux/macOS - requires compilation
git clone https://github.com/OpenShot/libopenshot.git
cd libopenshot
mkdir build && cd build
cmake ..
make
sudo make install
```

**Note**: This is optional. The basic integration works without it.

## Usage

### Method 1: Automatic Integration with Agent

Run the agent with OpenShot integration enabled:

```bash
# Create OpenShot project after extraction
python video_clip_agent.py "find moments where they catch fish" --openshot

# Create project and auto-open in OpenShot
python video_clip_agent.py "find moments where they catch fish" --openshot --openshot-auto-open
```

### Method 2: Programmatic Usage

```python
from agent import run_agent
from openshot_integration import create_openshot_project_from_agent_result

# Run agent
result = run_agent(
    "find moments where they catch fish",
    "camp_segment_tree.json",
    "camp.mp4",
    create_openshot_project=True,
    auto_open_openshot=True
)

# Or create project manually from agent result
project_path = create_openshot_project_from_agent_result(
    result,
    project_name="my_edits",
    auto_open=True,
    verbose=True
)
```

### Method 3: Standalone Usage

```python
from openshot_integration import create_openshot_project_from_clips

# Create project from list of clip paths
clip_paths = [
    "extracted_clips/clip_1_16s_to_21s.mp4",
    "extracted_clips/clip_2_37s_to_42s.mp4",
    "extracted_clips/clip_3_44s_to_49s.mp4",
]

project_path = create_openshot_project_from_clips(
    clip_paths,
    project_name="my_video_edits",
    auto_open=True,  # Opens in OpenShot if available
    verbose=True
)

print(f"Project saved to: {project_path}")
```

### Method 4: Using OpenShotProjectCreator Class

```python
from openshot_integration import OpenShotProjectCreator

# Create project creator
creator = OpenShotProjectCreator(project_name="my_project")

# Add clips
creator.add_clips([
    "extracted_clips/clip_1.mp4",
    "extracted_clips/clip_2.mp4",
], verbose=True)

# Create and save project
project_path = creator.create_project(
    output_path="my_project.osp",
    fps=30.0,
    verbose=True
)

# Open in OpenShot
creator.open_in_openshot(project_path, verbose=True)
```

## Workflow

1. **Agent Extracts Clips**: The video clip agent finds and extracts relevant moments
2. **Project Creation**: OpenShot project file is automatically created with all clips
3. **Open in OpenShot**: Project opens in OpenShot GUI (if auto-open enabled)
4. **Edit in GUI**: Use OpenShot's full-featured GUI to:
   - Trim clips
   - Add transitions
   - Apply effects
   - Add titles/text
   - Adjust audio
   - Export final video

## Project File Format

The integration creates OpenShot project files (.osp) which are JSON-based project definitions that OpenShot can open. The project includes:

- All extracted clips arranged on timeline
- Video track with clips in sequence
- Project settings (FPS, resolution, etc.)

## Troubleshooting

### OpenShot Not Found

If OpenShot doesn't open automatically:
1. Make sure OpenShot is installed
2. Manually open the .osp file in OpenShot:
   - Double-click the .osp file, or
   - Open OpenShot → File → Open → Select the .osp file

### Clips Not Loading

- Ensure clip paths are absolute (the integration handles this automatically)
- Check that clip files exist and are valid MP4 files
- OpenShot will detect actual clip durations when opening the project

### Project File Issues

- Project files are created in the current directory by default
- Check file permissions if you can't create/save projects
- Ensure you have write access to the directory

## Examples

### Example 1: Basic Usage

```bash
python video_clip_agent.py "find all fishing moments" --openshot
```

This will:
1. Extract clips matching "fishing moments"
2. Create an OpenShot project file
3. Print the project path

### Example 2: Auto-Open

```bash
python video_clip_agent.py "find all fishing moments" --openshot --openshot-auto-open
```

This will:
1. Extract clips
2. Create project
3. Automatically open in OpenShot (if installed)

### Example 3: Python Script

```python
from agent import run_agent

result = run_agent(
    query="find moments where they catch fish",
    json_path="camp_segment_tree.json",
    video_path="camp.mp4",
    create_openshot_project=True,
    auto_open_openshot=True
)

if result.get("openshot_project_path"):
    print(f"Project ready: {result['openshot_project_path']}")
```

## Next Steps

After opening the project in OpenShot:

1. **Review Clips**: Check that all clips are on the timeline
2. **Trim/Edit**: Adjust clip in/out points as needed
3. **Add Transitions**: Add transitions between clips
4. **Add Effects**: Apply color correction, filters, etc.
5. **Export**: Use OpenShot's export feature to create final video

## Integration Points

The integration is automatically called in:
- `agent/executor.py` - After clips are extracted
- `agent/__init__.py` - Via `run_agent()` function
- `video_clip_agent.py` - Command-line interface

## Support

For OpenShot-specific issues, refer to:
- OpenShot Documentation: https://www.openshot.org/user-guide/
- OpenShot Forums: https://www.openshot.org/forum/

