"""
Vidsor - Video Editor with MoviePy + Tkinter
Interactive video editor for vlog-style editing with automatic chunking,
fast-forward detection, and highlight marking.
"""

# Import models and utilities
from .models import Chunk, EditState
from .utils import format_time, format_time_detailed, get_chunk_color
from .timeline_manager import TimelineManager
from .chat_manager import ChatManager
from .export import VideoExporter
from .video_analyzer import VideoAnalyzer
from .project_manager import ProjectManager

# Re-export for convenience
__all__ = ['Vidsor', 'Chunk', 'EditState', 'TimelineManager', 'ChatManager', 
           'VideoExporter', 'VideoAnalyzer', 'ProjectManager']

# Import the main Vidsor class
# Note: The Vidsor class implementation is in vidsor.py (original file)
# This structure allows the codebase to be organized while maintaining functionality
import sys
import os

# Import from the original vidsor.py file in the parent directory
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_vidsor_path = os.path.join(_parent_dir, 'vidsor.py')

if os.path.exists(_vidsor_path):
    # Import the Vidsor class from the original file
    import importlib.util
    spec = importlib.util.spec_from_file_location("vidsor_module", _vidsor_path)
    vidsor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vidsor_module)
    Vidsor = vidsor_module.Vidsor
else:
    raise ImportError(f"Could not find vidsor.py at {_vidsor_path}")

