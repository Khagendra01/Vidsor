"""
Vidsor - Video Editor with MoviePy + Tkinter
Interactive video editor for vlog-style editing with automatic chunking,
fast-forward detection, and highlight marking.
"""

# Import models and utilities
from .models import Chunk, EditState
from .utils import format_time, format_time_detailed, get_chunk_color
from .managers.timeline_manager import TimelineManager
from .managers.chat_manager import ChatManager
from .export import VideoExporter
from .analyzers.video_analyzer import VideoAnalyzer
from .managers.project_manager import ProjectManager

# Re-export for convenience
__all__ = ['Vidsor', 'Chunk', 'EditState', 'TimelineManager', 'ChatManager', 
           'VideoExporter', 'VideoAnalyzer', 'ProjectManager']

# Import the main Vidsor class from the core module
from .core.vidsor import Vidsor

