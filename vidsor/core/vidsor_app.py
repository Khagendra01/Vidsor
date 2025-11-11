"""
Vidsor - Video Editor with MoviePy + Tkinter
Interactive video editor for vlog-style editing with automatic chunking,
fast-forward detection, and highlight marking.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import json
import shutil
from datetime import datetime
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[VIDSOR] Warning: PIL/Pillow not available. Video preview will not work. Install with: pip install pillow")

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("[VIDSOR] Warning: Pygame not available. Audio playback will not work. Install with: pip install pygame")

import numpy as np

try:
    # MoviePy 2.x imports (direct from moviepy)
    from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips
except ImportError:
    try:
        # Fallback for MoviePy 1.x
        from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
    except ImportError:
        raise ImportError("MoviePy is required. Install with: pip install moviepy")

from agent.utils.segment_tree_utils import load_segment_tree, SegmentTreeQuery
from agent.orchestrator_runner import run_orchestrator
from agent.utils.logging_utils import DualLogger, create_log_file
from extractor.pipeline import SegmentTreePipeline
from extractor.config import ExtractorConfig

# Import from organized modules
from ..models import Chunk, EditState
from ..utils import format_time, format_time_detailed, get_chunk_color
from ..managers.timeline_manager import TimelineManager
from ..managers.chat_manager import ChatManager
from ..export import VideoExporter
from ..managers.project_manager import ProjectManager
from ..analyzers.video_analyzer import VideoAnalyzer
from ..managers.playback_controller import PlaybackController
from ..managers.timeline_controller import TimelineController
from ..integrations.agent_integration import AgentIntegration
from ..ui.main_ui import (
    create_main_ui, update_project_list, on_new_project, 
    on_project_selected, update_ui_state, update_playback_controls
)

# Import split functions from organized modules
from .vidsor.timeline_drawing import draw_timeline
from .vidsor.highlight_processing import process_highlights
from .vidsor.playback import (
    start_playback_from_timeline,
    playback_loop_from_timeline,
    audio_playback_loop,
    seek_to_time
)
from .vidsor.agent_processing import (
    run_agent_thread,
    run_agent_thread_with_clarification
)
from .vidsor.chat_ui import (
    create_chat_ui,
    on_chat_input_return,
    on_send_message,
    continue_with_clarification
)
from .vidsor.video_analysis import (
    analyze_with_segment_tree,
    analyze_simple,
    merge_chunks
)
from .vidsor.preview import (
    update_preview_frame,
    render_preview_from_timeline
)
from .vidsor.timeline_handlers import (
    on_timeline_click,
    on_timeline_drag,
    on_timeline_release,
    on_timeline_motion,
    on_timeline_leave
)


class Vidsor:
    """
    Vidsor - Video Editor Class
    
    Features:
    - Automatic silence detection and fast-forward
    - Movement/activity-based highlight detection
    - Interactive timeline with trimming
    - Real-time preview
    - Export to MP4
    """
    
    def __init__(self, video_path: Optional[str] = None, segment_tree_path: Optional[str] = None):
        """
        Initialize Vidsor editor.
        
        Args:
            video_path: Optional path to video file (can be loaded later via UI)
            segment_tree_path: Optional path to segment tree JSON (auto-detected if None)
        """
        self.video_path = video_path
        self.segment_tree_path = segment_tree_path
        self.video_clip: Optional[VideoFileClip] = None
        self.segment_tree: Optional[SegmentTreeQuery] = None
        self.edit_state = EditState(chunks=[])
        
        # Project management
        self.current_project_path: Optional[str] = None
        # Use home directory for projects: ~/Data/projects
        home_dir = Path.home()
        data_dir = home_dir / "Data"
        self.projects_dir = str(data_dir / "projects")
        
        # Initialize controllers
        self.project_manager = ProjectManager(self.projects_dir)
        self.video_analyzer = VideoAnalyzer()
        self.playback_controller = PlaybackController(self)
        self.timeline_controller = TimelineController(self)
        self.agent_integration = AgentIntegration(self)
        
        # UI components
        self.root: Optional[tk.Tk] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_canvas: Optional[tk.Canvas] = None
        self.timeline_canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[ttk.Label] = None
        self.load_video_btn: Optional[ttk.Button] = None
        self.play_btn: Optional[ttk.Button] = None
        self.pause_btn: Optional[ttk.Button] = None
        self.export_btn: Optional[ttk.Button] = None
        self.project_label: Optional[ttk.Label] = None
        self.project_combo: Optional[ttk.Combobox] = None
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.extraction_thread: Optional[threading.Thread] = None
        self.is_extracting = False
        
        # Sync video analyzer state
        self.video_analyzer.video_path = self.video_path
        self.video_analyzer.segment_tree_path = self.segment_tree_path
        
        # Load video and segment tree if provided
        if self.video_path:
            self._load_video()
            if self.segment_tree_path:
                self._load_segment_tree()
    
    def _ensure_projects_dir(self):
        """Ensure projects directory exists."""
        self.project_manager._ensure_projects_dir()
    
    def create_new_project(self, project_name: str) -> str:
        """Create a new project folder structure."""
        return self.project_manager.create_new_project(project_name)
    
    def set_current_project(self, project_path: str):
        """Set the current active project."""
        if not os.path.exists(project_path):
            raise ValueError(f"Project path does not exist: {project_path}")
        
        self.current_project_path = project_path
        print(f"[VIDSOR] Set current project: {project_path}")
        
        # Try to load project if it has video and segment tree
        self._load_project()
    
    def _load_project(self):
        """Load project video and segment tree if they exist."""
        if not self.current_project_path:
            return
        
        config_path = os.path.join(self.current_project_path, "project_config.json")
        if not os.path.exists(config_path):
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for video
        video_dir = os.path.join(self.current_project_path, "video")
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
            if video_files:
                video_path = os.path.join(video_dir, video_files[0])
                try:
                    self._load_video(video_path)
                except Exception as e:
                    print(f"[VIDSOR] Failed to load project video: {e}")
        
        # Check for segment tree
        segment_tree_path = os.path.join(self.current_project_path, "segment_tree.json")
        if os.path.exists(segment_tree_path):
            self.segment_tree_path = segment_tree_path
            self._load_segment_tree()
        
        # Load timeline from timeline.json
        self._load_timeline()
        
        # Load chat history
        self._load_chat_history()
        
        # Display chat history if UI is ready
        if self.root and hasattr(self, 'chat_text') and self.chat_text:
            self._display_chat_history()
    
    def upload_video_to_project(self, video_path: str, project_path: str) -> str:
        """Copy video to project folder."""
        return self.project_manager.upload_video_to_project(video_path, project_path)
        """
        Copy video to project folder.
        
        Args:
            video_path: Source video path
            project_path: Project folder path
            
        Returns:
            Path to the copied video in project
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_dir = os.path.join(project_path, "video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        # Get original filename
        original_filename = os.path.basename(video_path)
        dest_path = os.path.join(video_dir, original_filename)
        
        # Copy video
        shutil.copy2(video_path, dest_path)
        print(f"[VIDSOR] Copied video to project: {dest_path}")
        
        # Update project config
        config_path = os.path.join(project_path, "project_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            config["video_filename"] = original_filename
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return dest_path
    
    def run_extractor_for_project(self, project_path: str, video_path: str):
        """
        Run extractor pipeline for a project.
        
        Args:
            project_path: Project folder path
            video_path: Path to video file in project
        """
        if self.is_extracting:
            raise Exception("Extraction already in progress")
        
        # Determine paths
        tracking_path = os.path.join(project_path, "tracking.json")
        output_path = os.path.join(project_path, "segment_tree.json")
        
        # Create extractor config
        config = ExtractorConfig(
            video_path=video_path,
            tracking_json_path=tracking_path,
            output_path=output_path
        )
        
        # Run in background thread
        self.is_extracting = True
        self.extraction_thread = threading.Thread(
            target=self._run_extractor_thread,
            args=(config, output_path),
            daemon=True
        )
        self.extraction_thread.start()
    
    def _run_extractor_thread(self, config: ExtractorConfig, output_path: str):
        """Run extractor in background thread."""
        try:
            # Update status
            if self.root:
                self.root.after(0, lambda: self.status_label.config(
                    text="Extracting video features... This may take several minutes."
                ))
                if self.progress_bar:
                    self.root.after(0, lambda: self.progress_bar.config(mode='indeterminate'))
                    self.root.after(0, self.progress_bar.start)
            
            # Run pipeline
            pipeline = SegmentTreePipeline(config)
            pipeline.run()
            
            # Update project config
            if self.current_project_path:
                config_path = os.path.join(self.current_project_path, "project_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        project_config = json.load(f)
                    project_config["segment_tree_path"] = output_path
                    with open(config_path, 'w') as f:
                        json.dump(project_config, f, indent=2)
            
            # Load segment tree
            if os.path.exists(output_path):
                self.segment_tree_path = output_path
                if self.root:
                    self.root.after(0, self._load_segment_tree)
                    self.root.after(0, lambda: self.status_label.config(
                        text="Extraction complete! Segment tree loaded."
                    ))
                    self.root.after(0, self._update_ui_state)
                    if self.progress_bar:
                        self.root.after(0, self.progress_bar.stop)
                        self.root.after(0, lambda: self.progress_bar.config(mode='determinate'))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success", 
                        "Video extraction complete!\n\nSegment tree has been generated and loaded.\n\nYou can now use the chat interface to query your video!"
                    ))
            
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            print(f"[VIDSOR] {error_msg}")
            if self.root:
                self.root.after(0, lambda: self.status_label.config(text=error_msg))
                if self.progress_bar:
                    self.root.after(0, self.progress_bar.stop)
                    self.root.after(0, lambda: self.progress_bar.config(mode='determinate'))
                self.root.after(0, lambda: messagebox.showerror("Extraction Error", error_msg))
        finally:
            self.is_extracting = False
    
    def get_available_projects(self) -> List[str]:
        """Get list of available project names."""
        return self.project_manager.get_available_projects()
    
    def _load_video(self, video_path: Optional[str] = None):
        """Load video file with MoviePy."""
        if video_path:
            self.video_path = video_path
            self.video_analyzer.video_path = video_path
        
        self.video_analyzer.load_video(video_path)
        self.video_clip = self.video_analyzer.video_clip
        self.segment_tree = self.video_analyzer.segment_tree
        
        # Update UI state
        self._update_ui_state()
    
    def _load_segment_tree(self):
        """Load segment tree for analysis."""
        self.video_analyzer.load_segment_tree(self.segment_tree_path)
        self.segment_tree = self.video_analyzer.segment_tree
        if self.segment_tree_path:
            self.video_analyzer.segment_tree_path = self.segment_tree_path
    
    def analyze_video(self, 
                     silence_threshold: float = 2.0,
                     fast_forward_speed: float = 4.0,
                     highlight_min_score: float = 0.6) -> List[Chunk]:
        """
        Analyze video and generate chunks.
        
        Args:
            silence_threshold: Minimum seconds of silence to fast-forward
            fast_forward_speed: Speed multiplier for silent sections
            highlight_min_score: Minimum score for highlight detection
            
        Returns:
            List of Chunk objects
        """
        if not self.video_clip:
            raise Exception("Video not loaded")
        
        chunks = []
        duration = self.video_clip.duration
        
        if self.segment_tree:
            # Use segment tree for intelligent analysis
            chunks = self._analyze_with_segment_tree(
                silence_threshold, fast_forward_speed, highlight_min_score
            )
        else:
            # Fallback: simple time-based chunking
            chunks = self._analyze_simple(duration)
        
        self.edit_state.chunks = chunks
        print(f"[VIDSOR] Analysis complete: {len(chunks)} chunks generated")
        
        # Save timeline to timeline.json
        self._save_timeline()
        
        return chunks
    
    def _analyze_with_segment_tree(self,
                                  silence_threshold: float,
                                  fast_forward_speed: float,
                                  highlight_min_score: float) -> List[Chunk]:
        """Analyze video using segment tree data."""
        return analyze_with_segment_tree(self, silence_threshold, fast_forward_speed, highlight_min_score)
    
    def _analyze_simple(self, duration: float) -> List[Chunk]:
        """Simple fallback analysis without segment tree."""
        return analyze_simple(self, duration)
    
    def _merge_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge overlapping chunks."""
        return merge_chunks(self, chunks)
    
    def create_ui(self):
        """Create Tkinter UI for video editing."""
        create_main_ui(self)
    
    def _update_project_list(self):
        """Update the project dropdown list."""
        update_project_list(self)
    
    def _on_new_project(self):
        """Handle new project button click."""
        on_new_project(self)
    
    def _on_project_selected(self, event=None):
        """Handle project selection from dropdown."""
        on_project_selected(self, event)
    
    def _on_load_video(self):
        """Load video button handler."""
        # Check if project is selected
        if not self.current_project_path:
            response = messagebox.askyesno(
                "No Project Selected",
                "No project is currently selected. Would you like to create a new project first?",
                parent=self.root
            )
            if response:
                self._on_new_project()
                if not self.current_project_path:
                    return  # User cancelled project creation
            else:
                return  # User chose not to create project
        
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if video_path:
            self.status_label.config(text="Uploading video to project...")
            self.root.update()
            
            try:
                # Upload video to project
                project_video_path = self.upload_video_to_project(video_path, self.current_project_path)
                
                # Load the video
                self._load_video(project_video_path)
                
                # Check if segment tree already exists
                segment_tree_path = os.path.join(self.current_project_path, "segment_tree.json")
                if os.path.exists(segment_tree_path):
                    self.segment_tree_path = segment_tree_path
                    self._load_segment_tree()
                    self.status_label.config(text=f"Video loaded: {os.path.basename(video_path)}")
                    self._update_ui_state()  # Enable chat button
                    messagebox.showinfo(
                        "Success", 
                        f"Video uploaded successfully!\n\n{os.path.basename(video_path)}\n\nSegment tree already exists and has been loaded.\n\nYou can now use the chat interface!"
                    )
                else:
                    # Ask if user wants to run extraction
                    response = messagebox.askyesno(
                        "Extract Video Features",
                        "No segment tree found for this video.\n\nWould you like to extract video features now?\n\nThis will analyze the video and may take several minutes.",
                        parent=self.root
                    )
                    
                    if response:
                        # Run extractor
                        self.run_extractor_for_project(self.current_project_path, project_video_path)
                    else:
                        self.status_label.config(text=f"Video loaded: {os.path.basename(video_path)}")
                        messagebox.showinfo(
                            "Success", 
                            f"Video uploaded successfully!\n\n{os.path.basename(video_path)}"
                        )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {str(e)}")
                self.status_label.config(text="Failed to load video")
    
    def _update_playback_controls(self):
        """Update playback control buttons (Play, Pause, Stop) based on current state."""
        has_video = self.video_clip is not None
        has_chunks = bool(self.edit_state.chunks)
        
        # Update Play button
        if self.play_btn:
            self.play_btn.config(
                state=tk.NORMAL if (has_video and has_chunks and not self.edit_state.is_playing) else tk.DISABLED
            )
        
        # Update Pause/Resume button - this is the critical one
        if self.pause_btn:
            if has_video and has_chunks:
                if self.edit_state.is_playing:
                    # Currently playing - show Pause
                    self.pause_btn.config(text="Pause", state=tk.NORMAL)
                elif self.edit_state.has_started_playback:
                    # Was playing, now paused - show Resume (ENABLED)
                    self.pause_btn.config(text="Resume", state=tk.NORMAL)
                else:
                    # Not started yet - disable
                    self.pause_btn.config(text="Pause", state=tk.DISABLED)
            else:
                # No video or chunks - disable
                self.pause_btn.config(text="Pause", state=tk.DISABLED)
    
    def _update_ui_state(self):
        """Update UI button states based on video loading status."""
        has_video = self.video_clip is not None
        
        # Update playback controls using dedicated method
        self._update_playback_controls()
        
        # Update other buttons
        if self.export_btn:
            self.export_btn.config(state=tk.NORMAL if has_video and self.edit_state.chunks else tk.DISABLED)
        
        # Update chat send button state
        if hasattr(self, 'chat_send_btn') and self.chat_send_btn:
            has_project = self.current_project_path is not None
            has_segment_tree = self.segment_tree_path is not None and os.path.exists(self.segment_tree_path) if self.segment_tree_path else False
            is_agent_running = hasattr(self, 'is_agent_running') and self.is_agent_running
            self.chat_send_btn.config(state=tk.NORMAL if (has_project and has_video and has_segment_tree and not is_agent_running) else tk.DISABLED)
        
        # Update preview label with timeline info
        if self.preview_label:
            if has_video:
                timeline_info = ""
                if self.edit_state.chunks:
                    highlight_count = sum(1 for c in self.edit_state.chunks if c.chunk_type == "highlight")
                    total_chunks = len(self.edit_state.chunks)
                    timeline_info = f"\nTimeline: {total_chunks} chunks ({highlight_count} highlights)"
                
                self.preview_label.config(
                    text=f"Video: {os.path.basename(self.video_path)}\n\n"
                         f"Duration: {self.video_clip.duration:.1f}s\n"
                         f"FPS: {self.video_clip.fps}{timeline_info}\n\n"
                         f"Click 'Play Preview' to view timeline"
                )
            elif self.current_project_path:
                project_name = os.path.basename(self.current_project_path)
                self.preview_label.config(text=f"Project: {project_name}\n\nNo video uploaded\n\nClick 'Upload Video' to add a video")
            else:
                self.preview_label.config(text="No project selected\n\nCreate a new project and upload a video to get started")
    
    def _on_play(self):
        """Play preview button handler - plays video directly from timeline.json chunks."""
        self.playback_controller.on_play()
    
    def _start_playback_from_timeline(self):
        """Start video playback directly from timeline chunks (no pre-rendering)."""
        return start_playback_from_timeline(self)
    
    def _playback_loop_from_timeline(self):
        """Main playback loop - plays directly from source video using timeline chunks."""
        return playback_loop_from_timeline(self)
    
    def _audio_playback_loop(self):
        """Audio playback loop - plays audio synchronized with video timeline."""
        return audio_playback_loop(self)
    
    def _seek_to_time(self, timeline_time: float):
        """
        Seek to a specific time in the timeline and update the video preview.
        
        Args:
            timeline_time: Time in the timeline (in seconds)
        """
        return seek_to_time(self, timeline_time)
    
    def _update_preview_frame(self, photo):
        """Update preview canvas with new frame (called from main thread)."""
        return update_preview_frame(self, photo)
    
    def _render_preview_from_timeline(self):
        """Render video preview from timeline.json chunks."""
        return render_preview_from_timeline(self)
    
    def _on_pause(self):
        """Pause/Resume preview button handler."""
        self.playback_controller.on_pause()
    
    def _on_stop(self):
        """Stop preview button handler."""
        self.playback_controller.on_stop()
    
    def _on_export(self):
        """Export video button handler."""
        if not self.edit_state.chunks:
            messagebox.showwarning("Warning", "No chunks to export. Analyze video first.")
            return
        
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if output_path:
            self.status_label.config(text="Exporting video...")
            self.root.update()
            try:
                self.export_video(output_path)
                self.status_label.config(text=f"Exported to: {output_path}")
                messagebox.showinfo("Success", f"Video exported to:\n{output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
                self.status_label.config(text="Export failed")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        return format_time(seconds)
    
    def _format_time_detailed(self, seconds: float) -> str:
        """Format time in MM:SS.mmm format for precise display."""
        return format_time_detailed(seconds)
    
    def _get_chunk_color(self, chunk_type: str, is_hovered: bool = False, is_selected: bool = False) -> tuple:
        """
        Get professional color scheme for chunks.
        Returns (fill_color, outline_color, gradient_color) tuple.
        """
        return get_chunk_color(chunk_type, is_hovered, is_selected)
    
    def _on_timeline_click(self, event):
        """Handle timeline click for chunk selection or playhead dragging."""
        return on_timeline_click(self, event)
    
    def _on_timeline_drag(self, event):
        """Handle timeline drag for playhead scrubbing."""
        return on_timeline_drag(self, event)
    
    def _on_timeline_release(self, event):
        """Handle mouse release after timeline drag."""
        return on_timeline_release(self, event)
    
    def _on_timeline_motion(self, event):
        """Handle mouse motion over timeline for hover effects."""
        return on_timeline_motion(self, event)
    
    def _on_timeline_leave(self, event):
        """Handle mouse leaving timeline."""
        return on_timeline_leave(self, event)
    
    def _draw_timeline(self):
        """Draw professional timeline with chunks, playhead, and modern styling."""
        return draw_timeline(self)
    
    def export_video(self, output_path: str):
        """
        Export edited video to file.
        
        Args:
            output_path: Path to save output video
        """
        VideoExporter.export_video(self.video_clip, self.edit_state, output_path)
    
    def _create_chat_ui(self, parent_frame):
        """Create chat interface UI components."""
        return create_chat_ui(self, parent_frame)
    
    def _on_chat_input_return(self, event):
        """Handle Enter key in chat input (send message, Shift+Enter for new line)."""
        return on_chat_input_return(self, event)
    
    def _on_send_message(self):
        """Handle send message button click."""
        return on_send_message(self)
    
    def _continue_with_clarification(self, user_response: str):
        """Continue operation with user's clarification response using preserved state."""
        return continue_with_clarification(self, user_response)
    
    def _run_agent_thread_with_clarification(self, clarification_response: str, segment_tree_path: str,
                                            operation: str, preserved_state: Dict, original_query: str):
        """Run orchestrator with clarification response, using preserved state to continue."""
        return run_agent_thread_with_clarification(self, clarification_response, segment_tree_path, operation, preserved_state, original_query)
    
    def _run_agent_thread(self, query: str, segment_tree_path: str):
        """Run orchestrator agent in background thread."""
        return run_agent_thread(self, query, segment_tree_path)
    
    def _process_highlights(self, result: Dict, clips: List[str], time_ranges: List[Tuple[float, float]], 
                           search_results: List[Dict]):
        """
        Process highlights from agent result: extract metadata, sort by timing, and add to timeline.
        Only adds to timeline if timeline.json is empty (preserves existing data).
        
        Args:
            result: Full agent result dictionary
            clips: List of extracted clip file paths
            time_ranges: List of (start, end) time tuples
            search_results: List of search result dictionaries with metadata
        """
        return process_highlights(self, result, clips, time_ranges, search_results)
    
    def _add_chat_message(self, role: str, content: str):
        """Add a message to chat history and display it."""
        self.agent_integration.add_chat_message(role, content)
        # Sync chat history
        self.chat_history = self.agent_integration.chat_history
    
    def _save_chat_history(self):
        """Save chat history to project folder."""
        ChatManager.save_chat_history(self.current_project_path, self.chat_history)
    
    def _load_timeline(self):
        """Load timeline from timeline.json in project folder."""
        chunks = TimelineManager.load_timeline(self.current_project_path)
        self.edit_state.chunks = chunks
        
        # Update timeline display if UI is ready
        if self.root and self.timeline_canvas:
            self.root.after(0, self._draw_timeline)
    
    def _save_timeline(self):
        """Save timeline to timeline.json in project folder."""
        TimelineManager.save_timeline(self.current_project_path, self.edit_state.chunks)
    
    def _load_chat_history(self):
        """Load chat history from project folder."""
        self.agent_integration.load_chat_history()
        self.chat_history = self.agent_integration.chat_history
    
    def _display_chat_history(self):
        """Display all chat history in the chat text widget."""
        self.agent_integration.display_chat_history()
    
    def run(self):
        """Run the Vidsor editor UI."""
        if not self.root:
            self.create_ui()
        
        self.root.mainloop()
    
    def close(self):
        """Cleanup and close."""
        if self.video_clip:
            self.video_clip.close()
        if self.root:
            self.root.destroy()


# Main entry point is in vidsor/main.py

