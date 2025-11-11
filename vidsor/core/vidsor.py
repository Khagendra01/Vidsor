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
        self.projects_dir = os.path.join(os.getcwd(), "projects")
        
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
        if self.root and self.chat_text:
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
        chunks = []
        duration = self.video_clip.duration
        
        # Get transcriptions to detect silence
        transcriptions = self.segment_tree.transcriptions
        transcription_ranges = []
        for tr in transcriptions:
            tr_range = tr.get("time_range", [])
            if tr_range and len(tr_range) >= 2:
                text = tr.get("transcription", "").strip()
                if text:  # Only non-empty transcriptions
                    transcription_ranges.append((tr_range[0], tr_range[1]))
        
        # Sort transcription ranges
        transcription_ranges.sort(key=lambda x: x[0])
        
        # Find silent gaps
        current_time = 0.0
        for tr_start, tr_end in transcription_ranges:
            # Check for silence before this transcription
            if tr_start - current_time >= silence_threshold:
                # Silent section - mark for fast-forward
                chunks.append(Chunk(
                    start_time=current_time,
                    end_time=tr_start,
                    chunk_type="fast_forward",
                    speed=fast_forward_speed,
                    description="Silent section"
                ))
            
            # Normal section with audio
            chunks.append(Chunk(
                start_time=max(current_time, tr_start),
                end_time=tr_end,
                chunk_type="normal",
                speed=1.0,
                description="Audio section"
            ))
            current_time = tr_end
        
        # Check for silence at the end
        if duration - current_time >= silence_threshold:
            chunks.append(Chunk(
                start_time=current_time,
                end_time=duration,
                chunk_type="fast_forward",
                speed=fast_forward_speed,
                description="Silent section"
            ))
        
        # Detect highlights using hierarchical tree or semantic search
        if self.segment_tree.hierarchical_tree:
            highlights = self.segment_tree.hierarchical_score_leaves_for_highlights(
                max_results=20
            )
            for highlight in highlights:
                tr = highlight.get("time_range", [])
                if tr and len(tr) >= 2:
                    score = highlight.get("score", 0)
                    if score >= highlight_min_score * 10:  # Adjust scale
                        # Mark as highlight
                        for chunk in chunks:
                            if (chunk.start_time <= tr[0] < chunk.end_time or
                                chunk.start_time < tr[1] <= chunk.end_time):
                                chunk.chunk_type = "highlight"
                                chunk.score = score
                                chunk.description = "Highlight moment"
        
        # Merge overlapping chunks and sort
        chunks = self._merge_chunks(chunks)
        chunks.sort(key=lambda x: x.start_time)
        
        return chunks
    
    def _analyze_simple(self, duration: float) -> List[Chunk]:
        """Simple fallback analysis without segment tree."""
        # Create a single chunk for the entire video
        return [Chunk(
            start_time=0.0,
            end_time=duration,
            chunk_type="normal",
            speed=1.0,
            description="Full video"
        )]
    
    def _merge_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge overlapping chunks."""
        if not chunks:
            return []
        
        # Sort by start time
        sorted_chunks = sorted(chunks, key=lambda x: x.start_time)
        merged = [sorted_chunks[0]]
        
        for chunk in sorted_chunks[1:]:
            last = merged[-1]
            if chunk.start_time <= last.end_time:
                # Overlapping - merge
                last.end_time = max(last.end_time, chunk.end_time)
                # Keep the more interesting type
                if chunk.chunk_type == "highlight":
                    last.chunk_type = "highlight"
                elif chunk.chunk_type == "fast_forward" and last.chunk_type == "normal":
                    last.chunk_type = "fast_forward"
                    last.speed = chunk.speed
            else:
                merged.append(chunk)
        
        return merged
    
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
        if self.chat_send_btn:
            has_project = self.current_project_path is not None
            has_segment_tree = self.segment_tree_path is not None and os.path.exists(self.segment_tree_path) if self.segment_tree_path else False
            self.chat_send_btn.config(state=tk.NORMAL if (has_project and has_video and has_segment_tree and not self.is_agent_running) else tk.DISABLED)
        
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
        if not self.video_clip or not self.edit_state.chunks:
            return
        
        if not HAS_PIL:
            messagebox.showerror(
                "Error",
                "PIL/Pillow is required for video preview.\n\nInstall with: pip install pillow"
            )
            return
        
        # Calculate total duration from timeline
        timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks) if self.edit_state.chunks else 0
        
        self.edit_state.is_playing = True
        self.edit_state.preview_time = 0.0
        self.edit_state.has_started_playback = True  # Mark that playback has started
        self.status_label.config(text=f"Playing preview... ({timeline_duration:.1f}s)")
        
        # Update playback controls
        self._update_playback_controls()
        
        # Show canvas, hide label
        if self.preview_canvas:
            self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        if self.preview_label:
            self.preview_label.pack_forget()
        
        # Initialize audio playback if available
        if HAS_PYGAME and self.video_clip.audio is not None:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                # Extract audio from video
                self.audio_clip = self.video_clip.audio
            except Exception as e:
                print(f"[VIDSOR] Warning: Could not initialize audio: {e}")
                self.audio_clip = None
        else:
            self.audio_clip = None
        
        # Start video playback thread
        if self.playback_thread and self.playback_thread.is_alive():
            return
        
        self.playback_thread = threading.Thread(
            target=self._playback_loop_from_timeline,
            daemon=True
        )
        self.playback_thread.start()
        
        # Start audio playback thread if audio is available
        if self.audio_clip and HAS_PYGAME:
            if self.audio_thread and self.audio_thread.is_alive():
                return
            
            self.audio_thread = threading.Thread(
                target=self._audio_playback_loop,
                daemon=True
            )
            self.audio_thread.start()
    
    def _playback_loop_from_timeline(self):
        """Main playback loop - plays directly from source video using timeline chunks."""
        if not self.video_clip or not self.edit_state.chunks:
            return
        
        if not HAS_PIL:
            return
        
        # Get video FPS for frame timing
        video_fps = self.video_clip.fps
        frame_duration = 1.0 / video_fps
        
        # Calculate total timeline duration
        timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
        
        try:
            while self.edit_state.preview_time < timeline_duration:
                # Check if paused - wait for resume
                if not self.edit_state.is_playing:
                    # Paused - wait in a loop until resumed
                    while not self.edit_state.is_playing:
                        time.sleep(0.1)  # Check every 100ms
                        # If we're no longer supposed to be playing (e.g., stopped), exit
                        if not self.edit_state.has_started_playback:
                            return
                    # Resumed - continue playback
                    continue
                
                start_time = time.time()
                
                # Find which chunk we're currently in
                current_chunk = None
                for chunk in self.edit_state.chunks:
                    if chunk.start_time <= self.edit_state.preview_time < chunk.end_time:
                        current_chunk = chunk
                        break
                
                if not current_chunk:
                    # Between chunks or past end - advance time and continue
                    self.edit_state.preview_time += frame_duration
                    time.sleep(frame_duration)
                    continue
                
                # Calculate position within this chunk (0.0 to 1.0)
                chunk_position = (self.edit_state.preview_time - current_chunk.start_time) / (current_chunk.end_time - current_chunk.start_time)
                
                # Map to original video time
                if current_chunk.original_start_time is not None and current_chunk.original_end_time is not None:
                    original_time = current_chunk.original_start_time + chunk_position * (current_chunk.original_end_time - current_chunk.original_start_time)
                else:
                    # Fallback to sequential timing
                    original_time = current_chunk.start_time + chunk_position * (current_chunk.end_time - current_chunk.start_time)
                
                # Ensure we're within video bounds
                original_time = max(0, min(original_time, self.video_clip.duration - 0.1))
                
                # Get frame from source video at original time
                frame = self.video_clip.get_frame(original_time)
                
                # Convert frame to PIL Image
                frame_pil = Image.fromarray(frame)
                
                # Resize to fit preview canvas (maintain aspect ratio)
                # Use reasonable size that matches our minimum size settings
                # The preview frame has minsize of 800x450
                max_width = 800
                max_height = 500
                
                # Calculate resize dimensions maintaining aspect ratio
                img_width, img_height = frame_pil.size
                aspect = img_width / img_height
                max_aspect = max_width / max_height
                
                if max_aspect > aspect:
                    # Max area is wider - fit to height
                    new_height = max_height
                    new_width = int(new_height * aspect)
                else:
                    # Max area is taller - fit to width
                    new_width = max_width
                    new_height = int(new_width / aspect)
                
                # Always resize to fit preview area (scale down if needed, scale up if smaller)
                frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image=frame_pil)
                
                # Update preview label in main thread
                if self.root:
                    self.root.after(0, lambda p=photo: self._update_preview_frame(p))
                
                # Advance timeline time
                self.edit_state.preview_time += frame_duration
                
                # Update timeline playhead in main thread (throttled to ~10fps for smooth animation)
                # Update every 3 frames (roughly 10 times per second at 30fps)
                if self.root:
                    self.timeline_update_counter += 1
                    if self.timeline_update_counter >= 3:
                        self.timeline_update_counter = 0
                        self.root.after(0, self._draw_timeline)
                
                # Sleep to maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_duration - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Playback finished
            if self.root:
                self.edit_state.is_playing = False
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Preview finished ({timeline_duration:.1f}s)"
                ))
                self.root.after(0, self._update_playback_controls)
                # Update timeline to show final playhead position
                self.root.after(0, self._draw_timeline)
                
        except Exception as e:
            print(f"[VIDSOR] Playback error: {e}")
            import traceback
            traceback.print_exc()
            if self.root:
                self.root.after(0, lambda: self.status_label.config(text=f"Playback error: {str(e)}"))
                self.edit_state.is_playing = False
    
    def _audio_playback_loop(self):
        """Audio playback loop - plays audio synchronized with video timeline."""
        if not self.audio_clip or not HAS_PYGAME or not self.edit_state.chunks:
            return
        
        try:
            # Determine start time for audio playback
            # Always start from current preview_time to keep audio in sync with video
            # (preview_time is 0.0 at initial start, or current position after scrubbing)
            start_timeline_time = self.edit_state.preview_time
            self.audio_needs_restart = False  # Reset flag after using it
            
            # Build composite audio from timeline chunks starting from start_timeline_time
            audio_segments = []
            for chunk in self.edit_state.chunks:
                # Skip chunks that end before the start time
                if chunk.end_time <= start_timeline_time:
                    continue
                
                # Use original timing to extract from source audio
                extract_start = chunk.original_start_time if chunk.original_start_time is not None else chunk.start_time
                extract_end = chunk.original_end_time if chunk.original_end_time is not None else chunk.end_time
                
                # If this chunk contains the start time, trim it
                if chunk.start_time <= start_timeline_time < chunk.end_time:
                    # Calculate position within this chunk
                    chunk_position = (start_timeline_time - chunk.start_time) / (chunk.end_time - chunk.start_time)
                    # Map to original video time
                    if chunk.original_start_time is not None and chunk.original_end_time is not None:
                        extract_start = chunk.original_start_time + chunk_position * (chunk.original_end_time - chunk.original_start_time)
                    else:
                        extract_start = chunk.start_time + chunk_position * (chunk.end_time - chunk.start_time)
                
                try:
                    # Extract audio segment
                    audio_segment = self.audio_clip.subclipped(extract_start, extract_end)
                    
                    # Note: Speed adjustment for audio is complex and may cause quality issues
                    # For now, we'll play audio at normal speed even if video chunk has speed != 1.0
                    # This ensures audio plays correctly synchronized with video
                    
                    audio_segments.append(audio_segment)
                except Exception as e:
                    print(f"[VIDSOR] Warning: Could not extract audio segment {extract_start}-{extract_end}s: {e}")
                    continue
            
            if not audio_segments:
                print("[VIDSOR] No audio segments to play")
                return
            
            # Concatenate audio segments
            from moviepy import concatenate_audioclips
            try:
                composite_audio = concatenate_audioclips(audio_segments)
            except ImportError:
                # Fallback for older MoviePy versions
                from moviepy.editor import concatenate_audioclips
                composite_audio = concatenate_audioclips(audio_segments)
            
            # Create a temporary audio file for playback
            import tempfile
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Write composite audio to temporary file
            try:
                # Try with verbose parameter (MoviePy 2.x)
                try:
                    composite_audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                except TypeError:
                    # Fallback for MoviePy 1.x which doesn't support verbose parameter
                    composite_audio.write_audiofile(temp_audio_path)
            except Exception as e:
                print(f"[VIDSOR] Warning: Could not write audio file: {e}")
                # Cleanup
                composite_audio.close()
                for seg in audio_segments:
                    seg.close()
                return
            
            # Cleanup audio segments
            composite_audio.close()
            for seg in audio_segments:
                seg.close()
            
            # Load and play audio
            pygame.mixer.music.load(temp_audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish or stop
            # Use a flag to track if we should continue playing
            audio_playing = True
            while audio_playing:
                # Check if music is still playing
                is_busy = pygame.mixer.music.get_busy()
                
                # Check if paused or stopped
                if not self.edit_state.is_playing:
                    # If paused, wait for resume
                    while not self.edit_state.is_playing:
                        time.sleep(0.1)
                        # Check if we should stop (e.g., user clicked stop)
                        if not self.edit_state.has_started_playback:
                            audio_playing = False
                            break
                        # Note: Don't break if music stops being busy while paused
                        # The music should stay "busy" when paused, but if it doesn't,
                        # we'll handle it when resuming
                    
                    # If resumed, try to unpause
                    if self.edit_state.is_playing and audio_playing:
                        try:
                            # Check if music is still playing/busy
                            if pygame.mixer.music.get_busy():
                                # Music is still active - unpause it
                                pygame.mixer.music.unpause()
                            else:
                                # Music stopped while paused - can't resume from same position
                                # This shouldn't normally happen, but if it does, exit
                                print("[VIDSOR] Warning: Audio stopped while paused, cannot resume")
                                audio_playing = False
                        except Exception as e:
                            print(f"[VIDSOR] Audio unpause error: {e}")
                            # If unpause fails, the music might have stopped
                            audio_playing = False
                else:
                    # Playing normally
                    if not is_busy:
                        # Music finished playing
                        audio_playing = False
                    else:
                        time.sleep(0.1)
            
            # Cleanup
            try:
                pygame.mixer.music.stop()
                os.unlink(temp_audio_path)
            except:
                pass
                
        except Exception as e:
            print(f"[VIDSOR] Audio playback error: {e}")
            import traceback
            traceback.print_exc()
            try:
                pygame.mixer.music.stop()
            except:
                pass
    
    def _seek_to_time(self, timeline_time: float):
        """Seek to a specific time in the timeline."""
        self.playback_controller.seek_to_time(timeline_time)
        """
        Seek to a specific time in the timeline and update the video preview.
        
        Args:
            timeline_time: Time in the timeline (in seconds)
        """
        if not self.video_clip:
            return
        
        if not HAS_PIL:
            return
        
        # Update preview time
        self.edit_state.preview_time = timeline_time
        
        # Stop audio if playing (audio will restart from new position when resumed)
        if self.edit_state.is_playing and HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except:
                pass
            self.audio_needs_restart = True
        
        # Mark that playback has started so Resume button appears
        self.edit_state.has_started_playback = True
        if self.root:
            self.root.after(0, self._update_playback_controls)
        
        # Calculate timeline duration
        if not self.edit_state.chunks:
            timeline_duration = self.video_clip.duration
            # If no chunks, just use the timeline time directly
            original_time = max(0, min(timeline_time, self.video_clip.duration - 0.1))
        else:
            timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
            
            # Find which chunk we're in
            current_chunk = None
            for chunk in self.edit_state.chunks:
                if chunk.start_time <= timeline_time < chunk.end_time:
                    current_chunk = chunk
                    break
            
            if current_chunk:
                # Calculate position within this chunk (0.0 to 1.0)
                chunk_position = (timeline_time - current_chunk.start_time) / (current_chunk.end_time - current_chunk.start_time)
                
                # Map to original video time
                if current_chunk.original_start_time is not None and current_chunk.original_end_time is not None:
                    original_time = current_chunk.original_start_time + chunk_position * (current_chunk.original_end_time - current_chunk.original_start_time)
                else:
                    # Fallback to sequential timing
                    original_time = current_chunk.start_time + chunk_position * (current_chunk.end_time - current_chunk.start_time)
            else:
                # Between chunks or past end - use last chunk's end or video duration
                if self.edit_state.chunks:
                    last_chunk = self.edit_state.chunks[-1]
                    if last_chunk.original_end_time is not None:
                        original_time = last_chunk.original_end_time
                    else:
                        original_time = last_chunk.end_time
                else:
                    original_time = timeline_time
            
            # Ensure we're within video bounds
            original_time = max(0, min(original_time, self.video_clip.duration - 0.1))
        
        try:
            # Get frame from source video at original time
            frame = self.video_clip.get_frame(original_time)
            
            # Convert frame to PIL Image
            frame_pil = Image.fromarray(frame)
            
            # Resize to fit preview canvas (maintain aspect ratio)
            max_width = 800
            max_height = 500
            
            # Calculate resize dimensions maintaining aspect ratio
            img_width, img_height = frame_pil.size
            aspect = img_width / img_height
            max_aspect = max_width / max_height
            
            if max_aspect > aspect:
                # Max area is wider - fit to height
                new_height = max_height
                new_width = int(new_height * aspect)
            else:
                # Max area is taller - fit to width
                new_width = max_width
                new_height = int(new_width / aspect)
            
            # Resize frame
            frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=frame_pil)
            
            # Update preview frame in main thread
            if self.root:
                self.root.after(0, lambda p=photo: self._update_preview_frame(p))
            
            # Update timeline to show new playhead position
            if self.timeline_canvas:
                self._draw_timeline()
                
        except Exception as e:
            print(f"[VIDSOR] Error seeking to time {timeline_time}: {e}")
    
    def _update_preview_frame(self, photo):
        """Update preview canvas with new frame (called from main thread)."""
        if self.preview_canvas:
            # Get canvas size (update first to get accurate size)
            self.preview_canvas.update_idletasks()
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Use actual canvas size or fallback
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 500
            
            # Get image size
            img_width = photo.width()
            img_height = photo.height()
            
            # Calculate position to center image
            x = (canvas_width - img_width) // 2
            y = (canvas_height - img_height) // 2
            
            # Clear canvas and draw image (centered)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.preview_canvas.image = photo  # Keep a reference to prevent garbage collection
            
            # Hide label, show canvas
            if self.preview_label:
                self.preview_label.pack_forget()
            if not self.preview_canvas.winfo_viewable():
                self.preview_canvas.pack(fill=tk.BOTH, expand=True)
    
    def _render_preview_from_timeline(self):
        """Render video preview from timeline.json chunks."""
        if not self.video_clip or not self.edit_state.chunks:
            return
        
        clips = []
        
        for chunk in self.edit_state.chunks:
            # Use original timing to extract from source video
            extract_start = chunk.original_start_time if chunk.original_start_time is not None else chunk.start_time
            extract_end = chunk.original_end_time if chunk.original_end_time is not None else chunk.end_time
            
            try:
                # Extract subclip from source video
                subclip = self.video_clip.subclipped(extract_start, extract_end)
                
                # Apply speed if needed
                if chunk.speed != 1.0:
                    original_fps = subclip.fps
                    original_duration = subclip.duration
                    subclip = subclip.set_fps(original_fps * chunk.speed)
                    subclip = subclip.set_duration(original_duration / chunk.speed)
                
                clips.append(subclip)
            except Exception as e:
                print(f"[VIDSOR] Warning: Failed to extract clip {extract_start}-{extract_end}s: {e}")
                continue
        
        if not clips:
            raise Exception("No clips could be extracted from timeline")
        
        # Concatenate all clips
        preview_clip = concatenate_videoclips(clips)
        
        # Store preview clip (will be used for playback)
        if hasattr(self, 'preview_clip') and self.preview_clip:
            self.preview_clip.close()
        self.preview_clip = preview_clip
        
        print(f"[VIDSOR] Preview rendered: {len(clips)} clips, total duration: {preview_clip.duration:.2f}s")
        
        # Clean up individual clips (preview_clip has its own copy)
        for clip in clips:
            clip.close()
    
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
        """Handle timeline click."""
        self.timeline_controller.on_timeline_click(event)
        """Handle timeline click for chunk selection or playhead dragging."""
        if not self.timeline_canvas:
            return
        
        # Pause playback if it's currently running for better drag handling
        if self.edit_state.is_playing:
            self.edit_state.is_playing = False
            self._update_playback_controls()
        
        # Get canvas coordinates
        canvas_x = self.timeline_canvas.canvasx(event.x)
        
        # Calculate timeline duration and scale
        if not self.edit_state.chunks:
            if not self.video_clip:
                return
            timeline_duration = self.video_clip.duration
        else:
            timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
        
        canvas_width = self.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
        
        # Check if clicking on/near the playhead (within 10 pixels)
        playhead_x = self.edit_state.preview_time * scale
        playhead_tolerance = 10
        
        if abs(canvas_x - playhead_x) <= playhead_tolerance:
            # Clicked on playhead - start dragging
            self.is_dragging_playhead = True
            # Stop playback and audio if playing
            if self.edit_state.is_playing:
                self.edit_state.is_playing = False
            # Stop audio completely
            if HAS_PYGAME:
                try:
                    pygame.mixer.music.stop()
                except:
                    pass
            # Mark that audio needs restart from new position
            self.audio_needs_restart = True
            # Mark that playback has started so Resume button appears
            self.edit_state.has_started_playback = True
            self._update_playback_controls()
        else:
            # Clicked elsewhere on timeline - seek to that position
            new_time = canvas_x / scale
            new_time = max(0, min(new_time, timeline_duration))
            self._seek_to_time(new_time)
            # Start dragging from this position
            self.is_dragging_playhead = True
    
    def _on_timeline_drag(self, event):
        """Handle timeline drag."""
        self.timeline_controller.on_timeline_drag(event)
        """Handle timeline drag for playhead scrubbing."""
        if not self.is_dragging_playhead or not self.timeline_canvas:
            return
        
        # Stop playback and audio if it's currently running for better drag handling
        if self.edit_state.is_playing:
            self.edit_state.is_playing = False
        # Stop audio completely
        if HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        # Mark that audio needs restart from new position
        self.audio_needs_restart = True
        # Mark that playback has started so Resume button appears
        self.edit_state.has_started_playback = True
        self._update_playback_controls()
        
        # Get canvas coordinates
        canvas_x = self.timeline_canvas.canvasx(event.x)
        
        # Calculate timeline duration and scale
        if not self.edit_state.chunks:
            if not self.video_clip:
                return
            timeline_duration = self.video_clip.duration
        else:
            timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
        
        canvas_width = self.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
        
        # Calculate new time from mouse position
        new_time = canvas_x / scale
        new_time = max(0, min(new_time, timeline_duration))
        
        # Seek to new time
        self._seek_to_time(new_time)
    
    def _on_timeline_release(self, event):
        """Handle mouse release after timeline drag."""
        self.timeline_controller.on_timeline_release(event)
    
    def _on_timeline_motion(self, event):
        """Handle mouse motion over timeline."""
        self.timeline_controller.on_timeline_motion(event)
        """Handle mouse motion over timeline for hover effects."""
        # Don't update hover if dragging playhead
        if self.is_dragging_playhead:
            return
        
        if not self.edit_state.chunks or not self.timeline_canvas:
            return
        
        # Get canvas coordinates
        canvas_x = self.timeline_canvas.canvasx(event.x)
        
        # Calculate timeline duration and scale
        timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
        canvas_width = self.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / timeline_duration
        
        # Find which chunk is being hovered
        hovered_chunk_idx = None
        for i, chunk in enumerate(self.edit_state.chunks):
            x_start = chunk.start_time * scale
            x_end = chunk.end_time * scale
            if x_start <= canvas_x <= x_end:
                hovered_chunk_idx = i
                break
        
        # Update hover state and redraw if changed
        if hovered_chunk_idx != self.timeline_hover_chunk:
            self.timeline_hover_chunk = hovered_chunk_idx
            self._draw_timeline()
    
    def _on_timeline_leave(self, event):
        """Handle mouse leaving timeline."""
        self.timeline_controller.on_timeline_leave(event)
    
    def _draw_timeline(self):
        """Draw professional timeline with chunks, playhead, and modern styling."""
        self.timeline_controller.draw_timeline()
        if not self.timeline_canvas:
            return
        
        self.timeline_canvas.delete("all")
        
        # Timeline dimensions
        TIMELINE_HEIGHT = 220
        RULER_HEIGHT = 35
        CHUNK_AREA_TOP = RULER_HEIGHT
        CHUNK_AREA_HEIGHT = 140
        CHUNK_AREA_BOTTOM = CHUNK_AREA_TOP + CHUNK_AREA_HEIGHT
        TIME_MARKER_HEIGHT = 12
        
        # Calculate timeline duration
        if not self.edit_state.chunks:
            if not self.video_clip:
                return
            timeline_duration = self.video_clip.duration
        else:
            timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
        
        canvas_width = self.timeline_canvas.winfo_width() or 1000
        if canvas_width <= 1:
            canvas_width = 1000
        scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
        
        # Draw background
        self.timeline_canvas.create_rectangle(
            0, 0, canvas_width, TIMELINE_HEIGHT,
            fill="#1a1a1a", outline="", tags="background"
        )
        
        # Draw ruler background (darker)
        self.timeline_canvas.create_rectangle(
            0, 0, canvas_width, RULER_HEIGHT,
            fill="#0f0f0f", outline="", tags="ruler_bg"
        )
        
        # Draw time markers with professional styling
        # Major markers every 10 seconds
        for t in range(0, int(timeline_duration) + 1, 10):
            x = t * scale
            if x > canvas_width:
                break
            
            # Major tick line
            self.timeline_canvas.create_line(
                x, RULER_HEIGHT - TIME_MARKER_HEIGHT, x, RULER_HEIGHT,
                fill="#666666", width=2, tags="time_marker"
            )
            
            # Time label
            time_str = self._format_time(t)
            self.timeline_canvas.create_text(
                x, RULER_HEIGHT - TIME_MARKER_HEIGHT - 8,
                text=time_str, font=("Segoe UI", 9, "normal"),
                fill="#cccccc", anchor="s", tags="time_label"
            )
        
        # Minor markers every 5 seconds (between major markers)
        for t in range(5, int(timeline_duration) + 1, 10):
            x = t * scale
            if x > canvas_width:
                break
            self.timeline_canvas.create_line(
                x, RULER_HEIGHT - TIME_MARKER_HEIGHT // 2, x, RULER_HEIGHT,
                fill="#444444", width=1, tags="time_marker_minor"
            )
        
        # Draw chunks with professional styling
        if self.edit_state.chunks:
            for i, chunk in enumerate(self.edit_state.chunks):
                x_start = chunk.start_time * scale
                x_end = chunk.end_time * scale
                width = max(x_end - x_start, 2)  # Minimum width
                
                if x_end < 0 or x_start > canvas_width:
                    continue  # Skip chunks outside visible area
                
                # Determine if this chunk is hovered or selected
                is_hovered = (self.timeline_hover_chunk == i)
                is_selected = (self.edit_state.selected_chunk == i)
                
                # Get colors
                fill_color, outline_color, gradient_color = self._get_chunk_color(
                    chunk.chunk_type, is_hovered, is_selected
                )
                
                # Draw chunk with gradient effect (simulated with multiple rectangles)
                chunk_y_top = CHUNK_AREA_TOP + 5
                chunk_y_bottom = CHUNK_AREA_BOTTOM - 5
                chunk_height = chunk_y_bottom - chunk_y_top
                
                # Main chunk rectangle with rounded corners effect (using polygon)
                # Draw shadow first (darker rectangle for depth effect)
                shadow_offset = 2
                self.timeline_canvas.create_rectangle(
                    x_start + shadow_offset, chunk_y_top + shadow_offset,
                    x_end + shadow_offset, chunk_y_bottom + shadow_offset,
                    fill="#0a0a0a", outline="", tags=f"chunk_shadow_{i}"
                )
                
                # Main chunk body
                self.timeline_canvas.create_rectangle(
                    x_start, chunk_y_top, x_end, chunk_y_bottom,
                    fill=fill_color, outline=outline_color, width=2,
                    tags=f"chunk_{i}"
                )
                
                # Gradient effect (top lighter, bottom darker)
                gradient_height = chunk_height // 3
                self.timeline_canvas.create_rectangle(
                    x_start, chunk_y_top, x_end, chunk_y_top + gradient_height,
                    fill=gradient_color, outline="", tags=f"chunk_gradient_{i}"
                )
                
                # Draw chunk label with better typography
                if width > 60:  # Only draw label if chunk is wide enough
                    # Chunk type label
                    type_label = chunk.chunk_type.replace("_", " ").title()
                    if chunk.chunk_type == "highlight":
                        type_label = "★ Highlight"
                    
                    self.timeline_canvas.create_text(
                        x_start + width/2, chunk_y_top + 20,
                        text=type_label, font=("Segoe UI", 10, "bold"),
                        fill="#ffffff", tags=f"chunk_label_{i}"
                    )
                    
                    # Time range
                    time_range = f"{self._format_time(chunk.start_time)} - {self._format_time(chunk.end_time)}"
                    self.timeline_canvas.create_text(
                        x_start + width/2, chunk_y_top + 40,
                        text=time_range, font=("Segoe UI", 8),
                        fill="#ffffff", tags=f"chunk_time_{i}"
                    )
                    
                    # Duration
                    duration = chunk.end_time - chunk.start_time
                    duration_str = f"{duration:.1f}s"
                    if chunk.speed != 1.0:
                        duration_str += f" @ {chunk.speed}x"
                    self.timeline_canvas.create_text(
                        x_start + width/2, chunk_y_top + 55,
                        text=duration_str, font=("Segoe UI", 8),
                        fill="#dddddd", tags=f"chunk_duration_{i}"
                    )
                    
                    # Description (truncated if too long)
                    if chunk.description and width > 100:
                        desc = chunk.description
                        max_chars = int(width / 6)  # Approximate chars that fit
                        if len(desc) > max_chars:
                            desc = desc[:max_chars-3] + "..."
                        self.timeline_canvas.create_text(
                            x_start + width/2, chunk_y_top + 75,
                            text=desc, font=("Segoe UI", 7),
                            fill="#cccccc", width=int(width - 10),
                            tags=f"chunk_desc_{i}"
                        )
                else:
                    # Very narrow chunk - just show type icon
                    if chunk.chunk_type == "highlight":
                        self.timeline_canvas.create_text(
                            x_start + width/2, chunk_y_top + chunk_height/2,
                            text="★", font=("Segoe UI", 14),
                            fill="#ffffff", tags=f"chunk_icon_{i}"
                        )
        
        # Draw playhead indicator (red line showing current playback position)
        if self.edit_state.preview_time >= 0:
            playhead_x = self.edit_state.preview_time * scale
            if 0 <= playhead_x <= canvas_width:
                # Playhead line
                self.timeline_canvas.create_line(
                    playhead_x, 0, playhead_x, TIMELINE_HEIGHT,
                    fill="#ff0000", width=2, tags="playhead"
                )
                
                # Playhead triangle at top
                triangle_size = 8
                self.timeline_canvas.create_polygon(
                    playhead_x, 0,
                    playhead_x - triangle_size, triangle_size,
                    playhead_x + triangle_size, triangle_size,
                    fill="#ff0000", outline="#cc0000", width=1, tags="playhead_triangle"
                )
                
                # Current time label above playhead
                current_time_str = self._format_time(self.edit_state.preview_time)
                self.timeline_canvas.create_text(
                    playhead_x, triangle_size + 5,
                    text=current_time_str, font=("Segoe UI", 9, "bold"),
                    fill="#ff0000", anchor="n", tags="playhead_time"
                )
        
        # Draw separator line between ruler and chunks
        self.timeline_canvas.create_line(
            0, RULER_HEIGHT, canvas_width, RULER_HEIGHT,
            fill="#333333", width=1, tags="separator"
        )
        
        # Update scroll region
        timeline_width = max(canvas_width, timeline_duration * scale)
        self.timeline_canvas.configure(scrollregion=(0, 0, timeline_width, TIMELINE_HEIGHT))
    
    def export_video(self, output_path: str):
        """
        Export edited video to file.
        
        Args:
            output_path: Path to save output video
        """
        VideoExporter.export_video(self.video_clip, self.edit_state, output_path)
    
    def _create_chat_ui(self, parent_frame):
        """Create chat interface UI components."""
        self.agent_integration.create_chat_ui(parent_frame)
        # Sync UI references
        self.chat_text = self.agent_integration.chat_text
        self.chat_input = self.agent_integration.chat_input
        self.chat_send_btn = self.agent_integration.chat_send_btn
        self.chat_status_label = self.agent_integration.chat_status_label
        # Chat frame
        chat_frame = ttk.LabelFrame(parent_frame, text="Chat Assistant", padding="10")
        chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat history display (scrollable text widget)
        chat_history_frame = ttk.Frame(chat_frame)
        chat_history_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_history_frame.columnconfigure(0, weight=1)
        chat_history_frame.rowconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(chat_history_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.chat_text = tk.Text(
            chat_history_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            height=30,
            font=("Arial", 10),
            bg="#f5f5f5"
        )
        self.chat_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.chat_text.yview)
        
        # Chat input frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        input_frame.columnconfigure(0, weight=1)
        
        # Input text widget (multi-line)
        self.chat_input = tk.Text(
            input_frame,
            wrap=tk.WORD,
            height=3,
            font=("Arial", 10)
        )
        self.chat_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Send button
        self.chat_send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._on_send_message,
            state=tk.DISABLED
        )
        self.chat_send_btn.grid(row=0, column=1, sticky=tk.E)
        
        # Bind Enter key (with Shift for new line)
        self.chat_input.bind("<Return>", self._on_chat_input_return)
        self.chat_input.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for new line
        
        # Chat status label
        self.chat_status_label = ttk.Label(chat_frame, text="Ready", foreground="gray")
        self.chat_status_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Display existing chat history if any
        self._display_chat_history()
    
    def _on_chat_input_return(self, event):
        """Handle Enter key in chat input (send message, Shift+Enter for new line)."""
        if event.state & 0x1:  # Shift key is pressed
            return  # Allow default behavior (new line)
        else:
            self._on_send_message()
            return "break"  # Prevent default behavior
    
    def _on_send_message(self):
        """Handle send message button click."""
        self.agent_integration.on_send_message()
        if self.is_agent_running:
            messagebox.showwarning("Warning", "Agent is already processing a query. Please wait.")
            return
        
        # Get message from input
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Check if this is a response to a clarification
        if self.pending_clarification:
            # User is responding to clarification - continue with preserved state
            self._continue_with_clarification(message)
            return
        
        # Check if project and video are loaded
        if not self.current_project_path:
            messagebox.showwarning("Warning", "Please select a project first.")
            return
        
        if not self.video_path:
            messagebox.showwarning("Warning", "Please upload a video first.")
            return
        
        segment_tree_path = os.path.join(self.current_project_path, "segment_tree.json")
        if not os.path.exists(segment_tree_path):
            messagebox.showwarning(
                "Warning",
                "Segment tree not found. Please extract video features first using 'Upload Video'."
            )
            return
        
        # Clear input
        self.chat_input.delete("1.0", tk.END)
        
        # Add user message to history
        self._add_chat_message("user", message)
        
        # Run agent in background thread
        self.is_agent_running = True
        self.chat_send_btn.config(state=tk.DISABLED)
        self.chat_status_label.config(text="Processing query...", foreground="blue")
        
        self.agent_thread = threading.Thread(
            target=self._run_agent_thread,
            args=(message, segment_tree_path),
            daemon=True
        )
        self.agent_thread.start()
    
    def _continue_with_clarification(self, user_response: str):
        """Continue operation with user's clarification response using preserved state."""
        if not self.pending_clarification:
            return
        
        # Clear input
        self.chat_input.delete("1.0", tk.END)
        
        # Add user response to history
        self._add_chat_message("user", user_response)
        
        # Get preserved state
        preserved = self.pending_clarification
        operation = preserved["operation"]
        preserved_state = preserved["preserved_state"]
        original_query = preserved["original_query"]
        segment_tree_path = preserved["segment_tree_path"]
        timeline_path = preserved["timeline_path"]
        
        # Clear pending clarification
        self.pending_clarification = None
        
        # Run agent thread with clarification response
        self.is_agent_running = True
        self.chat_send_btn.config(state=tk.DISABLED)
        self.chat_status_label.config(text="Processing clarification...", foreground="blue")
        
        self.agent_thread = threading.Thread(
            target=self._run_agent_thread_with_clarification,
            args=(user_response, segment_tree_path, operation, preserved_state, original_query),
            daemon=True
        )
        self.agent_thread.start()
    
    def _run_agent_thread_with_clarification(self, clarification_response: str, segment_tree_path: str,
                                            operation: str, preserved_state: Dict, original_query: str):
        """Run orchestrator with clarification response, using preserved state to continue."""
        # Create logger for this clarification response
        log_file = create_log_file(f"{original_query}_clarification_{clarification_response}", output_dir="logs")
        logger = DualLogger(log_file=log_file, verbose=True)
        
        logger.info("=" * 80)
        logger.info("VIDSOR: Continuing with Clarification Response")
        logger.info("=" * 80)
        logger.info(f"Original query: {original_query}")
        logger.info(f"Clarification response: {clarification_response}")
        logger.info(f"Operation: {operation}")
        logger.info(f"Preserved state keys: {list(preserved_state.keys()) if preserved_state else []}")
        
        try:
            timeline_path = os.path.join(self.current_project_path, "timeline.json")
            
            # Use preserved state to continue - planner will refine results instead of re-searching
            # Combine original query with clarification response
            combined_query = f"{original_query} ({clarification_response})"
            
            logger.info(f"Combined query: {combined_query}")
            logger.info("Calling orchestrator with preserved state (planner will refine instead of re-searching)")
            logger.info(f"Preserved time_ranges: {len(preserved_state.get('time_ranges', []))}")
            logger.info(f"Preserved search_results: {len(preserved_state.get('search_results', []))}")
            
            result = run_orchestrator(
                query=combined_query,
                timeline_path=timeline_path,
                json_path=segment_tree_path,
                video_path=self.video_path,
                model_name="gpt-4o-mini",
                verbose=False,
                logger=logger,
                preserved_state=preserved_state  # Pass preserved state so planner can refine
            )
            
            # Process result same as normal flow
            operation_result = result.get("operation_result", {})
            success = result.get("success", False)
            timeline_chunks = result.get("timeline_chunks", [])
            
            # Generate response
            if not success:
                error_msg = operation_result.get("error", "Operation failed")
                logger.error(f"Operation failed: {error_msg}")
                response = f"Error: {error_msg}"
            elif operation == "FIND_HIGHLIGHTS":
                chunks_created = operation_result.get("chunks_created", [])
                if chunks_created:
                    response = f"Found and added {len(chunks_created)} clip(s) to timeline based on your clarification."
                else:
                    response = "No matching clips found based on your clarification."
            else:
                response = f"Operation completed based on your clarification."
            
            # Update timeline UI if succeeded
            if success and timeline_chunks is not None:
                if self.root:
                    def update_timeline_ui():
                        try:
                            self._load_timeline()
                            if self.timeline_canvas:
                                self._draw_timeline()
                            # Update UI button states (especially play button)
                            self._update_ui_state()
                        except Exception as e:
                            logger.error(f"Error updating timeline UI: {e}")
                    
                    self.root.after(0, update_timeline_ui)
            
            # Update UI
            if self.root:
                self.root.after(0, lambda: self._add_chat_message("assistant", response))
                self.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            error_msg = f"Error processing clarification: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            if self.root:
                self.root.after(0, lambda: self._add_chat_message("assistant", error_msg))
                self.root.after(0, lambda: self.chat_status_label.config(text="Error occurred", foreground="red"))
                self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
        finally:
            self.is_agent_running = False
            logger.info("Clarification processing completed")
    
    def _run_agent_thread(self, query: str, segment_tree_path: str):
        """Run orchestrator agent in background thread."""
        # Create logger for this query
        log_file = create_log_file(query, output_dir="logs")
        logger = DualLogger(log_file=log_file, verbose=True)
        
        logger.info("=" * 80)
        logger.info("VIDSOR: Orchestrator Query Processing")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Project: {self.current_project_path}")
        logger.info(f"Video: {self.video_path}")
        logger.info(f"Segment tree: {segment_tree_path}")
        
        try:
            # Get timeline path from current project
            if not self.current_project_path:
                error_msg = "No project selected. Please select a project first."
                logger.error(error_msg)
                if self.root:
                    self.root.after(0, lambda: self._add_chat_message("assistant", error_msg))
                    self.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                    self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
                return
            
            timeline_path = os.path.join(self.current_project_path, "timeline.json")
            logger.info(f"Timeline path: {timeline_path}")
            
            # Check if there's a pending clarification and if this query looks like a follow-up
            if self.pending_clarification:
                preserved = self.pending_clarification
                previous_query = preserved.get("original_query", "")
                preserved_state = preserved.get("preserved_state", {})
                previous_time_ranges = preserved_state.get("time_ranges", [])
                
                # Simple heuristic: if query contains refinement keywords or numbers, treat as follow-up
                query_lower = query.lower()
                refinement_keywords = ["top", "best", "first", "most", "select", "give me", "show me", "only"]
                has_refinement = any(kw in query_lower for kw in refinement_keywords)
                has_number = any(char.isdigit() for char in query)
                
                if has_refinement or has_number:
                    logger.info(f"Detected follow-up query to clarification")
                    logger.info(f"  Previous query: {previous_query}")
                    logger.info(f"  Current query: {query}")
                    logger.info(f"  Previous results: {len(previous_time_ranges)} time ranges")
                    logger.info("  Treating as clarification response - using preserved_state")
                    
                    # Use the clarification handler instead
                    operation = preserved.get("operation")
                    original_query = preserved.get("original_query")
                    
                    # Clear pending clarification
                    self.pending_clarification = None
                    
                    # Run with preserved state
                    self._run_agent_thread_with_clarification(
                        query,  # Use current query as clarification response
                        segment_tree_path,
                        operation,
                        preserved_state,
                        original_query
                    )
                    return
            
            # Check if timeline exists and log current state
            if os.path.exists(timeline_path):
                try:
                    # Check if file is empty or whitespace only
                    with open(timeline_path, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            logger.info("Timeline.json is empty - starting with empty timeline")
                        else:
                            # Parse JSON
                            with open(timeline_path, 'r') as f2:
                                timeline_data = json.load(f2)
                                chunks_count = len(timeline_data.get("chunks", []))
                                logger.info(f"Existing timeline loaded: {chunks_count} chunks")
                except json.JSONDecodeError as e:
                    logger.warning(f"Timeline.json contains invalid JSON: {e}")
                    logger.info("Starting with empty timeline")
                except Exception as e:
                    logger.warning(f"Could not read existing timeline: {e}")
                    logger.info("Starting with empty timeline")
            else:
                logger.info("Timeline.json does not exist - will be created")
            
            # Run orchestrator (it handles timeline.json loading, operations, and saving)
            logger.info("\n" + "-" * 80)
            logger.info("CALLING ORCHESTRATOR")
            logger.info("-" * 80)
            result = run_orchestrator(
                query=query,
                timeline_path=timeline_path,
                json_path=segment_tree_path,
                video_path=self.video_path,
                model_name="gpt-4o-mini",
                verbose=False,  # Use logger instead
                logger=logger  # Pass logger to orchestrator
            )
            
            logger.info("\n" + "-" * 80)
            logger.info("ORCHESTRATOR RESULTS")
            logger.info("-" * 80)
            
            # Extract orchestrator results
            operation = result.get("operation", "UNKNOWN")
            operation_result = result.get("operation_result", {})
            success = result.get("success", False)
            timeline_chunks = result.get("timeline_chunks", [])
            
            logger.info(f"Operation: {operation}")
            logger.info(f"Success: {success}")
            logger.info(f"Timeline chunks after operation: {len(timeline_chunks) if timeline_chunks else 0}")
            
            if operation_result:
                logger.info(f"Operation result keys: {list(operation_result.keys())}")
                if "chunks_created" in operation_result:
                    logger.info(f"  Chunks created: {len(operation_result['chunks_created'])}")
                if "chunks_removed" in operation_result:
                    logger.info(f"  Chunks removed: {len(operation_result['chunks_removed'])}")
                if "chunks_added" in operation_result:
                    logger.info(f"  Chunks added: {len(operation_result['chunks_added'])}")
                if "chunks_inserted" in operation_result:
                    logger.info(f"  Chunks inserted: {len(operation_result['chunks_inserted'])}")
            
            # Check if this is a clarification request (not a real error)
            needs_clarification = operation_result.get("needs_clarification", False)
            clarification_question = operation_result.get("clarification_question")
            preserved_state = operation_result.get("preserved_state")
            
            # If clarification_question is in error field but needs_clarification flag is missing, extract it
            if not clarification_question and not success:
                error_msg = operation_result.get("error", "")
                # Check if error message looks like a clarification question
                if error_msg and ("Could you narrow down" in error_msg or "potential moments" in error_msg.lower() or "Found" in error_msg and "moment" in error_msg.lower()):
                    clarification_question = error_msg
                    needs_clarification = True
                    logger.info(f"Detected clarification question in error field: {clarification_question}")
                    # If we detected clarification but preserved_state wasn't set, try to get it from operation_result
                    if not preserved_state:
                        preserved_state = operation_result.get("preserved_state")
                        if preserved_state:
                            logger.info(f"Found preserved_state in operation_result: {len(preserved_state.get('time_ranges', []))} time ranges")
                        else:
                            logger.warning("Clarification detected but no preserved_state found - follow-up queries may not work correctly")
                    
                    # Ensure previous_time_ranges is set for refinement logic to work
                    if preserved_state and "previous_time_ranges" not in preserved_state:
                        # If time_ranges exists but previous_time_ranges doesn't, copy it
                        if "time_ranges" in preserved_state:
                            preserved_state["previous_time_ranges"] = preserved_state["time_ranges"]
                            logger.info(f"Set previous_time_ranges from time_ranges: {len(preserved_state['time_ranges'])} ranges")
                        # Also set previous_query if not set
                        if "previous_query" not in preserved_state:
                            preserved_state["previous_query"] = query
                            logger.info(f"Set previous_query: {query}")
            
            # Generate response based on operation type
            response_parts = []
            
            if needs_clarification and clarification_question:
                # This is a clarification request, not an error
                logger.info(f"Clarification needed: {clarification_question}")
                logger.info("Preserving state for continuation")
                
                # Store preserved state for when user responds
                if self.root:
                    def store_clarification_state():
                        # Ensure previous_time_ranges is set for refinement logic
                        if preserved_state:
                            if "previous_time_ranges" not in preserved_state and "time_ranges" in preserved_state:
                                preserved_state["previous_time_ranges"] = preserved_state["time_ranges"]
                            if "previous_query" not in preserved_state:
                                preserved_state["previous_query"] = query
                        
                        self.pending_clarification = {
                            "operation": operation,
                            "preserved_state": preserved_state,
                            "original_query": query,
                            "segment_tree_path": segment_tree_path,
                            "timeline_path": timeline_path
                        }
                        logger.info("Clarification state stored")
                        if preserved_state:
                            logger.info(f"  Preserved {len(preserved_state.get('time_ranges', preserved_state.get('previous_time_ranges', [])))} time ranges")
                    
                    self.root.after(0, store_clarification_state)
                
                # Show clarification question in chat (without "Error:" prefix)
                response = f"❓ {clarification_question}\n\nPlease respond to continue."
            elif not success:
                error_msg = operation_result.get("error", "Operation failed")
                logger.error(f"Operation failed: {error_msg}")
                response = f"Error: {error_msg}"
            elif operation == "FIND_HIGHLIGHTS":
                chunks_created = operation_result.get("chunks_created", [])
                if chunks_created:
                    logger.info(f"FIND_HIGHLIGHTS: Created {len(chunks_created)} chunks")
                    response_parts.append(f"Found and added {len(chunks_created)} clip(s) to timeline:")
                    for i, chunk in enumerate(chunks_created, 1):
                        start = chunk.get("original_start_time", 0)
                        end = chunk.get("original_end_time", 0)
                        timeline_start = chunk.get("start_time", 0)
                        timeline_end = chunk.get("end_time", 0)
                        response_parts.append(f"  {i}. {start:.1f}s - {end:.1f}s")
                        logger.debug(f"  Chunk {i}: source={start:.1f}s-{end:.1f}s, timeline={timeline_start:.1f}s-{timeline_end:.1f}s")
                else:
                    logger.warning("FIND_HIGHLIGHTS: No chunks created")
                    response_parts.append("No matching clips found.")
                response = "\n".join(response_parts)
            elif operation == "CUT":
                chunks_removed = operation_result.get("chunks_removed", [])
                logger.info(f"CUT: Removed {len(chunks_removed)} chunks")
                response = f"Removed {len(chunks_removed)} clip(s) from timeline."
            elif operation == "REPLACE":
                chunks_added = operation_result.get("chunks_added", [])
                chunks_removed = operation_result.get("chunks_removed", [])
                logger.info(f"REPLACE: Removed {len(chunks_removed)}, Added {len(chunks_added)} chunks")
                response = f"Replaced {len(chunks_removed)} clip(s) with {len(chunks_added)} new clip(s)."
            elif operation == "INSERT":
                chunks_inserted = operation_result.get("chunks_inserted", [])
                logger.info(f"INSERT: Inserted {len(chunks_inserted)} chunks")
                response = f"Inserted {len(chunks_inserted)} clip(s) into timeline."
            elif operation == "FIND_BROLL":
                chunks_created = operation_result.get("chunks_created", [])
                logger.info(f"FIND_BROLL: Created {len(chunks_created)} B-roll chunks")
                response = f"Found and added {len(chunks_created)} B-roll clip(s) to timeline."
            elif operation == "UNKNOWN":
                logger.warning("Operation classification: UNKNOWN")
                response = "I couldn't understand what you want to do. Please try rephrasing your query."
            else:
                logger.info(f"Operation '{operation}' completed")
                response = f"Operation '{operation}' completed."
            
            logger.info(f"Response message: {response[:100]}...")
            
            # Update timeline UI if operation succeeded and timeline changed
            if success and timeline_chunks is not None:
                logger.info("\n" + "-" * 80)
                logger.info("UPDATING TIMELINE UI")
                logger.info("-" * 80)
                logger.info(f"Timeline chunks to display: {len(timeline_chunks)}")
                
                if self.root:
                    def update_timeline_ui():
                        try:
                            logger.info("Reloading timeline from file...")
                            # Reload timeline from file (orchestrator already saved it)
                            self._load_timeline()
                            logger.info(f"Timeline loaded: {len(self.edit_state.chunks)} chunks")
                            
                            # Update timeline display
                            if self.timeline_canvas:
                                logger.info("Drawing timeline on canvas...")
                                self._draw_timeline()
                                logger.info("Timeline canvas updated")
                            
                            # Update UI button states (especially play button)
                            logger.info("Updating UI button states...")
                            self._update_ui_state()
                            
                            logger.info(f"[VIDSOR] Timeline updated: {len(timeline_chunks)} chunks")
                        except Exception as e:
                            logger.error(f"Error updating timeline UI: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    self.root.after(0, update_timeline_ui)
            else:
                logger.info("Skipping timeline UI update (success=False or no timeline_chunks)")
            
            # Update UI in main thread
            logger.info("\n" + "-" * 80)
            logger.info("UPDATING CHAT UI")
            logger.info("-" * 80)
            if self.root:
                self.root.after(0, lambda: self._add_chat_message("assistant", response))
                self.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
                logger.info("Chat UI update scheduled")
            
            logger.info("\n" + "=" * 80)
            logger.info("QUERY PROCESSING COMPLETED")
            logger.info("=" * 80)
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error("\n" + "=" * 80)
            logger.error("ERROR OCCURRED")
            logger.error("=" * 80)
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            print(f"[VIDSOR] {error_msg}")
            traceback.print_exc()
            if self.root:
                self.root.after(0, lambda: self._add_chat_message("assistant", error_msg))
                self.root.after(0, lambda: self.chat_status_label.config(text="Error occurred", foreground="red"))
                self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
        finally:
            self.is_agent_running = False
            logger.info("Agent thread completed")
    
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
        print(f"[VIDSOR] _process_highlights called with {len(clips) if clips else 0} clips and {len(time_ranges) if time_ranges else 0} time ranges")
        
        # Check if timeline.json exists and has data
        timeline_path = os.path.join(self.current_project_path, "timeline.json") if self.current_project_path else None
        timeline_has_data = False
        if timeline_path and os.path.exists(timeline_path):
            try:
                with open(timeline_path, 'r') as f:
                    timeline_data = json.load(f)
                    chunks_data = timeline_data.get("chunks", [])
                    if chunks_data:
                        timeline_has_data = True
                        print(f"[VIDSOR] Timeline.json already has {len(chunks_data)} chunks, preserving existing data")
            except Exception as e:
                print(f"[VIDSOR] Error checking timeline.json: {e}")
        
        # Only process highlights if timeline is empty
        if timeline_has_data:
            print("[VIDSOR] Timeline.json has existing data, skipping AI-generated highlights")
            return
        
        if not self.segment_tree:
            print("[VIDSOR] No segment tree available for metadata extraction")
            # Still process highlights even without segment tree for metadata
            if not time_ranges and not clips:
                return
        else:
            print(f"[VIDSOR] Segment tree available, extracting metadata")
        
        highlight_chunks = []
        
        # Create a mapping from time ranges to clips
        clip_map = {}
        if clips and time_ranges:
            for clip_path, (start, end) in zip(clips, time_ranges):
                clip_map[(start, end)] = clip_path
        
        # Process time ranges (use clips if available, otherwise time_ranges)
        ranges_to_process = []
        if clips and time_ranges and len(clips) == len(time_ranges):
            # Use clips with their corresponding time ranges (perfect match)
            for clip_path, (start, end) in zip(clips, time_ranges):
                ranges_to_process.append((start, end, clip_path))
        elif clips:
            # We have clips but maybe no time_ranges or mismatch - extract timing from filenames
            import re
            for clip_path in clips:
                # Try to extract timing from filename like "clip_1_22s_to_28s_..."
                match = re.search(r'(\d+)s_to_(\d+)s', os.path.basename(clip_path))
                if match:
                    start = float(match.group(1))
                    end = float(match.group(2))
                    ranges_to_process.append((start, end, clip_path))
                else:
                    print(f"[VIDSOR] Warning: Could not extract timing from clip filename: {clip_path}")
        elif time_ranges:
            # Only time ranges, no clips extracted yet
            for start, end in time_ranges:
                ranges_to_process.append((start, end, None))
        else:
            print("[VIDSOR] No time ranges or clips to process")
            return
        
        # Sort by start time
        ranges_to_process.sort(key=lambda x: x[0])
        
        # Extract metadata for each range
        for start, end, clip_path in ranges_to_process:
            # Get metadata from segment tree
            unified_desc = None
            audio_desc = None
            
            # Try to get description from search_results first
            if search_results:
                for search_result in search_results:
                    result_time_range = search_result.get("time_range", [])
                    if len(result_time_range) >= 2:
                        result_start, result_end = result_time_range[0], result_time_range[1]
                        # Check if this search result overlaps with our time range
                        if not (result_end < start or result_start > end):
                            # Get text from search result
                            text = search_result.get("text", "")
                            result_type = search_result.get("type", "")
                            
                            # Also check for transcription text in different formats
                            if not text:
                                text = search_result.get("transcription", "")
                            
                            if result_type == "unified" and not unified_desc and text:
                                unified_desc = text
                            elif (result_type == "transcription" or result_type == "audio") and not audio_desc and text:
                                audio_desc = text
                            # Handle generic search results
                            elif not result_type and text:
                                # If we don't have unified_desc yet, use this as visual
                                if not unified_desc:
                                    unified_desc = text
            
            # If not found in search_results, query segment tree directly
            # Sample multiple points in the time range to get better descriptions
            if not unified_desc or not audio_desc:
                sample_times = [
                    start,
                    start + (end - start) * 0.25,
                    (start + end) / 2,
                    start + (end - start) * 0.75,
                    end
                ]
                
                for sample_time in sample_times:
                    second_data = self.segment_tree.get_second_by_time(sample_time)
                    
                    if second_data:
                        if not unified_desc:
                            desc = second_data.get("unified_description", "")
                            if desc and desc != "0":
                                unified_desc = desc
                                break  # Found one, move on
                        
                        if not audio_desc:
                            # Try to get audio transcription
                            transcription_id = second_data.get("transcription_id")
                            if transcription_id and transcription_id in self.segment_tree._transcription_map:
                                transcription = self.segment_tree._transcription_map[transcription_id]
                                audio = transcription.get("transcription", "")
                                if audio:
                                    audio_desc = audio
                                    break  # Found one, move on
                    
                    # If we found both, no need to continue
                    if unified_desc and audio_desc:
                        break
            
            # Build description from available sources
            description_parts = []
            if unified_desc:
                description_parts.append(f"Visual: {unified_desc}")
            if audio_desc:
                description_parts.append(f"Audio: {audio_desc}")
            
            description = " | ".join(description_parts) if description_parts else "Highlight moment"
            
            # Calculate sequential timeline position (clips appear one after another in edited timeline)
            # start_time/end_time = position in edited timeline (sequential)
            # original_start_time/original_end_time = position in source video
            clip_duration = end - start
            if highlight_chunks:
                # Start after the last chunk
                timeline_start = highlight_chunks[-1].end_time
            else:
                # First chunk starts at 0
                timeline_start = 0.0
            timeline_end = timeline_start + clip_duration
            
            # Create chunk
            chunk = Chunk(
                start_time=timeline_start,  # Sequential position in edited timeline
                end_time=timeline_end,     # Sequential position in edited timeline
                chunk_type="highlight",
                speed=1.0,
                description=description,
                score=1.0,  # Highlights have high score
                original_start_time=start,  # Original position in source video
                original_end_time=end,      # Original position in source video
                unified_description=unified_desc,
                audio_description=audio_desc,
                clip_path=clip_path
            )
            
            highlight_chunks.append(chunk)
        
        # Add highlight chunks to edit state
        # Replace existing chunks if they're also highlights, otherwise append
        existing_highlights = [c for c in self.edit_state.chunks if c.chunk_type == "highlight"]
        if existing_highlights:
            # Replace existing highlights
            other_chunks = [c for c in self.edit_state.chunks if c.chunk_type != "highlight"]
            self.edit_state.chunks = other_chunks + highlight_chunks
        else:
            # Append to existing chunks
            self.edit_state.chunks.extend(highlight_chunks)
        
        # Sort all chunks by start_time (highlights are already sorted by original timing)
        self.edit_state.chunks.sort(key=lambda x: x.start_time)
        
        # Save timeline to timeline.json
        self._save_timeline()
        
        # Update timeline display
        print(f"[VIDSOR] Drawing timeline with {len(self.edit_state.chunks)} total chunks")
        self._draw_timeline()
        
        # Update UI state
        self._update_ui_state()
        
        # Force UI update
        if self.root:
            self.root.update_idletasks()
        
        print(f"[VIDSOR] Processed {len(highlight_chunks)} highlight clips and added to timeline")
        print(f"[VIDSOR] Total chunks in edit_state: {len(self.edit_state.chunks)}")
    
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

