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

from agent.segment_tree_utils import load_segment_tree, SegmentTreeQuery
from agent import run_agent
from extractor.pipeline import SegmentTreePipeline
from extractor.config import ExtractorConfig


@dataclass
class Chunk:
    """Represents a video chunk with metadata."""
    start_time: float
    end_time: float
    chunk_type: str  # "normal", "fast_forward", "highlight"
    speed: float = 1.0  # Playback speed (1.0 = normal, 2.0 = 2x speed)
    description: str = ""
    score: float = 0.0  # Interest score
    # Metadata for agent-extracted clips
    original_start_time: Optional[float] = None  # Original timing in source video
    original_end_time: Optional[float] = None
    unified_description: Optional[str] = None  # Visual description
    audio_description: Optional[str] = None  # Audio transcription
    clip_path: Optional[str] = None  # Path to extracted clip file


@dataclass
class EditState:
    """Current editing state."""
    chunks: List[Chunk]
    selected_chunk: Optional[int] = None
    preview_time: float = 0.0
    is_playing: bool = False
    has_started_playback: bool = False  # Track if playback has started (for resume button)


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
        self.preview_clip: Optional[VideoFileClip] = None  # Rendered preview from timeline
        self.playback_thread: Optional[threading.Thread] = None  # Thread for video playback
        self.audio_thread: Optional[threading.Thread] = None  # Thread for audio playback
        self.audio_clip = None  # Audio clip for playback
        self.segment_tree: Optional[SegmentTreeQuery] = None
        self.edit_state = EditState(chunks=[])
        
        # Project management
        self.current_project_path: Optional[str] = None
        self.projects_dir = os.path.join(os.getcwd(), "projects")
        self._ensure_projects_dir()
        
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
        
        # Chat interface components
        self.chat_history: List[Dict[str, str]] = []  # List of {"role": "user"/"assistant", "content": "..."}
        self.chat_text: Optional[tk.Text] = None
        self.chat_input: Optional[tk.Text] = None
        self.chat_send_btn: Optional[ttk.Button] = None
        self.chat_status_label: Optional[ttk.Label] = None
        self.agent_thread: Optional[threading.Thread] = None
        self.is_agent_running = False
        
        # Load video and segment tree if provided
        if self.video_path:
            self._load_video()
            if self.segment_tree_path:
                self._load_segment_tree()
    
    def _ensure_projects_dir(self):
        """Ensure projects directory exists."""
        if not os.path.exists(self.projects_dir):
            os.makedirs(self.projects_dir)
    
    def create_new_project(self, project_name: str) -> str:
        """
        Create a new project folder structure.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Path to the created project folder
        """
        # Sanitize project name
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            raise ValueError("Invalid project name")
        
        project_path = os.path.join(self.projects_dir, safe_name)
        
        if os.path.exists(project_path):
            raise ValueError(f"Project '{safe_name}' already exists")
        
        # Create project structure
        os.makedirs(project_path)
        os.makedirs(os.path.join(project_path, "video"))
        
        # Create project config
        config = {
            "project_name": safe_name,
            "created_at": datetime.now().isoformat(),
            "video_filename": None,
            "segment_tree_path": None
        }
        
        config_path = os.path.join(project_path, "project_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[VIDSOR] Created project: {project_path}")
        return project_path
    
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
    
    def upload_video_to_project(self, video_path: str, project_path: str) -> str:
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
        if not os.path.exists(self.projects_dir):
            return []
        
        projects = []
        for item in os.listdir(self.projects_dir):
            project_path = os.path.join(self.projects_dir, item)
            if os.path.isdir(project_path):
                config_path = os.path.join(project_path, "project_config.json")
                if os.path.exists(config_path):
                    projects.append(item)
        return sorted(projects)
    
    def _load_video(self, video_path: Optional[str] = None):
        """Load video file with MoviePy."""
        if video_path:
            self.video_path = video_path
        
        if not self.video_path:
            raise Exception("No video path provided")
        
        try:
            # Close existing video if any
            if self.video_clip:
                self.video_clip.close()
            
            self.video_clip = VideoFileClip(self.video_path)
            print(f"[VIDSOR] Video loaded: {self.video_path}")
            print(f"  Duration: {self.video_clip.duration:.2f}s")
            print(f"  FPS: {self.video_clip.fps}")
            print(f"  Resolution: {self.video_clip.size}")
            
            # Try to auto-detect segment tree
            if not self.segment_tree_path:
                video_dir = os.path.dirname(self.video_path)
                video_name = Path(self.video_path).stem
                potential_tree = os.path.join(video_dir, f"{video_name}_segment_tree.json")
                if os.path.exists(potential_tree):
                    self.segment_tree_path = potential_tree
                    self._load_segment_tree()
            
            # Update UI state
            self._update_ui_state()
            
        except Exception as e:
            raise Exception(f"Failed to load video: {str(e)}")
    
    def _load_segment_tree(self):
        """Load segment tree for analysis."""
        if not self.segment_tree_path or not os.path.exists(self.segment_tree_path):
            print("[VIDSOR] No segment tree provided. Will analyze video directly.")
            return
        
        try:
            self.segment_tree = load_segment_tree(self.segment_tree_path)
            print(f"[VIDSOR] Segment tree loaded: {self.segment_tree_path}")
        except Exception as e:
            print(f"[VIDSOR] Warning: Failed to load segment tree: {e}")
            self.segment_tree = None
    
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
        self.root = tk.Tk()
        self.root.title("Vidsor - Video Editor")
        self.root.geometry("1600x900")
        
        # Main container with paned window for resizable split
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left panel - Video editor
        left_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(left_frame, weight=2)
        
        # Right panel - Chat interface
        right_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(right_frame, weight=1)
        
        # Main container (for left panel)
        main_frame = ttk.Frame(left_frame, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        
        # Project management frame
        project_frame = ttk.LabelFrame(main_frame, text="Project", padding="15")
        project_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        project_frame.columnconfigure(0, weight=1)
        
        # New Project button - first row with generous spacing
        new_project_btn = ttk.Button(project_frame, text="New Project", command=self._on_new_project)
        new_project_btn.grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5, 15))
        
        # Project selection row - second row
        ttk.Label(project_frame, text="Project:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.project_combo = ttk.Combobox(project_frame, state="readonly", width=30)
        self.project_combo.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.project_combo.bind("<<ComboboxSelected>>", self._on_project_selected)
        
        self.project_label = ttk.Label(project_frame, text="No project selected")
        self.project_label.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        project_frame.columnconfigure(1, weight=1)
        
        # Update project list
        self._update_project_list()
        
        # Preview area
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        preview_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Preview label with fixed minimum size to prevent collapsing
        # Use a Canvas for better image display control
        self.preview_canvas = tk.Canvas(
            preview_frame,
            bg="black",
            highlightthickness=0
        )
        # Canvas initially hidden, will be shown when playing
        
        # Also keep label for text display
        self.preview_label = tk.Label(
            preview_frame,
            text="No project selected\n\nCreate a new project and upload a video to get started",
            bg="black",
            fg="white",
            font=("Arial", 12),
            anchor="center",
            justify="center"
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Set minimum size for preview to prevent collapsing
        preview_frame.grid_rowconfigure(0, weight=1, minsize=450)
        preview_frame.grid_columnconfigure(0, weight=1, minsize=800)
        
        # Timeline
        timeline_frame = ttk.LabelFrame(main_frame, text="Timeline", padding="10")
        timeline_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        timeline_frame.columnconfigure(0, weight=1)
        
        # Canvas for timeline
        canvas_frame = ttk.Frame(timeline_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        canvas_frame.columnconfigure(0, weight=1)
        
        self.timeline_canvas = tk.Canvas(
            canvas_frame,
            height=150,
            bg="white",
            scrollregion=(0, 0, 1000, 150)
        )
        self.timeline_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for timeline
        timeline_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.timeline_canvas.xview)
        timeline_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.timeline_canvas.configure(xscrollcommand=timeline_scroll.set)
        
        # Controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Buttons
        self.load_video_btn = ttk.Button(controls_frame, text="Upload Video", command=self._on_load_video)
        self.load_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(controls_frame, text="Play Preview", command=self._on_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(controls_frame, text="Pause", command=self._on_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=5)
        
        self.export_btn = ttk.Button(controls_frame, text="Export", command=self._on_export, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Ready - No project selected")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Bind timeline click
        self.timeline_canvas.bind("<Button-1>", self._on_timeline_click)
        
        # Initialize UI state
        self._update_ui_state()
        
        # Create chat interface
        self._create_chat_ui(right_frame)
    
    def _update_project_list(self):
        """Update the project dropdown list."""
        if self.project_combo:
            projects = self.get_available_projects()
            self.project_combo['values'] = projects
            
            # Select current project if any
            if self.current_project_path:
                project_name = os.path.basename(self.current_project_path)
                if project_name in projects:
                    self.project_combo.set(project_name)
                    self.project_label.config(text=f"Active: {project_name}")
                else:
                    self.project_label.config(text="No project selected")
            else:
                self.project_combo.set("")
                self.project_label.config(text="No project selected")
    
    def _on_new_project(self):
        """Handle new project button click."""
        project_name = simpledialog.askstring(
            "New Project",
            "Enter project name:",
            parent=self.root
        )
        
        if not project_name:
            return
        
        try:
            project_path = self.create_new_project(project_name)
            self.set_current_project(project_path)
            self._update_project_list()
            messagebox.showinfo("Success", f"Project '{project_name}' created successfully!")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def _on_project_selected(self, event=None):
        """Handle project selection from dropdown."""
        selected = self.project_combo.get()
        if not selected:
            return
        
        project_path = os.path.join(self.projects_dir, selected)
        if os.path.exists(project_path):
            self.set_current_project(project_path)
            self._update_project_list()
            self._update_ui_state()
            # Update status
            if self.status_label:
                project_name = os.path.basename(project_path)
                self.status_label.config(text=f"Project '{project_name}' selected")
            # Reload chat history for the new project
            self._load_chat_history()
            self._display_chat_history()
    
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
        if not self.video_clip:
            messagebox.showwarning("Warning", "No video loaded")
            return
        
        if not self.edit_state.chunks:
            messagebox.showwarning("Warning", "No clips in timeline. Load timeline.json or generate highlights first.")
            return
        
        # If already playing, do nothing
        if self.edit_state.is_playing:
            return
        
        # Start playback directly from source video using timeline chunks
        self._start_playback_from_timeline()
    
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
            while self.edit_state.is_playing and self.edit_state.preview_time < timeline_duration:
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
                
                # Sleep to maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_duration - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Check if we should stop
                if not self.edit_state.is_playing:
                    break
            
            # Playback finished
            if self.root:
                self.edit_state.is_playing = False
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Preview finished ({timeline_duration:.1f}s)"
                ))
                self.root.after(0, self._update_playback_controls)
                
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
            # Build composite audio from timeline chunks
            audio_segments = []
            for chunk in self.edit_state.chunks:
                # Use original timing to extract from source audio
                extract_start = chunk.original_start_time if chunk.original_start_time is not None else chunk.start_time
                extract_end = chunk.original_end_time if chunk.original_end_time is not None else chunk.end_time
                
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
            while pygame.mixer.music.get_busy():
                # Check if paused or stopped
                if not self.edit_state.is_playing:
                    # If paused, wait for resume
                    while not self.edit_state.is_playing:
                        time.sleep(0.1)
                        if not pygame.mixer.music.get_busy():
                            break
                    # If resumed, unpause
                    if self.edit_state.is_playing:
                        try:
                            pygame.mixer.music.unpause()
                        except:
                            pass
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
        if self.edit_state.is_playing:
            # Pause
            self.edit_state.is_playing = False
            # Keep has_started_playback = True so resume button stays enabled
            self.status_label.config(text="Paused")
            
            # Pause audio
            if HAS_PYGAME:
                try:
                    pygame.mixer.music.pause()
                except:
                    pass
            
            # Update playback controls - this will set Resume button correctly
            self._update_playback_controls()
            
            # Keep canvas visible when paused (don't switch back to label)
        else:
            # Resume
            self.edit_state.is_playing = True
            timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks) if self.edit_state.chunks else 0
            self.status_label.config(text=f"Playing preview... ({timeline_duration:.1f}s)")
            
            # Resume audio
            if HAS_PYGAME:
                try:
                    pygame.mixer.music.unpause()
                except:
                    pass
            
            # Update playback controls - this will set Pause button correctly
            self._update_playback_controls()
    
    def _on_stop(self):
        """Stop preview button handler."""
        self.edit_state.is_playing = False
        self.edit_state.preview_time = 0.0
        self.edit_state.has_started_playback = False  # Reset playback flag
        self.status_label.config(text="Stopped")
        
        # Stop audio
        if HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        
        # Update playback controls
        self._update_playback_controls()
        
        # Reset preview display
        if self.preview_canvas:
            self.preview_canvas.delete("all")
            self.preview_canvas.pack_forget()
        
        if self.preview_label:
            if has_video := self.video_clip is not None:
                timeline_info = ""
                if self.edit_state.chunks:
                    highlight_count = sum(1 for c in self.edit_state.chunks if c.chunk_type == "highlight")
                    total_chunks = len(self.edit_state.chunks)
                    timeline_info = f"\nTimeline: {total_chunks} chunks ({highlight_count} highlights)"
                
                self.preview_label.config(
                    text=f"Video: {os.path.basename(self.video_path)}\n\n"
                         f"Duration: {self.video_clip.duration:.1f}s\n"
                         f"FPS: {self.video_clip.fps}{timeline_info}\n\n"
                         f"Click Play Preview to view timeline"
                )
                self.preview_label.pack(fill=tk.BOTH, expand=True)
            else:
                self.preview_label.config(text="No video loaded")
                self.preview_label.pack(fill=tk.BOTH, expand=True)
    
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
    
    def _on_timeline_click(self, event):
        """Handle timeline click for chunk selection."""
        # TODO: Implement chunk selection and trimming
        pass
    
    def _draw_timeline(self):
        """Draw timeline with chunks."""
        print(f"[VIDSOR] _draw_timeline called - canvas: {self.timeline_canvas is not None}, chunks: {len(self.edit_state.chunks) if self.edit_state.chunks else 0}, video: {self.video_clip is not None}")
        
        if not self.timeline_canvas:
            print("[VIDSOR] Timeline canvas not available")
            return
        
        if not self.edit_state.chunks:
            print("[VIDSOR] No chunks to draw")
            # Still draw empty timeline with time markers
            if not self.video_clip:
                return
            self.timeline_canvas.delete("all")
            duration = self.video_clip.duration
            canvas_width = self.timeline_canvas.winfo_width() or 1000
            scale = canvas_width / duration
            # Draw time markers only
            y_start = 20
            for t in range(0, int(duration) + 1, 10):
                x = t * scale
                self.timeline_canvas.create_line(x, 0, x, y_start, fill="gray")
                self.timeline_canvas.create_text(x, y_start/2, text=f"{t}s", font=("Arial", 8))
            self.timeline_canvas.configure(scrollregion=(0, 0, duration * scale, 150))
            return
        
        self.timeline_canvas.delete("all")
        
        if not self.video_clip:
            print("[VIDSOR] No video clip available for timeline")
            return
        
        # Calculate timeline duration (sum of all chunks in edited timeline)
        if self.edit_state.chunks:
            timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
        else:
            timeline_duration = self.video_clip.duration if self.video_clip else 100
        
        canvas_width = self.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / timeline_duration
        print(f"[VIDSOR] Drawing timeline - timeline_duration: {timeline_duration}s, canvas_width: {canvas_width}, scale: {scale}")
        
        # Draw chunks
        y_start = 20
        y_height = 100
        
        for i, chunk in enumerate(self.edit_state.chunks):
            x_start = chunk.start_time * scale
            x_end = chunk.end_time * scale
            width = x_end - x_start
            print(f"[VIDSOR] Drawing chunk {i}: {chunk.chunk_type} at {chunk.start_time:.1f}s-{chunk.end_time:.1f}s (x: {x_start:.1f}-{x_end:.1f})")
            
            # Color based on chunk type
            if chunk.chunk_type == "highlight":
                color = "yellow"
            elif chunk.chunk_type == "fast_forward":
                color = "lightblue"
            else:
                color = "lightgreen"
            
            # Draw chunk rectangle
            self.timeline_canvas.create_rectangle(
                x_start, y_start, x_end, y_start + y_height,
                fill=color, outline="black", width=2, tags=f"chunk_{i}"
            )
            
            # Draw label
            if chunk.chunk_type == "highlight" and chunk.original_start_time is not None:
                # Show original timing and description for highlights
                label = f"Highlight\n{chunk.original_start_time:.1f}s-{chunk.original_end_time:.1f}s"
                if chunk.description:
                    # Truncate description if too long
                    desc = chunk.description[:40] + "..." if len(chunk.description) > 40 else chunk.description
                    label += f"\n{desc}"
            else:
                label = f"{chunk.chunk_type}\n{chunk.start_time:.1f}s-{chunk.end_time:.1f}s"
                if chunk.speed != 1.0:
                    label += f"\n{chunk.speed}x"
            
            self.timeline_canvas.create_text(
                x_start + width/2, y_start + y_height/2,
                text=label, font=("Arial", 8), tags=f"chunk_{i}"
            )
        
        # Draw time markers based on timeline duration
        for t in range(0, int(timeline_duration) + 1, 10):
            x = t * scale
            self.timeline_canvas.create_line(x, 0, x, y_start, fill="gray")
            self.timeline_canvas.create_text(x, y_start/2, text=f"{t}s", font=("Arial", 8))
        
        # Update scroll region
        self.timeline_canvas.configure(scrollregion=(0, 0, timeline_duration * scale, 150))
    
    def export_video(self, output_path: str):
        """
        Export edited video to file.
        
        Args:
            output_path: Path to save output video
        """
        if not self.video_clip or not self.edit_state.chunks:
            raise Exception("No video or chunks to export")
        
        clips = []
        
        for chunk in self.edit_state.chunks:
            # Use original timing to extract from source video
            # If original timing is not available, fall back to start_time/end_time
            extract_start = chunk.original_start_time if chunk.original_start_time is not None else chunk.start_time
            extract_end = chunk.original_end_time if chunk.original_end_time is not None else chunk.end_time
            
            # Extract subclip from source video using original timing
            subclip = self.video_clip.subclipped(extract_start, extract_end)
            
            # Apply speed if needed
            if chunk.speed != 1.0:
                # Adjust speed by changing FPS and duration
                original_fps = subclip.fps
                original_duration = subclip.duration
                subclip = subclip.set_fps(original_fps * chunk.speed)
                subclip = subclip.set_duration(original_duration / chunk.speed)
            
            clips.append(subclip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)
        
        # Write to file
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            remove_temp=True
        )
        
        # Cleanup
        final_clip.close()
        for clip in clips:
            clip.close()
        
        print(f"[VIDSOR] Video exported to: {output_path}")
    
    def _create_chat_ui(self, parent_frame):
        """Create chat interface UI components."""
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
        if self.is_agent_running:
            messagebox.showwarning("Warning", "Agent is already processing a query. Please wait.")
            return
        
        # Get message from input
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
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
    
    def _run_agent_thread(self, query: str, segment_tree_path: str):
        """Run agent in background thread."""
        try:
            # Run agent
            result = run_agent(
                query=query,
                json_path=segment_tree_path,
                video_path=self.video_path,
                model_name="gpt-4o-mini",
                verbose=False
            )
            
            # Extract response
            if result.get("needs_clarification"):
                response = f"Clarification needed: {result.get('clarification_question', 'Please provide more details.')}"
                if self.root:
                    self.root.after(0, lambda: self._add_chat_message("assistant", response))
                    self.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                    self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
            else:
                clips = result.get("output_clips", [])
                time_ranges = result.get("time_ranges", [])
                search_results = result.get("search_results", [])
                confidence = result.get("confidence", 0)
                
                response_parts = []
                if clips:
                    response_parts.append(f"Found {len(clips)} clip(s):")
                    for i, clip in enumerate(clips, 1):
                        clip_name = os.path.basename(clip)
                        response_parts.append(f"  {i}. {clip_name}")
                elif time_ranges:
                    response_parts.append(f"Found {len(time_ranges)} time range(s):")
                    for i, (start, end) in enumerate(time_ranges, 1):
                        response_parts.append(f"  {i}. {start:.1f}s - {end:.1f}s")
                else:
                    response_parts.append("No matching clips found.")
                
                if confidence is not None:
                    response_parts.append(f"\nConfidence: {confidence:.2%}")
                
                response = "\n".join(response_parts)
                
                # Check if this is a highlights query and process clips
                query_lower = query.lower()
                is_highlights_query = any(keyword in query_lower for keyword in [
                    "highlight", "highlights", "best moments", "best parts", 
                    "interesting moments", "key moments", "important moments",
                    "find the highlights", "find highlights", "show highlights"
                ])
                
                if is_highlights_query and (clips or time_ranges):
                    # Process highlights and add to timeline
                    # Use a proper closure to avoid lambda variable capture issues
                    if self.root:
                        def process_highlights_wrapper():
                            try:
                                num_items = len(clips) if clips else len(time_ranges) if time_ranges else 0
                                print(f"[VIDSOR] Processing {num_items} highlights for timeline")
                                if self.status_label:
                                    self.status_label.config(text=f"Processing {num_items} highlights and adding to timeline...")
                                self._process_highlights(result, clips, time_ranges, search_results)
                                if self.status_label:
                                    self.status_label.config(text=f"Added {num_items} highlights to timeline")
                            except Exception as e:
                                print(f"[VIDSOR] Error processing highlights: {e}")
                                import traceback
                                traceback.print_exc()
                                if self.status_label:
                                    self.status_label.config(text=f"Error processing highlights: {str(e)}")
                        
                        self.root.after(0, process_highlights_wrapper)
                
                # Update UI in main thread
                if self.root:
                    self.root.after(0, lambda: self._add_chat_message("assistant", response))
                    self.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                    self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"[VIDSOR] {error_msg}")
            if self.root:
                self.root.after(0, lambda: self._add_chat_message("assistant", error_msg))
                self.root.after(0, lambda: self.chat_status_label.config(text="Error occurred", foreground="red"))
                self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
        finally:
            self.is_agent_running = False
    
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
        # Add to history
        self.chat_history.append({"role": role, "content": content})
        
        # Save to file
        self._save_chat_history()
        
        # Display in chat text widget
        self.chat_text.config(state=tk.NORMAL)
        
        # Format message
        if role == "user":
            prefix = "You: "
            tag = "user"
        else:
            prefix = "Assistant: "
            tag = "assistant"
        
        # Get start position before inserting
        start_pos = self.chat_text.index(tk.END)
        self.chat_text.insert(tk.END, f"{prefix}{content}\n\n")
        # Get end position after inserting (before the two newlines)
        end_pos = self.chat_text.index(f"{start_pos}+{len(prefix)+len(content)}c")
        
        # Apply tags for styling
        self.chat_text.tag_add(tag, start_pos, end_pos)
        
        # Configure tag styles
        self.chat_text.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("assistant", foreground="green", font=("Arial", 10))
        
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)
    
    def _save_chat_history(self):
        """Save chat history to project folder."""
        if not self.current_project_path:
            return
        
        chat_history_path = os.path.join(self.current_project_path, "chat_history.json")
        try:
            with open(chat_history_path, 'w') as f:
                json.dump(self.chat_history, f, indent=2)
        except Exception as e:
            print(f"[VIDSOR] Failed to save chat history: {e}")
    
    def _load_timeline(self):
        """Load timeline from timeline.json in project folder."""
        if not self.current_project_path:
            self.edit_state.chunks = []
            return
        
        timeline_path = os.path.join(self.current_project_path, "timeline.json")
        if os.path.exists(timeline_path):
            try:
                with open(timeline_path, 'r') as f:
                    timeline_data = json.load(f)
                
                # Check if timeline is empty
                chunks_data = timeline_data.get("chunks", [])
                if not chunks_data:
                    print("[VIDSOR] Timeline.json is empty, will be filled with AI-generated clips")
                    self.edit_state.chunks = []
                    return
                
                # Load chunks from timeline.json
                chunks = []
                for chunk_data in chunks_data:
                    chunk = Chunk(
                        start_time=chunk_data.get("start_time", 0.0),
                        end_time=chunk_data.get("end_time", 0.0),
                        chunk_type=chunk_data.get("chunk_type", "normal"),
                        speed=chunk_data.get("speed", 1.0),
                        description=chunk_data.get("description", ""),
                        score=chunk_data.get("score", 0.0),
                        original_start_time=chunk_data.get("original_start_time"),
                        original_end_time=chunk_data.get("original_end_time"),
                        unified_description=chunk_data.get("unified_description"),
                        audio_description=chunk_data.get("audio_description"),
                        clip_path=chunk_data.get("clip_path")
                    )
                    chunks.append(chunk)
                
                self.edit_state.chunks = chunks
                print(f"[VIDSOR] Loaded {len(chunks)} chunks from timeline.json")
                
                # Update timeline display if UI is ready
                if self.root and self.timeline_canvas:
                    self.root.after(0, self._draw_timeline)
                
            except Exception as e:
                print(f"[VIDSOR] Failed to load timeline: {e}")
                import traceback
                traceback.print_exc()
                self.edit_state.chunks = []
        else:
            # Timeline.json doesn't exist, create empty one
            print("[VIDSOR] timeline.json not found, will be created when clips are added")
            self.edit_state.chunks = []
            self._save_timeline()  # Create empty timeline.json
    
    def _save_timeline(self):
        """Save timeline to timeline.json in project folder."""
        if not self.current_project_path:
            return
        
        timeline_path = os.path.join(self.current_project_path, "timeline.json")
        try:
            # Convert chunks to JSON-serializable format
            chunks_data = []
            for chunk in self.edit_state.chunks:
                chunk_dict = {
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "chunk_type": chunk.chunk_type,
                    "speed": chunk.speed,
                    "description": chunk.description,
                    "score": chunk.score,
                    "original_start_time": chunk.original_start_time,
                    "original_end_time": chunk.original_end_time,
                    "unified_description": chunk.unified_description,
                    "audio_description": chunk.audio_description,
                    "clip_path": chunk.clip_path
                }
                chunks_data.append(chunk_dict)
            
            timeline_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "chunks": chunks_data
            }
            
            with open(timeline_path, 'w') as f:
                json.dump(timeline_data, f, indent=2)
            
            print(f"[VIDSOR] Saved {len(chunks_data)} chunks to timeline.json")
            
        except Exception as e:
            print(f"[VIDSOR] Failed to save timeline: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_chat_history(self):
        """Load chat history from project folder."""
        if not self.current_project_path:
            self.chat_history = []
            return
        
        chat_history_path = os.path.join(self.current_project_path, "chat_history.json")
        if os.path.exists(chat_history_path):
            try:
                with open(chat_history_path, 'r') as f:
                    self.chat_history = json.load(f)
            except Exception as e:
                print(f"[VIDSOR] Failed to load chat history: {e}")
                self.chat_history = []
        else:
            self.chat_history = []
    
    def _display_chat_history(self):
        """Display all chat history in the chat text widget."""
        if not self.chat_text:
            return
        
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        
        for msg in self.chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                prefix = "You: "
                tag = "user"
            else:
                prefix = "Assistant: "
                tag = "assistant"
            
            start_pos = self.chat_text.index(tk.END)
            self.chat_text.insert(tk.END, f"{prefix}{content}\n\n")
            end_pos = self.chat_text.index(tk.END)
            
            # Apply tags
            self.chat_text.tag_add(tag, start_pos, f"{end_pos}-2c")
        
        # Configure tag styles
        self.chat_text.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("assistant", foreground="green", font=("Arial", 10))
        
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)
        
        # Update send button state
        if self.chat_send_btn:
            has_project = self.current_project_path is not None
            has_video = self.video_path is not None
            has_segment_tree = self.segment_tree_path is not None and os.path.exists(self.segment_tree_path)
            self.chat_send_btn.config(state=tk.NORMAL if (has_project and has_video and has_segment_tree) else tk.DISABLED)
    
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


def main():
    """Main entry point for Vidsor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vidsor - Video Editor")
    parser.add_argument("video", nargs="?", help="Path to video file (optional - can load via UI)")
    parser.add_argument("--segment-tree", help="Path to segment tree JSON (optional)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't open UI (requires video)")
    
    args = parser.parse_args()
    
    # If analyze-only mode, video is required
    if args.analyze_only:
        if not args.video:
            print("Error: Video file required for --analyze-only mode")
            return
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        
        editor = Vidsor(args.video, args.segment_tree)
        chunks = editor.analyze_video()
        print(f"\nGenerated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. {chunk.chunk_type}: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s "
                  f"(speed: {chunk.speed}x)")
        editor.close()
    else:
        # Start editor (empty or with video if provided)
        editor = Vidsor(args.video, args.segment_tree)
        editor.run()
        editor.close()


if __name__ == "__main__":
    main()

