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
    # MoviePy 2.x imports (direct from moviepy)
    from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips
except ImportError:
    try:
        # Fallback for MoviePy 1.x
        from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
    except ImportError:
        raise ImportError("MoviePy is required. Install with: pip install moviepy")

from agent.segment_tree_utils import load_segment_tree, SegmentTreeQuery
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


@dataclass
class EditState:
    """Current editing state."""
    chunks: List[Chunk]
    selected_chunk: Optional[int] = None
    preview_time: float = 0.0
    is_playing: bool = False


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
        self._ensure_projects_dir()
        
        # UI components
        self.root: Optional[tk.Tk] = None
        self.preview_label: Optional[tk.Label] = None
        self.timeline_canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[ttk.Label] = None
        self.load_video_btn: Optional[ttk.Button] = None
        self.analyze_btn: Optional[ttk.Button] = None
        self.play_btn: Optional[ttk.Button] = None
        self.export_btn: Optional[ttk.Button] = None
        self.project_label: Optional[ttk.Label] = None
        self.project_combo: Optional[ttk.Combobox] = None
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.extraction_thread: Optional[threading.Thread] = None
        self.is_extracting = False
        
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
                        "Video extraction complete!\n\nSegment tree has been generated and loaded."
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
        self.root.geometry("1200x800")
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
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
        
        self.preview_label = tk.Label(
            preview_frame,
            text="No project selected\n\nCreate a new project and upload a video to get started",
            bg="black",
            fg="white",
            width=80,
            height=20,
            font=("Arial", 12)
        )
        self.preview_label.pack()
        
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
        
        self.analyze_btn = ttk.Button(controls_frame, text="Analyze Video", command=self._on_analyze, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(controls_frame, text="Play Preview", command=self._on_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
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
                    messagebox.showinfo(
                        "Success", 
                        f"Video uploaded successfully!\n\n{os.path.basename(video_path)}\n\nSegment tree already exists and has been loaded."
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
                            f"Video uploaded successfully!\n\n{os.path.basename(video_path)}\n\nYou can extract features later using the 'Analyze Video' button."
                        )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {str(e)}")
                self.status_label.config(text="Failed to load video")
    
    def _update_ui_state(self):
        """Update UI button states based on video loading status."""
        has_video = self.video_clip is not None
        
        if self.analyze_btn:
            self.analyze_btn.config(state=tk.NORMAL if has_video else tk.DISABLED)
        if self.play_btn:
            self.play_btn.config(state=tk.NORMAL if has_video else tk.DISABLED)
        if self.export_btn:
            self.export_btn.config(state=tk.NORMAL if has_video and self.edit_state.chunks else tk.DISABLED)
        
        # Update preview label
        if self.preview_label:
            if has_video:
                self.preview_label.config(text=f"Video: {os.path.basename(self.video_path)}\n\nDuration: {self.video_clip.duration:.1f}s\nFPS: {self.video_clip.fps}")
            elif self.current_project_path:
                project_name = os.path.basename(self.current_project_path)
                self.preview_label.config(text=f"Project: {project_name}\n\nNo video uploaded\n\nClick 'Upload Video' to add a video")
            else:
                self.preview_label.config(text="No project selected\n\nCreate a new project and upload a video to get started")
    
    def _on_analyze(self):
        """Analyze video button handler."""
        if not self.video_clip:
            messagebox.showwarning("Warning", "No video loaded. Please load a video first.")
            return
        
        self.status_label.config(text="Analyzing video...")
        self.root.update()
        
        try:
            chunks = self.analyze_video()
            self._draw_timeline()
            self.status_label.config(text=f"Analysis complete: {len(chunks)} chunks")
            self._update_ui_state()  # Enable export button
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_label.config(text="Analysis failed")
    
    def _on_play(self):
        """Play preview button handler."""
        if not self.video_clip:
            messagebox.showwarning("Warning", "No video loaded")
            return
        
        self.edit_state.is_playing = True
        self.status_label.config(text="Playing...")
        # TODO: Implement video preview playback
    
    def _on_stop(self):
        """Stop preview button handler."""
        self.edit_state.is_playing = False
        self.status_label.config(text="Stopped")
    
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
        if not self.timeline_canvas or not self.edit_state.chunks:
            return
        
        self.timeline_canvas.delete("all")
        
        if not self.video_clip:
            return
        
        duration = self.video_clip.duration
        canvas_width = self.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / duration
        
        # Draw chunks
        y_start = 20
        y_height = 100
        
        for i, chunk in enumerate(self.edit_state.chunks):
            x_start = chunk.start_time * scale
            x_end = chunk.end_time * scale
            width = x_end - x_start
            
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
            label = f"{chunk.chunk_type}\n{chunk.start_time:.1f}s-{chunk.end_time:.1f}s"
            if chunk.speed != 1.0:
                label += f"\n{chunk.speed}x"
            
            self.timeline_canvas.create_text(
                x_start + width/2, y_start + y_height/2,
                text=label, font=("Arial", 8), tags=f"chunk_{i}"
            )
        
        # Draw time markers
        for t in range(0, int(duration) + 1, 10):
            x = t * scale
            self.timeline_canvas.create_line(x, 0, x, y_start, fill="gray")
            self.timeline_canvas.create_text(x, y_start/2, text=f"{t}s", font=("Arial", 8))
        
        # Update scroll region
        self.timeline_canvas.configure(scrollregion=(0, 0, duration * scale, 150))
    
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
            # Extract subclip
            subclip = self.video_clip.subclipped(chunk.start_time, chunk.end_time)
            
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

