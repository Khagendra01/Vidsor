"""
Video loading and analysis functionality for Vidsor.
"""

import os
from pathlib import Path
from typing import List, Optional
from moviepy import VideoFileClip
from agent.utils.segment_tree_utils import load_segment_tree, SegmentTreeQuery
from ..models import Chunk


class VideoAnalyzer:
    """Handles video loading and analysis."""
    
    def __init__(self):
        """Initialize video analyzer."""
        self.video_path: Optional[str] = None
        self.segment_tree_path: Optional[str] = None
        self.video_clip: Optional[VideoFileClip] = None
        self.segment_tree: Optional[SegmentTreeQuery] = None
    
    def load_video(self, video_path: Optional[str] = None) -> None:
        """
        Load video file with MoviePy.
        
        Args:
            video_path: Path to video file (if None, uses self.video_path)
        """
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
                    self.load_segment_tree()
            
        except Exception as e:
            raise Exception(f"Failed to load video: {str(e)}")
    
    def load_segment_tree(self, segment_tree_path: Optional[str] = None) -> None:
        """
        Load segment tree for analysis.
        
        Args:
            segment_tree_path: Path to segment tree JSON (if None, uses self.segment_tree_path)
        """
        if segment_tree_path:
            self.segment_tree_path = segment_tree_path
        
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
    
    def close(self):
        """Close video clip."""
        if self.video_clip:
            self.video_clip.close()

