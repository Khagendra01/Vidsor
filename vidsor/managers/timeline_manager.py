"""
Timeline persistence and management for Vidsor.
"""

import os
import json
from datetime import datetime
from typing import List, Optional
from ..models import Chunk


class TimelineManager:
    """Handles timeline loading and saving."""
    
    @staticmethod
    def load_timeline(project_path: Optional[str]) -> List[Chunk]:
        """
        Load timeline from timeline.json in project folder.
        
        Args:
            project_path: Path to project folder
            
        Returns:
            List of Chunk objects
        """
        if not project_path:
            return []
        
        timeline_path = os.path.join(project_path, "timeline.json")
        if not os.path.exists(timeline_path):
            # Timeline.json doesn't exist, create empty one
            print("[VIDSOR] timeline.json not found, will be created when clips are added")
            TimelineManager.save_timeline(project_path, [])
            return []
        
        try:
            # Check if file is empty or whitespace only
            with open(timeline_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    print("[VIDSOR] timeline.json is empty, starting with empty timeline")
                    return []
            
            # Parse JSON
            with open(timeline_path, 'r') as f:
                timeline_data = json.load(f)
            
            # Validate structure
            if not isinstance(timeline_data, dict):
                print("[VIDSOR] timeline.json has invalid structure, starting with empty timeline")
                return []
            
            # Check if timeline is empty
            chunks_data = timeline_data.get("chunks", [])
            if not chunks_data:
                print("[VIDSOR] Timeline.json has no chunks, will be filled with AI-generated clips")
                return []
            
            # Load chunks from timeline.json
            chunks = []
            for chunk_data in chunks_data:
                if not isinstance(chunk_data, dict):
                    continue  # Skip invalid chunk entries
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
            
            print(f"[VIDSOR] Loaded {len(chunks)} chunks from timeline.json")
            return chunks
            
        except json.JSONDecodeError as e:
            print(f"[VIDSOR] timeline.json contains invalid JSON, starting with empty timeline")
            return []
        except Exception as e:
            print(f"[VIDSOR] Failed to load timeline: {e}")
            return []
    
    @staticmethod
    def save_timeline(project_path: Optional[str], chunks: List[Chunk]) -> None:
        """
        Save timeline to timeline.json in project folder.
        
        Args:
            project_path: Path to project folder
            chunks: List of Chunk objects to save
        """
        if not project_path:
            return
        
        timeline_path = os.path.join(project_path, "timeline.json")
        try:
            # Convert chunks to JSON-serializable format
            chunks_data = []
            for chunk in chunks:
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

