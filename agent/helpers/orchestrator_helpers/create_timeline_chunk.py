"""Create a timeline chunk dictionary."""

from typing import Dict, Optional


def create_timeline_chunk(
    original_start: float,
    original_end: float,
    timeline_start: float,
    chunk_type: str = "highlight",
    description: str = "",
    unified_description: str = "",
    audio_description: str = "",
    score: float = 1.0,
    clip_path: Optional[str] = None
) -> Dict:
    """
    Create a timeline chunk dictionary.
    
    Args:
        original_start: Start time in source video
        original_end: End time in source video
        timeline_start: Start time in timeline
        chunk_type: Type of chunk (highlight, broll, etc.)
        description: Visual description
        unified_description: Unified visual description
        audio_description: Audio description
        score: Relevance score
        clip_path: Path to extracted clip file
        
    Returns:
        Chunk dictionary
    """
    duration = original_end - original_start
    return {
        "start_time": timeline_start,
        "end_time": timeline_start + duration,
        "chunk_type": chunk_type,
        "speed": 1.0,
        "description": description or f"Clip from {original_start:.1f}s to {original_end:.1f}s",
        "score": score,
        "original_start_time": original_start,
        "original_end_time": original_end,
        "unified_description": unified_description or description,
        "audio_description": audio_description or "",
        "clip_path": clip_path
    }

