"""Merge agent for intelligently grouping and merging video segments."""

from typing import Dict, List, Tuple, Optional, Any
from agent.timeline_manager import TimelineManager


def create_merge_agent():
    """
    Create a merge agent that intelligently groups time ranges based on:
    - Timeline state (empty vs existing content)
    - Operation type (highlights vs broll vs insert)
    - Duration constraints
    - Narrative coherence
    - Neighbor analysis
    """
    
    def merge_node(state: Dict) -> Dict:
        """
        Merge time ranges intelligently based on context.
        
        Args:
            state: State containing:
                - time_ranges: List of (start, end) tuples
                - timeline_manager: TimelineManager instance
                - operation_type: Type of operation (FIND_HIGHLIGHTS, FIND_BROLL, etc.)
                - video_duration: Total video duration
                - query: User query
        
        Returns:
            Updated state with merged_ranges
        """
        time_ranges = state.get("time_ranges", [])
        timeline_manager: Optional[TimelineManager] = state.get("timeline_manager")
        operation_type = state.get("operation_type", "FIND_HIGHLIGHTS")
        video_duration = state.get("video_duration", 600.0)
        query = state.get("query", "")
        
        if not time_ranges:
            return {**state, "merged_ranges": []}
        
        # Analyze timeline state
        is_empty_timeline = True
        existing_duration = 0.0
        existing_chunks = []
        
        if timeline_manager:
            try:
                timeline_manager.load()
                existing_chunks = timeline_manager.chunks
                existing_duration = timeline_manager.calculate_timeline_duration()
                is_empty_timeline = len(existing_chunks) == 0
            except:
                pass
        
        # Determine merge strategy based on context
        if operation_type == "FIND_HIGHLIGHTS":
            if is_empty_timeline:
                # New timeline: can create longer highlights
                merged_ranges = _merge_for_new_highlights(
                    time_ranges,
                    video_duration,
                    query
                )
            else:
                # Existing timeline: need to fit with neighbors
                merged_ranges = _merge_with_neighbors(
                    time_ranges,
                    existing_chunks,
                    query
                )
        elif operation_type == "FIND_BROLL":
            # B-roll: shorter clips, fit between main content
            merged_ranges = _merge_for_broll(
                time_ranges,
                existing_chunks,
                query
            )
        else:
            # Default: conservative merging
            merged_ranges = _merge_conservative(time_ranges)
        
        return {
            **state,
            "merged_ranges": merged_ranges,
            "merge_strategy": operation_type
        }
    
    return merge_node


def _merge_for_new_highlights(
    time_ranges: List[Tuple[float, float]],
    video_duration: float,
    query: str
) -> List[Tuple[float, float]]:
    """
    Merge strategy for new highlight timeline.
    
    Rules:
    - Target: 10-15% of video duration
    - Max clip duration: 15-20 seconds
    - Min gap between clips: 3 seconds
    - Merge nearby seconds (within 2s gap)
    """
    if not time_ranges:
        return []
    
    # Sort by start time
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    
    # Target duration: 12% of video
    target_duration = video_duration * 0.12
    max_clip_duration = 20.0  # Max 20 seconds per clip
    min_gap = 3.0  # Min 3 seconds between clips
    merge_gap = 2.0  # Merge if within 2 seconds
    
    merged = []
    current_start, current_end = sorted_ranges[0]
    total_duration = 0.0
    
    for start, end in sorted_ranges[1:]:
        # Check if we've exceeded target duration
        clip_duration = current_end - current_start
        if total_duration + clip_duration > target_duration:
            # Save current and start new (but still process remaining)
            if clip_duration <= max_clip_duration:
                merged.append((current_start, current_end))
                total_duration += clip_duration
            current_start, current_end = start, end
            continue
        
        # Check if can merge (within gap and not too long)
        gap = start - current_end
        if gap <= merge_gap:
            # Can merge
            new_end = end
            new_duration = new_end - current_start
            if new_duration <= max_clip_duration:
                current_end = new_end
            else:
                # Too long, save current and start new
                merged.append((current_start, current_end))
                total_duration += current_end - current_start
                current_start, current_end = start, end
        else:
            # Gap too large, save current and start new
            merged.append((current_start, current_end))
            total_duration += current_end - current_start
            current_start, current_end = start, end
    
    # Add final range
    if current_start is not None:
        clip_duration = current_end - current_start
        if total_duration + clip_duration <= target_duration * 1.2:  # Allow 20% over
            merged.append((current_start, current_end))
    
    return merged


def _merge_with_neighbors(
    time_ranges: List[Tuple[float, float]],
    existing_chunks: List[Dict],
    query: str
) -> List[Tuple[float, float]]:
    """
    Merge strategy when timeline has existing content.
    Considers neighbors and fits new content appropriately.
    """
    # For now, use conservative merging
    # TODO: Analyze neighbors, find gaps, fit content
    return _merge_conservative(time_ranges)


def _merge_for_broll(
    time_ranges: List[Tuple[float, float]],
    existing_chunks: List[Dict],
    query: str
) -> List[Tuple[float, float]]:
    """
    Merge strategy for B-roll: shorter clips, fit between main content.
    """
    # B-roll clips should be 4-8 seconds
    max_clip_duration = 8.0
    merge_gap = 1.5
    
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    merged = []
    current_start, current_end = sorted_ranges[0]
    
    for start, end in sorted_ranges[1:]:
        gap = start - current_end
        if gap <= merge_gap:
            new_end = end
            new_duration = new_end - current_start
            if new_duration <= max_clip_duration:
                current_end = new_end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    if current_start is not None:
        merged.append((current_start, current_end))
    
    return merged


def _merge_conservative(time_ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Conservative merging: only merge very close seconds (within 1s).
    """
    if not time_ranges:
        return []
    
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    merged = []
    current_start, current_end = sorted_ranges[0]
    
    for start, end in sorted_ranges[1:]:
        gap = start - current_end
        if gap <= 1.0:  # Very conservative: only merge if within 1 second
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    if current_start is not None:
        merged.append((current_start, current_end))
    
    return merged

