"""Create B-roll timeline chunks from adjusted ranges."""

from typing import Dict, List
from agent.timeline_manager import TimelineManager
from agent.helpers.orchestrator_helpers import create_timeline_chunk, recalculate_timeline_times


def create_broll_chunks(
    adjusted_ranges: List[Dict],
    timeline_manager: TimelineManager,
    indices: List[int],
    main_clip_total_duration: float,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Create B-roll timeline chunks from adjusted ranges.
    
    Args:
        adjusted_ranges: List of adjusted B-roll range dictionaries
        timeline_manager: Timeline manager instance
        indices: List of timeline indices where B-roll should be inserted
        main_clip_total_duration: Total duration of main clips for ratio calculation
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with:
        - chunks_created: List of created chunk dictionaries
        - insert_position: Position where chunks were inserted
    """
    chunks_created = []
    # Insert B-roll after the last selected chunk
    insert_position = max(indices) + 1
    if insert_position > len(timeline_manager.chunks):
        insert_position = len(timeline_manager.chunks)
    
    # Calculate timeline start time
    if insert_position > 0:
        timeline_start = timeline_manager.chunks[insert_position - 1]["end_time"]
    else:
        timeline_start = 0.0
    
    current_timeline_time = timeline_start
    
    for i, r in enumerate(adjusted_ranges):
        tr_start, tr_end = r["time_range"]
        chunk = create_timeline_chunk(
            original_start=tr_start,
            original_end=tr_end,
            timeline_start=current_timeline_time,
            chunk_type="broll",
            description=r["description"],
            unified_description=r["unified_description"],
            audio_description=r["audio_description"],
            score=r["combined_score"]
        )
        
        chunks_created.append(chunk)
        current_timeline_time = chunk["end_time"]
    
    # Insert B-roll chunks
    for i, chunk in enumerate(chunks_created):
        timeline_manager.chunks.insert(insert_position + i, chunk)
    
    # Recalculate timeline start_times
    recalculate_timeline_times(timeline_manager)
    
    if verbose:
        total_broll_duration = sum(r["duration"] for r in adjusted_ranges)
        ratio_final = (total_broll_duration / main_clip_total_duration * 100) if main_clip_total_duration > 0 else 0
        print(f"\n  ✓ Created {len(chunks_created)} B-roll chunk(s) at position {insert_position}")
        print(f"  ✓ Total B-roll duration: {total_broll_duration:.1f}s ({ratio_final:.0f}% of main content)")
        if len(chunks_created) > 0:
            print(f"  ✓ Average clip duration: {total_broll_duration/len(chunks_created):.1f}s per clip")
    
    return {
        "chunks_created": chunks_created,
        "insert_position": insert_position
    }

