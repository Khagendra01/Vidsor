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
    Inserts B-roll after each corresponding main clip.
    
    Args:
        adjusted_ranges: List of adjusted B-roll range dictionaries
        timeline_manager: Timeline manager instance
        indices: List of timeline indices where B-roll should be inserted (sorted)
        main_clip_total_duration: Total duration of main clips for ratio calculation
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with:
        - chunks_created: List of created chunk dictionaries
        - insert_position: List of positions where chunks were inserted (first position)
    """
    chunks_created = []
    
    # Sort indices to process in order
    sorted_indices = sorted(indices)
    num_main_clips = len(sorted_indices)
    num_broll_clips = len(adjusted_ranges)
    
    # Distribute B-roll clips among main clips (round-robin if counts don't match)
    # Create mapping: main_clip_index -> list of B-roll ranges to insert after it
    broll_distribution = {}
    for i, broll_range in enumerate(adjusted_ranges):
        # Round-robin distribution: assign B-roll to main clips in order
        main_clip_idx = sorted_indices[i % num_main_clips]
        if main_clip_idx not in broll_distribution:
            broll_distribution[main_clip_idx] = []
        broll_distribution[main_clip_idx].append((i, broll_range))
    
    if verbose:
        print(f"\n  Distributing {num_broll_clips} B-roll clip(s) among {num_main_clips} main clip(s)")
        for main_idx, broll_list in broll_distribution.items():
            print(f"    Main clip {main_idx}: {len(broll_list)} B-roll clip(s)")
    
    # Track offset as we insert (each insertion shifts subsequent indices)
    insertion_offset = 0
    first_insert_position = None
    
    # Process each main clip in order and insert its B-roll after it
    for main_clip_idx in sorted_indices:
        if main_clip_idx not in broll_distribution:
            continue  # No B-roll assigned to this main clip
        
        # Calculate insert position: after the main clip, accounting for previous insertions
        # main_clip_idx is the original index, but we need to account for B-roll we've already inserted
        insert_position = main_clip_idx + 1 + insertion_offset
        
        # Get the end time of the main clip to start B-roll timeline from
        if insert_position - 1 < len(timeline_manager.chunks):
            main_clip = timeline_manager.chunks[insert_position - 1]
            timeline_start = main_clip["end_time"]
        else:
            # Fallback: use end of last chunk
            if len(timeline_manager.chunks) > 0:
                timeline_start = timeline_manager.chunks[-1]["end_time"]
            else:
                timeline_start = 0.0
        
        current_timeline_time = timeline_start
        
        # Create and insert B-roll chunks for this main clip
        for broll_idx, broll_range in broll_distribution[main_clip_idx]:
            tr_start, tr_end = broll_range["time_range"]
            chunk = create_timeline_chunk(
                original_start=tr_start,
                original_end=tr_end,
                timeline_start=current_timeline_time,
                chunk_type="broll",
                description=broll_range["description"],
                unified_description=broll_range["unified_description"],
                audio_description=broll_range["audio_description"],
                score=broll_range["combined_score"]
            )
            
            chunks_created.append(chunk)
            
            # Insert this B-roll chunk after the main clip
            timeline_manager.chunks.insert(insert_position, chunk)
            
            # Track first insert position for return value
            if first_insert_position is None:
                first_insert_position = insert_position
            
            # Update timeline time for next B-roll (if multiple per main clip)
            current_timeline_time = chunk["end_time"]
            
            # Increment offset and position for next B-roll in this group
            insertion_offset += 1
            insert_position += 1
        
        # Note: insertion_offset already accounts for all B-roll inserted for this main clip
    
    # Recalculate timeline start_times for all chunks
    recalculate_timeline_times(timeline_manager)
    
    if verbose:
        total_broll_duration = sum(r["duration"] for r in adjusted_ranges)
        ratio_final = (total_broll_duration / main_clip_total_duration * 100) if main_clip_total_duration > 0 else 0
        print(f"\n  ✓ Created {len(chunks_created)} B-roll chunk(s) inserted after main clips")
        print(f"  ✓ Total B-roll duration: {total_broll_duration:.1f}s ({ratio_final:.0f}% of main content)")
        if len(chunks_created) > 0:
            print(f"  ✓ Average clip duration: {total_broll_duration/len(chunks_created):.1f}s per clip")
    
    return {
        "chunks_created": chunks_created,
        "insert_position": first_insert_position if first_insert_position is not None else max(sorted_indices) + 1
    }

