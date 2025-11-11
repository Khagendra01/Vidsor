"""Operation execution handlers for orchestrator agent."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState
from agent.utils.logging_utils import get_log_helper
from extractor.utils.video_utils import get_video_duration
from agent.helpers.orchestrator_helpers import (
    recalculate_timeline_times,
    validate_planner_result,
    create_clarification_response,
    match_search_result_to_time_range,
    create_timeline_chunk,
    create_chunks_from_time_ranges,
)
from agent.helpers.orchestrator_helpers.handle_find_broll import (
    calculate_broll_constraints,
    analyze_main_action_keywords,
    build_broll_query,
    filter_and_select_broll_clips,
    adjust_broll_durations,
    create_broll_chunks,
)


# ============================================================================
# Duration Constraint Functions
# ============================================================================

def _apply_duration_constraints(
    time_ranges: List[Tuple[float, float]],
    max_total_duration: float,
    max_clip_duration: float = 20.0,
    min_gap: float = 3.0,
    logger=None,
    verbose: bool = False
) -> List[Tuple[float, float]]:
    """
    Apply duration constraints to time ranges.
    
    Args:
        time_ranges: List of (start, end) tuples
        max_total_duration: Maximum total duration allowed
        max_clip_duration: Maximum duration per clip
        min_gap: Minimum gap between clips
        logger: Logger instance
        verbose: Whether to print verbose output
        
    Returns:
        Filtered and constrained time ranges
    """
    if not time_ranges:
        return []
    
    # Sort by start time
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    
    # Filter by max_clip_duration and accumulate until max_total_duration
    # Use SOFT limits: prefer shorter clips but don't hard-cut important long scenes
    filtered = []
    total_duration = 0.0
    last_end = -min_gap  # Track last end time for gap checking
    log = get_log_helper(logger, verbose)
    
    for start, end in sorted_ranges:
        duration = end - start
        
        # SOFT limit: Prefer shorter clips, but allow longer if they're important
        # Only skip if extremely long (>2x max_clip_duration)
        if duration > max_clip_duration * 2:
            log.info(f"  Skipping very long clip: {duration:.1f}s > {max_clip_duration * 2:.1f}s")
            continue
        
        # SOFT gap: Prefer spacing, but don't skip if clips are close
        # Only skip if overlapping or extremely close (<1s gap)
        gap = start - last_end
        if gap < 1.0 and gap >= 0:  # Overlapping or very close
            # Merge with previous if they're very close
            if filtered:
                prev_start, prev_end = filtered[-1]
                filtered[-1] = (prev_start, max(prev_end, end))
                total_duration = sum(e - s for s, e in filtered)
                last_end = max(prev_end, end)
                continue
        
        # SOFT total duration: Prefer staying under limit, but allow 20% over for important content
        if total_duration + duration > max_total_duration * 1.2:
            # Hard stop if way over
            break
        elif total_duration + duration > max_total_duration:
            # Over limit but within 20% - still add if high quality (check score if available)
            # For now, add it but log warning
            log(f"  [WARNING] Exceeding target duration: {total_duration + duration:.1f}s > {max_total_duration:.1f}s")
        
        filtered.append((start, end))
        total_duration += duration
        last_end = end
    if len(filtered) < len(time_ranges):
        log(f"  [DURATION] Filtered from {len(time_ranges)} to {len(filtered)} ranges "
            f"(total: {total_duration:.1f}s / {max_total_duration:.1f}s target)")
    elif total_duration > max_total_duration:
        log(f"  [DURATION] Total duration {total_duration:.1f}s exceeds target {max_total_duration:.1f}s (soft limit)")
    
    return filtered


def _apply_duration_constraints_with_neighbors(
    time_ranges: List[Tuple[float, float]],
    existing_chunks: List[Dict],
    max_clip_duration: float = 15.0,
    min_gap: float = 2.0,
    logger=None,
    verbose: bool = False
) -> List[Tuple[float, float]]:
    """
    Apply duration constraints considering existing timeline chunks.
    Analyzes neighbors to determine appropriate clip lengths.
    """
    if not time_ranges:
        return []
    
    # For now, use same logic as empty timeline but with stricter constraints
    # TODO: Analyze neighbors, find gaps, fit content appropriately
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    
    filtered = []
    last_end = -min_gap
    
    for start, end in sorted_ranges:
        duration = end - start
        
        # SOFT limits: prefer shorter but allow longer if needed
        if duration > max_clip_duration * 2:
            continue
        
        # SOFT gap: prefer spacing but don't skip close clips
        gap = start - last_end
        if gap < 1.0 and gap >= 0:  # Overlapping or very close - merge
            if filtered:
                prev_start, prev_end = filtered[-1]
                filtered[-1] = (prev_start, max(prev_end, end))
                last_end = max(prev_end, end)
                continue
        
        filtered.append((start, end))
        last_end = end
    
    return filtered


def handle_find_highlights(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    planner_agent,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle FIND_HIGHLIGHTS operation.
    Calls planner agent to find highlights, then creates timeline chunks.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        planner_agent: Planner agent function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and created chunks
    """
    logger = state.get("logger")
    log = get_log_helper(logger, verbose)
    
    log.info("\n[OPERATION] FIND_HIGHLIGHTS")
    log.info("  Calling planner agent to find highlights...")
    
    # Prepare state for planner (ensure all required fields)
    # Check if we have preserved_state from a previous clarification
    # This comes from orchestrator_runner when preserved_state is passed
    previous_time_ranges = state.get("previous_time_ranges")
    previous_query = state.get("previous_query")
    previous_scored_seconds = state.get("previous_scored_seconds")
    previous_search_results = state.get("previous_search_results")
    
    planner_state = {
        "user_query": state.get("user_query", "find highlights"),
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "logger": logger,  # Pass logger to planner
        "time_ranges": None,  # Will be set by planner
        "needs_clarification": False,
        "messages": state.get("messages", []),
        # Pass previous context for refinement logic
        "previous_time_ranges": previous_time_ranges,
        "previous_query": previous_query,
        "previous_scored_seconds": previous_scored_seconds,
        "previous_search_results": previous_search_results,
    }
    
    # Call planner agent
    try:
        planner_result = planner_agent(planner_state)
        
        # Validate planner result
        is_valid, error_msg = validate_planner_result(planner_result, verbose)
        if not is_valid:
            log.error(f"  ✗ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "chunks_created": []
            }
        
        if planner_result.get("needs_clarification"):
            return create_clarification_response(planner_result, "chunks_created")
        
        time_ranges = planner_result.get("time_ranges", [])
        if not time_ranges:
            return {
                "success": False,
                "error": "No highlights found",
                "chunks_created": []
            }
        
        log.info(f"  Planner found {len(time_ranges)} time ranges")
        
        # MERGE AGENT: Intelligently merge time ranges based on timeline state
        from agent.nodes.merge_agent import create_merge_agent
        
        video_path = state.get("video_path", "")
        video_duration = get_video_duration(video_path) if video_path else 600.0
        
        merge_state = {
            "time_ranges": time_ranges,
            "timeline_manager": timeline_manager,
            "operation_type": "FIND_HIGHLIGHTS",
            "video_duration": video_duration,
            "query": state.get("user_query", "")
        }
        
        merge_agent = create_merge_agent()
        merge_result = merge_agent(merge_state)
        merged_ranges = merge_result.get("merged_ranges", time_ranges)
        
        # Apply duration constraints based on timeline state
        is_empty_timeline = len(timeline_manager.chunks) == 0 if timeline_manager else True
        existing_duration = timeline_manager.calculate_timeline_duration() if timeline_manager else 0.0
        
        if is_empty_timeline:
            # New timeline: limit to 12-15% of video
            max_total_duration = video_duration * 0.15
            merged_ranges = _apply_duration_constraints(
                merged_ranges,
                max_total_duration=max_total_duration,
                max_clip_duration=20.0,
                min_gap=3.0,
                logger=logger,
                verbose=verbose
            )
        else:
            # Existing timeline: consider neighbors and available space
            merged_ranges = _apply_duration_constraints_with_neighbors(
                merged_ranges,
                existing_chunks=timeline_manager.chunks,
                max_clip_duration=15.0,
                min_gap=2.0,
                logger=logger,
                verbose=verbose
            )
        
        log.info(f"  After merging: {len(merged_ranges)} time ranges")
        total_duration = sum(end - start for start, end in merged_ranges)
        log.info(f"  Total highlight duration: {total_duration:.1f}s ({total_duration/video_duration*100:.1f}% of video)")
        
        # Create timeline chunks from merged time ranges
        current_timeline_time = timeline_manager.calculate_timeline_duration()
        search_results = planner_result.get("search_results", [])
        
        chunks_created = create_chunks_from_time_ranges(
            time_ranges=merged_ranges,
            search_results=search_results,
            timeline_start=current_timeline_time,
            chunk_type="highlight",
            default_score=planner_result.get("confidence", 0.7),
            name_prefix="Highlight"
        )
        
        # Log created chunks
        for i, (chunk, (start_time, end_time)) in enumerate(zip(chunks_created, merged_ranges), 1):
            log.info(f"    Created chunk {i}: timeline {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s "
                     f"(source: {start_time:.1f}s - {end_time:.1f}s)")
        
        # Add chunks to timeline
        timeline_manager.chunks.extend(chunks_created)
        
        log.info(f"  ✓ Created {len(chunks_created)} highlight chunks")
        
        return {
            "success": True,
            "chunks_created": chunks_created,
            "time_ranges": time_ranges
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error in FIND_HIGHLIGHTS: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_created": []
        }


def handle_trim(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle TRIM operation.
    Supports:
    - delta trims from start/end (trim_from with trim_seconds)
    - set exact duration (trim_target_length)
    - remove an internal range (remove_range with start/end offsets)
    
    Parameters (backward-compatible):
    - trim_index: int (required)
    - trim_seconds: float (optional; used with trim_from start/end)
    - trim_from: "start" | "end" (optional; default end for legacy)
    - trim_target_length: float (optional; set clip duration exactly)
    - remove_range: { "start_offset": float, "end_offset": float } (optional; within-clip offsets)
    """
    if verbose:
        print("\n[OPERATION] TRIM")

    idx = params.get("trim_index")
    if idx is None:
        return {"success": False, "error": "TRIM requires trim_index"}

    if idx < 0 or idx >= len(timeline_manager.chunks):
        return {"success": False, "error": f"Invalid trim_index: {idx}"}

    chunk = timeline_manager.chunks[idx]
    # Use original times as source-of-truth for content region
    orig_start = float(chunk.get("original_start_time", chunk.get("start_time", 0.0)))
    orig_end = float(chunk.get("original_end_time", chunk.get("end_time", 0.0)))
    tl_start = float(chunk.get("start_time", 0.0))
    tl_end = float(chunk.get("end_time", tl_start))

    if orig_end <= orig_start:
        return {"success": False, "error": "Chunk has invalid original time range"}

    # Determine trim mode
    trim_target_length = params.get("trim_target_length")
    remove_range = params.get("remove_range")
    trim_seconds = params.get("trim_seconds")
    trim_from = params.get("trim_from")

    new_chunks: List[Dict] = []

    def recalc_timeline_from(start_index: int, start_time: float) -> None:
        """Recalculate timeline start/end for all chunks from start_index onward."""
        current_time = start_time
        for i in range(start_index, len(timeline_manager.chunks)):
            c = timeline_manager.chunks[i]
            dur = max(0.0, float(c["original_end_time"]) - float(c["original_start_time"]))
            c["start_time"] = current_time
            c["end_time"] = current_time + dur
            current_time = c["end_time"]

    # Case 1: remove internal range → possibly split into two chunks
    if remove_range and isinstance(remove_range, dict):
        # Support either explicit offsets or centered length
        if "center_length" in remove_range and remove_range.get("center_length") is not None:
            clen = float(remove_range.get("center_length"))
            if clen <= 0 or clen >= (orig_end - orig_start):
                return {"success": False, "error": "Invalid center_length for remove_range"}
            clip_dur = (orig_end - orig_start)
            center_time = orig_start + clip_dur / 2.0
            half = clen / 2.0
            abs_remove_start = max(orig_start, center_time - half)
            abs_remove_end = min(orig_end, center_time + half)
            rs = abs_remove_start - orig_start
            re = abs_remove_end - orig_start
        else:
            rs = float(remove_range.get("start_offset", 0.0))
            re = float(remove_range.get("end_offset", 0.0))
        if rs < 0 or re < 0 or re <= rs:
            return {"success": False, "error": "Invalid remove_range offsets"}
        if rs >= (orig_end - orig_start) or re > (orig_end - orig_start):
            return {"success": False, "error": "remove_range exceeds clip duration"}

        abs_remove_start = orig_start + rs
        abs_remove_end = orig_start + re
        # If removal eats entire clip, just delete the chunk
        if abs_remove_start <= orig_start and abs_remove_end >= orig_end:
            timeline_manager.chunks.pop(idx)
            # Recalc following
            recalc_timeline_from(idx, tl_start)
            if verbose:
                print(f"  Removed entire chunk at index {idx}")
            return {"success": True, "chunks_modified": 0, "chunks_removed": 1}

        # Build kept segments
        kept_segments: List[Tuple[float, float]] = []
        if abs_remove_start > orig_start:
            kept_segments.append((orig_start, abs_remove_start))
        if abs_remove_end < orig_end:
            kept_segments.append((abs_remove_end, orig_end))

        # Replace current chunk with kept segments (1 or 2)
        # First, remove the original
        timeline_manager.chunks.pop(idx)

        insert_pos = idx
        current_time = tl_start
        for seg_start, seg_end in kept_segments:
            duration = seg_end - seg_start
            if duration <= 0:
                continue
            new_chunk = {
                **chunk,
                "start_time": current_time,
                "end_time": current_time + duration,
                "original_start_time": seg_start,
                "original_end_time": seg_end,
            }
            # Avoid carrying references that might cause duplication issues
            new_chunk = dict(new_chunk)
            timeline_manager.chunks.insert(insert_pos, new_chunk)
            insert_pos += 1
            current_time = new_chunk["end_time"]
            new_chunks.append(new_chunk)

        # Recalc following chunks from insert_pos
        recalc_timeline_from(insert_pos, current_time)

        if verbose:
            print(f"  ✓ Removed internal range {rs:.2f}s–{re:.2f}s (offsets) from chunk {idx}")
            print(f"  Resulted in {len(new_chunks)} chunk(s)")

        return {
            "success": True,
            "chunks_added": new_chunks,
            "chunks_modified": len(new_chunks),
            "removed_range": [rs, re]
        }

    # Case 2: set exact duration
    if trim_target_length is not None:
        target = float(trim_target_length)
        if target <= 0:
            return {"success": False, "error": "trim_target_length must be > 0"}
        new_orig_end = orig_start + target
        if new_orig_end > orig_end:
            # Extending is not supported by trim; cap to original end
            new_orig_end = orig_end
            target = new_orig_end - orig_start
            if target <= 0:
                return {"success": False, "error": "Target exceeds content; nothing to keep"}
        chunk["original_end_time"] = new_orig_end
        chunk["start_time"] = tl_start
        chunk["end_time"] = tl_start + target

        # Recalc following chunks
        recalc_timeline_from(idx + 1, chunk["end_time"])

        if verbose:
            print(f"  ✓ Set clip {idx} duration to {target:.2f}s")

        return {"success": True, "chunks_modified": 1}

    # Case 3: legacy delta trim from start/end using trim_seconds + trim_from
    if trim_seconds is not None:
        delta = float(trim_seconds)
        if delta <= 0:
            return {"success": False, "error": "trim_seconds must be > 0"}

        if trim_from == "start":
            new_orig_start = orig_start + delta
            if new_orig_start >= orig_end:
                return {"success": False, "error": "Trim exceeds or equals clip duration"}
            chunk["original_start_time"] = new_orig_start
            # Update timeline times to match new duration
            new_duration = orig_end - new_orig_start
            chunk["start_time"] = tl_start
            chunk["end_time"] = tl_start + new_duration

            recalc_timeline_from(idx + 1, chunk["end_time"])
            if verbose:
                print(f"  ✓ Trimmed {delta:.2f}s from start of clip {idx}")
            return {"success": True, "chunks_modified": 1}

        # default and explicit "end"
        new_orig_end = orig_end - delta
        if new_orig_end <= orig_start:
            return {"success": False, "error": "Trim exceeds or equals clip duration"}
        chunk["original_end_time"] = new_orig_end
        new_duration = new_orig_end - orig_start
        chunk["start_time"] = tl_start
        chunk["end_time"] = tl_start + new_duration

        recalc_timeline_from(idx + 1, chunk["end_time"])
        if verbose:
            print(f"  ✓ Trimmed {delta:.2f}s from end of clip {idx}")
        return {"success": True, "chunks_modified": 1}

    return {"success": False, "error": "No valid TRIM parameters provided"}


def handle_cut(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle CUT operation.
    Removes chunks at specified timeline indices.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with timeline_indices
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and removed chunks
    """
    if verbose:
        print("\n[OPERATION] CUT")
    
    indices = params.get("timeline_indices", [])
    if not indices:
        return {
            "success": False,
            "error": "No timeline indices provided",
            "chunks_removed": []
        }
    
    # Validate indices
    is_valid, error = timeline_manager.validate_indices(indices)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "chunks_removed": []
        }
    
    # Sort indices in descending order to remove from end to start
    # This prevents index shifting issues
    sorted_indices = sorted(indices, reverse=True)
    
    removed_chunks = []
    for idx in sorted_indices:
        if 0 <= idx < len(timeline_manager.chunks):
            removed_chunks.append(timeline_manager.chunks.pop(idx))
            if verbose:
                print(f"  Removed chunk at index {idx}")
    
    # Recalculate timeline start_times for remaining chunks
    recalculate_timeline_times(timeline_manager)
    
    if verbose:
        print(f"  ✓ Removed {len(removed_chunks)} chunk(s)")
        print(f"  Timeline now has {len(timeline_manager.chunks)} chunks")
    
    return {
        "success": True,
        "chunks_removed": removed_chunks,
        "remaining_chunks": len(timeline_manager.chunks)
    }


def handle_replace(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    planner_agent,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle REPLACE operation.
    Replaces chunks at specified indices with new content from planner.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with timeline_indices and search_query
        planner_agent: Planner agent function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and replaced chunks
    """
    if verbose:
        print("\n[OPERATION] REPLACE")
    
    indices = params.get("timeline_indices", [])
    search_query = params.get("search_query")
    temporal_constraint = params.get("temporal_constraint")
    temporal_type = params.get("temporal_type")
    
    if not indices:
        return {
            "success": False,
            "error": "No timeline indices provided",
            "chunks_replaced": []
        }
    
    if not search_query:
        return {
            "success": False,
            "error": "No search query provided for replacement",
            "chunks_replaced": []
        }
    
    # Validate indices
    is_valid, error = timeline_manager.validate_indices(indices)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "chunks_replaced": []
        }
    
    # Get chunks to be replaced (for reference)
    chunks_to_replace = timeline_manager.get_chunks(indices)
    
    # Combine search query with temporal constraint if present
    # This ensures the planner searches with full context
    full_query = search_query
    if temporal_constraint:
        # Incorporate temporal constraint into the search query
        # e.g., "helicopter clips" + "when they were in helicopter" → "helicopter clips when they were in helicopter"
        full_query = f"{search_query} {temporal_constraint}"
    
    if verbose:
        print(f"  Replacing {len(indices)} chunk(s) at indices {indices}")
        print(f"  Search query: '{search_query}'")
        if temporal_constraint:
            print(f"  Temporal constraint: '{temporal_constraint}' (type: {temporal_type})")
        print(f"  Full query to planner: '{full_query}'")
    
    # Call planner to find replacement content
    planner_state = {
        "user_query": full_query,  # Pass combined query with temporal constraint
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "time_ranges": None,
        "needs_clarification": False,
        "messages": state.get("messages", []),
        # Pass temporal constraint separately for potential filtering
        "temporal_constraint": temporal_constraint,
        "temporal_type": temporal_type,
    }
    
    try:
        planner_result = planner_agent(planner_state)
        
        # Validate planner result
        is_valid, error_msg = validate_planner_result(planner_result, verbose)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "chunks_replaced": []
            }
        
        if planner_result.get("needs_clarification"):
            return create_clarification_response(planner_result, "chunks_replaced")
        
        time_ranges = planner_result.get("time_ranges", [])
        if not time_ranges:
            return {
                "success": False,
                "error": "No replacement content found",
                "chunks_replaced": []
            }
        
        # Limit replacement to same number of chunks (or fewer)
        if len(time_ranges) > len(indices):
            time_ranges = time_ranges[:len(indices)]
            if verbose:
                print(f"  Limiting replacement to {len(indices)} chunk(s)")
        
        # Sort indices in descending order for replacement
        sorted_indices = sorted(indices, reverse=True)
        
        # Remove old chunks
        removed_chunks = []
        for idx in sorted_indices:
            if 0 <= idx < len(timeline_manager.chunks):
                removed_chunks.append(timeline_manager.chunks.pop(idx))
        
        # Create new chunks from planner results
        # Insert at the position of the first removed chunk
        insert_position = min(indices) if indices else 0
        current_timeline_time = timeline_manager.chunks[insert_position - 1]["end_time"] if insert_position > 0 else 0.0
        search_results = planner_result.get("search_results", [])
        
        new_chunks = create_chunks_from_time_ranges(
            time_ranges=time_ranges,
            search_results=search_results,
            timeline_start=current_timeline_time,
            chunk_type="highlight",
            default_score=planner_result.get("confidence", 0.7),
            name_prefix="Replacement clip"
        )
        
        # Insert new chunks at the position
        for i, chunk in enumerate(new_chunks):
            timeline_manager.chunks.insert(insert_position + i, chunk)
        
        # Recalculate timeline start_times
        recalculate_timeline_times(timeline_manager)
        
        if verbose:
            print(f"  ✓ Replaced {len(removed_chunks)} chunk(s) with {len(new_chunks)} new chunk(s)")
        
        return {
            "success": True,
            "chunks_removed": removed_chunks,
            "chunks_added": new_chunks,
            "chunks_replaced": len(new_chunks)
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error in REPLACE: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_replaced": []
        }


def handle_insert(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    planner_agent,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle INSERT operation.
    Inserts new clips at specified position in timeline.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with insert position and search_query
        planner_agent: Planner agent function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and inserted chunks
    """
    if verbose:
        print("\n[OPERATION] INSERT")
    
    search_query = params.get("search_query")
    insert_before_index = params.get("insert_before_index")
    insert_after_index = params.get("insert_after_index")
    insert_between_indices = params.get("insert_between_indices")
    
    # Determine insert position
    insert_position = None
    
    if insert_between_indices:
        # Insert between two indices
        idx1, idx2 = insert_between_indices[0], insert_between_indices[1]
        if 0 <= idx1 < len(timeline_manager.chunks) and 0 <= idx2 < len(timeline_manager.chunks):
            insert_position = idx2  # Insert after first index, before second
    elif insert_after_index is not None:
        # Insert after specified index
        if 0 <= insert_after_index < len(timeline_manager.chunks):
            insert_position = insert_after_index + 1
    elif insert_before_index is not None:
        # Insert before specified index
        if 0 <= insert_before_index < len(timeline_manager.chunks):
            insert_position = insert_before_index
    
    if insert_position is None:
        return {
            "success": False,
            "error": "Invalid insert position",
            "chunks_inserted": []
        }
    
    if not search_query:
        # Default query if none provided
        search_query = "interesting moments"
    
    if verbose:
        print(f"  Inserting at position {insert_position}")
        print(f"  Search query: '{search_query}'")
    
    # Call planner to find content
    planner_state = {
        "user_query": search_query,
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "time_ranges": None,
        "needs_clarification": False,
        "messages": state.get("messages", []),
    }
    
    try:
        planner_result = planner_agent(planner_state)
        
        # Validate planner result
        is_valid, error_msg = validate_planner_result(planner_result, verbose)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "chunks_inserted": []
            }
        
        if planner_result.get("needs_clarification"):
            return create_clarification_response(planner_result, "chunks_inserted")
        
        time_ranges = planner_result.get("time_ranges", [])
        if not time_ranges:
            return {
                "success": False,
                "error": "No content found to insert",
                "chunks_inserted": []
            }
        
        # Calculate timeline start time for insertion
        timeline_start = timeline_manager.chunks[insert_position - 1]["end_time"] if insert_position > 0 else 0.0
        search_results = planner_result.get("search_results", [])
        
        # Create chunks from planner results
        new_chunks = create_chunks_from_time_ranges(
            time_ranges=time_ranges,
            search_results=search_results,
            timeline_start=timeline_start,
            chunk_type="highlight",
            default_score=planner_result.get("confidence", 0.7),
            name_prefix="Inserted clip"
        )
        
        # Insert chunks at position
        for i, chunk in enumerate(new_chunks):
            timeline_manager.chunks.insert(insert_position + i, chunk)
        
        # Recalculate timeline start_times for all chunks after insertion
        recalculate_timeline_times(timeline_manager)
        
        if verbose:
            print(f"  ✓ Inserted {len(new_chunks)} chunk(s) at position {insert_position}")
        
        return {
            "success": True,
            "chunks_inserted": new_chunks,
            "insert_position": insert_position
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error in INSERT: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_inserted": []
        }


def handle_find_broll(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    planner_agent,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle FIND_BROLL operation.
    Finds complementary B-roll for selected timeline segments.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with timeline_indices
        planner_agent: Planner agent function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and B-roll chunks
    """
    if verbose:
        print("\n[OPERATION] FIND_BROLL")
    
    indices = params.get("timeline_indices", [])
    if not indices:
        return {
            "success": False,
            "error": "No timeline indices provided",
            "chunks_created": []
        }
    
    # Validate indices
    is_valid, error = timeline_manager.validate_indices(indices)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "chunks_created": []
        }
    
    # Get selected chunks and analyze main action
    selected_chunks = timeline_manager.get_chunks(indices)
    
    # PHASE 1: Calculate main clip statistics and target constraints
    constraints = calculate_broll_constraints(selected_chunks, verbose=verbose)
    main_clip_total_duration = constraints["main_clip_total_duration"]
    
    # Get time range from selected chunks
    time_range = timeline_manager.get_timeline_range(indices)
    if not time_range:
        return {
            "success": False,
            "error": "Could not determine time range from selected chunks",
            "chunks_created": []
        }
    
    start_time, end_time = time_range
    
    # PHASE 2: Analyze main action keywords
    segment_tree = state.get("segment_tree")
    main_keywords = analyze_main_action_keywords(selected_chunks, segment_tree, verbose=verbose)
    
    # Build B-roll search query with director guidelines
    broll_query = build_broll_query(
        start_time=start_time,
        end_time=end_time,
        main_keywords=main_keywords,
        constraints=constraints,
        verbose=verbose
    )
    
    if verbose:
        print(f"  Selected chunks: indices {indices}")
    
    # Call planner to find B-roll
    planner_state = {
        "user_query": broll_query,
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "time_ranges": None,
        "needs_clarification": False,
        "messages": state.get("messages", []),
    }
    
    try:
        planner_result = planner_agent(planner_state)
        
        # Validate planner result
        is_valid, error_msg = validate_planner_result(planner_result, verbose)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "chunks_created": []
            }
        
        if planner_result.get("needs_clarification"):
            return create_clarification_response(planner_result, "chunks_created")
        
        time_ranges = planner_result.get("time_ranges", [])
        if not time_ranges:
            return {
                "success": False,
                "error": "No B-roll found",
                "chunks_created": []
            }
        
        search_results = planner_result.get("search_results", [])
        
        # PHASE 3: Filter and select best B-roll clips
        selected_ranges = filter_and_select_broll_clips(
            time_ranges=time_ranges,
            search_results=search_results,
            constraints=constraints,
            main_clip_total_duration=main_clip_total_duration,
            verbose=verbose
        )
        
        # PHASE 4: Adjust clip durations to 4-8 seconds
        video_path = state.get("video_path", "")
        adjusted_ranges = adjust_broll_durations(
            selected_ranges=selected_ranges,
            constraints=constraints,
            video_path=video_path,
            verbose=verbose
        )
        
        # PHASE 5: Create B-roll chunks
        result = create_broll_chunks(
            adjusted_ranges=adjusted_ranges,
            timeline_manager=timeline_manager,
            indices=indices,
            main_clip_total_duration=main_clip_total_duration,
            verbose=verbose
        )
        
        return {
            "success": True,
            "chunks_created": result["chunks_created"],
            "insert_position": result["insert_position"]
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error in FIND_BROLL: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_created": []
        }

