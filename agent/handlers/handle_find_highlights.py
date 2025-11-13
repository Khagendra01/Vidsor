"""Handler for FIND_HIGHLIGHTS operation."""

from typing import Dict, Any
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState
from agent.utils.logging_utils import get_log_helper
from extractor.utils.video_utils import get_video_duration
from agent.helpers.orchestrator_helpers import (
    validate_planner_result,
    create_clarification_response,
    create_chunks_from_time_ranges,
)
from agent.handlers.duration_constraints import (
    apply_duration_constraints,
    apply_duration_constraints_with_neighbors,
)


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
    
    # Get video_path and video_duration before using them
    video_path = state.get("video_path", "")
    video_duration = get_video_duration(video_path) if video_path else 600.0
    
    planner_state = {
        "user_query": state.get("user_query", "find highlights"),
        "video_path": video_path,
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
        # Pass operation type for operation-aware threshold adjustment
        "operation_type": "FIND_HIGHLIGHTS",
        "video_duration": video_duration,
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
        
        # video_path and video_duration already defined above
        
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
            merged_ranges = apply_duration_constraints(
                merged_ranges,
                max_total_duration=max_total_duration,
                max_clip_duration=20.0,
                min_gap=3.0,
                logger=logger,
                verbose=verbose
            )
        else:
            # Existing timeline: consider neighbors and available space
            merged_ranges = apply_duration_constraints_with_neighbors(
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

