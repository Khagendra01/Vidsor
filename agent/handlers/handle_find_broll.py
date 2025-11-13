"""Handler for FIND_BROLL operation."""

from typing import Dict, Any
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState
from agent.helpers.orchestrator_helpers import (
    validate_planner_result,
    create_clarification_response,
    gather_clip_contexts,
)
from agent.helpers.orchestrator_helpers.handle_find_broll import (
    calculate_broll_constraints,
    analyze_main_action_keywords,
    build_broll_query,
    filter_and_select_broll_clips,
    adjust_broll_durations,
    create_broll_chunks,
)
from agent.helpers.orchestrator_helpers.handle_find_broll.calculate_constraints import detect_multiple_broll_intent


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
    
    # Detect if user explicitly wants multiple B-roll clips
    user_query = state.get("user_query", "")
    user_wants_multiple = detect_multiple_broll_intent(user_query)
    
    # PHASE 1: Calculate main clip statistics and target constraints
    constraints = calculate_broll_constraints(
        selected_chunks, 
        user_wants_multiple=user_wants_multiple,
        verbose=verbose
    )
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
    clip_contexts = gather_clip_contexts(segment_tree, timeline_manager, indices)

    planner_state = {
        "user_query": broll_query,
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "time_ranges": None,
        "needs_clarification": False,
        "messages": state.get("messages", []),
        "clip_contexts": clip_contexts,
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
        
        # Validate and filter time ranges to only include those within requested range (with buffer)
        # Buffer allows B-roll up to 60 seconds before or after the requested range
        buffer_seconds = 60.0
        valid_time_ranges = []
        for tr in time_ranges:
            if len(tr) < 2:
                continue
            tr_start, tr_end = tr[0], tr[1]
            # Check if range overlaps with or is near the requested range
            # Range is valid if it overlaps with [start_time - buffer, end_time + buffer]
            range_start = start_time - buffer_seconds
            range_end = end_time + buffer_seconds
            # Check for overlap: range overlaps if tr_start < range_end and tr_end > range_start
            if tr_start < range_end and tr_end > range_start:
                valid_time_ranges.append(tr)
        
        if not valid_time_ranges:
            if verbose:
                print(f"  ✗ No B-roll found within requested range ({start_time:.1f}s - {end_time:.1f}s ± {buffer_seconds}s)")
                print(f"  ✗ Planner returned {len(time_ranges)} ranges, but none were within valid time window")
            return {
                "success": False,
                "error": f"No B-roll found within requested time range ({start_time:.1f}s - {end_time:.1f}s)",
                "chunks_created": []
            }
        
        if verbose and len(valid_time_ranges) < len(time_ranges):
            filtered_count = len(time_ranges) - len(valid_time_ranges)
            print(f"  ⚠ Filtered out {filtered_count} time range(s) outside requested range ({start_time:.1f}s - {end_time:.1f}s)")
            print(f"  ✓ {len(valid_time_ranges)} valid time range(s) remaining")
        
        # Use filtered time ranges
        time_ranges = valid_time_ranges
        
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

