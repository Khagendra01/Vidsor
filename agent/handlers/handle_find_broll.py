"""Handler for FIND_BROLL operation."""

from typing import Dict, Any
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState
from agent.helpers.orchestrator_helpers import (
    validate_planner_result,
    create_clarification_response,
)
from agent.helpers.orchestrator_helpers.handle_find_broll import (
    calculate_broll_constraints,
    analyze_main_action_keywords,
    build_broll_query,
    filter_and_select_broll_clips,
    adjust_broll_durations,
    create_broll_chunks,
)


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

