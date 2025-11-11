"""Handler for INSERT operation."""

from typing import Dict, Any
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState
from agent.helpers.orchestrator_helpers import (
    validate_planner_result,
    create_clarification_response,
    create_chunks_from_time_ranges,
    recalculate_timeline_times,
)


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

