"""Handler for REPLACE operation."""

from typing import Dict, Any
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState
from agent.helpers.orchestrator_helpers import (
    validate_planner_result,
    create_clarification_response,
    create_chunks_from_time_ranges,
    recalculate_timeline_times,
)


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

