"""Handler for CUT operation."""

from typing import Dict, Any
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState
from agent.helpers.orchestrator_helpers import recalculate_timeline_times


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

