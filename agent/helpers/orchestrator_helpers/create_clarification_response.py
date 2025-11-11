"""Create standardized clarification response from planner result."""

from typing import Dict, Any


def create_clarification_response(planner_result: Dict, error_key: str = "chunks_created") -> Dict[str, Any]:
    """
    Create standardized clarification response from planner result.
    
    Args:
        planner_result: Planner result dictionary
        error_key: Key to use for empty list in error response
        
    Returns:
        Clarification response dictionary
    """
    return {
        "success": False,
        "error": planner_result.get("clarification_question", "Clarification needed"),
        error_key: [],
        "needs_clarification": True,
        "clarification_question": planner_result.get("clarification_question", "Clarification needed"),
        "preserved_state": {
            "time_ranges": planner_result.get("time_ranges", []),
            "search_results": planner_result.get("search_results", []),
            "previous_time_ranges": planner_result.get("previous_time_ranges"),
            "previous_scored_seconds": planner_result.get("previous_scored_seconds"),
            "previous_query": planner_result.get("previous_query"),
            "previous_search_results": planner_result.get("previous_search_results")
        }
    }

