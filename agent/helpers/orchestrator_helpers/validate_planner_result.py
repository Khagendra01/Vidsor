"""Validate planner agent result."""

from typing import Any, Tuple, Optional


def validate_planner_result(planner_result: Any, verbose: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate planner agent result.
    
    Args:
        planner_result: Result from planner agent
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if planner_result is None:
        error_msg = "Planner agent returned None - this may indicate an error in the planner"
        if verbose:
            print(f"  ✗ {error_msg}")
        return False, error_msg
    
    if not isinstance(planner_result, dict):
        error_msg = f"Planner agent returned invalid type: {type(planner_result).__name__}, expected dict"
        if verbose:
            print(f"  ✗ {error_msg}")
        return False, error_msg
    
    return True, None

