"""Operation execution handlers for orchestrator agent.

This module is maintained for backward compatibility.
All handlers have been moved to agent/handlers/ directory.
"""

# Import all handlers from the new handlers module for backward compatibility
from agent.handlers import (
    handle_find_highlights,
    handle_trim,
    handle_cut,
    handle_replace,
    handle_insert,
    handle_find_broll,
    handle_apply_effect,
    apply_duration_constraints as _apply_duration_constraints,
    apply_duration_constraints_with_neighbors as _apply_duration_constraints_with_neighbors,
)

__all__ = [
    "handle_find_highlights",
    "handle_trim",
    "handle_cut",
    "handle_replace",
    "handle_insert",
    "handle_find_broll",
    "handle_apply_effect",
    "_apply_duration_constraints",
    "_apply_duration_constraints_with_neighbors",
]
