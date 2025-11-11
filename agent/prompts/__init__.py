"""Prompts package for agent modules."""

# Re-export planner prompts for backward compatibility
from agent.prompts.planner_prompts import (
    PLANNER_SYSTEM_PROMPT,
    SEGMENT_TREE_INSPECTION_PROMPT,
    QUERY_REASONING_PROMPT,
    SEARCH_QUERY_GENERATION_PROMPT,
    VIDEO_NARRATION_PROMPT,
    CLARIFICATION_DECISION_PROMPT
)

# Re-export orchestrator prompts if available
try:
    from agent.prompts.orchestrator_prompts import (
        ORCHESTRATOR_SYSTEM_PROMPT,
        OPERATION_CLASSIFICATION_PROMPT
    )
    __all__ = [
        "PLANNER_SYSTEM_PROMPT",
        "SEGMENT_TREE_INSPECTION_PROMPT",
        "QUERY_REASONING_PROMPT",
        "SEARCH_QUERY_GENERATION_PROMPT",
        "VIDEO_NARRATION_PROMPT",
        "CLARIFICATION_DECISION_PROMPT",
        "ORCHESTRATOR_SYSTEM_PROMPT",
        "OPERATION_CLASSIFICATION_PROMPT",
    ]
except ImportError:
    # orchestrator_prompts might still be in agent/ directory
    __all__ = [
        "PLANNER_SYSTEM_PROMPT",
        "SEGMENT_TREE_INSPECTION_PROMPT",
        "QUERY_REASONING_PROMPT",
        "SEARCH_QUERY_GENERATION_PROMPT",
        "VIDEO_NARRATION_PROMPT",
        "CLARIFICATION_DECISION_PROMPT",
    ]

