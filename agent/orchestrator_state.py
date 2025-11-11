"""Extended state for orchestrator agent with timeline management."""

from typing import Dict, List, Optional, Tuple, Annotated, TypedDict, Any
from agent.state import AgentState


class OrchestratorState(AgentState):
    """Extended state for orchestrator agent that manages timeline editing."""
    
    # Timeline management
    timeline_path: Optional[str]  # Path to timeline.json
    timeline_chunks: Optional[List[Dict]]  # Current timeline chunks
    timeline_version: Optional[str]  # Version for tracking changes
    
    # Operation context
    current_operation: Optional[str]  # "FIND_HIGHLIGHTS", "CUT", "REPLACE", "INSERT", "FIND_BROLL", etc.
    operation_params: Optional[Dict]  # Operation-specific parameters
    
    # B-roll context
    selected_timeline_indices: Optional[List[int]]  # For B-roll operations
    broll_time_range: Optional[Tuple[float, float]]  # Source time range for B-roll
    
    # Narrative context
    editing_history: Optional[List[Dict]]  # History of operations
    narrative_notes: Optional[str]  # Director's notes about narrative flow

