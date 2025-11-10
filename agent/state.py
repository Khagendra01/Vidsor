"""Agent state definitions."""

from typing import Dict, List, Optional, Tuple, Annotated, TypedDict
from langgraph.graph.message import add_messages
from segment_tree_utils import SegmentTreeQuery


class AgentState(TypedDict):
    """State for the video clip extraction agent."""
    messages: Annotated[list, add_messages]
    user_query: str
    video_path: str
    json_path: str
    query_type: Optional[str]  # "visual", "audio", "combined", "object", "activity"
    search_results: Optional[List[Dict]]
    time_ranges: Optional[List[Tuple[float, float]]]
    confidence: Optional[float]  # 0-1, how confident we are about the results
    needs_clarification: bool
    clarification_question: Optional[str]
    output_clips: List[str]  # Paths to saved clip files
    segment_tree: Optional[SegmentTreeQuery]
    verbose: bool  # Whether to print verbose output
    # Memory/context fields for agentic behavior
    previous_time_ranges: Optional[List[Tuple[float, float]]]  # Previous search results
    previous_scored_seconds: Optional[List[Dict]]  # Previous scored seconds with scores
    previous_query: Optional[str]  # Original query before clarification
    previous_search_results: Optional[List[Dict]]  # Previous search results

