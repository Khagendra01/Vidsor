"""Create timeline chunks from time ranges with descriptions from search results."""

from typing import List, Dict, Tuple
from agent.helpers.orchestrator_helpers.create_timeline_chunk import create_timeline_chunk
from agent.helpers.orchestrator_helpers.match_search_result_to_time_range import match_search_result_to_time_range


def create_chunks_from_time_ranges(
    time_ranges: List[Tuple[float, float]],
    search_results: List[Dict],
    timeline_start: float,
    chunk_type: str = "highlight",
    default_score: float = 0.7,
    name_prefix: str = "Clip"
) -> List[Dict]:
    """
    Create timeline chunks from time ranges with descriptions from search results.
    
    Args:
        time_ranges: List of (start, end) time tuples
        search_results: List of search result dictionaries
        timeline_start: Starting timeline time
        chunk_type: Type of chunk (highlight, broll, etc.)
        default_score: Default relevance score
        name_prefix: Prefix for default descriptions
        
    Returns:
        List of created chunk dictionaries
    """
    chunks = []
    current_timeline_time = timeline_start
    
    for i, (start_time, end_time) in enumerate(time_ranges):
        default_desc = f"{name_prefix} {i+1}: {start_time:.1f}s - {end_time:.1f}s"
        description, unified_description, audio_description = match_search_result_to_time_range(
            search_results, start_time, default_desc
        )
        
        chunk = create_timeline_chunk(
            original_start=start_time,
            original_end=end_time,
            timeline_start=current_timeline_time,
            chunk_type=chunk_type,
            description=description,
            unified_description=unified_description,
            audio_description=audio_description,
            score=default_score
        )
        
        chunks.append(chunk)
        current_timeline_time = chunk["end_time"]
    
    return chunks

