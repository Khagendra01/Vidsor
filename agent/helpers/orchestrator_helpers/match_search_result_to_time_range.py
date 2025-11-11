"""Match a time range to a search result and extract descriptions."""

from typing import List, Dict, Tuple


def match_search_result_to_time_range(
    search_results: List[Dict],
    start_time: float,
    default_description: str = ""
) -> Tuple[str, str, str]:
    """
    Match a time range to a search result and extract descriptions.
    
    Args:
        search_results: List of search result dictionaries
        start_time: Start time to match
        default_description: Default description if no match found
        
    Returns:
        Tuple of (description, unified_description, audio_description)
    """
    description = default_description
    unified_description = default_description
    audio_description = ""
    
    for result in search_results:
        if result is None:
            continue
        result_tr = result.get("time_range", [])
        if result_tr and len(result_tr) >= 2:
            if abs(result_tr[0] - start_time) < 1.0:  # Close match
                description = result.get("description", default_description)
                unified_description = result.get("unified_description", description)
                audio_description = result.get("audio_description", "")
                break
    
    return description, unified_description, audio_description

