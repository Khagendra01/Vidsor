"""Utility functions for the video clip agent."""

import re
from typing import List, Tuple


def extract_json(text: str) -> str:
    """Extract JSON from LLM response."""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    else:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
    return text.strip()


def merge_time_ranges(time_ranges: List[Tuple[float, float]], 
                     padding: float = 2.0) -> List[Tuple[float, float]]:
    """Merge overlapping time ranges and add padding."""
    if not time_ranges:
        return []
    
    # Sort by start time
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    merged = []
    
    current_start, current_end = sorted_ranges[0]
    
    for start, end in sorted_ranges[1:]:
        # Add padding
        if start - padding <= current_end:
            # Overlapping or close, merge
            current_end = max(current_end, end)
        else:
            # Not overlapping, save current and start new
            merged.append((max(0, current_start - padding), current_end + padding))
            current_start, current_end = start, end
    
    # Add last range
    merged.append((max(0, current_start - padding), current_end + padding))
    
    return merged

