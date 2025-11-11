"""Adjust B-roll clip durations to fit within 4-8 second range."""

from typing import Dict, List
from extractor.utils.video_utils import get_video_duration


def adjust_broll_durations(
    selected_ranges: List[Dict],
    constraints: Dict[str, any],
    video_path: str = "",
    verbose: bool = False
) -> List[Dict]:
    """
    Adjust clip durations to fit within 4-8 second range.
    
    Args:
        selected_ranges: List of selected B-roll range dictionaries
        constraints: Dictionary with constraint values from calculate_broll_constraints
        video_path: Path to video file for boundary checks
        verbose: Whether to print verbose output
        
    Returns:
        List of adjusted B-roll range dictionaries with updated durations
    """
    min_clip_duration = constraints["min_clip_duration"]
    max_clip_duration = constraints["max_clip_duration"]
    preferred_clip_duration = constraints["preferred_clip_duration"]
    
    # Get video duration for boundary checks
    video_duration = None
    if video_path:
        video_duration = get_video_duration(video_path)
    
    adjusted_ranges = []
    for r in selected_ranges:
        tr_start, tr_end = r["time_range"]
        duration = r["duration"]
        adjusted_start = tr_start
        adjusted_end = tr_end
        
        # Adjust if too short (< 4s) - try to extend
        if duration < min_clip_duration:
            extension_needed = min_clip_duration - duration
            # Try to extend from end (check video boundary)
            if video_duration is None or (tr_end + extension_needed) <= video_duration:
                adjusted_end = tr_end + extension_needed
            elif video_duration and tr_start > extension_needed:
                # Try to extend from start
                adjusted_start = max(0, tr_start - extension_needed)
        
        # Adjust if too long (> 8s) - trim to 8s
        elif duration > max_clip_duration:
            # Trim from end, keeping the start
            adjusted_end = adjusted_start + max_clip_duration
        
        # If in good range, keep as-is (maybe slight adjustment to preferred)
        elif duration < preferred_clip_duration:
            # Optionally extend slightly toward preferred, but don't force
            pass  # Keep as-is for now
        
        adjusted_duration = adjusted_end - adjusted_start
        adjusted_ranges.append({
            **r,
            "time_range": (adjusted_start, adjusted_end),
            "duration": adjusted_duration,
            "was_adjusted": (adjusted_start != tr_start or adjusted_end != tr_end)
        })
    
    if verbose:
        print(f"\n[FINAL SELECTION]")
        for i, r in enumerate(adjusted_ranges, 1):
            adj_note = " (adjusted)" if r.get("was_adjusted", False) else ""
            print(f"  {i}. {r['duration']:.1f}s ({r['time_range'][0]:.1f}s - {r['time_range'][1]:.1f}s){adj_note}")
    
    return adjusted_ranges

