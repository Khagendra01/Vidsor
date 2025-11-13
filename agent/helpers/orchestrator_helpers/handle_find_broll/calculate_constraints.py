"""Calculate B-roll constraints based on main clip statistics."""

import re
from typing import Dict, List, Tuple


def detect_multiple_broll_intent(user_query: str) -> bool:
    """
    Detect if user explicitly wants multiple B-roll clips per main clip.
    
    Args:
        user_query: Original user query string
        
    Returns:
        True if user explicitly requests multiple B-rolls, False otherwise
    """
    if not user_query:
        return False
    
    query_lower = user_query.lower()
    
    # Keywords that indicate user wants multiple B-rolls
    multiple_keywords = [
        "multiple", "several", "many", "more", "extra",
        "2 broll", "two broll", "3 broll", "three broll",
        "few broll", "a few broll", "couple broll",
        "multiple broll", "several broll", "many broll"
    ]
    
    # Check if any multiple keywords are present
    for keyword in multiple_keywords:
        if keyword in query_lower:
            return True
    
    # Check for explicit numbers (e.g., "2 brolls", "3 b-roll clips")
    number_pattern = r'\b(\d+)\s*(?:broll|b-roll|b roll)'
    match = re.search(number_pattern, query_lower)
    if match:
        count = int(match.group(1))
        if count > 1:
            return True
    
    return False


def calculate_broll_constraints(
    selected_chunks: List[Dict],
    user_wants_multiple: bool = False,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Calculate main clip statistics and target B-roll constraints.
    
    Args:
        selected_chunks: List of selected timeline chunks
        user_wants_multiple: If True, allows multiple B-roll clips per main clip (1-2x).
                            If False (default), generates exactly 1 B-roll per main clip (1:1 ratio).
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with constraints and statistics:
        - main_clip_count: Number of main clips
        - main_clip_total_duration: Total duration of main clips
        - min_broll_count: Minimum number of B-roll clips
        - max_broll_count: Maximum number of B-roll clips
        - target_broll_count: Target number of B-roll clips
        - target_broll_total_min: Minimum total B-roll duration
        - target_broll_total_max: Maximum total B-roll duration
        - min_clip_duration: Minimum duration per clip
        - max_clip_duration: Maximum duration per clip
        - preferred_clip_duration: Preferred duration per clip
        - ratio_min: Minimum ratio (20%)
        - ratio_max: Maximum ratio (40%)
    """
    main_clip_count = len(selected_chunks)
    main_clip_total_duration = 0.0
    
    for chunk in selected_chunks:
        orig_start = chunk.get("original_start_time", chunk.get("start_time"))
        orig_end = chunk.get("original_end_time", chunk.get("end_time"))
        if orig_start is not None and orig_end is not None:
            main_clip_total_duration += (orig_end - orig_start)
    
    # Calculate target B-roll quantity
    # Default: 1 B-roll per main clip (1:1 ratio)
    # Only use multiple if user explicitly requests it
    if user_wants_multiple:
        # User explicitly wants multiple: 1-2 broll per main clip (target: 1.5x)
        min_broll_count = max(1, main_clip_count)  # At least 1 broll
        max_broll_count = main_clip_count * 2  # Max 2 per main clip
        target_broll_count = int(round(main_clip_count * 1.5))  # Target: 1.5x
    else:
        # Default: 1 B-roll per main clip (1:1 ratio)
        min_broll_count = main_clip_count  # Exactly 1 per clip
        max_broll_count = main_clip_count  # Exactly 1 per clip
        target_broll_count = main_clip_count  # Exactly 1 per clip
    
    # Calculate target B-roll duration (20-40% of main content)
    ratio_min = 0.20  # 20%
    ratio_max = 0.40  # 40%
    target_broll_total_min = main_clip_total_duration * ratio_min
    target_broll_total_max = main_clip_total_duration * ratio_max
    
    # Per-clip duration guidelines: 2-4 seconds (prefer 3)
    min_clip_duration = 2.0
    max_clip_duration = 4.0
    preferred_clip_duration = 3.0
    
    if verbose:
        print("\n[DIRECTOR GUIDELINES]")
        print(f"  Main clips: {main_clip_count} clips, {main_clip_total_duration:.1f}s total")
        if user_wants_multiple:
            print(f"  Target: {min_broll_count}-{max_broll_count} broll clips (aiming for {target_broll_count}) [User requested multiple]")
        else:
            print(f"  Target: {target_broll_count} broll clip(s) (1 per main clip) [Default: 1:1 ratio]")
        print(f"  Duration per clip: {min_clip_duration}-{max_clip_duration}s (prefer {preferred_clip_duration}s)")
        print(f"  Total target: {target_broll_total_min:.1f}-{target_broll_total_max:.1f}s ({ratio_min*100:.0f}-{ratio_max*100:.0f}% of main content)")
    
    return {
        "main_clip_count": main_clip_count,
        "main_clip_total_duration": main_clip_total_duration,
        "min_broll_count": min_broll_count,
        "max_broll_count": max_broll_count,
        "target_broll_count": target_broll_count,
        "target_broll_total_min": target_broll_total_min,
        "target_broll_total_max": target_broll_total_max,
        "min_clip_duration": min_clip_duration,
        "max_clip_duration": max_clip_duration,
        "preferred_clip_duration": preferred_clip_duration,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
    }

