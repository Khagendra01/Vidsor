"""Calculate B-roll constraints based on main clip statistics."""

from typing import Dict, List, Tuple


def calculate_broll_constraints(
    selected_chunks: List[Dict],
    verbose: bool = False
) -> Dict[str, any]:
    """
    Calculate main clip statistics and target B-roll constraints.
    
    Args:
        selected_chunks: List of selected timeline chunks
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
    # Formula: 1-2 broll per main clip (target: 1.5x)
    min_broll_count = max(1, main_clip_count)  # At least 1 broll
    max_broll_count = main_clip_count * 2  # Max 2 per main clip
    target_broll_count = int(round(main_clip_count * 1.5))  # Target: 1.5x
    
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
        print(f"  Target: {min_broll_count}-{max_broll_count} broll clips (aiming for {target_broll_count})")
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

