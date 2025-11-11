"""Build B-roll search query with director guidelines."""

from typing import Dict


def build_broll_query(
    start_time: float,
    end_time: float,
    main_keywords: list,
    constraints: Dict[str, any],
    verbose: bool = False
) -> str:
    """
    Build B-roll search query with director guidelines.
    
    Args:
        start_time: Start time of source range
        end_time: End time of source range
        main_keywords: List of main action keywords to exclude
        constraints: Dictionary with constraint values from calculate_broll_constraints
        verbose: Whether to print verbose output
        
    Returns:
        B-roll search query string with director guidelines
    """
    # Build B-roll search query with director guidelines
    broll_query = f"find B-roll between {start_time:.1f}s and {end_time:.1f}s, show nature, scenery, wide shots, different from main action"
    if main_keywords:
        # Exclude main action keywords
        exclude_terms = ", ".join(set(main_keywords[:5]))  # Top 5 unique keywords
        broll_query += f", not {exclude_terms}"
    
    # Add director guidelines to query
    main_clip_count = constraints["main_clip_count"]
    main_clip_total_duration = constraints["main_clip_total_duration"]
    target_broll_count = constraints["target_broll_count"]
    min_broll_count = constraints["min_broll_count"]
    max_broll_count = constraints["max_broll_count"]
    min_clip_duration = constraints["min_clip_duration"]
    max_clip_duration = constraints["max_clip_duration"]
    preferred_clip_duration = constraints["preferred_clip_duration"]
    target_broll_total_min = constraints["target_broll_total_min"]
    target_broll_total_max = constraints["target_broll_total_max"]
    ratio_min = constraints["ratio_min"]
    ratio_max = constraints["ratio_max"]
    
    director_guidelines = (
        f"\n\nDirector's Guidelines:\n"
        f"- Main clips: {main_clip_count} clips, {main_clip_total_duration:.1f}s total\n"
        f"- Target: {target_broll_count} broll clips (range: {min_broll_count}-{max_broll_count})\n"
        f"- Duration per clip: {min_clip_duration}-{max_clip_duration}s (prefer {preferred_clip_duration}s)\n"
        f"- Total broll duration: {target_broll_total_min:.1f}-{target_broll_total_max:.1f}s ({ratio_min*100:.0f}-{ratio_max*100:.0f}% of main content)\n"
        f"- Purpose: Complementary footage, not overwhelming the main story\n"
        f"- Quality over quantity: Select the most relevant and complementary footage"
    )
    broll_query += director_guidelines
    
    if verbose:
        print(f"  Selected chunks: indices (provided in handler)")
        print(f"  Source time range: {start_time:.1f}s - {end_time:.1f}s")
        print(f"  B-roll query: '{broll_query[:200]}...' (with director guidelines)")
    
    return broll_query

