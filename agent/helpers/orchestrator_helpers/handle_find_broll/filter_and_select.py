"""Filter and select best B-roll clips based on scoring."""

from typing import Dict, List, Tuple
from agent.helpers.orchestrator_helpers import match_search_result_to_time_range


def filter_and_select_broll_clips(
    time_ranges: List[Tuple[float, float]],
    search_results: List[Dict],
    constraints: Dict[str, any],
    main_clip_total_duration: float,
    verbose: bool = False
) -> List[Dict]:
    """
    Score, filter, and select best B-roll clips based on relevance and duration.
    
    Args:
        time_ranges: List of (start, end) time range tuples
        search_results: List of search result dictionaries
        constraints: Dictionary with constraint values from calculate_broll_constraints
        main_clip_total_duration: Total duration of main clips for ratio calculation
        verbose: Whether to print verbose output
        
    Returns:
        List of selected B-roll range dictionaries with metadata
    """
    min_clip_duration = constraints["min_clip_duration"]
    max_clip_duration = constraints["max_clip_duration"]
    preferred_clip_duration = constraints["preferred_clip_duration"]
    min_broll_count = constraints["min_broll_count"]
    max_broll_count = constraints["max_broll_count"]
    target_broll_count = constraints["target_broll_count"]
    target_broll_total_max = constraints["target_broll_total_max"]
    
    if verbose:
        print(f"\n[PLANNER RESULTS]")
        print(f"  Found: {len(time_ranges)} candidate clips")
    
    # Score and prepare time ranges with metadata
    scored_ranges = []
    for tr in time_ranges:
        if len(tr) < 2:
            continue
        tr_start, tr_end = tr[0], tr[1]
        duration = tr_end - tr_start
        
        # Get description and score from search results
        default_desc = f"B-roll: {tr_start:.1f}s - {tr_end:.1f}s"
        description, unified_description, audio_description = match_search_result_to_time_range(
            search_results, tr_start, default_desc
        )
        relevance_score = 0.7  # Default
        # Try to get score from matched result
        for result in search_results:
            if result is None:
                continue
            result_tr = result.get("time_range", [])
            if result_tr and len(result_tr) >= 2:
                if abs(result_tr[0] - tr_start) < 1.0:
                    relevance_score = result.get("score", result.get("relevance_score", 0.7))
                    break
        
        # Calculate duration score (prefer 4-8 seconds)
        if duration < min_clip_duration:
            duration_score = 0.3  # Too short
        elif duration > max_clip_duration:
            duration_score = 0.5  # Too long
        elif min_clip_duration <= duration <= preferred_clip_duration:
            duration_score = 1.0  # Perfect range
        else:
            duration_score = 0.8  # Good but not ideal
        
        # Combined score (weighted: 70% relevance, 30% duration)
        combined_score = (relevance_score * 0.7) + (duration_score * 0.3)
        
        scored_ranges.append({
            "time_range": (tr_start, tr_end),
            "duration": duration,
            "description": description,
            "unified_description": unified_description,
            "audio_description": audio_description,
            "relevance_score": relevance_score,
            "duration_score": duration_score,
            "combined_score": combined_score
        })
    
    # Sort by combined score (best first)
    scored_ranges.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Select top N within target range
    selected_ranges = scored_ranges[:max_broll_count]  # Take up to max
    
    # If we have more than target, try to get closer to target
    if len(selected_ranges) > target_broll_count:
        # Calculate total duration with target count
        test_total = sum(r["duration"] for r in selected_ranges[:target_broll_count])
        if test_total <= target_broll_total_max:
            selected_ranges = selected_ranges[:target_broll_count]
        else:
            # Need fewer clips to stay within duration limit
            # Find the best combination that fits
            best_combination = []
            best_total = 0.0
            for i in range(min_broll_count, min(target_broll_count + 1, len(selected_ranges) + 1)):
                combo = selected_ranges[:i]
                combo_total = sum(r["duration"] for r in combo)
                if combo_total <= target_broll_total_max and combo_total > best_total:
                    best_combination = combo
                    best_total = combo_total
            if best_combination:
                selected_ranges = best_combination
            else:
                # Fallback: take minimum count
                selected_ranges = selected_ranges[:min_broll_count]
    
    # Ensure we have at least minimum
    if len(selected_ranges) < min_broll_count and len(scored_ranges) >= min_broll_count:
        selected_ranges = scored_ranges[:min_broll_count]
    
    if verbose:
        print(f"  Selected: {len(selected_ranges)} clips (within target range {min_broll_count}-{max_broll_count})")
        total_selected_duration = sum(r["duration"] for r in selected_ranges)
        ratio_actual = (total_selected_duration / main_clip_total_duration * 100) if main_clip_total_duration > 0 else 0
        print(f"  Total duration: {total_selected_duration:.1f}s ({ratio_actual:.0f}% of main content)")
    
    return selected_ranges

