"""Duration constraint utilities for timeline operations."""

from typing import List, Tuple, Dict
from agent.utils.logging_utils import get_log_helper


def apply_duration_constraints(
    time_ranges: List[Tuple[float, float]],
    max_total_duration: float,
    max_clip_duration: float = 20.0,
    min_gap: float = 3.0,
    logger=None,
    verbose: bool = False
) -> List[Tuple[float, float]]:
    """
    Apply duration constraints to time ranges.
    
    Args:
        time_ranges: List of (start, end) tuples
        max_total_duration: Maximum total duration allowed
        max_clip_duration: Maximum duration per clip
        min_gap: Minimum gap between clips
        logger: Logger instance
        verbose: Whether to print verbose output
        
    Returns:
        Filtered and constrained time ranges
    """
    if not time_ranges:
        return []
    
    # Sort by start time
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    
    # Filter by max_clip_duration and accumulate until max_total_duration
    # Use SOFT limits: prefer shorter clips but don't hard-cut important long scenes
    filtered = []
    total_duration = 0.0
    last_end = -min_gap  # Track last end time for gap checking
    log = get_log_helper(logger, verbose)
    
    for start, end in sorted_ranges:
        duration = end - start
        
        # SOFT limit: Prefer shorter clips, but allow longer if they're important
        # Only skip if extremely long (>2x max_clip_duration)
        if duration > max_clip_duration * 2:
            log.info(f"  Skipping very long clip: {duration:.1f}s > {max_clip_duration * 2:.1f}s")
            continue
        
        # SOFT gap: Prefer spacing, but don't skip if clips are close
        # Only skip if overlapping or extremely close (<1s gap)
        gap = start - last_end
        if gap < 1.0 and gap >= 0:  # Overlapping or very close
            # Merge with previous if they're very close
            if filtered:
                prev_start, prev_end = filtered[-1]
                filtered[-1] = (prev_start, max(prev_end, end))
                total_duration = sum(e - s for s, e in filtered)
                last_end = max(prev_end, end)
                continue
        
        # SOFT total duration: Prefer staying under limit, but allow 20% over for important content
        if total_duration + duration > max_total_duration * 1.2:
            # Hard stop if way over
            break
        elif total_duration + duration > max_total_duration:
            # Over limit but within 20% - still add if high quality (check score if available)
            # For now, add it but log warning
            log(f"  [WARNING] Exceeding target duration: {total_duration + duration:.1f}s > {max_total_duration:.1f}s")
        
        filtered.append((start, end))
        total_duration += duration
        last_end = end
    if len(filtered) < len(time_ranges):
        log(f"  [DURATION] Filtered from {len(time_ranges)} to {len(filtered)} ranges "
            f"(total: {total_duration:.1f}s / {max_total_duration:.1f}s target)")
    elif total_duration > max_total_duration:
        log(f"  [DURATION] Total duration {total_duration:.1f}s exceeds target {max_total_duration:.1f}s (soft limit)")
    
    return filtered


def apply_duration_constraints_with_neighbors(
    time_ranges: List[Tuple[float, float]],
    existing_chunks: List[Dict],
    max_clip_duration: float = 15.0,
    min_gap: float = 2.0,
    logger=None,
    verbose: bool = False
) -> List[Tuple[float, float]]:
    """
    Apply duration constraints considering existing timeline chunks.
    Analyzes neighbors to determine appropriate clip lengths.
    """
    if not time_ranges:
        return []
    
    # For now, use same logic as empty timeline but with stricter constraints
    # TODO: Analyze neighbors, find gaps, fit content appropriately
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    
    filtered = []
    last_end = -min_gap
    
    for start, end in sorted_ranges:
        duration = end - start
        
        # SOFT limits: prefer shorter but allow longer if needed
        if duration > max_clip_duration * 2:
            continue
        
        # SOFT gap: prefer spacing but don't skip close clips
        gap = start - last_end
        if gap < 1.0 and gap >= 0:  # Overlapping or very close - merge
            if filtered:
                prev_start, prev_end = filtered[-1]
                filtered[-1] = (prev_start, max(prev_end, end))
                last_end = max(prev_end, end)
                continue
        
        filtered.append((start, end))
        last_end = end
    
    return filtered

