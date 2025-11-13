"""Score all seconds and filter by adaptive threshold."""

import time
from typing import Optional, Set
from agent.utils.logging_utils import get_log_helper
from agent.utils.search.scoring import score_seconds


def score_and_filter(
    segment_tree,
    feature_extractor,
    all_search_results: list,
    query_intent: dict,
    weights: dict,
    semantic_analysis: dict,
    state: dict = None,
    logger=None,
    verbose: bool = False
) -> tuple[list, float]:
    """
    Score all seconds and filter by adaptive threshold.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        feature_extractor: PerSecondFeatureExtractor instance
        all_search_results: List of all search results
        query_intent: Query intent dictionary
        weights: Weights dictionary
        semantic_analysis: Semantic analysis dictionary
        state: Optional state dictionary (may contain operation_type, video_duration)
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (filtered_seconds, adaptive_threshold)
    """
    log = get_log_helper(logger, verbose)
    
    log.info(f"\n[STEP 3] Scoring all seconds with weighted features...")
    log.info(f"  Total seconds to score: {len(segment_tree.seconds)}")
    start_time = time.time()
    
    # Separate results by type for scoring
    semantic_results = [r for r in all_search_results if r is not None and r.get("search_type") == "semantic"]
    activity_results = [r for r in all_search_results if r is not None and r.get("search_type") == "activity"]
    hierarchical_results = [r for r in all_search_results if r is not None and r.get("search_type") in ["hierarchical", "hierarchical_highlight"]]
    negated_results = [r for r in all_search_results if r is not None and r.get("search_type") == "object_negated"]
    
    # CORE FIX: For negative queries, only score the seconds identified by negation logic
    is_negative = False
    if semantic_analysis:
        is_negative = semantic_analysis.get("query_type") == "NEGATIVE" or semantic_analysis.get("special_handling", {}).get("negation", False)
    negated_second_indices: Optional[Set[int]] = None
    
    if is_negative and negated_results:
        # Extract the second indices from negated results - these are the ONLY seconds to score
        negated_second_indices = set()
        for result in negated_results:
            second_idx = result.get("second")
            if second_idx is not None:
                negated_second_indices.add(second_idx)
        
        log.info(f"\n[NEGATION] Only scoring {len(negated_second_indices)} seconds identified by negation logic")
    
    # Score seconds (only negated ones if negative query)
    scored_seconds = score_seconds(
        feature_extractor,
        query_intent,
        weights,
        semantic_results,
        activity_results,
        hierarchical_results,
        semantic_analysis=semantic_analysis,
        negated_second_indices=negated_second_indices,
        verbose=verbose
    )
    elapsed = time.time() - start_time
    log.info(f"[STEP 3] Scoring completed in {elapsed:.2f}s")
    
    # Get operation type and video duration from state (if available)
    operation_type = state.get("operation_type") if state else None
    video_duration_from_state = state.get("video_duration") if state else None
    
    # STRATEGY 1: Adaptive threshold (percentile-based)
    # Calculate score distribution and use top percentile as threshold
    if scored_seconds:
        all_scores = [s["score"] for s in scored_seconds]
        all_scores.sort(reverse=True)
        
        # Use top 12% of video as target (conservative but adaptive)
        video_duration = video_duration_from_state if video_duration_from_state else (len(segment_tree.seconds) if segment_tree else 600)
        target_seconds = max(30, int(video_duration * 0.12))  # At least 30 seconds, or 12% of video
        
        # Determine quality floor based on operation type
        if operation_type == "FIND_HIGHLIGHTS":
            # For highlights, allow lower quality floor to ensure coverage
            quality_floor = 0.20  # Lower floor for highlights
            min_threshold_multiplier = 0.5  # Allow 50% below min threshold
        else:
            # For other operations (FIND_BROLL, etc.), maintain strict quality
            quality_floor = 0.35  # Higher floor for precision
            min_threshold_multiplier = 0.7  # Only allow 30% below min threshold
        
        if len(all_scores) >= target_seconds:
            # Use score at target_seconds position as threshold
            adaptive_threshold = all_scores[target_seconds - 1]
        else:
            # Not enough scores, use fixed threshold as fallback
            adaptive_threshold = weights["threshold"]
        
        # Ensure threshold respects quality floor and maximum
        adaptive_threshold = max(quality_floor, min(0.65, adaptive_threshold))
        
        # Also respect minimum threshold from weights (operation-aware)
        min_threshold = weights["threshold"]
        adaptive_threshold = max(min_threshold * min_threshold_multiplier, adaptive_threshold)
        
        log.info(f"\n[ADAPTIVE THRESHOLD] Score distribution analysis:")
        log.info(f"  Total seconds scored: {len(scored_seconds)}")
        log.info(f"  Target: top {target_seconds} seconds ({target_seconds/video_duration*100:.1f}% of video)")
        log.info(f"  Score range: {all_scores[-1]:.3f} - {all_scores[0]:.3f}")
        log.info(f"  Score at position {target_seconds}: {adaptive_threshold:.3f}")
        log.info(f"  Operation type: {operation_type or 'UNKNOWN'}")
        log.info(f"  Quality floor: {quality_floor:.2f} (operation-aware)")
        log.info(f"  Using adaptive threshold: {adaptive_threshold:.3f} (min from weights: {min_threshold:.3f})")
        
        # OPERATION-AWARE: For FIND_HIGHLIGHTS, check if results are sufficient
        # If not, lower threshold incrementally to meet target
        if operation_type == "FIND_HIGHLIGHTS":
            filtered_seconds_initial = [s for s in scored_seconds if s["score"] >= adaptive_threshold]
            initial_count = len(filtered_seconds_initial)
            target_minimum = int(target_seconds * 0.7)  # 70% of target is acceptable minimum
            
            if initial_count < target_minimum and len(all_scores) > initial_count:
                log.info(f"\n[THRESHOLD ADJUSTMENT] FIND_HIGHLIGHTS: Only {initial_count} seconds found (target: {target_minimum})")
                log.info(f"  Lowering threshold incrementally to meet coverage target...")
                
                # Try progressively lower thresholds
                threshold_steps = [
                    adaptive_threshold * 0.9,  # 10% lower
                    adaptive_threshold * 0.8,  # 20% lower
                    adaptive_threshold * 0.7,  # 30% lower
                    adaptive_threshold * 0.6,  # 40% lower
                    max(quality_floor, adaptive_threshold * 0.5),  # 50% lower (but not below floor)
                ]
                
                best_threshold = adaptive_threshold
                best_count = initial_count
                
                for step_threshold in threshold_steps:
                    if step_threshold < quality_floor:
                        break
                    
                    filtered_test = [s for s in scored_seconds if s["score"] >= step_threshold]
                    test_count = len(filtered_test)
                    
                    if test_count >= target_minimum:
                        # Found threshold that meets target - use it
                        adaptive_threshold = step_threshold
                        log.info(f"  ✓ Adjusted threshold to {adaptive_threshold:.3f} → {test_count} seconds (meets target)")
                        break
                    elif test_count > best_count:
                        # Better than current best, keep it
                        best_threshold = step_threshold
                        best_count = test_count
                        log.info(f"  → Lowered threshold to {step_threshold:.3f} → {test_count} seconds (improved, continuing...)")
                
                # Use best threshold found (either target-met or best-improvement)
                if adaptive_threshold != best_threshold:
                    adaptive_threshold = best_threshold
                    log.info(f"  → Using best threshold: {adaptive_threshold:.3f} → {best_count} seconds")
    else:
        adaptive_threshold = weights["threshold"]
        log.info(f"\n[FILTERING] No scores to analyze, using fixed threshold: {adaptive_threshold:.2f}")
    
    # Filter by adaptive threshold and sort
    filtered_seconds = [s for s in scored_seconds if s["score"] >= adaptive_threshold]
    filtered_seconds.sort(key=lambda x: x["score"], reverse=True)
    
    log.info(f"\n[FILTERING] Filtered to {len(filtered_seconds)} seconds above adaptive threshold {adaptive_threshold:.3f}")
    if filtered_seconds:
        log.info(f"  Top 10 scored seconds:")
        for i, sec in enumerate(filtered_seconds[:10], 1):
            log.info(f"    {i}. Second {sec['second']}: score={sec['score']:.3f} "
                      f"(sem={sec['semantic_score']:.2f}, act={sec['activity_score']:.2f}, "
                      f"obj={sec['object_score']:.2f}, hier={sec.get('hierarchical_score', 0):.2f})")
    
    return filtered_seconds, adaptive_threshold

