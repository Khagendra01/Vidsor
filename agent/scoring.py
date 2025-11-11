"""Scoring and grouping functions for video segments."""

from typing import Dict, List, Tuple, Optional, Set
from agent.feature_extractor import PerSecondFeatureExtractor


def score_seconds(feature_extractor: PerSecondFeatureExtractor, 
                  query_intent: Dict,
                  weights: Dict,
                  semantic_results: List[Dict],
                  activity_results: List[Dict],
                  hierarchical_results: List[Dict],
                  semantic_analysis: Optional[Dict] = None,
                  negated_second_indices: Optional[Set[int]] = None,
                  verbose: bool = False) -> List[Dict]:
    """
    Score seconds using weighted features.
    
    If negated_second_indices is provided (for negative queries), only score those specific seconds.
    Otherwise, score all seconds.
    
    Returns list of {second_idx, score, features, time_range, ...}
    """
    scored_seconds = []
    
    # Helper function to map time to second index
    def time_to_second_idx(time_seconds: float) -> Optional[int]:
        """Map a time in seconds to the corresponding second index."""
        for idx, second_data in enumerate(feature_extractor.segment_tree.seconds):
            tr = second_data.get("time_range", [])
            if tr and len(tr) >= 2 and tr[0] <= time_seconds <= tr[1]:
                return idx
        # Fallback: approximate by rounding
        return int(time_seconds) if 0 <= int(time_seconds) < len(feature_extractor.segment_tree.seconds) else None
    
    # Build lookup maps for search results
    semantic_map = {}  # second_idx -> max_score
    for result in semantic_results:
        tr = result.get("time_range", [])
        if tr and len(tr) >= 2:
            second_idx = time_to_second_idx(tr[0])
            if second_idx is not None:
                score = result.get("score", 0)
                semantic_map[second_idx] = max(semantic_map.get(second_idx, 0), score)
    
    activity_map = {}  # second_idx -> score
    for result in activity_results:
        for evidence in result.get("evidence", []):
            tr = evidence.get("time_range", [])
            if tr and len(tr) >= 2:
                second_idx = time_to_second_idx(tr[0])
                if second_idx is not None:
                    activity_map[second_idx] = 1.0  # Activity is binary
    
    hierarchical_map = {}  # second_idx -> score
    for result in hierarchical_results:
        tr = result.get("time_range", [])
        if tr and len(tr) >= 2:
            second_idx = time_to_second_idx(tr[0])
            if second_idx is not None:
                hierarchical_map[second_idx] = result.get("score", 1.0)
    
    # Determine which seconds to score
    seconds_to_score = range(len(feature_extractor.segment_tree.seconds))
    if negated_second_indices is not None:
        # CORE FIX: Only score the seconds identified by negation logic
        seconds_to_score = sorted(negated_second_indices)
        if verbose:
            print(f"\n[SCORING] Scoring {len(seconds_to_score)} negated seconds (restricted set)...")
    else:
        if verbose:
            print(f"\n[SCORING] Scoring {len(feature_extractor.segment_tree.seconds)} seconds...")
    
    if verbose:
        print(f"  Semantic matches: {len(semantic_map)} seconds")
        print(f"  Activity matches: {len(activity_map)} seconds")
        print(f"  Hierarchical matches: {len(hierarchical_map)} seconds")
    
    # Score each second (only negated ones if negative query)
    for second_idx in seconds_to_score:
        features = feature_extractor.extract_features_for_second(second_idx)
        if not features:
            continue
        
        # Get search result scores
        semantic_score = semantic_map.get(second_idx, 0.0)
        activity_score = activity_map.get(second_idx, 0.0)
        hierarchical_score = hierarchical_map.get(second_idx, 0.0)
        
        # Compute object score
        object_score = 0.0
        is_negative = False
        if semantic_analysis:
            is_negative = semantic_analysis.get("query_type") == "NEGATIVE" or semantic_analysis.get("special_handling", {}).get("negation", False)
        
        for class_name, normalized_count in features["object_presence"].items():
            class_weight = weights["object_weights"].get(class_name, 0.1)
            # Normalize count to [0, 1] (max 3 -> 1.0)
            normalized = normalized_count / 3.0
            if is_negative:
                # For negative queries: score HIGH when objects are ABSENT (low count)
                # Invert: 1.0 - normalized means absence = high score
                object_score += class_weight * (1.0 - normalized)
            else:
                # For positive queries: score HIGH when objects are PRESENT (high count)
                object_score += class_weight * normalized
        
        # Final weighted score
        final_score = (
            weights["semantic_weight"] * semantic_score +
            weights["activity_weight"] * activity_score +
            weights["hierarchical_weight"] * hierarchical_score +
            object_score
        )
        
        # Contextual boost: if object-centric and object present, ensure minimum score
        if query_intent.get("is_object_centric") and object_score > 0:
            final_score = max(final_score, weights["threshold"])
        
        scored_seconds.append({
            "second": second_idx,
            "score": final_score,
            "time_range": features["time_range"],
            "semantic_score": semantic_score,
            "activity_score": activity_score,
            "hierarchical_score": hierarchical_score,
            "object_score": object_score,
            "features": features
        })
    
    if verbose:
        if scored_seconds:
            sorted_scores = sorted(scored_seconds, key=lambda x: x["score"], reverse=True)
            print(f"  Score range: {sorted_scores[-1]['score']:.3f} - {sorted_scores[0]['score']:.3f}")
            top_scores = [f"{s['second']}:{s['score']:.3f}" for s in sorted_scores[:5]]
            print(f"  Top 5 scores: {top_scores}")
    
    return scored_seconds


def group_contiguous_seconds(scored_seconds: List[Dict], 
                              min_duration: float = 1.0,
                              gap_tolerance: float = 1.0) -> List[Tuple[float, float]]:
    """
    Group contiguous high-scoring seconds into time ranges.
    """
    if not scored_seconds:
        return []
    
    # Sort by second index
    sorted_seconds = sorted(scored_seconds, key=lambda x: x["second"])
    
    time_ranges = []
    current_start = None
    current_end = None
    
    for sec in sorted_seconds:
        tr = sec.get("time_range", [])
        if not tr or len(tr) < 2:
            continue
        
        start, end = tr[0], tr[1]
        
        if current_start is None:
            current_start = start
            current_end = end
        elif start - current_end <= gap_tolerance:
            # Contiguous or close enough, merge
            current_end = end
        else:
            # Gap detected, save current range and start new
            if current_end - current_start >= min_duration:
                time_ranges.append((current_start, current_end))
            current_start = start
            current_end = end
    
    # Add final range
    if current_start is not None and current_end - current_start >= min_duration:
        time_ranges.append((current_start, current_end))
    
    return time_ranges

