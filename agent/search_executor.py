"""Search execution module for different search types."""

import time
from typing import List, Dict, Optional, Any
from agent.logging_utils import get_log_helper


def execute_hierarchical_search(
    segment_tree,
    keywords: List[str],
    match_mode: str = "any",
    max_results: int = 20,
    logger=None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Execute hierarchical keyword search.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        keywords: List of keywords to search for
        match_mode: "any" (OR) or "all" (AND) - whether to match any or all keywords
        max_results: Maximum number of results to return
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        List of search results with standardized format
    """
    log = get_log_helper(logger, verbose)
    results = []
    
    if not keywords or not segment_tree.hierarchical_tree:
        return results
    
    log.info(f"\n  [SEARCH TYPE: Hierarchical Tree (Fast Keyword Lookup)]")
    for keyword in keywords:
        log.info(f"    Keyword: '{keyword}'")
        search_results = segment_tree.hierarchical_keyword_search(
            [keyword],
            match_mode=match_mode,
            max_results=max_results
        )
        for result in search_results:
            # Prefer leaf nodes
            if result.get("level", -1) == 0:
                tr = result.get("time_range", [])
                if tr and len(tr) >= 2:
                    results.append({
                        "search_type": "hierarchical",
                        "time_range": tr,
                        "score": result.get("match_count", 1),
                        "node_id": result.get("node_id"),
                        "matched_keyword": keyword
                    })
        log.info(f"      Found {len([r for r in search_results if r.get('level') == 0])} leaf node matches")
    
    return results


def execute_hierarchical_highlight_search(
    segment_tree,
    action_keywords: Optional[List[str]] = None,
    max_results: int = 20,
    logger=None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Execute hierarchical search for general highlights.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        action_keywords: Optional list of action keywords for highlights
        max_results: Maximum number of results to return
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        List of highlight search results
    """
    log = get_log_helper(logger, verbose)
    results = []
    
    if not segment_tree.hierarchical_tree:
        return results
    
    log.info(f"\n  [SEARCH TYPE: General Highlights (Hierarchical Tree)]")
    scored_leaves = segment_tree.hierarchical_score_leaves_for_highlights(
        action_keywords=action_keywords,
        max_results=max_results
    )
    for leaf in scored_leaves:
        tr = leaf.get("time_range", [])
        if tr and len(tr) >= 2:
            results.append({
                "search_type": "hierarchical_highlight",
                "time_range": tr,
                "score": leaf.get("score", 0),
                "node_id": leaf.get("node_id"),
                "keyword_count": leaf.get("keyword_count", 0)
            })
    log.info(f"      Found {len(scored_leaves)} highlight candidates")
    
    return results


def execute_semantic_search(
    segment_tree,
    queries: List[str],
    threshold: float = 0.4,
    top_k: int = 50,
    search_transcriptions: bool = True,
    search_unified: bool = True,
    logger=None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Execute semantic search with adaptive threshold filtering.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        queries: List of semantic queries to search for
        threshold: Initial threshold for candidate selection
        top_k: Maximum number of candidates to retrieve
        search_transcriptions: Whether to search transcriptions
        search_unified: Whether to search unified descriptions
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        List of semantic search results with standardized format
    """
    log = get_log_helper(logger, verbose)
    all_results = []
    
    if not queries:
        return all_results
    
    log.info(f"\n  [SEARCH TYPE: Semantic Search (Visual + Audio)]")
    for sem_query in queries:
        log.info(f"    Query: '{sem_query}'")
        start_time = time.time()
        
        # Call semantic_search once with lowest threshold to get all candidates
        # This avoids redundant embedding computation and similarity calculations
        all_candidates = segment_tree.semantic_search(
            sem_query,
            top_k=top_k,
            threshold=threshold,
            search_transcriptions=search_transcriptions,
            search_unified=search_unified,
            verbose=False
        )
        
        # Adaptive threshold filtering based on result quality
        if all_candidates:
            # Sort by score (highest first)
            all_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Count high-quality results (score >= 0.5)
            high_quality = [r for r in all_candidates if r.get("score", 0) >= 0.5]
            
            # Adaptive selection:
            # - If we have many high-quality results, prefer those
            # - Otherwise, use all results above threshold
            if len(high_quality) >= 5:
                # We have enough high-quality results, use those
                results = high_quality[:15]  # Top 15 high-quality
                log.info(f"      Using {len(results)} high-quality results (score >= 0.5)")
            elif len(high_quality) > 0:
                # Mix of high and medium quality
                results = high_quality + [r for r in all_candidates if 0.45 <= r.get("score", 0) < 0.5][:10]
                results = results[:15]  # Limit to top 15
                log.info(f"      Using {len(results)} results (mix of high and medium quality)")
            else:
                # No high-quality results, use all above threshold
                results = all_candidates[:15]  # Top 15 by score
                log.info(f"      Using {len(results)} results (all above threshold {threshold})")
        else:
            # No results found even with threshold
            results = []
            log.info(f"      No results found (threshold {threshold})")
        
        elapsed = time.time() - start_time
        
        for result in results:
            result["search_type"] = "semantic"
            result["search_query"] = sem_query
            all_results.append(result)
        log.info(f"      Final: {len(results)} matches (took {elapsed:.2f}s, optimized single call)")
    
    return all_results


def execute_object_search(
    segment_tree,
    object_classes: List[str],
    is_negative: bool = False,
    logger=None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Execute object detection search with optional negation.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        object_classes: List of object class names to search for
        is_negative: If True, find seconds WITHOUT the objects (negation)
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        List of object search results with standardized format
    """
    log = get_log_helper(logger, verbose)
    results = []
    
    if not object_classes:
        return results
    
    log.info(f"\n  [SEARCH TYPE: Object Detection (YOLO)]")
    if is_negative:
        log.info(f"    [NEGATION] Inverting search - finding seconds WITHOUT objects")
    
    if is_negative:
        # NEGATIVE QUERY: Find seconds WITHOUT the objects
        # Strategy: Get all seconds, then remove those with object detections
        all_seconds_with_objects = set()
        total_seconds = len(segment_tree.seconds)
        
        # Helper to map time to second index
        def time_to_second_idx(time_seconds: float) -> Optional[int]:
            """Map a time in seconds to the corresponding second index."""
            for idx, second_data in enumerate(segment_tree.seconds):
                # Skip None values that might be in the seconds list
                if second_data is None:
                    continue
                tr = second_data.get("time_range", [])
                if tr and len(tr) >= 2 and tr[0] <= time_seconds <= tr[1]:
                    return idx
            # Fallback: approximate by rounding
            return int(time_seconds) if 0 <= int(time_seconds) < total_seconds else None
        
        for obj_class in object_classes:
            log.info(f"    Class: '{obj_class}' (inverted)")
            search_results = segment_tree.find_objects_by_class(obj_class)
            for result in search_results:
                tr = result.get("time_range", [])
                if tr and len(tr) >= 2:
                    # Map time to second index properly
                    start_idx = time_to_second_idx(tr[0])
                    end_idx = time_to_second_idx(tr[1])
                    # Mark all seconds in the range as having the object
                    if start_idx is not None and end_idx is not None:
                        for idx in range(start_idx, end_idx + 1):
                            if 0 <= idx < total_seconds:
                                all_seconds_with_objects.add(idx)
                    elif start_idx is not None:
                        all_seconds_with_objects.add(start_idx)
        
        # Create results for seconds WITHOUT objects
        for second_idx in range(total_seconds):
            if second_idx not in all_seconds_with_objects:
                second_data = segment_tree.get_second_by_index(second_idx)
                if second_data:
                    tr = second_data.get("time_range", [])
                    if tr and len(tr) >= 2:
                        results.append({
                            "search_type": "object_negated",
                            "object_class": object_classes[0],  # Primary class
                            "time_range": tr,
                            "second": second_idx,
                            "inverted": True
                        })
        
        log.info(f"      Found {total_seconds - len(all_seconds_with_objects)} seconds WITHOUT objects (inverted from {len(all_seconds_with_objects)} with objects)")
    else:
        # POSITIVE QUERY: Normal object detection
        for obj_class in object_classes:
            log.info(f"    Class: '{obj_class}'")
            search_results = segment_tree.find_objects_by_class(obj_class)
            for result in search_results:
                result["search_type"] = "object"
                result["object_class"] = obj_class
                results.append(result)
            log.info(f"      Found {len(search_results)} detections")
    
    return results


def execute_activity_search(
    segment_tree,
    activity_name: str,
    activity_keywords: List[str],
    evidence_keywords: List[str],
    query: str,
    llm,
    validate_activity_evidence_fn,
    logger=None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Execute activity pattern matching search.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        activity_name: Name of the activity to search for
        activity_keywords: Keywords related to the activity
        evidence_keywords: Keywords that provide evidence for the activity
        query: Original user query (for validation)
        llm: Language model instance (for validation)
        validate_activity_evidence_fn: Function to validate activity evidence
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        List of activity search results with standardized format
    """
    log = get_log_helper(logger, verbose)
    results = []
    
    if not activity_name and not activity_keywords:
        return results
    
    log.info(f"\n  [SEARCH TYPE: Activity Pattern Matching]")
    query_lower = query.lower()
    
    # Specialized handling for fish catching
    if "fish" in query_lower and ("catch" in query_lower or "caught" in query_lower):
        log.info(f"    Activity: Fish catching (specialized)")
        result = segment_tree.check_fish_caught()
        result["search_type"] = "activity"
        # Add metadata for validation
        result["activity_name"] = "fishing"
        result["evidence_name"] = "fish caught"
        # Validate evidence descriptions to filter false positives
        log.info(f"      Evidence scenes before validation: {result.get('fish_holding_count', 0)}")
        validated_result = validate_activity_evidence_fn(query, result, llm, verbose=verbose)
        if validated_result is None:
            validated_result = result  # Fallback to original if validation returns None
        # Update fish-specific fields after validation
        validated_result["fish_holding_count"] = validated_result.get("evidence_count", 0)
        validated_result["fish_caught"] = validated_result.get("detected", False)
        results.append(validated_result)
        log.info(f"      Evidence scenes after validation: {validated_result.get('fish_holding_count', 0)}")
    elif activity_keywords:
        log.info(f"    Activity: {activity_name or 'general'}")
        result = segment_tree.check_activity(
            activity_keywords=activity_keywords,
            evidence_keywords=evidence_keywords or activity_keywords,
            activity_name=activity_name or "activity"
        )
        result["search_type"] = "activity"
        # Add metadata for validation
        result["activity_name"] = activity_name or "activity"
        result["evidence_name"] = activity_name or "activity"
        # Validate evidence descriptions to filter false positives
        log.info(f"      Evidence scenes before validation: {result.get('evidence_count', 0)}")
        validated_result = validate_activity_evidence_fn(query, result, llm, verbose=verbose)
        if validated_result is None:
            validated_result = result  # Fallback to original if validation returns None
        results.append(validated_result)
        log.info(f"      Evidence scenes after validation: {validated_result.get('evidence_count', 0)}")
    
    return results

