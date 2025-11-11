"""Execute all search types and collect results."""

from agent.utils.logging_utils import get_log_helper
from agent.utils.processing.refinement import validate_activity_evidence
from agent.utils.search.search_executor import (
    execute_hierarchical_search,
    execute_hierarchical_highlight_search,
    execute_semantic_search,
    execute_object_search,
    execute_activity_search
)


def execute_searches(
    segment_tree,
    search_plan: dict,
    semantic_analysis: dict,
    query: str,
    llm,
    logger=None,
    verbose: bool = False
) -> list:
    """
    Execute all search types and collect results.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        search_plan: Search plan dictionary
        semantic_analysis: Semantic analysis dictionary
        query: User query string
        llm: Language model instance
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        List of all search results
    """
    log = get_log_helper(logger, verbose)
    
    log.info("\n[STEP 2] Executing search with hierarchical tree + semantic search...")
    
    all_search_results = []
    query_lower = query.lower()
    
    # Check if this is a general highlight query
    is_general_highlight = search_plan.get("is_general_highlight_query", False)
    if "highlight" in query_lower and ("all" in query_lower or "find" in query_lower):
        is_general_highlight = True
    
    # Execute all search types using search executor module
    # 0. General highlight detection (using hierarchical tree)
    if is_general_highlight:
        hierarchical_keywords = search_plan.get("hierarchical_keywords", [])
        highlight_results = execute_hierarchical_highlight_search(
            segment_tree,
            action_keywords=hierarchical_keywords if hierarchical_keywords else None,
            max_results=20,
            logger=logger,
            verbose=verbose
        )
        all_search_results.extend(highlight_results)
    
    # 1. Hierarchical tree keyword search (fast pre-filter)
    hierarchical_keywords = search_plan.get("hierarchical_keywords", [])
    if hierarchical_keywords:
        hierarchical_results = execute_hierarchical_search(
            segment_tree,
            keywords=hierarchical_keywords,
            match_mode="any",
            max_results=20,
            logger=logger,
            verbose=verbose
        )
        all_search_results.extend(hierarchical_results)
    
    # 2. Semantic search (replaces old visual/audio keyword search)
    semantic_queries = search_plan.get("semantic_queries", [query])
    if semantic_queries:
        semantic_results = execute_semantic_search(
            segment_tree,
            queries=semantic_queries,
            threshold=0.4,
            top_k=50,
            search_transcriptions=True,
            search_unified=True,
            logger=logger,
            verbose=verbose
        )
        all_search_results.extend(semantic_results)
    
    # 3. Object search (with negation handling)
    object_classes = search_plan.get("object_classes", [])
    # Also check semantic analysis for object classes
    semantic_objects = semantic_analysis.get("target_entities", {}).get("objects", [])
    all_object_classes = list(set(object_classes + semantic_objects))
    
    # Check if this is a negative query
    is_negative = semantic_analysis.get("query_type") == "NEGATIVE" or semantic_analysis.get("special_handling", {}).get("negation", False)
    
    if all_object_classes:
        object_results = execute_object_search(
            segment_tree,
            object_classes=all_object_classes,
            is_negative=is_negative,
            logger=logger,
            verbose=verbose
        )
        all_search_results.extend(object_results)
    
    # 4. Activity search (pattern matching)
    activity_name = search_plan.get("activity_name", "")
    activity_keywords = search_plan.get("activity_keywords", [])
    evidence_keywords = search_plan.get("evidence_keywords", [])
    if activity_name or activity_keywords:
        activity_results = execute_activity_search(
            segment_tree,
            activity_name=activity_name,
            activity_keywords=activity_keywords,
            evidence_keywords=evidence_keywords,
            query=query,
            llm=llm,
            validate_activity_evidence_fn=validate_activity_evidence,
            logger=logger,
            verbose=verbose
        )
        all_search_results.extend(activity_results)
    
    log.info(f"\n[AGGREGATION] Collected results from all search types:")
    log.info(f"  Total results: {len(all_search_results)}")
    
    return all_search_results

