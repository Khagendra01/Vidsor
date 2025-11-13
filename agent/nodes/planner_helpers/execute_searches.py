"""Execute all search types and collect results."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    log.info("[PARALLEL SEARCH] Running all search types in parallel for better performance...")
    
    all_search_results = []
    query_lower = query.lower()
    search_start_time = time.time()
    
    # Check if this is a general highlight query
    is_general_highlight = search_plan.get("is_general_highlight_query", False)
    if "highlight" in query_lower and ("all" in query_lower or "find" in query_lower):
        is_general_highlight = True
    
    # Prepare search parameters
    hierarchical_keywords = search_plan.get("hierarchical_keywords", [])
    semantic_queries = search_plan.get("semantic_queries", [query])
    object_classes = search_plan.get("object_classes", [])
    semantic_objects = semantic_analysis.get("target_entities", {}).get("objects", [])
    all_object_classes = list(set(object_classes + semantic_objects))
    is_negative = semantic_analysis.get("query_type") == "NEGATIVE" or semantic_analysis.get("special_handling", {}).get("negation", False)
    activity_name = search_plan.get("activity_name", "")
    activity_keywords = search_plan.get("activity_keywords", [])
    evidence_keywords = search_plan.get("evidence_keywords", [])
    
    # CRITICAL FIX: Execute searches in parallel using ThreadPoolExecutor
    # This significantly improves performance by running independent searches concurrently
    search_tasks = []
    
    # Task 0: General highlight detection (if needed)
    if is_general_highlight:
        search_tasks.append((
            "highlight",
            lambda: execute_hierarchical_highlight_search(
                segment_tree,
                action_keywords=hierarchical_keywords if hierarchical_keywords else None,
                max_results=20,
                logger=logger,
                verbose=verbose
            )
        ))
    
    # Task 1: Hierarchical search
    if hierarchical_keywords:
        search_tasks.append((
            "hierarchical",
            lambda: execute_hierarchical_search(
                segment_tree,
                keywords=hierarchical_keywords,
                match_mode="any",
                max_results=20,
                logger=logger,
                verbose=verbose
            )
        ))
    
    # Task 2: Semantic search
    if semantic_queries:
        search_tasks.append((
            "semantic",
            lambda: execute_semantic_search(
                segment_tree,
                queries=semantic_queries,
                threshold=0.4,
                top_k=50,
                search_transcriptions=True,
                search_unified=True,
                logger=logger,
                verbose=verbose
            )
        ))
    
    # Task 3: Object search
    if all_object_classes:
        search_tasks.append((
            "object",
            lambda: execute_object_search(
                segment_tree,
                object_classes=all_object_classes,
                is_negative=is_negative,
                logger=logger,
                verbose=verbose
            )
        ))
    
    # Task 4: Activity search
    if activity_name or activity_keywords:
        search_tasks.append((
            "activity",
            lambda: execute_activity_search(
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
        ))
    
    # Execute all searches in parallel
    if search_tasks:
        with ThreadPoolExecutor(max_workers=len(search_tasks)) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(task_func): task_name 
                for task_name, task_func in search_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    results = future.result()
                    all_search_results.extend(results)
                    if verbose:
                        log.info(f"  ✓ {task_name} search completed: {len(results)} results")
                except Exception as e:
                    log.warning(f"  ✗ {task_name} search failed: {e}")
                    if verbose:
                        import traceback
                        log.warning(traceback.format_exc())
    
    search_elapsed = time.time() - search_start_time
    log.info(f"\n[PARALLEL SEARCH] All searches completed in {search_elapsed:.2f}s (parallel execution)")
    log.info(f"[AGGREGATION] Collected results from all search types:")
    log.info(f"  Total results: {len(all_search_results)}")
    
    return all_search_results

