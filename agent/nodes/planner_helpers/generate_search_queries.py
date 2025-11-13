"""Generate search queries and perform semantic analysis in parallel."""

import re
import time
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from agent.utils.logging_utils import get_log_helper
from agent.utils.llm_utils import parse_json_response
from agent.utils.search.query_analysis import (
    analyze_query_semantics,
    plan_search_strategy,
    validate_and_adjust_intent
)
from agent.utils.search.query_builder import build_search_query_message
from agent.prompts.planner_prompts import (
    PLANNER_SYSTEM_PROMPT,
    SEARCH_QUERY_GENERATION_PROMPT
)


def generate_search_queries(
    query: str,
    content_inspection: Optional[dict],
    video_narrative: Optional[dict],
    llm,
    clip_contexts: Optional[List[Dict[str, Any]]] = None,
    logger=None,
    verbose: bool = False
) -> tuple[dict, dict, dict]:
    """
    Generate search queries and perform semantic analysis in parallel.
    
    Args:
        query: User query string
        content_inspection: Optional content inspection data
        video_narrative: Optional video narrative data
        llm: Language model instance
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (search_plan, semantic_analysis, query_intent)
    """
    log = get_log_helper(logger, verbose)
    
    # PARALLEL LLM CALLS: Query generation and semantic analysis can run in parallel
    log.info("\n[PARALLEL LLM] Starting parallel query generation and semantic analysis...")
    overall_start = time.time()
    
    # Helper function for query generation
    def generate_search_queries():
        """Generate search queries for all modalities."""
        system_prompt_step1 = PLANNER_SYSTEM_PROMPT + "\n\n" + SEARCH_QUERY_GENERATION_PROMPT
        
        # Build user message using template pattern
        user_message_content = build_search_query_message(
            query=query,
            content_inspection=content_inspection,
            video_narrative=video_narrative,
            clip_contexts=clip_contexts
        )
        
        messages_step1 = [
            SystemMessage(content=system_prompt_step1),
            HumanMessage(content=user_message_content)
        ]
        
        response = llm.invoke(messages_step1)
        response_text = response.content.strip()
        
        if logger:
            logger.debug(f"[LLM RESPONSE - Query Gen] {response_text[:300]}...")
        elif verbose:
            print(f"[LLM RESPONSE - Query Gen] {response_text[:300]}...")
        
        # Extract and parse JSON from response
        fallback_search_plan = {
            "semantic_queries": [query],
            "hierarchical_keywords": [w for w in re.findall(r'\b\w+\b', query.lower()) if len(w) >= 3][:5],
            "object_classes": [],
            "activity_name": "",
            "activity_keywords": [],
            "evidence_keywords": [],
            "is_general_highlight_query": "highlight" in query.lower(),
            "needs_clarification": False
        }
        return parse_json_response(response_text, fallback=fallback_search_plan)
    
    # Helper function for semantic analysis
    def analyze_semantics():
        """Analyze query semantics."""
        result = analyze_query_semantics(query, llm)
        # Ensure semantic_analysis is not None
        if result is None:
            result = {
                "query_type": "POSITIVE",
                "search_intent": "hybrid",
                "target_entities": {},
                "special_handling": {},
                "reasoning": "Default analysis (LLM returned None)"
            }
        return result
    
    # Execute both in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_query_gen = executor.submit(generate_search_queries)
        future_semantic = executor.submit(analyze_semantics)
        
        # Wait for both to complete
        search_plan = future_query_gen.result()
        semantic_analysis = future_semantic.result()
    
    elapsed_parallel = time.time() - overall_start
    log.info(f"[PARALLEL LLM] Both calls completed in {elapsed_parallel:.2f}s (parallel execution)")
    
    log.info("\n[SEARCH PLAN] Generated search queries:")
    if content_inspection:
        log.info(f"  [INSPECTION] Used segment tree inspection: Yes")
    log.info(f"  Semantic queries: {search_plan.get('semantic_queries', [])}")
    log.info(f"  Hierarchical keywords: {search_plan.get('hierarchical_keywords', [])}")
    log.info(f"  Object classes: {search_plan.get('object_classes', [])}")
    if clip_contexts:
        log.info(f"  Clip contexts provided: {len(clip_contexts)}")
    log.info(f"  Activity: {search_plan.get('activity_name', 'N/A')}")
    log.info(f"  Is general highlight query: {search_plan.get('is_general_highlight_query', False)}")
    
    log.info("\n[AGENTIC] Phase 1: Semantic Query Analysis (completed in parallel):")
    log.info(f"  Query type: {semantic_analysis.get('query_type', 'POSITIVE')}")
    log.info(f"  Search intent: {semantic_analysis.get('search_intent', 'hybrid')}")
    log.info(f"  Target entities: {semantic_analysis.get('target_entities', {})}")
    log.info(f"  Special handling: {semantic_analysis.get('special_handling', {})}")
    log.info(f"  Reasoning: {semantic_analysis.get('reasoning', 'N/A')}")
    
    # AGENTIC PHASE 2: Dynamic Strategy Planning
    log.info("\n[AGENTIC] Phase 2: Dynamic Strategy Planning...")
    start_time = time.time()
    strategy = plan_search_strategy(semantic_analysis, llm, verbose=verbose)
    elapsed = time.time() - start_time
    log.info(f"  Strategy operations: {len(strategy.get('search_operations', []))}")
    log.info(f"  Post-processing: {strategy.get('post_processing', [])}")
    log.info(f"  Reasoning: {strategy.get('reasoning', 'N/A')}")
    log.info(f"  Strategy planning completed in {elapsed:.2f}s")
    
    # Convert strategy to query_intent format for backward compatibility
    query_intent = {
        "is_object_centric": semantic_analysis.get("search_intent") == "object",
        "mentioned_classes": semantic_analysis.get("target_entities", {}).get("objects", []),
        "primary_intent": semantic_analysis.get("search_intent", "hybrid"),
        "object_priority": semantic_analysis.get("object_priority", {}),
        "needs_semantic": semantic_analysis.get("modalities", {}).get("semantic_search", True),
        "needs_activity": semantic_analysis.get("modalities", {}).get("activity_detection", False),
        "needs_hierarchical": semantic_analysis.get("modalities", {}).get("hierarchical_search", True),
        "confidence": semantic_analysis.get("confidence", 0.7),
        "_semantic_analysis": semantic_analysis,
        "_strategy": strategy
    }
    
    # Cross-validate with search plan (keep for compatibility)
    log.info("\n[VALIDATION] Cross-validating with search plan...")
    query_intent = validate_and_adjust_intent(query_intent, search_plan, verbose=verbose)
    
    return search_plan, semantic_analysis, query_intent

