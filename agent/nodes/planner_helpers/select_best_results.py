"""Select best results, handle reranking, and determine if clarification is needed."""

import re
import time
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState
from agent.utils.logging_utils import get_log_helper
from agent.utils.llm_utils import parse_json_response
from agent.utils.processing.refinement import validate_search_results
from agent.prompts.planner_prompts import CLARIFICATION_DECISION_PROMPT


def select_best_results(
    filtered_seconds: list,
    search_results: list,
    query: str,
    llm,
    state: AgentState,
    adaptive_threshold: float,
    logger=None,
    verbose: bool = False
) -> tuple[list, float, bool, Optional[str]]:
    """
    Select best results, handle reranking, and determine if clarification is needed.
    
    Args:
        filtered_seconds: List of filtered and scored seconds
        search_results: List of all search results
        query: User query string
        llm: Language model instance
        state: Agent state dictionary
        adaptive_threshold: Adaptive threshold value
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (time_ranges, confidence, needs_clarification, clarification_question)
    """
    log = get_log_helper(logger, verbose)
    
    # AGENTIC PHASE 3: Self-Validation
    log.info(f"\n[AGENTIC] Phase 3: Self-Validation...")
    
    semantic_analysis = state.get("_semantic_analysis", {})
    validation_result = validate_search_results(
        query=query,
        semantic_analysis=semantic_analysis,
        filtered_seconds=filtered_seconds[:20] if filtered_seconds else [],
        llm=llm,
        verbose=verbose
    )
    
    if validation_result.get("needs_refinement", False):
        log.info(f"  [VALIDATION] Issues detected: {validation_result.get('issues', [])}")
        log.info(f"  [VALIDATION] Suggestions: {validation_result.get('suggestions', [])}")
    
    # STEP 4: Select best seconds with semantic prioritization
    log.info(f"\n[STEP 4] Selecting best highlights...")
    
    from agent.utils.search.selection import select_best_of
    
    # Select best seconds with semantic prioritization
    best_seconds = select_best_of(
        filtered_seconds,
        top_k=None,  # No limit, will be handled by orchestrator duration constraints
        min_score=adaptive_threshold,
        prioritize_semantic=True,
        verbose=verbose
    )
    
    log.info(f"  Selected {len(best_seconds)} best seconds (from {len(filtered_seconds)} above threshold)")
    if best_seconds:
        log.info(f"  Top 10 selected seconds:")
        for i, sec in enumerate(best_seconds[:10], 1):
            tr = sec.get("time_range", [])
            time_str = f"{tr[0]:.1f}s" if tr else "N/A"
            log.info(f"    {i}. Second {sec['second']} ({time_str}): score={sec['score']:.3f} "
                      f"(sem={sec['semantic_score']:.2f})")
    
    # Convert seconds to time ranges
    time_ranges = []
    for sec in best_seconds:
        tr = sec.get("time_range", [])
        if tr and len(tr) >= 2:
            time_ranges.append((tr[0], tr[1]))
    
    # Determine if user wants one or multiple results
    query_lower = query.lower()
    
    # Indicators for multiple results (strong signals)
    multiple_indicators = [
        "all", "every", "each", "multiple", "several",
        "moments", "instances", "scenes", "clips", "times"
    ]
    has_multiple_indicator = any(indicator in query_lower for indicator in multiple_indicators)
    
    # Indicators for single result (strong signals)
    single_indicators = [
        "the moment", "the best", "the first", "the one",
        "a moment", "one moment", "single moment"
    ]
    has_single_indicator = any(indicator in query_lower for indicator in single_indicators)
    
    # Check for plural vs singular "moment"
    has_plural_moment = "moments" in query_lower
    has_singular_moment = "moment" in query_lower and "moments" not in query_lower
    
    # Determine intent
    if has_multiple_indicator:
        user_wants_one = False
        reason = "multiple indicator detected"
    elif has_single_indicator:
        user_wants_one = True
        reason = "single indicator detected"
    elif has_plural_moment:
        user_wants_one = False
        reason = "plural 'moments' detected"
    elif has_singular_moment:
        user_wants_one = True
        reason = "singular 'moment' detected"
    else:
        # Default: if we found many results, assume user wants multiple
        user_wants_one = len(time_ranges) <= 3
        reason = f"default (found {len(time_ranges)} ranges)"
    
    log.info(f"\n[SELECTION] User intent detection:")
    log.info(f"  Multiple indicators: {has_multiple_indicator}")
    log.info(f"  Single indicators: {has_single_indicator}")
    log.info(f"  Plural 'moments': {has_plural_moment}")
    log.info(f"  Singular 'moment': {has_singular_moment}")
    log.info(f"  Decision: {'ONE result' if user_wants_one else 'MULTIPLE results'} ({reason})")
    
    if user_wants_one and time_ranges:
        time_ranges = [time_ranges[0]]
        log.info(f"  [ACTION] Selecting top match only")
    elif not user_wants_one:
        log.info(f"  [ACTION] Returning all {len(time_ranges)} time ranges")
    
    # Calculate confidence based on scores
    if filtered_seconds:
        top_score = filtered_seconds[0]["score"]
        confidence = min(0.95, max(0.3, top_score))
    else:
        confidence = 0.3
    
    # AUTOMATIC RERANKING AND FILTERING: Check if user specified "top N" or "all"
    top_n_match = re.search(r'(?:proceed\s+)?(?:top|first|best|select)\s+(\d+)', query_lower)
    extract_number = None
    if top_n_match:
        extract_number = int(top_n_match.group(1))
        log.info(f"\n[AUTO-RERANK] Detected 'top {extract_number}' in query - will automatically rerank and filter")
    elif "all" in query_lower or "every" in query_lower:
        extract_number = None
        log.info(f"\n[AUTO-RERANK] Detected 'all/every' in query - will automatically rerank and filter false positives")
    
    # If we have results and user specified a number or "all", automatically rerank and filter
    if time_ranges and (extract_number is not None or "all" in query_lower or "every" in query_lower):
        log.info(f"\n[AUTO-RERANK] Automatically reranking and filtering {len(time_ranges)} results...")
        
        # Map time ranges to their descriptions from search_results
        range_descriptions = {}
        for result in search_results:
            if result is None:
                continue
            result_tr = result.get("time_range", [])
            if result_tr and len(result_tr) >= 2:
                tr_key = (result_tr[0], result_tr[1])
                desc = result.get("unified_description") or result.get("description") or result.get("text", "")
                if desc:
                    if tr_key not in range_descriptions:
                        range_descriptions[tr_key] = []
                    range_descriptions[tr_key].append(desc)
        
        # Match time ranges with their descriptions
        ranges_with_descriptions = []
        for tr in time_ranges:
            tr_key = (tr[0], tr[1])
            descriptions = range_descriptions.get(tr_key, [])
            combined_desc = " | ".join(descriptions[:2]) if descriptions else f"Time range {tr[0]:.1f}s-{tr[1]:.1f}s"
            ranges_with_descriptions.append({
                "time_range": tr,
                "description": combined_desc
            })
        
        # Use LLM to rerank and filter
        if len(ranges_with_descriptions) > 1:
            from agent.utils.processing.refinement import rank_ranges_with_llm
            
            top_n_for_rerank = extract_number if extract_number else len(ranges_with_descriptions)
            
            try:
                ranked_ranges = rank_ranges_with_llm(
                    ranges_with_descriptions,
                    query,
                    query,
                    top_n_for_rerank,
                    logger,
                    verbose=verbose
                )
                
                reranked_time_ranges = [r["time_range"] for r in ranked_ranges]
                
                # Filter out false positives
                log.info(f"  [AUTO-FILTER] Validating {len(reranked_time_ranges)} results to remove false positives...")
                validated_ranges = []
                
                video_narrative = state.get("video_narrative", {})
                narrative_theme = video_narrative.get("theme", "")
                
                for r in ranked_ranges:
                    tr = r["time_range"]
                    desc = r["description"]
                    
                    validation_prompt = f"""You are validating video search results. Be LENIENT - only filter out OBVIOUS false positives.

Video context: {narrative_theme or "General video content"}
User query: "{query}"
Clip time: {tr[0]:.1f}s - {tr[1]:.1f}s
Clip description: "{desc[:300]}"

Instructions:
- If the clip is REMOTELY related to the query, mark it as VALID
- Only filter if it's CLEARLY unrelated (e.g., completely different topic)
- When in doubt, KEEP the result
- For "find highlights" queries, be very lenient - most results should be valid

Return JSON only:
{{"is_valid": true/false, "reasoning": "brief explanation"}}"""
                    
                    try:
                        validation_response = llm.invoke([
                            SystemMessage(content="You are a lenient video search validator. Only filter out OBVIOUS false positives. When unsure, keep the result. Be especially lenient for highlight queries."),
                            HumanMessage(content=validation_prompt)
                        ])
                        
                        validation_text = validation_response.content.strip()
                        validation_result = parse_json_response(validation_text, fallback={"is_valid": True, "reasoning": ""})
                        
                        is_valid = validation_result.get("is_valid", True)
                        reasoning = validation_result.get("reasoning", "")
                        
                        # Extra safety: if reasoning suggests uncertainty, keep it
                        if not is_valid and any(word in reasoning.lower() for word in ["might", "could", "possibly", "maybe", "uncertain"]):
                            is_valid = True
                        
                        if is_valid:
                            validated_ranges.append(tr)
                        else:
                            log.info(f"    Filtered out {tr[0]:.1f}s-{tr[1]:.1f}s: {reasoning[:80]}")
                    except Exception as e:
                        validated_ranges.append(tr)
                        if verbose:
                            log.info(f"    Validation error for {tr[0]:.1f}s-{tr[1]:.1f}s, keeping result")
                
                if validated_ranges:
                    time_ranges = validated_ranges
                    log.info(f"  [AUTO-RERANK] Reranked and filtered to {len(time_ranges)} valid results")
                else:
                    log.info(f"  [AUTO-RERANK] Warning: All results filtered out, keeping original top results")
                    time_ranges = reranked_time_ranges[:min(extract_number or 10, len(reranked_time_ranges))]
            except Exception as e:
                log.info(f"  [AUTO-RERANK] Error during reranking: {e}, using original results")
                if extract_number and len(time_ranges) > extract_number:
                    time_ranges = time_ranges[:extract_number]
    
    log.info(f"\n[FINAL SELECTION]")
    log.info(f"  Selected {len(time_ranges)} time range(s)")
    log.info(f"  Confidence: {confidence:.2f}")
    if time_ranges:
        log.info(f"  Time ranges:")
        for i, (start, end) in enumerate(time_ranges, 1):
            log.info(f"    {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    # Determine if clarification is needed
    needs_clarification = False
    clarification_question = None
    
    if not time_ranges:
        confidence = 0.3
        needs_clarification = True
        clarification_question = "No relevant results found. Could you rephrase your query or provide more details?"
        log.info("  [WARNING] No results found - will ask for clarification")
    elif len(time_ranges) > 15:
        # Use LLM to decide if clarification is needed
        log.info(f"\n[CLARIFICATION DECISION] Found {len(time_ranges)} results - asking LLM if clarification needed...")
        
        clarification_prompt = CLARIFICATION_DECISION_PROMPT.format(
            query=query,
            result_count=len(time_ranges)
        )
        
        try:
            start_time = time.time()
            clarification_response = llm.invoke([
                SystemMessage(content="You are a helpful assistant that determines if users need clarification for video search queries."),
                HumanMessage(content=clarification_prompt)
            ])
            elapsed = time.time() - start_time
            
            clarification_text = clarification_response.content.strip()
            fallback_clarification = {
                "needs_clarification": True,
                "clarification_question": f"Found {len(time_ranges)} potential moments. Could you narrow down what you're looking for?",
                "reasoning": "Fallback due to parsing error"
            }
            clarification_decision = parse_json_response(clarification_text, fallback=fallback_clarification)
            needs_clarification = clarification_decision.get("needs_clarification", False)
            clarification_question = clarification_decision.get("clarification_question", f"Found {len(time_ranges)} potential moments. Could you narrow down what you're looking for?")
            reasoning = clarification_decision.get("reasoning", "")
            
            log.info(f"  LLM Decision: {'Needs clarification' if needs_clarification else 'Proceed with all results'} (took {elapsed:.2f}s)")
            log.info(f"  Reasoning: {reasoning}")
            
            if needs_clarification:
                confidence = 0.6
                log.info(f"  [WARNING] Too many results ({len(time_ranges)}) - will ask for clarification")
            else:
                confidence = min(0.95, max(0.6, confidence))
                log.info(f"  [SUCCESS] User wants all results - proceeding with {len(time_ranges)} time range(s)")
        except Exception as e:
            log.info(f"  [FALLBACK] LLM clarification decision failed: {e}, using default logic")
            confidence = 0.6
            needs_clarification = True
            clarification_question = f"Found {len(time_ranges)} potential moments. Could you narrow down what you're looking for?"
            log.info(f"  [WARNING] Too many results ({len(time_ranges)}) - will ask for clarification")
    else:
        log.info(f"  [SUCCESS] Found {len(time_ranges)} time range(s) - proceeding with extraction")
    
    return time_ranges, confidence, needs_clarification, clarification_question

