"""Planner agent for analyzing queries and retrieving relevant moments."""

from agent.state import AgentState
from agent.utils.processing.feature_extractor import PerSecondFeatureExtractor
from agent.utils.search.query_analysis import configure_weights
from agent.utils.processing.refinement import decide_refine_or_research, refine_existing_results
from agent.utils.utils import merge_time_ranges
from agent.utils.llm_utils import create_llm
from agent.utils.logging_utils import get_log_helper
from agent.utils.weight_config import configure_search_weights
from agent.nodes.planner_helpers import (
    inspect_segment_tree,
    generate_search_queries,
    execute_searches,
    score_and_filter,
    select_best_results
)


def create_planner_agent(model_name: str = "gpt-4o-mini"):
    """Create the planner agent that analyzes queries and retrieves relevant moments."""
    
    # Use shared LLM creation utility
    llm = create_llm(model_name)
    
    def planner_node(state: AgentState) -> AgentState:
        """Planner agent: Analyzes user query and retrieves relevant moments using ALL search types."""
        query = state["user_query"]
        segment_tree = state["segment_tree"]
        verbose = state.get("verbose", False)
        logger = state.get("logger")
        
        # Use shared logging helper
        log_info = get_log_helper(logger, verbose)
        
        # AGENTIC DECISION: Check if we should refine existing results or do new search
        if state.get("previous_time_ranges") and state.get("previous_query"):
            log_info("\n" + "=" * 60)
            log_info("PLANNER AGENT: Context-Aware Decision")
            log_info("=" * 60)
            log_info(f"Previous query: {state.get('previous_query')}")
            log_info(f"Previous results: {len(state.get('previous_time_ranges', []))} time ranges")
            log_info(f"Current query: {query}")
            log_info("\n[DECISION] Analyzing user intent...")
            
            decision = decide_refine_or_research(state, llm)
            
            log_info(f"  Decision: {decision.get('action')}")
            log_info(f"  Reason: {decision.get('reason')}")
            if decision.get('extract_number'):
                log_info(f"  Extract number: {decision.get('extract_number')}")
            
            if decision.get("action") == "REFINE":
                # Refine existing results
                return refine_existing_results(state, decision, verbose=verbose)
            # Otherwise, continue with new search (fall through)
        
        # Initialize variables
        search_results = []
        time_ranges = []
        confidence = 0.5
        needs_clarification = False
        clarification_question = None
        filtered_seconds = []  # For preserving scored seconds
        
        log_info("\n" + "=" * 60)
        log_info("PLANNER AGENT: Multi-Modal Search Strategy")
        log_info("=" * 60)
        log_info(f"Query: {query}")
        
        # Check for temporal constraints
        temporal_constraint = state.get("temporal_constraint")
        temporal_type = state.get("temporal_type")
        if temporal_constraint:
            log_info(f"Temporal constraint: '{temporal_constraint}' (type: {temporal_type})")
            log_info("  → Query includes temporal/conditional constraint - will search with full context")
        
        # STEP 0: Inspect segment tree and create video narrative
        content_inspection, video_narrative = inspect_segment_tree(
            segment_tree, query, llm, logger=logger, verbose=verbose
        )
        
        # STEP 1: Generate search queries and perform semantic analysis
        search_plan, semantic_analysis, query_intent = generate_search_queries(
            query, content_inspection, video_narrative, llm, logger=logger, verbose=verbose
        )
        
        # Store semantic_analysis and strategy in state for later use
        strategy = query_intent.get("_strategy", {})
        
        # NEW: Initialize feature extractor
        feature_extractor = PerSecondFeatureExtractor(segment_tree)
        
        # NEW: Get all object classes and configure weights (use strategy scoring if available)
        all_object_classes = set(segment_tree.get_all_classes().keys())
        
        # Use shared weight configuration utility
        weights = configure_search_weights(
            strategy=strategy,
            query_intent=query_intent,
            all_object_classes=all_object_classes,
            search_plan=search_plan,
            configure_weights_fn=configure_weights,
            verbose=verbose,
            log_info=log_info,
            logger=logger
        )
        
        log_info(f"\n[WEIGHT CONFIGURATION]")
        log_info(f"  Semantic weight: {weights['semantic_weight']:.2f}")
        log_info(f"  Activity weight: {weights['activity_weight']:.2f}")
        log_info(f"  Hierarchical weight: {weights['hierarchical_weight']:.2f}")
        log_info(f"  Threshold: {weights['threshold']:.2f}")
        high_priority_objects = {k: v for k, v in weights['object_weights'].items() if v > 0.3}
        if high_priority_objects:
            log_info(f"  High priority objects: {high_priority_objects}")
        
        # Check for clarification needs from search plan
        needs_clarification = search_plan.get("needs_clarification", False)
        clarification_question = search_plan.get("clarification_question")
        
        # Check for audio query ambiguity
        query_lower = query.lower()
        audio_indicators = ["say", "said", "talk", "speak", "mention", "discuss", "conversation", "audio", "transcription"]
        is_audio_query = any(indicator in query_lower for indicator in audio_indicators)
        ambiguous_pronouns = ["they", "he", "she", "it", "that", "this", "those", "these"]
        has_ambiguous_refs = any(pronoun in query_lower.split() for pronoun in ambiguous_pronouns)
        
        if is_audio_query and has_ambiguous_refs and not needs_clarification:
            needs_clarification = True
            clarification_question = "I notice your query mentions audio/transcription but uses ambiguous references. Could you clarify who or what you're looking for?"
            log_info("[WARNING] Ambiguous audio query detected - will ask for clarification")
        
        if needs_clarification and not is_audio_query:
            log_info("[INFO] LLM suggested clarification, but query seems clear enough. Will try searching first.")
            needs_clarification = False
        
        if not needs_clarification:
            # STEP 2: Execute all search types
            all_search_results = execute_searches(
                segment_tree, search_plan, semantic_analysis, query, llm, logger=logger, verbose=verbose
            )
            
            # STEP 3: Score and filter results
            filtered_seconds, adaptive_threshold = score_and_filter(
                segment_tree,
                feature_extractor,
                all_search_results,
                query_intent,
                weights,
                semantic_analysis,
                logger=logger,
                verbose=verbose
            )
            
            # Store semantic_analysis in state for select_best_results
            state_with_analysis = {**state, "_semantic_analysis": semantic_analysis, "video_narrative": video_narrative}
            
            # STEP 4: Select best results and handle clarification
            time_ranges, confidence, needs_clarification, clarification_question = select_best_results(
                filtered_seconds,
                all_search_results,
                query,
                llm,
                state_with_analysis,
                adaptive_threshold,
                logger=logger,
                verbose=verbose
            )
            
            # Store all search results
            search_results = all_search_results
            
        # Merge overlapping time ranges
        if time_ranges:
            original_count = len(time_ranges)
            log_info(f"\n[MERGING] Merging {original_count} time ranges...")
            log_info(f"  Before merge: {original_count} ranges")
            time_ranges = merge_time_ranges(time_ranges)
            log_info(f"  After merge: {len(time_ranges)} non-overlapping ranges")
            log_info(f"  Merged ranges with 2.0s padding:")
            for i, (start, end) in enumerate(time_ranges, 1):
                log_info(f"    Range {i}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        else:
            log_info("\n[MERGING] No time ranges to merge")
        
        # Preserve state for potential refinement (preserve even if clarification needed)
        # This allows user to refine results after clarification
        preserved_scored_seconds = filtered_seconds if filtered_seconds else state.get("previous_scored_seconds")
        
        return {
            **state,
            "query_type": "multi_modal",  # Indicate we used all search types
            "search_results": search_results,
            "time_ranges": time_ranges,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            # Preserve for future refinement (always preserve if we have new results)
            "previous_time_ranges": time_ranges if time_ranges else state.get("previous_time_ranges"),
            "previous_scored_seconds": preserved_scored_seconds,
            "previous_query": query if time_ranges else state.get("previous_query"),
            "previous_search_results": search_results if search_results else state.get("previous_search_results")
        }
    
    return planner_node

