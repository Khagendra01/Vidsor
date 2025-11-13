"""Orchestrator agent for managing timeline editing operations."""

from typing import Dict, Any
from agent.state import OrchestratorState
from agent.timeline_manager import TimelineManager
from agent.orchestrator_operations import classify_operation, validate_operation_params
from agent.orchestrator_handlers import (
    handle_find_highlights,
    handle_cut,
    handle_replace,
    handle_insert,
    handle_find_broll,
    handle_trim,
    handle_apply_effect
)
from agent.nodes.planner import create_planner_agent
from agent.utils.llm_utils import create_llm


def create_orchestrator_agent(model_name: str = "gpt-4o-mini"):
    """
    Create the orchestrator agent that manages timeline editing operations.
    
    Args:
        model_name: LLM model name to use
        
    Returns:
        Orchestrator node function
    """
    
    # Initialize LLM using shared utility
    llm = create_llm(model_name)
    
    def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
        """
        Orchestrator agent: Manages timeline editing operations.
        
        Step 2: Operation classification and parameter extraction.
        """
        query = state.get("user_query", "")
        timeline_path = state.get("timeline_path")
        verbose = state.get("verbose", False)
        logger = state.get("logger")
        
        # Use logger if available, otherwise use print
        log = logger if logger else None
        
        if log:
            log.info("\n" + "=" * 80)
            log.info("ORCHESTRATOR AGENT: Timeline Management")
            log.info("=" * 80)
            log.info(f"Query: {query}")
            log.info(f"Timeline path: {timeline_path}")
        elif verbose:
            print("\n" + "=" * 60)
            print("ORCHESTRATOR AGENT: Timeline Management")
            print("=" * 60)
            print(f"Query: {query}")
            print(f"Timeline path: {timeline_path}")
        
        # Initialize timeline manager if path is provided
        timeline_manager = None
        chunk_count = 0
        duration = 0.0
        
        if timeline_path:
            timeline_manager = TimelineManager(timeline_path, verbose=verbose)
            try:
                timeline_manager.load()
                chunk_count = timeline_manager.get_chunk_count()
                duration = timeline_manager.calculate_timeline_duration()
                if log:
                    log.info(f"Timeline loaded: {chunk_count} chunks, {duration:.2f}s duration")
                elif verbose:
                    print(f"Timeline loaded: {chunk_count} chunks, {duration:.2f}s duration")
            except Exception as e:
                error_msg = f"[ERROR] Failed to load timeline: {e}"
                if log:
                    log.error(error_msg)
                elif verbose:
                    print(error_msg)
        
        # Step 2: Classify operation and extract parameters
        operation_result = None
        if query:
            operation_result = classify_operation(
                query=query,
                chunk_count=chunk_count,
                duration=duration,
                llm=llm,
                verbose=verbose
            )
            
            # Validate parameters
            operation = operation_result.get("operation")
            params = operation_result.get("parameters", {})
            
            # If LLM returned UNKNOWN but we have effect parameters, try to use APPLY_EFFECT
            if operation == "UNKNOWN" and params.get("effect_type") and params.get("effect_object"):
                # Try to extract timeline indices if missing
                indices = params.get("timeline_indices", [])
                if not indices:
                    # Use heuristic to extract indices from the query
                    # Clean query first (remove clip references) then extract indices
                    from agent.orchestrator_operations import _clean_query, _extract_timeline_indices
                    cleaned_query = _clean_query(query)
                    indices = _extract_timeline_indices(cleaned_query, chunk_count)
                    if indices:
                        params["timeline_indices"] = indices
                        operation_result["parameters"]["timeline_indices"] = indices
                        if log:
                            log.info(f"  [OVERRIDE] Extracted timeline indices from query: {indices}")
                            if cleaned_query != query:
                                log.info(f"  [OVERRIDE] Cleaned query: '{cleaned_query}' (from: '{query}')")
                        elif verbose:
                            print(f"  [OVERRIDE] Extracted timeline indices from query: {indices}")
                            if cleaned_query != query:
                                print(f"  [OVERRIDE] Cleaned query: '{cleaned_query}' (from: '{query}')")
                
                # Check if we have timeline indices (required for APPLY_EFFECT)
                if indices:
                    if log:
                        log.info(f"  [OVERRIDE] LLM returned UNKNOWN but effect parameters detected, using APPLY_EFFECT")
                        log.info(f"  [OVERRIDE] Timeline indices: {indices}")
                    elif verbose:
                        print(f"  [OVERRIDE] LLM returned UNKNOWN but effect parameters detected, using APPLY_EFFECT")
                        print(f"  [OVERRIDE] Timeline indices: {indices}")
                    operation = "APPLY_EFFECT"
                    operation_result["operation"] = "APPLY_EFFECT"
                    operation_result["confidence"] = 0.7  # Medium confidence for override
                else:
                    if log:
                        log.warning(f"  [OVERRIDE] Effect parameters detected but no timeline indices found in query")
                    elif verbose:
                        print(f"  [OVERRIDE] Effect parameters detected but no timeline indices found in query")
            
            if operation != "UNKNOWN" and operation != "FIND_HIGHLIGHTS":
                is_valid, error = validate_operation_params(
                    operation=operation,
                    params=params,
                    chunk_count=chunk_count,
                    verbose=verbose
                )
                
                if not is_valid:
                    if log:
                        log.warning(f"[WARNING] Invalid parameters: {error}")
                    elif verbose:
                        print(f"[WARNING] Invalid parameters: {error}")
                    operation_result["operation"] = "UNKNOWN"
                    operation_result["error"] = error
        
        # Store operation result in state
        current_operation = operation_result.get("operation") if operation_result else None
        operation_params = operation_result.get("parameters", {}) if operation_result else None
        
        if log and operation_result:
            log.info(f"\n[OPERATION CLASSIFICATION]")
            log.info(f"  Operation: {current_operation}")
            log.info(f"  Confidence: {operation_result.get('confidence', 0.0):.2f}")
            if operation_params:
                log.info(f"  Parameters: {operation_params}")
        elif verbose and operation_result:
            print(f"\n[OPERATION CLASSIFICATION]")
            print(f"  Operation: {current_operation}")
            print(f"  Confidence: {operation_result.get('confidence', 0.0):.2f}")
            if operation_params:
                print(f"  Parameters: {operation_params}")
        
        # Step 3: Execute operation
        operation_result_dict = None
        if current_operation == "UNKNOWN":
            # Handle UNKNOWN operation - return error result
            operation_result_dict = {
                "success": False,
                "error": operation_result.get("error") if operation_result else "Unknown operation",
                "reasoning": operation_result.get("reasoning") if operation_result else "Could not classify operation"
            }
            if log:
                log.warning(f"[WARNING] Unknown operation - cannot execute")
            elif verbose:
                print(f"[WARNING] Unknown operation - cannot execute")
        elif current_operation and timeline_manager:
            # Create planner agent node for operations that need it
            planner_node = create_planner_agent(model_name)
            
            # Helper function to call planner
            def call_planner(planner_state):
                return planner_node(planner_state)
            
            try:
                if current_operation == "FIND_HIGHLIGHTS":
                    operation_result_dict = handle_find_highlights(
                        state, timeline_manager, call_planner, verbose=verbose
                    )
                elif current_operation == "CUT":
                    operation_result_dict = handle_cut(
                        state, timeline_manager, operation_params, verbose=verbose
                    )
                elif current_operation == "REPLACE":
                    operation_result_dict = handle_replace(
                        state, timeline_manager, operation_params, call_planner, verbose=verbose
                    )
                elif current_operation == "INSERT":
                    operation_result_dict = handle_insert(
                        state, timeline_manager, operation_params, call_planner, verbose=verbose
                    )
                elif current_operation == "FIND_BROLL":
                    operation_result_dict = handle_find_broll(
                        state, timeline_manager, operation_params, call_planner, verbose=verbose
                    )
                elif current_operation == "TRIM":
                    operation_result_dict = handle_trim(
                        state, timeline_manager, operation_params, verbose=verbose
                    )
                elif current_operation == "APPLY_EFFECT":
                    operation_result_dict = handle_apply_effect(
                        state, timeline_manager, operation_params, verbose=verbose
                    )
                else:
                    if verbose:
                        print(f"[WARNING] Operation '{current_operation}' not yet implemented")
                    operation_result_dict = {
                        "success": False,
                        "error": f"Operation '{current_operation}' not yet implemented"
                    }
                
                # Save timeline if operation was successful
                if operation_result_dict and operation_result_dict.get("success"):
                    timeline_manager.save()
                    if log:
                        log.info(f"\n[SAVED] Timeline updated and saved to {timeline_path}")
                    elif verbose:
                        print(f"\n[SAVED] Timeline updated and saved to {timeline_path}")
                
            except Exception as e:
                error_msg = f"[ERROR] Operation execution failed: {e}"
                if log:
                    log.error(error_msg)
                    import traceback
                    log.error(traceback.format_exc())
                elif verbose:
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                operation_result_dict = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            **state,
            "timeline_chunks": timeline_manager.chunks if timeline_manager else None,
            "current_operation": current_operation,
            "operation_params": operation_params,
            "operation_result": operation_result_dict,
        }
    
    return orchestrator_node

