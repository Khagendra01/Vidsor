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
from agent.utils.transaction import TimelineTransaction
from agent.utils.self_correction import self_correct_loop
from agent.utils.multi_step_planner import (
    create_multi_step_plan,
    execute_multi_step_plan
)


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
    
    # CRITICAL FIX: Cache planner agent instance to avoid recreating on every call
    # This significantly improves performance by reusing the LLM instance
    planner_agent = create_planner_agent(model_name)
    
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
        
        # Step 1.5: Check if query needs multi-step planning
        enable_multi_step = state.get("enable_multi_step_planning", True)
        multi_step_plan = None
        
        if enable_multi_step and query:
            # Check if query is complex enough for multi-step planning
            context = {
                "chunk_count": chunk_count,
                "duration": duration
            }
            multi_step_plan = create_multi_step_plan(
                query=query,
                llm=llm,
                context=context,
                verbose=verbose
            )
            
            is_multi_step = multi_step_plan.get("is_multi_step", False)
            steps = multi_step_plan.get("steps", [])
            
            if is_multi_step and len(steps) > 1:
                # Execute multi-step plan
                if log:
                    log.info(f"\n[MULTI-STEP PLANNING] Detected complex query, executing {len(steps)} steps")
                elif verbose:
                    print(f"\n[MULTI-STEP PLANNING] Detected complex query, executing {len(steps)} steps")
                
                # Execute steps sequentially (each step will go through normal orchestrator flow)
                # We'll handle this by processing each step's query through the orchestrator
                # Note: classify_operation and handlers are already imported at the top
                
                step_results = []
                current_step_state = state.copy()
                
                for step in steps:
                    step_num = step.get("step_number")
                    step_query = step.get("query", "")
                    step_operation = step.get("operation")
                    step_params = step.get("parameters", {})
                    
                    if log:
                        log.info(f"\n[STEP {step_num}] {step_operation}: {step_query}")
                    elif verbose:
                        print(f"\n[STEP {step_num}] {step_operation}: {step_query}")
                    
                    # Update state for this step
                    step_state = current_step_state.copy()
                    step_state["user_query"] = step_query
                    
                    # Classify this step's operation
                    step_operation_result = classify_operation(
                        query=step_query,
                        chunk_count=timeline_manager.get_chunk_count() if timeline_manager else 0,
                        duration=timeline_manager.calculate_timeline_duration() if timeline_manager else 0.0,
                        llm=llm,
                        verbose=verbose
                    )
                    
                    # Use step's operation if classification matches, otherwise use classified operation
                    classified_op = step_operation_result.get("operation")
                    if classified_op != "UNKNOWN":
                        step_operation = classified_op
                        # Merge step params with classified params
                        classified_params = step_operation_result.get("parameters", {})
                        step_params = {**classified_params, **step_params}
                    
                    # Execute step using transaction and handlers (same as single-step execution)
                    with TimelineTransaction(timeline_manager, verbose=verbose) as step_tx:
                        try:
                            step_tx.add_operation({"operation": step_operation, "parameters": step_params})
                            
                            # Execute step operation
                            def call_planner(planner_state):
                                return planner_agent(planner_state)
                            
                            if step_operation == "FIND_HIGHLIGHTS":
                                step_result = handle_find_highlights(
                                    step_state, timeline_manager, call_planner, verbose=verbose
                                )
                            elif step_operation == "CUT":
                                step_result = handle_cut(
                                    step_state, timeline_manager, step_params, verbose=verbose
                                )
                            elif step_operation == "REPLACE":
                                step_result = handle_replace(
                                    step_state, timeline_manager, step_params, call_planner, verbose=verbose
                                )
                            elif step_operation == "INSERT":
                                step_result = handle_insert(
                                    step_state, timeline_manager, step_params, call_planner, verbose=verbose
                                )
                            elif step_operation == "FIND_BROLL":
                                step_result = handle_find_broll(
                                    step_state, timeline_manager, step_params, call_planner, verbose=verbose
                                )
                            elif step_operation == "TRIM":
                                step_result = handle_trim(
                                    step_state, timeline_manager, step_params, verbose=verbose
                                )
                            elif step_operation == "APPLY_EFFECT":
                                step_result = handle_apply_effect(
                                    step_state, timeline_manager, step_params, verbose=verbose
                                )
                            else:
                                step_result = {
                                    "success": False,
                                    "error": f"Operation '{step_operation}' not supported in multi-step"
                                }
                            
                            # Commit if successful
                            if step_result and step_result.get("success"):
                                is_valid, errors = timeline_manager.validate_timeline()
                                if is_valid:
                                    step_tx.commit()
                                else:
                                    step_result["success"] = False
                                    step_result["error"] = f"Validation failed: {', '.join(errors)}"
                            
                            step_results.append({
                                "step": step_num,
                                "operation": step_operation,
                                "success": step_result.get("success", False),
                                "result": step_result
                            })
                            
                            # Update state for next step
                            if step_result.get("success"):
                                current_step_state["timeline_chunks"] = timeline_manager.chunks
                        
                        except Exception as e:
                            step_tx.rollback()
                            error_msg = f"Exception in step {step_num}: {e}"
                            if log:
                                log.error(f"  ✗ {error_msg}")
                            elif verbose:
                                print(f"  ✗ {error_msg}")
                            step_results.append({
                                "step": step_num,
                                "operation": step_operation,
                                "success": False,
                                "error": error_msg,
                                "result": None
                            })
                
                multi_step_result = {
                    "success": all(r.get("success") for r in step_results),
                    "steps_total": len(steps),
                    "steps_completed": len([r for r in step_results if r.get("success")]),
                    "steps_failed": len([r for r in step_results if not r.get("success")]),
                    "results": step_results
                }
                
                # Return multi-step result
                return {
                    **state,
                    "timeline_chunks": timeline_manager.chunks if timeline_manager else None,
                    "current_operation": "MULTI_STEP",
                    "operation_params": {"steps": steps},
                    "operation_result": multi_step_result,
                    "multi_step_plan": multi_step_plan
                }
        
        # Step 2: Classify operation and extract parameters (single-step execution)
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
            # Use cached planner agent instance (created once, reused)
            # Helper function to call planner
            def call_planner(planner_state):
                return planner_agent(planner_state)
            
            # Check if self-correction is enabled (default: True for search-based operations)
            enable_self_correction = state.get("enable_self_correction", True)
            operations_with_self_correction = ["FIND_HIGHLIGHTS", "REPLACE", "INSERT", "FIND_BROLL"]
            use_self_correction = enable_self_correction and current_operation in operations_with_self_correction
            
            # TRANSACTION SUPPORT: Wrap operation in transaction for rollback capability
            with TimelineTransaction(timeline_manager, verbose=verbose) as tx:
                try:
                    # Track operation in transaction
                    tx.add_operation(operation_result)
                    
                    # Execute operation with or without self-correction
                    if use_self_correction:
                        if log:
                            log.info(f"[SELF-CORRECTION] Enabled for {current_operation}")
                        elif verbose:
                            print(f"[SELF-CORRECTION] Enabled for {current_operation}")
                        
                        # Create operation handler wrapper for self-correction
                        def create_operation_handler(op_type):
                            if op_type == "FIND_HIGHLIGHTS":
                                def handler(s, tm, p, verbose=False):
                                    return handle_find_highlights(s, tm, call_planner, verbose=verbose)
                                return handler
                            elif op_type == "REPLACE":
                                def handler(s, tm, p, verbose=False):
                                    return handle_replace(s, tm, p, call_planner, verbose=verbose)
                                return handler
                            elif op_type == "INSERT":
                                def handler(s, tm, p, verbose=False):
                                    return handle_insert(s, tm, p, call_planner, verbose=verbose)
                                return handler
                            elif op_type == "FIND_BROLL":
                                def handler(s, tm, p, verbose=False):
                                    return handle_find_broll(s, tm, p, call_planner, verbose=verbose)
                                return handler
                            return None
                        
                        handler = create_operation_handler(current_operation)
                        if handler:
                            operation_result_dict = self_correct_loop(
                                state=state,
                                timeline_manager=timeline_manager,
                                operation=current_operation,
                                params=operation_params,
                                operation_handler=handler,
                                max_iterations=3,
                                confidence_threshold=0.7,
                                verbose=verbose
                            )
                        else:
                            # Fallback to normal execution
                            use_self_correction = False
                    
                    # Normal execution (without self-correction or if self-correction not applicable)
                    if not use_self_correction:
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
                    
                    # Commit transaction if operation was successful
                    if operation_result_dict and operation_result_dict.get("success"):
                        # Validate timeline before committing
                        is_valid, errors = timeline_manager.validate_timeline()
                        if is_valid:
                            if tx.commit():
                                if log:
                                    log.info(f"\n[SAVED] Timeline updated and saved to {timeline_path}")
                                elif verbose:
                                    print(f"\n[SAVED] Timeline updated and saved to {timeline_path}")
                                
                                # Add transaction info to result
                                operation_result_dict["transaction"] = {
                                    "committed": True,
                                    "operations": len(tx.get_operations())
                                }
                            else:
                                # Commit failed, mark operation as failed
                                if log:
                                    log.error("[TRANSACTION] Failed to commit transaction")
                                elif verbose:
                                    print("[TRANSACTION] Failed to commit transaction")
                                operation_result_dict["success"] = False
                                operation_result_dict["error"] = "Failed to save timeline"
                                operation_result_dict["transaction"] = {
                                    "committed": False,
                                    "rolled_back": True
                                }
                        else:
                            # Validation failed, transaction will auto-rollback
                            if log:
                                log.error(f"[TRANSACTION] Timeline validation failed:")
                                for error in errors:
                                    log.error(f"  - {error}")
                            elif verbose:
                                print(f"[TRANSACTION] Timeline validation failed:")
                                for error in errors:
                                    print(f"  - {error}")
                            operation_result_dict["success"] = False
                            operation_result_dict["error"] = f"Validation failed: {', '.join(errors)}"
                            operation_result_dict["transaction"] = {
                                "committed": False,
                                "rolled_back": True,
                                "validation_errors": errors
                            }
                    else:
                        # Operation failed, transaction will auto-rollback
                        if log:
                            log.warning("[TRANSACTION] Operation failed, rolling back")
                        elif verbose:
                            print("[TRANSACTION] Operation failed, rolling back")
                        operation_result_dict["transaction"] = {
                            "committed": False,
                            "rolled_back": True,
                            "reason": "Operation failed"
                        }
                
                except Exception as e:
                    # Exception occurred, manually rollback transaction
                    tx.rollback()
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
                        "error": str(e),
                        "transaction": {
                            "committed": False,
                            "rolled_back": True,
                            "exception": type(e).__name__
                        }
                    }
        
        return {
            **state,
            "timeline_chunks": timeline_manager.chunks if timeline_manager else None,
            "current_operation": current_operation,
            "operation_params": operation_params,
            "operation_result": operation_result_dict,
        }
    
    return orchestrator_node

