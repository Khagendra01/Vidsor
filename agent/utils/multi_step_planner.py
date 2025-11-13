"""Multi-step planning utilities for breaking complex queries into executable steps."""

import json
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.state import OrchestratorState

from agent.utils.llm_utils import invoke_llm_with_json, create_llm
from agent.utils.logging_utils import get_log_helper


MULTI_STEP_PLANNING_PROMPT = """You are a video editing workflow planner. Your job is to break complex user queries into a sequence of executable steps.

Available Operations:
- FIND_HIGHLIGHTS: Find and add highlights to timeline
- CUT: Remove chunks from timeline
- REPLACE: Replace timeline segments with new content
- INSERT: Add clips between existing chunks
- FIND_BROLL: Find complementary B-roll for selected segments
- TRIM: Adjust clip boundaries
- APPLY_EFFECT: Apply visual effects to clips
- REORDER: Change clip sequence

Guidelines:
1. Break complex queries into logical steps
2. Identify dependencies between steps (e.g., step 2 depends on step 1)
3. Each step should be a single, clear operation
4. Steps should build on previous results when needed
5. Use timeline indices from previous steps when appropriate

Return JSON only:
{
    "is_multi_step": true/false,
    "steps": [
        {
            "step_number": 1,
            "operation": "FIND_HIGHLIGHTS",
            "query": "find highlights",
            "parameters": {},
            "depends_on": [],
            "reasoning": "First find the highlights"
        },
        {
            "step_number": 2,
            "operation": "FIND_BROLL",
            "query": "find B-roll for timeline 0-2",
            "parameters": {
                "timeline_indices": [0, 1, 2]
            },
            "depends_on": [1],
            "reasoning": "Then add B-roll for the highlights found in step 1"
        }
    ]
}

If the query is simple and can be handled in one step, set "is_multi_step": false and return a single step."""


def create_multi_step_plan(
    query: str,
    llm,
    context: Optional[Dict] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Break complex query into executable steps.
    
    Args:
        query: User query string
        llm: Language model instance
        context: Optional context dictionary (timeline state, etc.)
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with is_multi_step flag and steps list
    """
    log = get_log_helper(None, verbose)
    
    log.info(f"\n[MULTI-STEP PLANNING] Analyzing query: {query}")
    
    # Build context information for planning
    context_info = ""
    if context:
        chunk_count = context.get("chunk_count", 0)
        duration = context.get("duration", 0.0)
        if chunk_count > 0 or duration > 0:
            context_info = f"\nCurrent Timeline Context:\n- Chunks: {chunk_count}\n- Duration: {duration:.2f}s"
    
    user_message = f"Query: {query}{context_info}\n\nBreak this into executable steps if it's complex, or return a single step if it's simple."
    
    fallback = {
        "is_multi_step": False,
        "steps": [{
            "step_number": 1,
            "operation": "UNKNOWN",
            "query": query,
            "parameters": {},
            "depends_on": [],
            "reasoning": "Fallback: could not parse query"
        }]
    }
    
    try:
        result = invoke_llm_with_json(
            llm=llm,
            system_prompt=MULTI_STEP_PLANNING_PROMPT,
            user_message=user_message,
            fallback=fallback,
            verbose=verbose
        )
        
        is_multi_step = result.get("is_multi_step", False)
        steps = result.get("steps", [])
        
        if verbose:
            if is_multi_step:
                log.info(f"  → Multi-step plan: {len(steps)} steps")
                for step in steps:
                    log.info(f"    Step {step.get('step_number')}: {step.get('operation')} - {step.get('reasoning', '')}")
            else:
                log.info(f"  → Single-step plan: {steps[0].get('operation') if steps else 'UNKNOWN'}")
        
        return result
    except Exception as e:
        if verbose:
            log.error(f"  ✗ Planning failed: {e}, using fallback")
        return fallback


def resolve_dependencies(steps: List[Dict]) -> List[Dict]:
    """
    Sort steps by dependencies using topological sort.
    
    Args:
        steps: List of step dictionaries with depends_on fields
        
    Returns:
        Sorted list of steps in execution order
    """
    # Build dependency graph
    step_map = {step["step_number"]: step for step in steps}
    dependencies = {step["step_number"]: set(step.get("depends_on", [])) for step in steps}
    
    # Topological sort
    sorted_steps = []
    visited = set()
    visiting = set()
    
    def visit(step_num: int):
        if step_num in visiting:
            # Circular dependency detected, break it
            return
        if step_num in visited:
            return
        
        visiting.add(step_num)
        
        # Visit dependencies first
        for dep in dependencies.get(step_num, []):
            if dep in step_map:
                visit(dep)
        
        visiting.remove(step_num)
        visited.add(step_num)
        
        if step_num in step_map:
            sorted_steps.append(step_map[step_num])
    
    # Visit all steps
    for step in steps:
        visit(step["step_number"])
    
    return sorted_steps


def update_state_for_next_step(
    state: "OrchestratorState",
    step_result: Dict[str, Any],
    step: Dict[str, Any]
) -> "OrchestratorState":
    """
    Update state with results from a completed step for use in next steps.
    
    Args:
        state: Current orchestrator state
        step_result: Result from executing the step
        step: Step dictionary that was executed
        
    Returns:
        Updated state dictionary
    """
    updated_state = state.copy()
    
    # Update timeline chunks if operation modified timeline
    if step_result.get("success") and "chunks_created" in step_result:
        # Timeline was modified, chunks will be updated by timeline_manager
        # Just mark that we need to reload timeline chunks
        pass
    
    # Update operation history
    if "operation_history" not in updated_state:
        updated_state["operation_history"] = []
    
    updated_state["operation_history"].append({
        "step": step.get("step_number"),
        "operation": step.get("operation"),
        "result": step_result,
        "timestamp": step.get("timestamp")
    })
    
    return updated_state


def execute_multi_step_plan(
    state: "OrchestratorState",
    steps: List[Dict],
    orchestrator_node,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Execute multi-step plan with dependency resolution.
    
    Args:
        state: Initial orchestrator state
        steps: List of step dictionaries
        orchestrator_node: Orchestrator node function to execute steps
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with execution results
    """
    log = get_log_helper(state.get("logger"), verbose)
    
    log.info(f"\n[MULTI-STEP EXECUTION] Executing {len(steps)} steps")
    
    # Sort steps by dependencies
    sorted_steps = resolve_dependencies(steps)
    
    results = []
    completed_steps: Set[int] = set()
    current_state = state
    
    for step in sorted_steps:
        step_num = step.get("step_number")
        dependencies = step.get("depends_on", [])
        operation = step.get("operation")
        step_query = step.get("query", "")
        step_params = step.get("parameters", {})
        
        log.info(f"\n[STEP {step_num}/{len(sorted_steps)}] {operation}")
        log.info(f"  Query: {step_query}")
        if dependencies:
            log.info(f"  Depends on: {dependencies}")
        
        # Check dependencies are complete
        if not all(dep in completed_steps for dep in dependencies):
            missing_deps = [dep for dep in dependencies if dep not in completed_steps]
            error_msg = f"Step {step_num} depends on incomplete steps: {missing_deps}"
            log.error(f"  ✗ {error_msg}")
            results.append({
                "step": step_num,
                "success": False,
                "error": error_msg,
                "result": None
            })
            continue
        
        # Update state with step query and parameters
        step_state = current_state.copy()
        step_state["user_query"] = step_query
        step_state["current_operation"] = operation
        step_state["operation_params"] = step_params
        
        # Execute step using orchestrator
        try:
            step_result_state = orchestrator_node(step_state)
            step_result = step_result_state.get("operation_result", {})
            
            success = step_result.get("success", False)
            
            if success:
                log.info(f"  ✓ Step {step_num} completed successfully")
                completed_steps.add(step_num)
            else:
                error = step_result.get("error", "Unknown error")
                log.error(f"  ✗ Step {step_num} failed: {error}")
            
            results.append({
                "step": step_num,
                "operation": operation,
                "success": success,
                "result": step_result,
                "error": step_result.get("error") if not success else None
            })
            
            # Update state for next step
            current_state = update_state_for_next_step(current_state, step_result, step)
            
            # If step failed and it's critical, stop execution
            if not success and step.get("critical", False):
                log.warning(f"  Critical step {step_num} failed, stopping execution")
                break
        
        except Exception as e:
            error_msg = f"Exception in step {step_num}: {e}"
            log.error(f"  ✗ {error_msg}")
            results.append({
                "step": step_num,
                "operation": operation,
                "success": False,
                "error": error_msg,
                "result": None
            })
            # Continue with next step unless this was critical
            if not step.get("critical", False):
                continue
            else:
                break
    
    # Summary
    successful_steps = [r for r in results if r.get("success")]
    failed_steps = [r for r in results if not r.get("success")]
    
    log.info(f"\n[MULTI-STEP EXECUTION] Summary:")
    log.info(f"  Completed: {len(successful_steps)}/{len(results)} steps")
    if failed_steps:
        log.info(f"  Failed: {len(failed_steps)} steps")
        for failed in failed_steps:
            log.info(f"    Step {failed.get('step')}: {failed.get('error', 'Unknown error')}")
    
    return {
        "success": len(failed_steps) == 0,
        "steps_total": len(steps),
        "steps_completed": len(successful_steps),
        "steps_failed": len(failed_steps),
        "results": results,
        "final_state": current_state
    }

