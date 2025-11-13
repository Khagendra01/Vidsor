"""Self-correction utilities for automatic operation refinement."""

import json
from typing import Dict, List, Optional, Any, Callable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.state import OrchestratorState

from agent.timeline_manager import TimelineManager
from agent.utils.llm_utils import invoke_llm_with_json, create_llm
from agent.utils.logging_utils import get_log_helper


def validate_operation_result(
    result: Dict[str, Any],
    operation: str,
    params: Dict[str, Any],
    state: "OrchestratorState",
    llm,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate an operation result and determine if refinement is needed.
    
    Args:
        result: Operation result dictionary
        operation: Operation type (e.g., "FIND_HIGHLIGHTS", "REPLACE")
        params: Operation parameters
        state: Orchestrator state
        llm: Language model instance
        verbose: Whether to print verbose output
        
    Returns:
        Validation dictionary with is_valid, confidence, issues, suggestions
    """
    log = get_log_helper(state.get("logger"), verbose)
    
    # Basic validation: check if operation succeeded
    if not result.get("success", False):
        return {
            "is_valid": False,
            "confidence": 0.0,
            "needs_refinement": True,
            "issues": [result.get("error", "Operation failed")],
            "suggestions": ["Check operation parameters and try again"],
            "reasoning": "Operation returned success=False"
        }
    
    # Operation-specific validation
    if operation == "FIND_HIGHLIGHTS":
        return _validate_find_highlights_result(result, params, state, llm, verbose)
    elif operation == "REPLACE":
        return _validate_replace_result(result, params, state, llm, verbose)
    elif operation == "INSERT":
        return _validate_insert_result(result, params, state, llm, verbose)
    elif operation == "FIND_BROLL":
        return _validate_find_broll_result(result, params, state, llm, verbose)
    elif operation in ["CUT", "TRIM", "APPLY_EFFECT"]:
        # These operations are simpler, basic validation is usually enough
        return {
            "is_valid": True,
            "confidence": 0.9,
            "needs_refinement": False,
            "issues": [],
            "suggestions": [],
            "reasoning": "Operation completed successfully"
        }
    
    # Default: assume valid if operation succeeded
    return {
        "is_valid": True,
        "confidence": 0.7,
        "needs_refinement": False,
        "issues": [],
        "suggestions": [],
        "reasoning": "Operation completed, no specific validation"
    }


def _validate_find_highlights_result(
    result: Dict[str, Any],
    params: Dict[str, Any],
    state: "OrchestratorState",
    llm,
    verbose: bool = False
) -> Dict[str, Any]:
    """Validate FIND_HIGHLIGHTS operation result."""
    chunks_created = result.get("chunks_created", [])
    query = state.get("user_query", "")
    
    if not chunks_created:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "needs_refinement": True,
            "issues": ["No highlights found"],
            "suggestions": [
                "Try expanding search query",
                "Lower search threshold",
                "Check if video contains requested content"
            ],
            "reasoning": "No chunks were created"
        }
    
    # Use LLM to validate if highlights match query intent
    system_prompt = """You are a video editing operation validator. Validate if the operation results match the user's intent.

Return JSON:
{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "needs_refinement": true/false,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "reasoning": "brief explanation"
}"""
    
    chunks_summary = [
        {
            "index": i,
            "duration": chunk.get("end_time", 0) - chunk.get("start_time", 0),
            "original_range": f"{chunk.get('original_start_time', 0):.1f}s-{chunk.get('original_end_time', 0):.1f}s",
            "description": chunk.get("description", ""),
            "unified_description": chunk.get("unified_description", ""),
            "audio_description": chunk.get("audio_description", ""),
            "score": chunk.get("score", 0.0)
        }
        for i, chunk in enumerate(chunks_created[:10])  # Top 10 for validation
    ]
    
    user_message = f"""
Query: {query}
Operation: FIND_HIGHLIGHTS
Number of chunks created: {len(chunks_created)}
Chunks summary: {json.dumps(chunks_summary, indent=2)}

Validate if these highlights match what the user asked for. Consider:
1. Are there enough highlights?
2. Do the highlights match the query intent (check descriptions)?
3. Are the durations appropriate?
4. Do the descriptions indicate these are actually highlight moments?

Note: Each chunk includes description, unified_description, and audio_description fields that describe the content.
Use these descriptions to assess if the highlights are relevant to the query.

Return JSON only.
"""
    
    fallback = {
        "is_valid": len(chunks_created) > 0,
        "confidence": 0.7 if len(chunks_created) > 0 else 0.3,
        "needs_refinement": len(chunks_created) == 0,
        "issues": [] if len(chunks_created) > 0 else ["No highlights found"],
        "suggestions": [],
        "reasoning": "Fallback validation"
    }
    
    try:
        validation = invoke_llm_with_json(
            llm=llm,
            system_prompt=system_prompt,
            user_message=user_message,
            fallback=fallback,
            verbose=verbose
        )
        return validation
    except Exception as e:
        if verbose:
            print(f"  [VALIDATION] Error: {e}, using fallback")
        return fallback


def _validate_replace_result(
    result: Dict[str, Any],
    params: Dict[str, Any],
    state: "OrchestratorState",
    llm,
    verbose: bool = False
) -> Dict[str, Any]:
    """Validate REPLACE operation result."""
    chunks_created = result.get("chunks_created", [])
    timeline_indices = params.get("timeline_indices", [])
    
    if not chunks_created:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "needs_refinement": True,
            "issues": ["No replacement content found"],
            "suggestions": [
                "Try different search query",
                "Expand search parameters",
                "Check if replacement content exists in video"
            ],
            "reasoning": "No chunks created for replacement"
        }
    
    # Check if we replaced the right number of indices
    expected_count = len(timeline_indices)
    actual_count = len(chunks_created)
    
    if actual_count < expected_count:
        return {
            "is_valid": False,
            "confidence": 0.5,
            "needs_refinement": True,
            "issues": [f"Expected {expected_count} replacements, got {actual_count}"],
            "suggestions": ["Try expanding search to find more content"],
            "reasoning": "Incomplete replacement"
        }
    
    return {
        "is_valid": True,
        "confidence": 0.8,
        "needs_refinement": False,
        "issues": [],
        "suggestions": [],
        "reasoning": "Replacement completed successfully"
    }


def _validate_insert_result(
    result: Dict[str, Any],
    params: Dict[str, Any],
    state: "OrchestratorState",
    llm,
    verbose: bool = False
) -> Dict[str, Any]:
    """Validate INSERT operation result."""
    chunks_created = result.get("chunks_created", [])
    
    if not chunks_created:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "needs_refinement": True,
            "issues": ["No content found to insert"],
            "suggestions": [
                "Try different search query",
                "Check if content exists in video"
            ],
            "reasoning": "No chunks created for insertion"
        }
    
    return {
        "is_valid": True,
        "confidence": 0.8,
        "needs_refinement": False,
        "issues": [],
        "suggestions": [],
        "reasoning": "Insertion completed successfully"
    }


def _validate_find_broll_result(
    result: Dict[str, Any],
    params: Dict[str, Any],
    state: "OrchestratorState",
    llm,
    verbose: bool = False
) -> Dict[str, Any]:
    """Validate FIND_BROLL operation result."""
    chunks_created = result.get("chunks_created", [])
    
    if not chunks_created:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "needs_refinement": True,
            "issues": ["No B-roll found"],
            "suggestions": [
                "Try broader search terms",
                "Look for complementary footage",
                "Check if B-roll content exists"
            ],
            "reasoning": "No B-roll chunks created"
        }
    
    return {
        "is_valid": True,
        "confidence": 0.8,
        "needs_refinement": False,
        "issues": [],
        "suggestions": [],
        "reasoning": "B-roll found successfully"
    }


def suggest_refinement(
    validation: Dict[str, Any],
    result: Dict[str, Any],
    operation: str,
    params: Dict[str, Any],
    state: "OrchestratorState",
    llm,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Suggest how to refine the operation based on validation results.
    
    Args:
        validation: Validation result dictionary
        result: Operation result dictionary
        operation: Operation type
        params: Current operation parameters
        state: Orchestrator state
        llm: Language model instance
        verbose: Whether to print verbose output
        
    Returns:
        Refinement dictionary with adjustments to make, or None if no refinement possible
    """
    if not validation.get("needs_refinement", False):
        return None
    
    issues = validation.get("issues", [])
    suggestions = validation.get("suggestions", [])
    
    if not issues and not suggestions:
        return None
    
    # Use LLM to convert suggestions into parameter adjustments
    system_prompt = """You are a video editing operation refiner. Convert validation issues and suggestions into specific parameter adjustments.

Return JSON:
{
    "adjustments": {
        "search_query": "new or modified query",
        "timeline_indices": [0, 1, 2],
        "threshold": 0.5,
        "max_results": 20,
        // ... other parameter adjustments
    },
    "reasoning": "why these adjustments"
}"""
    
    user_message = f"""
Operation: {operation}
Current parameters: {json.dumps(params, indent=2)}
Validation issues: {json.dumps(issues, indent=2)}
Suggestions: {json.dumps(suggestions, indent=2)}

Convert these into specific parameter adjustments. Return JSON only.
"""
    
    fallback = {
        "adjustments": {},
        "reasoning": "Fallback: no specific adjustments"
    }
    
    try:
        refinement = invoke_llm_with_json(
            llm=llm,
            system_prompt=system_prompt,
            user_message=user_message,
            fallback=fallback,
            verbose=verbose
        )
        return refinement
    except Exception as e:
        if verbose:
            print(f"  [REFINEMENT] Error: {e}, using fallback")
        return fallback


def apply_refinement(
    params: Dict[str, Any],
    refinement: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply refinement adjustments to operation parameters.
    
    Args:
        params: Current operation parameters
        refinement: Refinement dictionary with adjustments
        
    Returns:
        Updated parameters dictionary
    """
    adjustments = refinement.get("adjustments", {})
    
    # Merge adjustments into params
    updated_params = params.copy()
    for key, value in adjustments.items():
        if key in updated_params:
            # Update existing parameter
            if isinstance(updated_params[key], list) and isinstance(value, list):
                # Merge lists
                updated_params[key] = value
            elif isinstance(updated_params[key], dict) and isinstance(value, dict):
                # Merge dicts
                updated_params[key].update(value)
            else:
                # Replace value
                updated_params[key] = value
        else:
            # Add new parameter
            updated_params[key] = value
    
    return updated_params


def should_stop_early(
    validation: Dict[str, Any],
    iteration: int,
    max_iterations: int,
    previous_confidence: float = 0.0,
    confidence_threshold: float = 0.7,
    enable_early_stopping: bool = True,
    operation: str = None,
    result: Dict[str, Any] = None,
    video_duration: float = None
) -> Tuple[bool, str]:
    """
    Determine if we should stop early based on quality metrics and duration adequacy.
    
    Args:
        validation: Validation result dictionary
        iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        previous_confidence: Confidence from previous iteration
        confidence_threshold: Minimum acceptable confidence
        enable_early_stopping: Whether early stopping is enabled
        operation: Operation type (e.g., "FIND_HIGHLIGHTS")
        result: Operation result dictionary (may contain chunks_created, time_ranges)
        video_duration: Total video duration in seconds
        
    Returns:
        Tuple of (should_stop: bool, reason: str)
    """
    if not enable_early_stopping:
        return False, "Early stopping disabled"
    
    confidence = validation.get("confidence", 0.0)
    is_valid = validation.get("is_valid", False)
    issues = validation.get("issues", [])
    needs_refinement = validation.get("needs_refinement", True)
    
    # Tier 1: Perfect result - stop immediately
    if is_valid and confidence >= 0.9:
        return True, f"Perfect result (confidence: {confidence:.2f})"
    
    # Tier 2: High confidence with minor issues or no issues - stop early
    # Minor issues: missing descriptions, lack diversity (fixable later)
    # Critical issues: no results, wrong operation, invalid parameters
    minor_issue_keywords = ["description", "context", "diversity", "summary", "timestamps"]
    critical_issue_keywords = ["no results", "failed", "error", "invalid", "no highlights", "operation failed"]
    
    # Check if we have only minor issues (or no issues at all)
    has_minor_issues = any(
        any(keyword in issue.lower() for keyword in minor_issue_keywords)
        for issue in issues
    ) if issues else False
    
    has_critical_issues = any(
        any(keyword in issue.lower() for keyword in critical_issue_keywords)
        for issue in issues
    ) if issues else False
    
    has_minor_issues_only = (has_minor_issues or len(issues) == 0) and not has_critical_issues
    
    if confidence >= 0.85 and has_minor_issues_only:
        if len(issues) == 0:
            return True, f"High confidence ({confidence:.2f}) with no issues"
        return True, f"High confidence ({confidence:.2f}) with minor fixable issues"
    
    # OPERATION-AWARE: For FIND_HIGHLIGHTS, check duration adequacy
    duration_adequate = True
    duration_check_reason = ""
    if operation == "FIND_HIGHLIGHTS" and video_duration and result:
        chunks_created = result.get("chunks_created", [])
        if chunks_created:
            total_duration = sum(
                chunk.get("end_time", 0) - chunk.get("start_time", 0)
                for chunk in chunks_created
            )
            target_duration = video_duration * 0.12  # 12% target
            duration_ratio = total_duration / target_duration if target_duration > 0 else 0
            
            if duration_ratio < 0.7:  # Less than 70% of target
                duration_adequate = False
                duration_check_reason = f"Duration insufficient: {total_duration:.1f}s ({total_duration/video_duration*100:.1f}% of video) < 70% of target ({target_duration:.1f}s)"
            else:
                duration_check_reason = f"Duration adequate: {total_duration:.1f}s ({total_duration/video_duration*100:.1f}% of video) >= 70% of target"
    
    # Tier 3: Acceptable result - stop if valid AND duration adequate (for FIND_HIGHLIGHTS)
    if is_valid and confidence >= confidence_threshold:
        if operation == "FIND_HIGHLIGHTS" and not duration_adequate:
            # Don't stop early if duration is insufficient for highlights
            return False, f"Confidence acceptable ({confidence:.2f}) but {duration_check_reason} - continuing"
        return True, f"Acceptable result (confidence: {confidence:.2f} >= {confidence_threshold}" + (f", {duration_check_reason}" if duration_check_reason else "") + ")"
    
    # Tier 4: Good confidence even if not "valid" - stop if minor issues or no issues
    # This handles cases where confidence meets threshold but validation has minor issues
    if confidence >= confidence_threshold and has_minor_issues_only:
        if operation == "FIND_HIGHLIGHTS" and not duration_adequate:
            # Don't stop early if duration is insufficient for highlights
            return False, f"Confidence good ({confidence:.2f}) but {duration_check_reason} - continuing"
        if len(issues) == 0:
            return True, f"Good confidence ({confidence:.2f}) with no issues (meets threshold)" + (f", {duration_check_reason}" if duration_check_reason else "")
        return True, f"Good confidence ({confidence:.2f}) with minor issues (meets threshold, acceptable)" + (f", {duration_check_reason}" if duration_check_reason else "")
    
    # Tier 5: Confidence plateau - no improvement between iterations
    if iteration >= 2 and abs(confidence - previous_confidence) < 0.05:
        if confidence >= confidence_threshold * 0.9:  # Close to threshold
            return True, f"Confidence plateaued at {confidence:.2f}, unlikely to improve"
    
    # Tier 6: No refinement possible - stop early
    if not needs_refinement:
        return True, "No refinement needed or possible"
    
    # Tier 7: Very low confidence - unlikely to improve
    if iteration >= 2 and confidence < 0.3:
        return True, f"Very low confidence ({confidence:.2f}), unlikely to improve"
    
    # Continue iteration
    return False, "Continue iteration"


def self_correct_loop(
    state: "OrchestratorState",
    timeline_manager: TimelineManager,
    operation: str,
    params: Dict[str, Any],
    operation_handler: Callable,
    max_iterations: int = 3,
    confidence_threshold: float = 0.7,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Execute operation with automatic self-correction.
    
    Attempts to improve results through iterative refinement.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        operation: Operation type
        params: Initial operation parameters
        operation_handler: Function to execute the operation
        max_iterations: Maximum number of refinement iterations
        confidence_threshold: Minimum confidence to accept result
        verbose: Whether to print verbose output
        
    Returns:
        Best result dictionary with self-correction metadata
    """
    log = get_log_helper(state.get("logger"), verbose)
    
    # Get LLM for validation and refinement
    model_name = state.get("model_name", "gpt-4o-mini")
    llm = create_llm(model_name)
    
    best_result = None
    best_confidence = 0.0
    best_iteration = 0
    previous_confidence = 0.0
    
    # Get early stopping setting (default: True)
    enable_early_stopping = state.get("enable_early_stopping", True)
    
    log.info(f"\n[SELF-CORRECTION] Starting self-correction loop (max {max_iterations} iterations)")
    if enable_early_stopping:
        log.info(f"  Early stopping: ENABLED")
    
    for iteration in range(1, max_iterations + 1):
        log.info(f"\n[SELF-CORRECTION] Iteration {iteration}/{max_iterations}")
        
        try:
            # Execute operation
            current_params = params.copy()
            result = operation_handler(state, timeline_manager, current_params, verbose=verbose)
            
            # Validate result
            validation = validate_operation_result(
                result=result,
                operation=operation,
                params=current_params,
                state=state,
                llm=llm,
                verbose=verbose
            )
            
            confidence = validation.get("confidence", 0.0)
            is_valid = validation.get("is_valid", False)
            
            log.info(f"  Confidence: {confidence:.2f}")
            log.info(f"  Valid: {is_valid}")
            
            # Track best result
            if confidence > best_confidence:
                best_result = result.copy()
                best_confidence = confidence
                best_iteration = iteration
                if verbose:
                    print(f"  [BEST] New best result (confidence: {confidence:.2f})")
            
            # NEW: Check for early stopping before standard check
            if enable_early_stopping:
                # Get video duration for duration adequacy check
                video_path = state.get("video_path", "")
                video_duration = None
                if video_path:
                    try:
                        from extractor.utils.video_utils import get_video_duration
                        video_duration = get_video_duration(video_path)
                    except:
                        pass
                
                should_stop, reason = should_stop_early(
                    validation=validation,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    previous_confidence=previous_confidence,
                    confidence_threshold=confidence_threshold,
                    enable_early_stopping=enable_early_stopping,
                    operation=operation,
                    result=result,
                    video_duration=video_duration
                )
                
                if should_stop:
                    log.info(f"  ✓ Early stopping: {reason}")
                    return {
                        **result,
                        "iterations": iteration,
                        "self_corrected": iteration > 1,
                        "final_confidence": confidence,
                        "validation": validation,
                        "early_stopped": True,
                        "early_stop_reason": reason
                    }
            
            # Check if result is good enough (original check, kept as fallback)
            if is_valid and confidence >= confidence_threshold:
                log.info(f"  ✓ Result is acceptable (confidence: {confidence:.2f} >= {confidence_threshold})")
                return {
                    **result,
                    "iterations": iteration,
                    "self_corrected": iteration > 1,
                    "final_confidence": confidence,
                    "validation": validation
                }
            
            # Store confidence for plateau detection
            previous_confidence = confidence
            
            # If not last iteration, refine
            if iteration < max_iterations:
                refinement = suggest_refinement(
                    validation=validation,
                    result=result,
                    operation=operation,
                    params=current_params,
                    state=state,
                    llm=llm,
                    verbose=verbose
                )
                
                if refinement and refinement.get("adjustments"):
                    log.info(f"  → Refining parameters: {refinement.get('reasoning', 'N/A')}")
                    params = apply_refinement(params, refinement)
                else:
                    log.info(f"  → No refinement possible, stopping")
                    break
            else:
                log.info(f"  → Max iterations reached")
        
        except Exception as e:
            log.error(f"  ✗ Error in iteration {iteration}: {e}")
            if best_result:
                # Return best result we found
                break
            else:
                # No good result yet, re-raise
                raise
    
    # Return best result found
    if best_result:
        log.info(f"\n[SELF-CORRECTION] Returning best result from iteration {best_iteration} (confidence: {best_confidence:.2f})")
        return {
            **best_result,
            "iterations": best_iteration,
            "self_corrected": True,
            "final_confidence": best_confidence,
            "max_iterations_reached": True
        }
    else:
        # No result found
        return {
            "success": False,
            "error": "Self-correction loop failed to produce valid result",
            "iterations": max_iterations,
            "self_corrected": True,
            "final_confidence": 0.0
        }

