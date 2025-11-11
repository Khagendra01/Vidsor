"""Weight configuration utilities for search scoring."""

from typing import Dict, Any, Set, Optional
from agent.logging_utils import get_log_helper


def validate_weights(
    weights: Dict[str, Any],
    verbose: bool = False,
    logger=None
) -> None:
    """
    Validate and cap weights to safe ranges.
    Modifies weights dict in place.
    
    Args:
        weights: Weights dictionary to validate
        verbose: Whether to print verbose output
        logger: Optional logger instance
    """
    log = get_log_helper(logger, verbose)
    
    # Validation 1: Cap hierarchical weight to reasonable maximum (0.3)
    if weights.get("hierarchical_weight", 0) > 0.3:
        if verbose:
            log.info(f"  [VALIDATION] Capping hierarchical weight from {weights['hierarchical_weight']:.2f} to 0.30")
        weights["hierarchical_weight"] = 0.3
    
    # Validation 2: Cap any individual weight to 0.5 maximum (safety check)
    max_main_weight = 0.5
    if weights.get("semantic_weight", 0) > max_main_weight:
        if verbose:
            log.info(f"  [VALIDATION] Capping semantic weight from {weights['semantic_weight']:.2f} to {max_main_weight:.2f}")
        weights["semantic_weight"] = max_main_weight
    if weights.get("activity_weight", 0) > max_main_weight:
        if verbose:
            log.info(f"  [VALIDATION] Capping activity weight from {weights['activity_weight']:.2f} to {max_main_weight:.2f}")
        weights["activity_weight"] = max_main_weight
    if weights.get("hierarchical_weight", 0) > max_main_weight:
        if verbose:
            log.info(f"  [VALIDATION] Capping hierarchical weight from {weights['hierarchical_weight']:.2f} to {max_main_weight:.2f}")
        weights["hierarchical_weight"] = max_main_weight
    
    # Validation 3: Ensure threshold is reasonable
    if weights.get("threshold", 0.5) < 0.3:
        if verbose:
            log.info(f"  [VALIDATION] Raising threshold from {weights.get('threshold', 0.5):.2f} to 0.30")
        weights["threshold"] = 0.3


def initialize_weights_from_strategy(
    strategy: Dict[str, Any],
    all_object_classes: Set[str],
    search_plan: Dict[str, Any],
    verbose: bool = False,
    log_info=None,
    logger=None
) -> Dict[str, Any]:
    """
    Initialize weights from strategy scoring configuration.
    
    Args:
        strategy: Strategy dict with scoring configuration
        all_object_classes: Set of all object class names
        search_plan: Search plan dict (for checking semantic queries, highlight queries)
        verbose: Whether to print verbose output
        log_info: Optional logging function (deprecated, use logger instead)
        logger: Optional DualLogger instance
        
    Returns:
        Weights dictionary
    """
    # Use get_log_helper if logger provided, otherwise fall back to log_info or print
    if logger is not None:
        log = get_log_helper(logger, verbose)
        log_info = log.info
    elif not log_info:
        log_info = print if verbose else lambda x: None
    
    strategy_scoring = strategy.get("scoring", {})
    weights = {
        "semantic_weight": strategy_scoring.get("weights", {}).get("semantic", 0.4),
        "activity_weight": strategy_scoring.get("weights", {}).get("activity", 0.3),
        "hierarchical_weight": strategy_scoring.get("weights", {}).get("hierarchical", 0.1),
        "object_weight": strategy_scoring.get("weights", {}).get("object", 0.2),
        "object_weights": strategy_scoring.get("object_weights", {}).copy(),
        "threshold": max(0.5, strategy_scoring.get("threshold", 0.5))
    }
    
    # Initialize all object classes with default low weight
    for class_name in all_object_classes:
        if class_name not in weights["object_weights"]:
            weights["object_weights"][class_name] = 0.1
    
    # Apply weight fixes
    apply_weight_fixes(weights, search_plan, verbose=verbose, log_info=log_info, logger=logger)
    
    return weights


def apply_weight_fixes(
    weights: Dict[str, Any],
    search_plan: Dict[str, Any],
    verbose: bool = False,
    log_info=None,
    logger=None
) -> None:
    """
    Apply fixes and adjustments to weights based on search plan.
    Modifies weights dict in place, then validates.
    
    Args:
        weights: Weights dictionary to modify
        search_plan: Search plan dict (for checking semantic queries, highlight queries)
        verbose: Whether to print verbose output
        log_info: Optional logging function (deprecated, use logger instead)
        logger: Optional DualLogger instance
    """
    # Use get_log_helper if logger provided, otherwise fall back to log_info or print
    if logger is not None:
        log = get_log_helper(logger, verbose)
        log_info = log.info
    elif not log_info:
        log_info = print if verbose else lambda x: None
    
    # Fix 1: Ensure semantic weight is set if semantic queries were generated
    has_semantic_queries = bool(search_plan.get("semantic_queries"))
    if has_semantic_queries and weights.get("semantic_weight", 0) == 0.0:
        # Override: if semantic queries exist, semantic weight should be > 0
        if weights.get("hierarchical_weight", 0) > 0:
            # Reduce hierarchical to make room for semantic
            weights["hierarchical_weight"] = max(0.05, weights["hierarchical_weight"] * 0.5)
        weights["semantic_weight"] = 0.4  # Set reasonable default
        if verbose:
            log_info(f"  [FIX] Set semantic weight to 0.4 (semantic queries detected)")
    
    # Fix 2: Boost semantic weight for highlight queries
    if search_plan.get("is_general_highlight_query"):
        # For highlight queries, prioritize semantic relevance
        weights["semantic_weight"] = max(weights.get("semantic_weight", 0.4), 0.5)  # Boost to at least 0.5
        weights["hierarchical_weight"] = min(weights.get("hierarchical_weight", 0.1), 0.1)  # Reduce to max 0.1
        weights["object_weight"] = min(weights.get("object_weight", 0.2), 0.15)  # Reduce to max 0.15
        if verbose:
            log_info(f"  [HIGHLIGHT BOOST] Semantic weight: {weights.get('semantic_weight', 0.4):.2f}, "
                     f"Hierarchical: {weights.get('hierarchical_weight', 0.1):.2f}, "
                     f"Object: {weights.get('object_weight', 0.2):.2f}")
    
    # Apply validation at the end
    validate_weights(weights, verbose=verbose, logger=logger)


def configure_search_weights(
    strategy: Optional[Dict[str, Any]],
    query_intent: Dict[str, Any],
    all_object_classes: Set[str],
    search_plan: Dict[str, Any],
    configure_weights_fn,
    verbose: bool = False,
    log_info=None,
    logger=None
) -> Dict[str, Any]:
    """
    Configure search weights using strategy if available, otherwise fallback to configure_weights_fn.
    Applies all weight fixes and validation consistently.
    
    This is the main entry point for weight configuration.
    
    Args:
        strategy: Optional strategy dict with scoring configuration
        query_intent: Query intent dictionary
        all_object_classes: Set of all object class names
        search_plan: Search plan dict
        configure_weights_fn: Fallback function to configure weights (from query_analysis)
        verbose: Whether to print verbose output
        log_info: Optional logging function (deprecated, use logger instead)
        logger: Optional DualLogger instance
        
    Returns:
        Configured and validated weights dictionary
    """
    # Use get_log_helper if logger provided, otherwise fall back to log_info or print
    if logger is not None:
        log = get_log_helper(logger, verbose)
        log_info = log.info
    elif not log_info:
        log_info = print if verbose else lambda x: None
    
    if strategy and strategy.get("scoring"):
        # Use strategy-based weights
        weights = initialize_weights_from_strategy(
            strategy, all_object_classes, search_plan, verbose=verbose, log_info=log_info, logger=logger
        )
    else:
        # Fallback to old weight configuration
        weights = configure_weights_fn(query_intent, all_object_classes)
        # Ensure object_weight is set
        if "object_weight" not in weights:
            weights["object_weight"] = 0.2
        
        # Apply same fixes to fallback weights
        apply_weight_fixes(weights, search_plan, verbose=verbose, log_info=log_info, logger=logger)
    
    return weights


# Backward compatibility alias
configure_weights_with_fallback = configure_search_weights

