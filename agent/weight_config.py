"""Weight configuration utilities for search scoring."""

from typing import Dict, Any, Set, Optional


def initialize_weights_from_strategy(
    strategy: Dict[str, Any],
    all_object_classes: Set[str],
    search_plan: Dict[str, Any],
    verbose: bool = False,
    log_info=None
) -> Dict[str, Any]:
    """
    Initialize weights from strategy scoring configuration.
    
    Args:
        strategy: Strategy dict with scoring configuration
        all_object_classes: Set of all object class names
        search_plan: Search plan dict (for checking semantic queries, highlight queries)
        verbose: Whether to print verbose output
        log_info: Optional logging function
        
    Returns:
        Weights dictionary
    """
    if not log_info:
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
    apply_weight_fixes(weights, search_plan, verbose=verbose, log_info=log_info)
    
    return weights


def apply_weight_fixes(
    weights: Dict[str, Any],
    search_plan: Dict[str, Any],
    verbose: bool = False,
    log_info=None
) -> None:
    """
    Apply fixes and adjustments to weights.
    Modifies weights dict in place.
    
    Args:
        weights: Weights dictionary to modify
        search_plan: Search plan dict (for checking semantic queries, highlight queries)
        verbose: Whether to print verbose output
        log_info: Optional logging function
    """
    if not log_info:
        log_info = print if verbose else lambda x: None
    
    # Fix 1: Cap hierarchical weight to reasonable maximum (0.3)
    if weights["hierarchical_weight"] > 0.3:
        if verbose:
            log_info(f"  [FIX] Capping hierarchical weight from {weights['hierarchical_weight']:.2f} to 0.30")
        weights["hierarchical_weight"] = 0.3
    
    # Fix 2: Ensure semantic weight is set if semantic queries were generated
    has_semantic_queries = bool(search_plan.get("semantic_queries"))
    if has_semantic_queries and weights["semantic_weight"] == 0.0:
        # Override: if semantic queries exist, semantic weight should be > 0
        if weights["hierarchical_weight"] > 0:
            # Reduce hierarchical to make room for semantic
            weights["hierarchical_weight"] = max(0.05, weights["hierarchical_weight"] * 0.5)
        weights["semantic_weight"] = 0.4  # Set reasonable default
    
    # Fix 3: Boost semantic weight for highlight queries
    if search_plan.get("is_general_highlight_query"):
        # For highlight queries, prioritize semantic relevance
        weights["semantic_weight"] = max(weights["semantic_weight"], 0.5)  # Boost to at least 0.5
        weights["hierarchical_weight"] = min(weights["hierarchical_weight"], 0.1)  # Reduce to max 0.1
        weights["object_weight"] = min(weights.get("object_weight", 0.2), 0.15)  # Reduce to max 0.15
        if verbose:
            log_info(f"  [HIGHLIGHT BOOST] Semantic weight: {weights['semantic_weight']:.2f}, "
                     f"Hierarchical: {weights['hierarchical_weight']:.2f}, "
                     f"Object: {weights.get('object_weight', 0.2):.2f}")
    
    # Fix 4: Final safety check - cap any individual weight to 0.5 maximum
    max_main_weight = 0.5
    if weights["semantic_weight"] > max_main_weight:
        if verbose:
            log_info(f"  [FIX] Capping semantic weight from {weights['semantic_weight']:.2f} to {max_main_weight:.2f}")
        weights["semantic_weight"] = max_main_weight
    if weights["activity_weight"] > max_main_weight:
        if verbose:
            log_info(f"  [FIX] Capping activity weight from {weights['activity_weight']:.2f} to {max_main_weight:.2f}")
        weights["activity_weight"] = max_main_weight
    if weights["hierarchical_weight"] > max_main_weight:
        if verbose:
            log_info(f"  [FIX] Capping hierarchical weight from {weights['hierarchical_weight']:.2f} to {max_main_weight:.2f}")
        weights["hierarchical_weight"] = max_main_weight


def configure_weights_with_fallback(
    strategy: Optional[Dict[str, Any]],
    query_intent: Dict[str, Any],
    all_object_classes: Set[str],
    search_plan: Dict[str, Any],
    configure_weights_fn,
    verbose: bool = False,
    log_info=None
) -> Dict[str, Any]:
    """
    Configure weights using strategy if available, otherwise fallback to configure_weights_fn.
    Applies all weight fixes consistently.
    
    Args:
        strategy: Optional strategy dict with scoring configuration
        query_intent: Query intent dictionary
        all_object_classes: Set of all object class names
        search_plan: Search plan dict
        configure_weights_fn: Fallback function to configure weights (from query_analysis)
        verbose: Whether to print verbose output
        log_info: Optional logging function
        
    Returns:
        Configured weights dictionary
    """
    if not log_info:
        log_info = print if verbose else lambda x: None
    
    if strategy and strategy.get("scoring"):
        # Use strategy-based weights
        weights = initialize_weights_from_strategy(
            strategy, all_object_classes, search_plan, verbose=verbose, log_info=log_info
        )
    else:
        # Fallback to old weight configuration
        weights = configure_weights_fn(query_intent, all_object_classes)
        # Ensure object_weight is set
        if "object_weight" not in weights:
            weights["object_weight"] = 0.2
        
        # Apply same fixes to fallback weights
        apply_weight_fixes(weights, search_plan, verbose=verbose, log_info=log_info)
    
    return weights

