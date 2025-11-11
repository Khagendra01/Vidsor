"""Query analysis and semantic understanding functions."""

import json
from typing import Dict, Any, Set, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm_utils import parse_json_response, invoke_llm_with_json


def analyze_query_semantics(query: str, llm) -> Dict[str, Any]:
    """
    Agentic semantic query analyzer - understands query meaning, not just keywords.
    Replaces heuristic-based intent detection with LLM-driven semantic understanding.
    """
    system_prompt = """You are a semantic query analyzer for video search. Your job is to deeply understand what the user is asking for, not just extract keywords.

Analyze the query and determine:

1. QUERY TYPE (choose the most specific):
   - POSITIVE: Find moments WITH something (e.g., "find person", "show fish")
   - NEGATIVE: Find moments WITHOUT something (e.g., "find moments without person", "no people")
   - TEMPORAL: Find moments relative to time/events (e.g., "after they catch fish", "before sunset")
   - CONDITIONAL: Find moments with multiple conditions (e.g., "person AND fish", "person but no boat")
   - COMPARATIVE: Find moments with comparisons (e.g., "more fish than people", "longest clip")
   - AGGREGATE: Find all/any/some instances (e.g., "all moments", "any time")

2. TARGET ENTITIES: What are we searching for?
   - Objects: ["person", "fish", "boat"]
   - Activities: ["catching", "running", "cooking"]
   - Concepts: ["happiness", "conflict", "nature"]

3. CONSTRAINTS: Any conditions or filters?
   - Duration: "longer than 5 seconds"
   - Quality: "best", "most relevant"
   - Count: "top 10", "first 5"

4. MODALITIES NEEDED: Which search types are relevant?
   - object_detection: For object queries
   - activity_detection: For action/activity queries
   - semantic_search: For concept/description queries
   - temporal_reasoning: For before/after queries
   - hierarchical_search: For keyword-based fast lookup

5. SPECIAL HANDLING: Does this query need custom logic?
   - negation: true if query contains "not", "without", "no", "absence"
   - temporal: true if query contains "before", "after", "during", "while"
   - conditional: true if query contains "and", "but", "or", "except"
   - counting: true if query contains numbers or "top", "first", "best"

6. SEARCH INTENT: What's the primary focus?
   - object: Finding specific objects
   - activity: Finding specific actions/activities
   - semantic: Finding concepts/descriptions
   - hybrid: Combination of above

Return JSON:
{
    "query_type": "POSITIVE" | "NEGATIVE" | "TEMPORAL" | "CONDITIONAL" | "COMPARATIVE" | "AGGREGATE",
    "target_entities": {
        "objects": ["person", "fish"],
        "activities": ["catching"],
        "concepts": []
    },
    "constraints": {
        "duration": null or {"min": 5, "max": null},
        "quality": null or "best" | "most_relevant",
        "count": null or 10
    },
    "modalities": {
        "object_detection": true/false,
        "activity_detection": true/false,
        "semantic_search": true/false,
        "temporal_reasoning": true/false,
        "hierarchical_search": true/false
    },
    "special_handling": {
        "negation": true/false,
        "temporal": true/false,
        "conditional": true/false,
        "counting": true/false
    },
    "search_intent": "object" | "activity" | "semantic" | "hybrid",
    "object_priority": {"person": 1.0},  // For object classes mentioned
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of your analysis"
}"""
    
    # Fallback dict with default values
    fallback_result = {
        "query_type": "POSITIVE",
        "target_entities": {"objects": [], "activities": [], "concepts": []},
        "constraints": {},
        "modalities": {},
        "special_handling": {},
        "object_priority": {},
        "confidence": 0.5,
        "reasoning": "Fallback due to parsing error"
    }
    
    try:
        result = invoke_llm_with_json(
            llm=llm,
            system_prompt=system_prompt,
            user_message=f"Query: {query}\n\nPerform deep semantic analysis. Understand the true meaning, not just keywords. Return JSON only.",
            fallback=fallback_result
        )
        
        # Ensure all required fields exist with defaults
        if "confidence" not in result:
            result["confidence"] = 0.7
        if "target_entities" not in result:
            result["target_entities"] = {"objects": [], "activities": [], "concepts": []}
        if "constraints" not in result:
            result["constraints"] = {}
        if "modalities" not in result:
            result["modalities"] = {}
        if "special_handling" not in result:
            result["special_handling"] = {}
        if "object_priority" not in result:
            result["object_priority"] = {}
        
        return result
    except Exception as e:
        # Fallback: basic heuristic detection
        query_lower = query.lower()
        
        # Detect negation
        negation_keywords = ["not", "without", "no", "absence", "lack", "missing"]
        has_negation = any(kw in query_lower for kw in negation_keywords)
        
        # Detect temporal
        temporal_keywords = ["before", "after", "during", "while", "when", "then"]
        has_temporal = any(kw in query_lower for kw in temporal_keywords)
        
        # Extract objects
        mentioned_classes = []
        common_classes = ["person", "car", "fish", "boat", "dog", "cat", "bird", "bike", "truck"]
        for cls in common_classes:
            if cls in query_lower:
                mentioned_classes.append(cls)
        
        # Determine query type
        if has_negation:
            query_type = "NEGATIVE"
        elif has_temporal:
            query_type = "TEMPORAL"
        else:
            query_type = "POSITIVE"
        
        return {
            "query_type": query_type,
            "target_entities": {
                "objects": mentioned_classes,
                "activities": [],
                "concepts": []
            },
            "constraints": {},
            "modalities": {
                "object_detection": bool(mentioned_classes),
                "activity_detection": False,
                "semantic_search": True,
                "temporal_reasoning": has_temporal,
                "hierarchical_search": True
            },
            "special_handling": {
                "negation": has_negation,
                "temporal": has_temporal,
                "conditional": False,
                "counting": False
            },
            "search_intent": "object" if mentioned_classes else "hybrid",
            "object_priority": {cls: 1.0 for cls in mentioned_classes},
            "confidence": 0.5,  # Low confidence for fallback
            "reasoning": "Fallback heuristic analysis"
        }


def analyze_query_intent(query: str, llm) -> Dict[str, Any]:
    """
    Legacy function - now calls semantic analyzer and converts to old format for compatibility.
    """
    semantic_analysis = analyze_query_semantics(query, llm)
    
    # Convert to old format for backward compatibility
    is_object_centric = semantic_analysis.get("search_intent") == "object"
    mentioned_classes = semantic_analysis.get("target_entities", {}).get("objects", [])
    
    return {
        "is_object_centric": is_object_centric,
        "mentioned_classes": mentioned_classes,
        "primary_intent": semantic_analysis.get("search_intent", "hybrid"),
        "object_priority": semantic_analysis.get("object_priority", {}),
        "needs_semantic": semantic_analysis.get("modalities", {}).get("semantic_search", True),
        "needs_activity": semantic_analysis.get("modalities", {}).get("activity_detection", False),
        "needs_hierarchical": semantic_analysis.get("modalities", {}).get("hierarchical_search", True),
        "confidence": semantic_analysis.get("confidence", 0.7),
        # New fields from semantic analysis
        "_semantic_analysis": semantic_analysis  # Store full analysis for strategy planner
    }


def validate_and_adjust_intent(query_intent: Dict, search_plan: Dict, verbose: bool = False) -> Dict[str, Any]:
    """
    Cross-validate query intent with search plan and adjust if there's a conflict.
    This catches cases where LLM misclassified but search plan generator got it right.
    """
    adjusted_intent = query_intent.copy()
    
    # Check if search plan indicates activity but intent says object-centric
    has_activity_in_plan = bool(
        search_plan.get("activity_name") or 
        search_plan.get("activity_keywords") or
        search_plan.get("evidence_keywords")
    )
    
    # Check if search plan has semantic queries (indicates semantic intent)
    has_semantic_in_plan = bool(search_plan.get("semantic_queries"))
    
    # Conflict detection
    conflicts = []
    
    # Conflict 1: Search plan has activity but intent is object-centric
    if has_activity_in_plan and adjusted_intent.get("is_object_centric") and adjusted_intent.get("primary_intent") == "object":
        conflicts.append("Search plan has activity keywords but intent is object-centric")
        if verbose:
            print(f"  [CONFLICT DETECTED] {conflicts[-1]}")
        # Override: This is likely activity-centric or hybrid
        if adjusted_intent.get("confidence", 0.7) < 0.8:  # Only override if low confidence
            adjusted_intent["is_object_centric"] = False
            adjusted_intent["primary_intent"] = "activity" if has_activity_in_plan else "hybrid"
            adjusted_intent["needs_activity"] = True
            adjusted_intent["needs_semantic"] = True
            if verbose:
                print(f"  [ADJUSTED] Changed to {adjusted_intent['primary_intent']} intent")
    
    # Conflict 2: Search plan has semantic queries but intent disabled semantic
    if has_semantic_in_plan and not adjusted_intent.get("needs_semantic"):
        conflicts.append("Search plan has semantic queries but intent disabled semantic")
        if verbose:
            print(f"  [CONFLICT DETECTED] {conflicts[-1]}")
        # Enable semantic
        adjusted_intent["needs_semantic"] = True
        if adjusted_intent.get("primary_intent") == "object":
            adjusted_intent["primary_intent"] = "hybrid"
        if verbose:
            print(f"  [ADJUSTED] Enabled semantic search")
    
    # Conflict 3: Low confidence + ambiguous → default to hybrid
    if adjusted_intent.get("confidence", 0.7) < 0.6 and not conflicts:
        # If confidence is low and no clear conflicts, default to hybrid for safety
        if adjusted_intent.get("primary_intent") in ["object", "activity"]:
            if verbose:
                print(f"  [LOW CONFIDENCE] Defaulting to hybrid for safety")
            adjusted_intent["primary_intent"] = "hybrid"
            adjusted_intent["is_object_centric"] = False
            adjusted_intent["needs_semantic"] = True
            adjusted_intent["needs_activity"] = True
    
    if verbose and conflicts:
        print(f"  [VALIDATION] Found {len(conflicts)} conflict(s), adjusted intent")
    elif verbose:
        print(f"  [VALIDATION] No conflicts detected")
    
    return adjusted_intent


def plan_search_strategy(semantic_analysis: Dict, llm, verbose: bool = False) -> Dict[str, Any]:
    """
    Agentic dynamic strategy planner - generates custom search strategy based on semantic analysis.
    Replaces fixed search plan generation with LLM-driven strategy planning.
    """
    query_type = semantic_analysis.get("query_type", "POSITIVE")
    target_entities = semantic_analysis.get("target_entities", {})
    special_handling = semantic_analysis.get("special_handling", {})
    modalities = semantic_analysis.get("modalities", {})
    
    system_prompt = """You are a video search strategy planner. Based on the semantic analysis of a query, generate a custom search strategy.

Available search operations:
1. object_detection: Search for object classes using YOLO detections
2. semantic_search: Search using embeddings (visual descriptions + audio transcriptions)
3. activity_detection: Search for activities/actions using pattern matching
4. hierarchical_search: Fast keyword-based tree lookup
5. temporal_reasoning: Find moments relative to other events (before/after)

Special operations for complex queries:
- NEGATIVE queries: Invert object detection results (find seconds WITHOUT objects)
- CONDITIONAL queries: Combine multiple searches with AND/OR logic
- TEMPORAL queries: Use temporal reasoning to find moments relative to events

Generate a strategy JSON:
{
    "search_operations": [
        {
            "type": "object_detection" | "semantic_search" | "activity_detection" | "hierarchical_search" | "temporal_reasoning",
            "params": {
                // Type-specific parameters
                // For object_detection: {"classes": ["person"], "invert": true/false}
                // For semantic_search: {"queries": ["..."], "threshold": 0.3}
                // For activity_detection: {"activity_name": "...", "keywords": [...]}
                // For hierarchical_search: {"keywords": [...]}
            },
            "weight": 0.0-1.0,  // How much this operation contributes to final score
            "required": true/false  // Must succeed for query to work
        }
    ],
    "scoring": {
        "formula": "description of how to combine operation results",
        "weights": {
            "semantic": 0.4,
            "activity": 0.3,
            "hierarchical": 0.1,
            "object": 0.2  # Weight for object scores (not per-class)
        },
        "threshold": 0.5,  # Increased default threshold
        "object_weights": {"person": 1.0}  // Per-class weights
    },
    "post_processing": [
        "filter_by_duration" | "remove_overlaps" | "rank_by_relevance" | "invert_results"
    ],
    "reasoning": "brief explanation of why this strategy fits the query"
}"""
    
    analysis_summary = f"""
Query Type: {query_type}
Target Entities: {target_entities}
Special Handling: {special_handling}
Modalities Needed: {modalities}
"""
    
    # Fallback strategy
    fallback_strategy = {
        "search_operations": [],
        "scoring": {},
        "post_processing": [],
        "reasoning": "Fallback strategy due to parsing error"
    }
    
    try:
        strategy = invoke_llm_with_json(
            llm=llm,
            system_prompt=system_prompt,
            user_message=f"Semantic Analysis:\n{analysis_summary}\n\nGenerate a search strategy that handles this query type. Return JSON only.",
            fallback=fallback_strategy
        )
        
        # Validate and set defaults
        if "search_operations" not in strategy:
            strategy["search_operations"] = []
        if "scoring" not in strategy:
            strategy["scoring"] = {}
        if "post_processing" not in strategy:
            strategy["post_processing"] = []
        
        return strategy
    except Exception as e:
        # Fallback: generate basic strategy from semantic analysis
        if verbose:
            print(f"  [FALLBACK] Strategy planning failed, using default strategy")
        
        search_operations = []
        
        # Add object detection if needed
        if modalities.get("object_detection") and target_entities.get("objects"):
            invert = special_handling.get("negation", False)
            search_operations.append({
                "type": "object_detection",
                "params": {
                    "classes": target_entities["objects"],
                    "invert": invert
                },
                "weight": 1.0 if query_type == "NEGATIVE" and invert else 0.8,
                "required": True
            })
        
        # Add semantic search if needed
        if modalities.get("semantic_search"):
            search_operations.append({
                "type": "semantic_search",
                "params": {
                    "queries": [],  # Will be filled by search plan generator
                    "threshold": 0.3
                },
                "weight": 0.4,
                "required": False
            })
        
        # Add activity detection if needed
        if modalities.get("activity_detection") and target_entities.get("activities"):
            search_operations.append({
                "type": "activity_detection",
                "params": {
                    "activity_name": target_entities["activities"][0] if target_entities["activities"] else "",
                    "keywords": target_entities["activities"]
                },
                "weight": 0.3,
                "required": False
            })
        
        # Default scoring
        scoring = {
            "weights": {
                "semantic": 0.4 if modalities.get("semantic_search") else 0.0,
                "activity": 0.3 if modalities.get("activity_detection") else 0.0,
                "hierarchical": 0.1 if modalities.get("hierarchical_search") else 0.0,
                "object": 0.2 if modalities.get("object_detection") else 0.0  # Weight for object scores
            },
            "threshold": 0.5,  # Increased default threshold
            "object_weights": semantic_analysis.get("object_priority", {})
        }
        
        post_processing = []
        if special_handling.get("negation"):
            post_processing.append("invert_results")
        post_processing.extend(["remove_overlaps", "rank_by_relevance"])
        
        return {
            "search_operations": search_operations,
            "scoring": scoring,
            "post_processing": post_processing,
            "reasoning": "Fallback strategy based on semantic analysis"
        }


def configure_weights(query_intent: Dict, all_object_classes: Set[str]) -> Dict[str, Any]:
    """
    Configure scoring weights based on query intent.
    
    Returns:
    {
        "semantic_weight": 0.4,
        "activity_weight": 0.3,
        "hierarchical_weight": 0.1,
        "object_weights": {"person": 0.1, "fish": 0.8},
        "threshold": 0.3
    }
    """
    weights = {
        "semantic_weight": 0.0,
        "activity_weight": 0.0,
        "hierarchical_weight": 0.0,
        "object_weight": 0.2,  # NEW: Weight for object scores (not per-class)
        "object_weights": {},
        "threshold": 0.5  # Increased from 0.3 to be more selective
    }
    
    # Initialize all object classes with default low weight
    for class_name in all_object_classes:
        weights["object_weights"][class_name] = 0.1  # Default low
    
    # If object-centric query
    if query_intent.get("is_object_centric"):
        # Set high weights for mentioned classes
        for class_name, priority in query_intent.get("object_priority", {}).items():
            weights["object_weights"][class_name] = priority
        
        # Lower threshold for pure object queries (but still higher than before)
        weights["threshold"] = 0.3  # Increased from 0.1
        weights["semantic_weight"] = 0.0
        weights["activity_weight"] = 0.0
        weights["hierarchical_weight"] = 0.0
    
    # If hybrid query
    elif query_intent.get("primary_intent") == "hybrid":
        # Balance all modalities
        weights["semantic_weight"] = 0.3
        weights["activity_weight"] = 0.3
        weights["hierarchical_weight"] = 0.1
        
        # Boost mentioned object classes
        for class_name, priority in query_intent.get("object_priority", {}).items():
            weights["object_weights"][class_name] = priority
    
    # If activity/semantic query (default)
    else:
        weights["semantic_weight"] = 0.4
        weights["activity_weight"] = 0.3
        weights["hierarchical_weight"] = 0.1
        
        # Only boost rare/mentioned objects
        for class_name, priority in query_intent.get("object_priority", {}).items():
            weights["object_weights"][class_name] = priority
    
    return weights

