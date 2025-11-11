"""Operation classification and parameter extraction for orchestrator agent."""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils import extract_json


def classify_operation(query: str, chunk_count: int, duration: float, llm, verbose: bool = False) -> Dict[str, Any]:
    """
    Classify user query into an operation type and extract parameters.
    
    Args:
        query: User query string
        chunk_count: Number of chunks in current timeline
        duration: Total duration of timeline in seconds
        llm: Language model instance
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with operation classification and parameters
    """
    from agent.orchestrator_prompts import ORCHESTRATOR_SYSTEM_PROMPT, OPERATION_CLASSIFICATION_PROMPT
    
    if verbose:
        print("\n[CLASSIFICATION] Analyzing user query...")
        print(f"  Query: {query}")
        print(f"  Timeline context: {chunk_count} chunks, {duration:.2f}s")
    
    system_prompt = ORCHESTRATOR_SYSTEM_PROMPT + "\n\n" + OPERATION_CLASSIFICATION_PROMPT
    
    user_message = OPERATION_CLASSIFICATION_PROMPT.format(
        query=query,
        chunk_count=chunk_count,
        duration=duration
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])
        
        response_text = response.content.strip()
        json_text = extract_json(response_text)
        result = json.loads(json_text)
        
        if verbose:
            print(f"  Operation: {result.get('operation')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  [FALLBACK] Classification failed: {e}, using heuristic")
        
        # Fallback to heuristic classification
        return _classify_operation_heuristic(query, chunk_count, verbose)


def _classify_operation_heuristic(query: str, chunk_count: int, verbose: bool = False) -> Dict[str, Any]:
    """
    Heuristic fallback for operation classification.
    
    Args:
        query: User query string
        chunk_count: Number of chunks in timeline
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with operation classification and parameters
    """
    query_lower = query.lower()
    
    # Extract timeline indices
    indices = _extract_timeline_indices(query, chunk_count)
    
    # Classify operation
    if any(kw in query_lower for kw in ["highlight", "best moment", "show me", "find"]):
        if "b-roll" in query_lower or "broll" in query_lower or "b roll" in query_lower:
            operation = "FIND_BROLL"
        else:
            operation = "FIND_HIGHLIGHTS"
    elif any(kw in query_lower for kw in ["cut", "remove", "delete"]):
        operation = "CUT"
    elif "replace" in query_lower:
        operation = "REPLACE"
    elif any(kw in query_lower for kw in ["add", "insert"]):
        operation = "INSERT"
    elif any(kw in query_lower for kw in ["move", "reorder"]):
        operation = "REORDER"
    elif any(kw in query_lower for kw in ["trim", "shorten"]):
        operation = "TRIM"
    else:
        operation = "UNKNOWN"
    
    # Extract search query if present
    search_query = _extract_search_query(query)
    
    # Extract insert position
    insert_position = None
    insert_before_index = None
    insert_after_index = None
    insert_between_indices = None
    
    if operation == "INSERT":
        if "between" in query_lower:
            insert_position = "between"
            # Try to extract two indices
            between_match = re.search(r'between\s+(?:timeline\s+)?(?:index\s+)?(\d+)\s+and\s+(?:timeline\s+)?(?:index\s+)?(\d+)', query_lower)
            if between_match:
                idx1, idx2 = int(between_match.group(1)), int(between_match.group(2))
                insert_between_indices = [min(idx1, idx2), max(idx1, idx2)]
        elif "before" in query_lower:
            insert_position = "before"
            before_match = re.search(r'before\s+(?:timeline\s+)?(?:index\s+)?(\d+)', query_lower)
            if before_match:
                insert_before_index = int(before_match.group(1))
        elif "after" in query_lower:
            insert_position = "after"
            after_match = re.search(r'after\s+(?:timeline\s+)?(?:index\s+)?(\d+)', query_lower)
            if after_match:
                insert_after_index = int(after_match.group(1))
    
    # Extract trim parameters
    trim_index = None
    trim_seconds = None
    trim_from = None
    trim_target_length = None
    remove_range = None
    
    if operation == "TRIM":
        # Try to extract index
        trim_match = re.search(r'(?:timeline\s+)?(?:index\s+)?(\d+)', query_lower)
        if trim_match:
            trim_index = int(trim_match.group(1))
        
        # Try to extract explicit "set to X sec" target length
        set_match = re.search(r'(?:set\s+(?:to|at)|to)\s+(\d+(?:\.\d+)?)\s*(?:second|sec|s)', query_lower)
        if set_match:
            try:
                trim_target_length = float(set_match.group(1))
            except:
                trim_target_length = None
        
        # Try to extract seconds (delta trim)
        seconds_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s)', query_lower)
        if seconds_match:
            trim_seconds = float(seconds_match.group(1))
        
        if "from start" in query_lower or "from beginning" in query_lower:
            trim_from = "start"
        elif "from end" in query_lower:
            trim_from = "end"
        else:
            trim_from = "end"  # Default
        
        # Remove internal range: "remove X seconds from the middle"
        if "remove" in query_lower and "middle" in query_lower:
            # If only a single X seconds specified, center it
            if seconds_match:
                # We'll compute centered removal in handler; here mark as request
                remove_range = {"start_offset": -1.0, "end_offset": -1.0, "center_length": float(seconds_match.group(1))}
        
        # Remove internal range: "remove between 2s and 5s" (offsets within the clip)
        between_match = re.search(r'remove.*?between\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\s*(?:and|to)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)', query_lower)
        if between_match:
            try:
                off_start = float(between_match.group(1))
                off_end = float(between_match.group(2))
                remove_range = {"start_offset": off_start, "end_offset": off_end}
            except:
                pass
    
    result = {
        "operation": operation,
        "confidence": 0.6,  # Lower confidence for heuristic
        "parameters": {
            "timeline_indices": indices,
            "insert_position": insert_position,
            "insert_before_index": insert_before_index,
            "insert_after_index": insert_after_index,
            "insert_between_indices": insert_between_indices,
            "search_query": search_query,
            "trim_index": trim_index,
            "trim_seconds": trim_seconds,
            "trim_from": trim_from,
            "trim_target_length": trim_target_length,
            "remove_range": remove_range
        },
        "reasoning": "Heuristic classification (LLM classification failed)"
    }
    
    if verbose:
        print(f"  [HEURISTIC] Operation: {operation}")
        print(f"  [HEURISTIC] Indices: {indices}")
    
    return result


def _extract_timeline_indices(query: str, chunk_count: int) -> List[int]:
    """
    Extract timeline indices from query using regex patterns.
    
    Args:
        query: User query string
        chunk_count: Number of chunks in timeline
        
    Returns:
        List of timeline indices
    """
    query_lower = query.lower()
    indices = []
    
    # Pattern 1: "first two", "first 3", etc. (check this first to avoid conflicts)
    # Handle both numeric and word numbers
    word_to_number = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    
    first_match = re.search(r'first\s+(\d+|\w+)', query_lower)
    if first_match:
        count_str = first_match.group(1)
        if count_str.isdigit():
            count = int(count_str)
        elif count_str in word_to_number:
            count = word_to_number[count_str]
        else:
            count = 0
        
        if count > 0:
            for i in range(min(count, chunk_count)):
                indices.append(i)
            return sorted(indices)  # Return early, don't process other patterns
    
    # Pattern 2: "last two", "last 3", etc.
    last_match = re.search(r'last\s+(\d+)', query_lower)
    if last_match:
        count = int(last_match.group(1))
        for i in range(max(0, chunk_count - count), chunk_count):
            indices.append(i)
        return sorted(indices)  # Return early, don't process other patterns
    
    # Pattern 3: "0-2" or "0 to 2" or "index 0 to 4" (range) - check this before single indices
    # More flexible pattern to catch "timeline index 0 to 4"
    pattern2 = r'(?:timeline\s+)?(?:index\s+)?(\d+)\s+(?:to|and|-)\s+(?:timeline\s+)?(?:index\s+)?(\d+)'
    range_matches = re.findall(pattern2, query_lower)
    if range_matches:
        # Use range matches, clear any single index matches
        indices = []
        for start, end in range_matches:
            start_idx = int(start)
            end_idx = int(end)
            for idx in range(start_idx, end_idx + 1):
                if 0 <= idx < chunk_count and idx not in indices:
                    indices.append(idx)
        if indices:  # Only return if we found valid indices
            return sorted(indices)  # Return early if range found
    
    # Also check for dash pattern: "0-2"
    dash_pattern = r'(\d+)\s*-\s*(\d+)'
    dash_matches = re.findall(dash_pattern, query_lower)
    if dash_matches:
        indices = []
        for start, end in dash_matches:
            start_idx = int(start)
            end_idx = int(end)
            for idx in range(start_idx, end_idx + 1):
                if 0 <= idx < chunk_count and idx not in indices:
                    indices.append(idx)
        if indices:
            return sorted(indices)
    
    # Pattern 4: "timeline index 0" or "index 0" or "0 and 1" (single indices)
    # Handle "and" pattern: "index 0 and 1"
    and_pattern = r'(?:timeline\s+)?(?:index\s+)?(\d+)(?:\s+and\s+(?:timeline\s+)?(?:index\s+)?(\d+))+'
    and_match = re.search(and_pattern, query_lower)
    if and_match:
        # Extract all numbers from "X and Y and Z" pattern
        all_numbers = re.findall(r'(?:timeline\s+)?(?:index\s+)?(\d+)', query_lower)
        for num_str in all_numbers:
            idx = int(num_str)
            if 0 <= idx < chunk_count and idx not in indices:
                indices.append(idx)
        return sorted(indices)
    
    # Pattern 5: Single index mentions
    pattern1 = r'(?:timeline\s+)?(?:index\s+)?(\d+)'
    matches = re.findall(pattern1, query_lower)
    for match in matches:
        idx = int(match)
        if 0 <= idx < chunk_count and idx not in indices:
            indices.append(idx)
    
    # Remove duplicates and sort
    indices = sorted(list(set(indices)))
    
    return indices


def _extract_search_query(query: str) -> Optional[str]:
    """
    Extract search query from user input.
    
    Args:
        query: User query string
        
    Returns:
        Search query string or None
    """
    query_lower = query.lower()
    
    # Pattern 1: "replace ... with [query]"
    replace_match = re.search(r'replace.*?with\s+(.+)', query_lower)
    if replace_match:
        return replace_match.group(1).strip()
    
    # Pattern 2: "find [query]"
    find_match = re.search(r'find\s+(.+)', query_lower)
    if find_match:
        return find_match.group(1).strip()
    
    # Pattern 3: "add [query]"
    add_match = re.search(r'add\s+(?:a\s+)?(?:clip\s+of\s+)?(.+)', query_lower)
    if add_match:
        return add_match.group(1).strip()
    
    # Pattern 4: "show me [query]"
    show_match = re.search(r'show\s+me\s+(.+)', query_lower)
    if show_match:
        return show_match.group(1).strip()
    
    # If query contains "highlight" or similar, return the full query
    if any(kw in query_lower for kw in ["highlight", "best", "moment"]):
        return query
    
    return None


def validate_operation_params(operation: str, params: Dict, chunk_count: int, verbose: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate operation parameters.
    
    Args:
        operation: Operation type
        params: Operation parameters
        chunk_count: Number of chunks in timeline
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if operation == "CUT" or operation == "REPLACE" or operation == "FIND_BROLL":
        indices = params.get("timeline_indices", [])
        if not indices:
            return False, f"{operation} requires timeline indices"
        for idx in indices:
            if not isinstance(idx, int) or idx < 0 or idx >= chunk_count:
                return False, f"Invalid timeline index: {idx} (timeline has {chunk_count} chunks)"
    
    elif operation == "INSERT":
        # Must have one of: insert_before_index, insert_after_index, or insert_between_indices
        has_position = (
            params.get("insert_before_index") is not None or
            params.get("insert_after_index") is not None or
            params.get("insert_between_indices") is not None
        )
        if not has_position:
            return False, "INSERT requires insert position (before, after, or between indices)"
    
    elif operation == "REORDER":
        from_idx = params.get("reorder_from_index")
        to_idx = params.get("reorder_to_index")
        if from_idx is None or to_idx is None:
            return False, "REORDER requires both from_index and to_index"
        if from_idx < 0 or from_idx >= chunk_count:
            return False, f"Invalid reorder_from_index: {from_idx}"
        if to_idx < 0 or to_idx >= chunk_count:
            return False, f"Invalid reorder_to_index: {to_idx}"
    
    elif operation == "TRIM":
        trim_index = params.get("trim_index")
        if trim_index is None:
            return False, "TRIM requires trim_index"
        if trim_index < 0 or trim_index >= chunk_count:
            return False, f"Invalid trim_index: {trim_index}"
        
        # At least one trim spec must be provided
        has_delta = params.get("trim_seconds") is not None
        has_target = params.get("trim_target_length") is not None
        has_remove = params.get("remove_range") is not None
        if not (has_delta or has_target or has_remove):
            # Legacy simple phrasing like "trim the first clip" without numbers is ambiguous
            return False, "TRIM requires trim_seconds, trim_target_length, or remove_range"
    
    return True, None

