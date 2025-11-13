"""Operation classification and parameter extraction for orchestrator agent."""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm_utils import parse_json_response, invoke_llm_with_json


def _clean_query(query: str) -> str:
    """
    Clean user query by removing clip references like "clip5", "@clip5", etc.
    Handles clip references anywhere in the text and in various formats.
    
    Clip references are UI metadata (like @clip1, clip5) that users might include
    when referring to timeline chunks. These should be removed so the actual
    command can be processed correctly.
    
    Examples:
    - "clip5 Make timeline start zoomed" → "Make timeline start zoomed"
    - "Make @clip4 start zoomed in" → "Make start zoomed in"
    - "find clip moments" → "find clip moments" (NOT removed - "clip" not followed by digits)
    - "clip number 3 make it shorter" → "make it shorter"
    
    Args:
        query: Raw user query
        
    Returns:
        Cleaned query string
    """
    # Patterns to match various clip reference formats:
    # - @clip5, @clip 5, @clip-5 (with @ symbol)
    # - clip5, clip 5, clip-5 (without @)
    # - clip number 5, clip #5 (with "number" or "#")
    # - the 5th clip, 5th clip (ordinal format)
    # Works anywhere in text (start, middle, end)
    
    # Pattern 1: @clip followed by optional space/dash and digits (anywhere in text)
    # Matches: @clip5, @clip 5, @clip-5, @CLIP5, etc.
    # Word boundary ensures we don't match "@clipping" or similar
    cleaned = re.sub(r'@clip\s*-?\s*\d+\b', '', query, flags=re.IGNORECASE)
    
    # Pattern 2: "clip" followed by optional space/dash and digits (standalone word)
    # Matches: clip5, clip 5, clip-5, but NOT "clip moments" or "clip editing"
    # Word boundaries (\b) ensure "clip" is a complete word, and \d+ ensures digits follow
    # This safely distinguishes "clip5" (reference) from "clip moments" (command)
    cleaned = re.sub(r'\bclip\s*-?\s*\d+\b', '', cleaned, flags=re.IGNORECASE)
    
    # Pattern 3: "clip number X", "clip #X" (explicit number format)
    cleaned = re.sub(r'\bclip\s+number\s+\d+\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bclip\s+#\s*\d+\b', '', cleaned, flags=re.IGNORECASE)
    
    # Pattern 4: "the Xth clip", "Xth clip" (ordinal format)
    cleaned = re.sub(r'\bthe\s+\d+th\s+clip\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d+th\s+clip\b', '', cleaned, flags=re.IGNORECASE)
    
    # Clean up multiple spaces that might result from removals
    # Replace 2+ spaces with single space, trim edges
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Return cleaned query, or original if cleaning resulted in empty string
    return cleaned if cleaned else query


def _extract_clip_references(query: str) -> List[Dict[str, Any]]:
    """
    Extract clip references (clip numbers) from the raw query text.

    Returns:
        List of dicts with keys:
            - raw: original matched text
            - label: normalized label (e.g., "clip3")
            - clip_number: 1-based clip number as int
            - index: 0-based timeline index (clip_number - 1)
    """
    references: List[Dict[str, Any]] = []
    seen_indices: set[int] = set()

    def add_reference(raw_text: str, clip_number: int):
        index = clip_number - 1
        if index < 0:
            return
        if index in seen_indices:
            return
        seen_indices.add(index)
        references.append(
            {
                "raw": raw_text.strip(),
                "label": f"clip{clip_number}",
                "clip_number": clip_number,
                "index": index,
            }
        )

    range_pattern = re.compile(
        r'\bclip\s*(?:number\s*)?(\d+)\s*(?:to|through|-)\s*(\d+)\b',
        re.IGNORECASE,
    )
    for match in range_pattern.finditer(query):
        start = int(match.group(1))
        end = int(match.group(2))
        if start <= end:
            numbers = range(start, end + 1)
        else:
            numbers = range(start, end - 1, -1)
        for num in numbers:
            add_reference(match.group(0), num)

    multi_pattern = re.compile(
        r'\bclip\s*(?:number\s*)?((?:\d+\s*,\s*)+\d+)\b',
        re.IGNORECASE,
    )
    for match in multi_pattern.finditer(query):
        segment = match.group(1)
        for num_text in re.findall(r'\d+', segment):
            add_reference(match.group(0), int(num_text))

    and_pattern = re.compile(
        r'\bclip\s*(?:number\s*)?(\d+)\s+(?:and|&)\s+(\d+)\b',
        re.IGNORECASE,
    )
    for match in and_pattern.finditer(query):
        add_reference(match.group(0), int(match.group(1)))
        add_reference(match.group(0), int(match.group(2)))

    direct_pattern = re.compile(r'@?clip\s*-?\s*(\d+)\b', re.IGNORECASE)
    for match in direct_pattern.finditer(query):
        add_reference(match.group(0), int(match.group(1)))

    hash_pattern = re.compile(r'\bclip\s*#\s*(\d+)\b', re.IGNORECASE)
    for match in hash_pattern.finditer(query):
        add_reference(match.group(0), int(match.group(1)))

    number_pattern = re.compile(r'\bclip\s+number\s+(\d+)\b', re.IGNORECASE)
    for match in number_pattern.finditer(query):
        add_reference(match.group(0), int(match.group(1)))

    ordinal_pattern = re.compile(r'\b(\d+)(?:st|nd|rd|th)\s+clip\b', re.IGNORECASE)
    for match in ordinal_pattern.finditer(query):
        add_reference(match.group(0), int(match.group(1)))

    return references


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
    from agent.prompts.orchestrator_prompts import ORCHESTRATOR_SYSTEM_PROMPT, OPERATION_CLASSIFICATION_PROMPT
    
    original_query = query
    clip_references = _extract_clip_references(original_query)
    clip_reference_indices = [ref["index"] for ref in clip_references]
    
    if verbose:
        print("\n[CLASSIFICATION] Analyzing user query...")
        print(f"  Query: {original_query}")
        print(f"  Timeline context: {chunk_count} chunks, {duration:.2f}s")
        if clip_references:
            mapped = ", ".join(f"{ref['label']}→{ref['index']}" for ref in clip_references)
            print(f"  Detected clip references: {mapped}")
    
    system_prompt = ORCHESTRATOR_SYSTEM_PROMPT + "\n\n" + OPERATION_CLASSIFICATION_PROMPT
    
    user_message = OPERATION_CLASSIFICATION_PROMPT.format(
        query=original_query,
        chunk_count=chunk_count,
        duration=duration
    )

    if clip_references:
        reference_lines = [
            f"- {ref['label']} (raw: \"{ref['raw']}\") → timeline index {ref['index']}"
            for ref in clip_references
        ]
        user_message += (
            "\nDetected clip references (timeline indices are 0-based):\n"
            + "\n".join(reference_lines)
            + "\nPrefer these indices if they align with the user's intent."
        )
    
    fallback_result = {
        "operation": "UNKNOWN",
        "confidence": 0.0,
        "parameters": {},
        "reasoning": "Fallback due to parsing error"
    }
    
    clip_indices_valid = [
        idx for idx in clip_reference_indices if 0 <= idx < chunk_count
    ]

    try:
        result = invoke_llm_with_json(
            llm=llm,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            user_message=user_message,
            fallback=fallback_result
        )

        if clip_indices_valid:
            parameters = result.setdefault("parameters", {})
            existing_indices = parameters.get("timeline_indices") or []

            combined_indices: List[int] = []
            for idx in existing_indices:
                if isinstance(idx, int) and 0 <= idx < chunk_count and idx not in combined_indices:
                    combined_indices.append(idx)
            for idx in clip_indices_valid:
                if idx not in combined_indices:
                    combined_indices.append(idx)

            if combined_indices:
                parameters["timeline_indices"] = combined_indices
                if verbose:
                    print(f"  Applied clip references → timeline_indices: {combined_indices}")
        
        if verbose:
            print(f"  Operation: {result.get('operation')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  [FALLBACK] Classification failed: {e}, using heuristic")
        
        # Fallback to heuristic classification
        return _classify_operation_heuristic(
            original_query,
            chunk_count,
            clip_indices=clip_indices_valid,
            verbose=verbose
        )


def _classify_operation_heuristic(
    query: str,
    chunk_count: int,
    clip_indices: Optional[List[int]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Heuristic fallback for operation classification.
    
    Args:
        query: User query string
        chunk_count: Number of chunks in timeline
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with operation classification and parameters
    """
    # Clean query by removing clip prefixes
    query = _clean_query(query)
    query_lower = query.lower()
    
    # Extract timeline indices
    indices = _extract_timeline_indices(query, chunk_count)
    clip_indices = clip_indices or []
    for clip_idx in clip_indices:
        if 0 <= clip_idx < chunk_count and clip_idx not in indices:
            indices.append(clip_idx)
    indices = sorted(indices)
    
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
    elif any(kw in query_lower for kw in ["zoom", "effect", "apply"]):
        if any(kw in query_lower for kw in ["zoom", "zoomed"]):
            operation = "APPLY_EFFECT"
        else:
            operation = "UNKNOWN"
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
    
    # Extract effect parameters
    effect_type = None
    effect_object = None
    effect_duration = None
    
    if operation == "APPLY_EFFECT":
        # Extract effect type (zoom in/out, etc.)
        if "zoom" in query_lower:
            if "zoom in" in query_lower or "zoomed in" in query_lower:
                effect_type = "zoom_in"
            elif "zoom out" in query_lower or "zoomed out" in query_lower:
                effect_type = "zoom_out"
            elif "zoom" in query_lower:
                # Default to zoom in then out
                effect_type = "zoom_in_to_out"
        
        # Extract object name (man, plane, person, etc.)
        # Common object patterns
        object_patterns = [
            r'(?:on|to|the)\s+(man|woman|person|people|plane|airplane|car|truck|boat|dog|cat|bird)',
            r'(?:zoom|effect).*?(?:on|to|the)\s+(\w+)',
        ]
        for pattern in object_patterns:
            match = re.search(pattern, query_lower)
            if match:
                effect_object = match.group(1)
                break
        
        # Extract duration (default 1.0 second)
        duration_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s)', query_lower)
        if duration_match:
            effect_duration = float(duration_match.group(1))
        else:
            effect_duration = 1.0  # Default
    
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
            "remove_range": remove_range,
            "effect_type": effect_type,
            "effect_object": effect_object,
            "effect_duration": effect_duration
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
    
    # Pattern 0: "timeline start", "start", "beginning" - means index 0
    # Check various ways to refer to the first/timeline start
    start_phrases = [
        "timeline start", "timeline beginning", 
        "at start", "at beginning",
        "make timeline start", "make start",
        "the start", "the beginning",
        "from start", "from beginning"
    ]
    if any(phrase in query_lower for phrase in start_phrases):
        if chunk_count > 0:
            indices.append(0)
            return sorted(indices)
    
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
    
    # Helper function to check if extracted query is a general highlight query
    def is_general_highlight_query(text: str) -> bool:
        """Check if text is just a general highlight query without specific content."""
        text_lower = text.lower().strip()
        highlight_keywords = ["highlight", "highlights", "best moment", "best moments", 
                            "important moment", "important moments", "key moment", "key moments"]
        highlight_words = ["highlight", "highlights", "moment", "moments", "best", "important", "key"]
        
        # Check if it contains highlight keywords/phrases
        has_highlight_kw = any(kw in text_lower for kw in highlight_keywords)
        if not has_highlight_kw:
            return False
        
        # Check if there's specific content mentioned (e.g., "highlights of fishing" has "fishing")
        # Remove common filler words and highlight-related words
        text_clean = re.sub(r'\b(all|the|of|in|video|clip|clips|moments?|scenes?|highlight|highlights|best|important|key)\b', '', text_lower).strip()
        
        # If after removing fillers and highlight words, there's nothing left or only very short words, it's general
        # Also check for patterns like "highlights of X" or "best moments with X" - these have specific content
        has_specific_content = bool(re.search(r'(?:highlight|moment).*?(?:of|with|featuring|showing)\s+\w+', text_lower))
        if has_specific_content:
            return False  # Has specific content, not general
        
        # If text_clean is empty or only has very short words (< 3 chars), it's a general highlight query
        remaining_words = [w for w in text_clean.split() if len(w) >= 3]
        return len(remaining_words) == 0
    
    # Pattern 1: "replace ... with [query]"
    replace_match = re.search(r'replace.*?with\s+(.+)', query_lower)
    if replace_match:
        extracted = replace_match.group(1).strip()
        if not is_general_highlight_query(extracted):
            return extracted
    
    # Pattern 2: "find [query]"
    find_match = re.search(r'find\s+(.+)', query_lower)
    if find_match:
        extracted = find_match.group(1).strip()
        # Check if it's a general highlight query
        if is_general_highlight_query(extracted):
            # Check if there's specific content mentioned (e.g., "highlights of fishing")
            specific_match = re.search(r'(?:highlight|moment).*?(?:of|with|featuring|showing)\s+(.+)', extracted)
            if specific_match:
                return specific_match.group(1).strip()
            else:
                # General highlight query - return None
                return None
        return extracted
    
    # Pattern 3: "add [query]"
    add_match = re.search(r'add\s+(?:a\s+)?(?:clip\s+of\s+)?(.+)', query_lower)
    if add_match:
        extracted = add_match.group(1).strip()
        if not is_general_highlight_query(extracted):
            return extracted
    
    # Pattern 4: "show me [query]"
    show_match = re.search(r'show\s+me\s+(.+)', query_lower)
    if show_match:
        extracted = show_match.group(1).strip()
        if is_general_highlight_query(extracted):
            # Check for specific content
            specific_match = re.search(r'(?:highlight|moment).*?(?:of|with|featuring|showing)\s+(.+)', extracted)
            if specific_match:
                return specific_match.group(1).strip()
            else:
                return None
        return extracted
    
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
    
    elif operation == "APPLY_EFFECT":
        indices = params.get("timeline_indices", [])
        if not indices:
            return False, "APPLY_EFFECT requires timeline indices"
        for idx in indices:
            if not isinstance(idx, int) or idx < 0 or idx >= chunk_count:
                return False, f"Invalid timeline index: {idx} (timeline has {chunk_count} chunks)"
        
        effect_type = params.get("effect_type")
        if not effect_type:
            return False, "APPLY_EFFECT requires effect_type"
        
        effect_object = params.get("effect_object")
        if not effect_object:
            return False, "APPLY_EFFECT requires effect_object"
    
    return True, None

