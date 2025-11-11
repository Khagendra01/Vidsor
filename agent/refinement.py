"""Refinement and validation functions for search results."""

import json
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState
from agent.utils import extract_json


def decide_refine_or_research(state: AgentState, llm) -> Dict[str, Any]:
    """
    Agentic decision: Should we refine existing results or do a new search?
    Uses LLM to understand context and make intelligent decision.
    """
    previous_time_ranges = state.get("previous_time_ranges")
    previous_query = state.get("previous_query")
    current_query = state.get("user_query", "")
    
    # If no previous results, always do new search
    if not previous_time_ranges or not previous_query:
        return {
            "action": "NEW_SEARCH",
            "reason": "No previous results to refine",
            "extract_number": None
        }
    
    # Use LLM to decide based on context
    system_prompt = """You are a video search agent assistant. You need to decide whether the user wants you to:
1. REFINE existing results - select/filter from results you already found
2. NEW_SEARCH - perform a completely new search

IMPORTANT: If the current response is a clarification or refinement of the previous query (e.g., adding context, specifying details, narrowing down), it should be REFINE, not NEW_SEARCH.

Context:
- Previous query: "{previous_query}"
- Previous results: Found {num_results} time ranges
- Current user response: "{current_query}"

Examples:
- User said "top 10" after you found 28 results → REFINE (select top 10 from 28)
- User said "best 5" after you found results → REFINE (select best 5)
- User said "fishing context" after asking about fish → REFINE (clarifying/narrowing the search)
- User said "only the big ones" after asking about fish → REFINE (filtering existing results)
- User said "actually, find cooking moments" → NEW_SEARCH (completely different topic)
- User said "show me longer clips" → REFINE (filter existing by duration)
- User said "find moments where they run" → NEW_SEARCH (different activity)

Return JSON:
{{
    "action": "REFINE" | "NEW_SEARCH",
    "reason": "brief explanation",
    "extract_number": null or number  // If action is REFINE and user mentioned a number (e.g., "top 10" → 10)
}}"""
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt.format(
                previous_query=previous_query,
                num_results=len(previous_time_ranges),
                current_query=current_query
            )),
            HumanMessage(content=f"Analyze the user's intent and decide. Return JSON only.")
        ])
        
        response_text = response.content.strip()
        json_text = extract_json(response_text)
        decision = json.loads(json_text)
        
        return decision
    except Exception as e:
        # Fallback: heuristic detection
        query_lower = current_query.lower()
        
        # Check for refinement indicators
        refinement_keywords = ["top", "best", "first", "most", "select", "give me", "show me"]
        has_refinement = any(kw in query_lower for kw in refinement_keywords)
        
        # Check for number
        import re
        numbers = re.findall(r'\d+', current_query)
        extract_number = int(numbers[0]) if numbers else None
        
        # Check if it's a completely different topic (simple heuristic)
        previous_words = set(previous_query.lower().split())
        current_words = set(query_lower.split())
        overlap = len(previous_words & current_words) / max(len(previous_words), 1)
        
        if overlap < 0.3 and not has_refinement:
            # Very different words, likely new search
            return {
                "action": "NEW_SEARCH",
                "reason": "Query seems to be about different topic",
                "extract_number": None
            }
        elif has_refinement:
            return {
                "action": "REFINE",
                "reason": "User wants to refine/select from previous results",
                "extract_number": extract_number
            }
        else:
            # Default to refine if we have previous results
            return {
                "action": "REFINE",
                "reason": "Default: refine existing results",
                "extract_number": extract_number
            }


def refine_existing_results(state: AgentState, decision: Dict, verbose: bool = False) -> AgentState:
    """
    Refine existing results based on user's clarification.
    Selects top N, filters, or applies other refinements.
    Uses LLM-based ranking when descriptions are available.
    """
    previous_time_ranges = state.get("previous_time_ranges", [])
    previous_scored_seconds = state.get("previous_scored_seconds", [])
    previous_search_results = state.get("previous_search_results", [])
    current_query = state.get("user_query", "")
    previous_query = state.get("previous_query", "")
    extract_number = decision.get("extract_number")
    
    if verbose:
        print("\n[REFINEMENT] Refining existing results...")
        print(f"  Previous results: {len(previous_time_ranges)} time ranges")
        print(f"  User request: {decision.get('reason', 'N/A')}")
        if extract_number:
            print(f"  Extracting top {extract_number} results")
    
    # If we have search results with descriptions, use LLM-based ranking
    if previous_search_results and extract_number:
        # Map time ranges to their descriptions from search_results
        range_descriptions = {}
        for result in previous_search_results:
            result_tr = result.get("time_range", [])
            if result_tr and len(result_tr) >= 2:
                tr_key = (result_tr[0], result_tr[1])
                # Get description (prefer unified_description, fallback to description or text)
                desc = result.get("unified_description") or result.get("description") or result.get("text", "")
                if desc:
                    if tr_key not in range_descriptions:
                        range_descriptions[tr_key] = []
                    range_descriptions[tr_key].append(desc)
        
        # Match time ranges with their descriptions
        ranges_with_descriptions = []
        for tr in previous_time_ranges:
            tr_key = (tr[0], tr[1])
            descriptions = range_descriptions.get(tr_key, [])
            # Combine descriptions or use first one
            combined_desc = " | ".join(descriptions[:2]) if descriptions else f"Time range {tr[0]:.1f}s-{tr[1]:.1f}s"
            ranges_with_descriptions.append({
                "time_range": tr,
                "description": combined_desc
            })
        
        # Use LLM to rank based on descriptions and user query
        if len(ranges_with_descriptions) > extract_number:
            if verbose:
                print(f"  Using LLM-based ranking for {len(ranges_with_descriptions)} results...")
            
            ranked_ranges = rank_ranges_with_llm(
                ranges_with_descriptions,
                current_query,
                previous_query,
                extract_number,
                state.get("logger"),
                verbose=verbose
            )
            refined_ranges = [r["time_range"] for r in ranked_ranges]
        else:
            refined_ranges = previous_time_ranges
    
    # If we have scored seconds but no search results, use score-based ranking
    elif previous_scored_seconds:
        # Build a map: second_index -> score
        second_scores = {}
        for scored_sec in previous_scored_seconds:
            second_idx = scored_sec.get("second")
            score = scored_sec.get("score", 0)
            if second_idx is not None:
                second_scores[second_idx] = max(second_scores.get(second_idx, 0), score)
        
        # Score each time range (use max score of any second in the range)
        scored_ranges = []
        for tr in previous_time_ranges:
            start, end = tr[0], tr[1]
            # Find seconds that overlap with this time range
            max_score = 0
            for second_idx, score in second_scores.items():
                # Get the second's time range from segment tree
                segment_tree = state.get("segment_tree")
                if segment_tree:
                    second_data = segment_tree.get_second_by_index(second_idx)
                    if second_data:
                        sec_tr = second_data.get("time_range", [])
                        if sec_tr and len(sec_tr) >= 2:
                            sec_start, sec_end = sec_tr[0], sec_tr[1]
                            # Check if this second overlaps with the time range
                            if sec_start <= end and sec_end >= start:
                                max_score = max(max_score, score)
            
            scored_ranges.append((tr, max_score))
        
        # Sort by score
        scored_ranges.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top N if specified
        if extract_number:
            scored_ranges = scored_ranges[:extract_number]
            if verbose:
                print(f"  Selected top {extract_number} by score")
                print(f"  Score range: {scored_ranges[-1][1]:.3f} - {scored_ranges[0][1]:.3f}")
        
        # Extract just the time ranges
        refined_ranges = [tr for tr, _ in scored_ranges]
    else:
        # No scores, just take first N or all
        if extract_number:
            refined_ranges = previous_time_ranges[:extract_number]
        else:
            refined_ranges = previous_time_ranges
    
    if verbose:
        print(f"  Refined to {len(refined_ranges)} time ranges")
    
    return {
        **state,
        "time_ranges": refined_ranges,
        "needs_clarification": False,
        "confidence": state.get("confidence", 0.7)
    }


def rank_ranges_with_llm(
    ranges_with_descriptions: List[Dict],
    current_query: str,
    previous_query: str,
    top_n: int,
    logger: Optional[Any] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Use LLM to intelligently rank time ranges based on descriptions and user query.
    
    Args:
        ranges_with_descriptions: List of dicts with 'time_range' and 'description'
        current_query: Current user query/clarification
        previous_query: Original query
        top_n: Number of top results to return
        logger: Optional logger
        verbose: Print debug info
        
    Returns:
        Ranked list of ranges (top N)
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from agent.utils import extract_json
        import json
        
        # Get LLM (try to get from state or create new)
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except:
            try:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
            except:
                if verbose:
                    print("  [WARNING] No LLM available, falling back to score-based ranking")
                # Fallback: just return first N
                return ranges_with_descriptions[:top_n]
        
        # Prepare context for LLM
        ranges_summary = []
        for i, r in enumerate(ranges_with_descriptions):
            tr = r["time_range"]
            desc = r["description"]
            ranges_summary.append({
                "index": i,
                "time_range": f"{tr[0]:.1f}s - {tr[1]:.1f}s",
                "description": desc[:200]  # Truncate long descriptions
            })
        
        system_prompt = """You are a video search assistant. Your job is to rank video time ranges based on their descriptions and relevance to the user's query.

Context:
- Original query: "{previous_query}"
- User clarification/refinement: "{current_query}"
- Total results: {total_count}
- Need to select: top {top_n}

You will receive a list of time ranges with descriptions. Rank them by relevance to the user's query/clarification and return the top {top_n} most relevant ones.

Return JSON array with ranked indices:
[
    {{"index": 0, "rank": 1, "reasoning": "why this is relevant"}},
    {{"index": 5, "rank": 2, "reasoning": "why this is relevant"}},
    ...
]

Return exactly {top_n} results, ranked from most relevant (rank 1) to least relevant (rank {top_n})."""
        
        prompt = f"""Time ranges to rank:
{json.dumps(ranges_summary, indent=2)}

Rank these by relevance to: "{current_query}" (clarification of "{previous_query}").
Return JSON array with top {top_n} ranked results. Return JSON only."""
        
        if logger:
            logger.info(f"  [LLM RANKING] Ranking {len(ranges_with_descriptions)} results to select top {top_n}")
        elif verbose:
            print(f"  [LLM RANKING] Ranking {len(ranges_with_descriptions)} results to select top {top_n}")
        
        response = llm.invoke([
            SystemMessage(content=system_prompt.format(
                previous_query=previous_query,
                current_query=current_query,
                total_count=len(ranges_with_descriptions),
                top_n=top_n
            )),
            HumanMessage(content=prompt)
        ])
        
        response_text = response.content.strip()
        json_text = extract_json(response_text)
        ranking_results = json.loads(json_text)
        
        # Convert to list if single dict
        if isinstance(ranking_results, dict):
            ranking_results = [ranking_results]
        
        # Sort by rank and extract indices
        ranking_results.sort(key=lambda x: x.get("rank", 999))
        ranked_indices = [r["index"] for r in ranking_results[:top_n]]
        
        # Return ranked ranges
        ranked_ranges = [ranges_with_descriptions[i] for i in ranked_indices if 0 <= i < len(ranges_with_descriptions)]
        
        if logger:
            logger.info(f"  [LLM RANKING] Selected top {len(ranked_ranges)} results")
            for i, r in enumerate(ranking_results[:top_n], 1):
                idx = r.get("index", -1)
                reasoning = r.get("reasoning", "No reasoning")
                if 0 <= idx < len(ranges_with_descriptions):
                    tr = ranges_with_descriptions[idx]["time_range"]
                    logger.info(f"    {i}. {tr[0]:.1f}s-{tr[1]:.1f}s: {reasoning[:80]}")
        elif verbose:
            print(f"  [LLM RANKING] Selected top {len(ranked_ranges)} results")
        
        return ranked_ranges
        
    except Exception as e:
        if logger:
            logger.warning(f"  [LLM RANKING] Error: {e}, falling back to first {top_n} results")
        elif verbose:
            print(f"  [LLM RANKING] Error: {e}, falling back to first {top_n} results")
        # Fallback: return first N
        return ranges_with_descriptions[:top_n]


def validate_activity_evidence(query: str, activity_result: Dict, llm, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate activity evidence by examining actual descriptions.
    Filters out false positives (e.g., "eating fish" when query is "catching fish").
    
    Args:
        query: Original user query
        activity_result: Activity detection result with evidence list
        llm: Language model for validation
        verbose: Print debug info
    
    Returns:
        Filtered activity result with validated evidence
    """
    evidence_list = activity_result.get("evidence", [])
    if not evidence_list:
        return activity_result
    
    if verbose:
        print(f"  [VALIDATION] Validating {len(evidence_list)} evidence descriptions...")
    
    # Prepare descriptions for validation
    evidence_to_validate = []
    for ev in evidence_list:
        desc = ev.get("description", "")
        if desc:
            evidence_to_validate.append({
                "second": ev.get("second"),
                "time_range": ev.get("time_range"),
                "description": desc,
                "type": ev.get("type", "unknown")
            })
    
    if not evidence_to_validate:
        return activity_result
    
    # Batch validate descriptions (process in chunks to avoid token limits)
    system_prompt = """You are a video activity validator. Your job is to check if video descriptions actually match the user's query intent.

You will receive:
- User's query (what they're looking for)
- A list of video descriptions that matched keywords

For each description, determine if it TRULY matches the query intent, not just keywords.

Examples:
- Query: "find moments where they catch fish"
  - "person holding a fish they just caught" → VALID (catching)
  - "person eating a fish" → INVALID (eating, not catching)
  - "person with a fish in their hands" → VALID (likely caught)
  - "person cooking fish" → INVALID (cooking, not catching)

- Query: "find moments where they eat"
  - "person eating a fish" → VALID (eating)
  - "person holding a fish" → INVALID (holding, not eating)

Return JSON array with validation results:
[
    {
        "index": 0,
        "is_valid": true/false,
        "reasoning": "brief explanation"
    },
    ...
]"""
    
    validation_prompt = f"""
Query: {query}

Evidence descriptions to validate:
{json.dumps(evidence_to_validate, indent=2)}

For each description, determine if it truly matches the query "{query}".
Return JSON array with validation results. Return JSON only.
"""
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=validation_prompt)
        ])
        
        response_text = response.content.strip()
        json_text = extract_json(response_text)
        
        # FIX: Handle cases where JSON might have extra data or multiple objects
        validation_results = []
        try:
            validation_results = json.loads(json_text)
        except json.JSONDecodeError as e:
            # Try to extract first valid JSON object
            if verbose:
                print(f"  [VALIDATION] JSON parsing error: {e}, attempting recovery...")
            # Try to find first complete JSON array/object
            import re
            # Look for JSON array pattern
            array_match = re.search(r'\[.*?\]', json_text, re.DOTALL)
            if array_match:
                try:
                    validation_results = json.loads(array_match.group())
                except:
                    pass
            # If still fails, use empty list (will keep all evidence)
            if not validation_results:
                if verbose:
                    print(f"  [VALIDATION] Could not parse JSON, using all evidence")
                validation_results = []
        
        # Convert to dict for easier lookup
        validation_map = {}
        if isinstance(validation_results, list):
            for val_result in validation_results:
                idx = val_result.get("index")
                if idx is not None:
                    validation_map[idx] = val_result
        elif isinstance(validation_results, dict):
            # Handle case where LLM returns dict with list
            if "results" in validation_results:
                for i, val_result in enumerate(validation_results["results"]):
                    validation_map[i] = val_result
            else:
                # Single result
                validation_map[0] = validation_results
        
        # Filter evidence based on validation
        validated_evidence = []
        validated_evidence_scenes = []
        filtered_count = 0
        
        # Build a set of valid evidence keys (second + description) for matching
        valid_evidence_keys = set()
        
        for i, ev in enumerate(evidence_list):
            val_result = validation_map.get(i, {})
            is_valid = val_result.get("is_valid", True)  # Default to valid if validation failed
            
            if is_valid:
                validated_evidence.append(ev)
                # Create key for matching evidence_scenes
                ev_key = (ev.get("second"), ev.get("description", ""))
                valid_evidence_keys.add(ev_key)
            else:
                filtered_count += 1
                if verbose:
                    reasoning = val_result.get("reasoning", "No reasoning provided")
                    desc_preview = ev.get("description", "")[:50]
                    print(f"    Filtered out second {ev.get('second')} ({desc_preview}...): {reasoning}")
        
        # Filter evidence_scenes to match validated evidence
        # Handle both standard format (evidence_scenes) and fish-specific format (fish_holding_scenes)
        evidence_scenes = activity_result.get("evidence_scenes", [])
        fish_holding_scenes = activity_result.get("fish_holding_scenes", [])
        
        validated_evidence_scenes = []
        validated_fish_holding_scenes = []
        
        # Filter evidence_scenes (standard format)
        for scene in evidence_scenes:
            scene_key = (scene.get("second"), scene.get("description", ""))
            if scene_key in valid_evidence_keys:
                validated_evidence_scenes.append(scene)
        
        # Filter fish_holding_scenes (fish-specific format)
        for scene in fish_holding_scenes:
            scene_key = (scene.get("second"), scene.get("description", ""))
            if scene_key in valid_evidence_keys:
                validated_fish_holding_scenes.append(scene)
        
        if verbose:
            print(f"  [VALIDATION] Kept {len(validated_evidence)}/{len(evidence_list)} evidence (filtered {filtered_count})")
        
        # Update activity result with filtered evidence
        updated_result = activity_result.copy()
        updated_result["evidence"] = validated_evidence
        updated_result["evidence_count"] = len(validated_evidence)
        updated_result["detected"] = len(validated_evidence) > 0
        
        # Update both formats if they exist in original result
        if "evidence_scenes" in activity_result:
            updated_result["evidence_scenes"] = validated_evidence_scenes
        if "fish_holding_scenes" in activity_result:
            updated_result["fish_holding_scenes"] = validated_fish_holding_scenes
        
        # Update summary
        if len(validated_evidence) > 0:
            evidence_name = activity_result.get("evidence_name", "evidence")
            updated_result["summary"] = f"YES - {evidence_name.capitalize()} detected! Found {len(validated_evidence)} scene(s) with validated evidence."
        else:
            activity_name = activity_result.get("activity_name", "activity")
            updated_result["summary"] = f"NO - No validated {activity_name} evidence found after filtering false positives."
        
        return updated_result
        
    except Exception as e:
        if verbose:
            print(f"  [VALIDATION] Error validating evidence: {str(e)}, keeping all evidence")
        # On error, return original result (don't filter)
        return activity_result


def validate_search_results(query: str, semantic_analysis: Dict, filtered_seconds: List[Dict], 
                            llm, verbose: bool = False) -> Dict[str, Any]:
    """
    Agentic self-validation - checks if search results match query intent.
    Uses LLM to validate results and suggest improvements.
    """
    if not filtered_seconds:
        return {
            "is_valid": False,
            "needs_refinement": True,
            "issues": ["No results found"],
            "suggestions": ["Try lowering threshold or expanding search"],
            "confidence": 0.0
        }
    
    # Prepare summary of results for validation
    top_results_summary = []
    for sec in filtered_seconds[:10]:  # Top 10 for validation
        top_results_summary.append({
            "second": sec.get("second"),
            "score": sec.get("score", 0),
            "semantic_score": sec.get("semantic_score", 0),
            "activity_score": sec.get("activity_score", 0),
            "object_score": sec.get("object_score", 0)
        })
    
    system_prompt = """You are a search result validator. Your job is to check if the search results match the user's query intent.

Analyze:
1. Do the results match what the user asked for?
2. Are there obvious false positives?
3. Are the results relevant to the query type (positive, negative, temporal, etc.)?
4. Should the search be refined?

Return JSON:
{
    "is_valid": true/false,
    "needs_refinement": true/false,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""
    
    query_type = semantic_analysis.get("query_type", "POSITIVE")
    search_intent = semantic_analysis.get("search_intent", "hybrid")
    
    validation_prompt = f"""
Query: {query}
Query Type: {query_type}
Search Intent: {search_intent}
Number of results: {len(filtered_seconds)}
Top 10 results summary: {json.dumps(top_results_summary, indent=2)}

Validate if these results make sense for the query. Return JSON only.
"""
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=validation_prompt)
        ])
        
        response_text = response.content.strip()
        json_text = extract_json(response_text)
        result = json.loads(json_text)
        
        if verbose:
            print(f"  Validation confidence: {result.get('confidence', 0.7):.2f}")
            if result.get("is_valid"):
                print(f"  [VALIDATION] Results are valid")
            else:
                print(f"  [VALIDATION] Results need refinement")
        
        return result
    except Exception as e:
        # Fallback: basic validation
        if verbose:
            print(f"  [FALLBACK] Validation failed, assuming valid")
        return {
            "is_valid": True,
            "needs_refinement": False,
            "issues": [],
            "suggestions": [],
            "confidence": 0.5,
            "reasoning": "Fallback validation"
        }

