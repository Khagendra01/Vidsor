"""
LangGraph-based video clip extraction agent with planner and execution agents.
Takes user queries, retrieves relevant moments from video, and saves clips as MP4.
"""

import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Tuple
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # MoviePy 2.x uses different import path
    from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import time
from datetime import datetime
from segment_tree_utils import SegmentTreeQuery, load_segment_tree


class AgentState(TypedDict):
    """State for the video clip extraction agent."""
    messages: Annotated[list, add_messages]
    user_query: str
    video_path: str
    json_path: str
    query_type: Optional[str]  # "visual", "audio", "combined", "object", "activity"
    search_results: Optional[List[Dict]]
    time_ranges: Optional[List[Tuple[float, float]]]
    confidence: Optional[float]  # 0-1, how confident we are about the results
    needs_clarification: bool
    clarification_question: Optional[str]
    output_clips: List[str]  # Paths to saved clip files
    segment_tree: Optional[SegmentTreeQuery]
    verbose: bool  # Whether to print verbose output


def create_planner_agent(model_name: str = "gpt-4o-mini"):
    """Create the planner agent that analyzes queries and retrieves relevant moments."""
    
    # Try OpenAI first, fallback to Anthropic if needed
    if HAS_OPENAI:
        try:
            llm = ChatOpenAI(model=model_name, temperature=0)
        except:
            if HAS_ANTHROPIC:
                llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
            else:
                raise ValueError("Need either OpenAI or Anthropic API key configured")
    elif HAS_ANTHROPIC:
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    else:
        raise ValueError("Need either langchain-openai or langchain-anthropic installed")
    
    def planner_node(state: AgentState) -> AgentState:
        """Planner agent: Analyzes user query and retrieves relevant moments using ALL search types."""
        query = state["user_query"]
        segment_tree = state["segment_tree"]
        verbose = state.get("verbose", False)
        
        # Initialize variables
        search_results = []
        time_ranges = []
        confidence = 0.5
        needs_clarification = False
        clarification_question = None
        
        if verbose:
            print("\n" + "=" * 60)
            print("PLANNER AGENT: Multi-Modal Search Strategy")
            print("=" * 60)
            print(f"Query: {query}")
            print("\n[STEP 1] Generating search queries for all modalities...")
        
        # STEP 1: Generate search queries/keywords for ALL search types
        system_prompt_step1 = """You are a video analysis planner. Generate search queries and keywords for search types based on the user query.

Available search types:
1. "semantic": Natural language semantic search (searches both visual descriptions AND audio transcriptions using embeddings)
2. "hierarchical_keywords": Keywords for fast hierarchical tree lookup (extract key nouns/verbs from query)
3. "object": Object class names to search for (e.g., "person", "boat", "car") - uses YOLO detection data
4. "activity": Activity names and evidence keywords (e.g., "cooking", "fishing") - pattern matching

Note: Semantic search replaces the old visual/audio keyword search - it searches both automatically.

Return JSON with:
{
    "semantic_queries": ["query1", "query2"],  // Natural language queries for semantic search
    "hierarchical_keywords": ["keyword1", "keyword2"],  // Key nouns/verbs for fast tree lookup
    "object_classes": ["class1", "class2"],  // If objects are mentioned
    "activity_name": "activity",  // If activity is mentioned
    "activity_keywords": ["keyword1", "keyword2"],  // Keywords for activity search
    "evidence_keywords": ["keyword1"],  // Strong evidence keywords for activities
    "is_general_highlight_query": true/false,  // True if query is asking for general highlights
    "needs_clarification": true/false,
    "clarification_question": "question" (if needed)
}

Generate comprehensive search terms - be creative and think of synonyms, related terms, and different phrasings."""
        
        messages_step1 = [
            SystemMessage(content=system_prompt_step1),
            HumanMessage(content=f"User query: {query}\n\nGenerate search queries and keywords for ALL search types. Return JSON only.")
        ]
        
        if verbose:
            print("[LLM] Calling LLM to generate search queries...")
        
        response_step1 = llm.invoke(messages_step1)
        response_text_step1 = response_step1.content.strip()
        
        if verbose:
            print(f"[LLM RESPONSE] {response_text_step1[:300]}...")
        
        # Extract JSON from response
        if "```json" in response_text_step1:
            response_text_step1 = response_text_step1.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text_step1:
            response_text_step1 = response_text_step1.split("```")[1].split("```")[0].strip()
        
        try:
            search_plan = json.loads(response_text_step1)
        except:
            import re
            json_match = re.search(r'\{[^}]+\}', response_text_step1)
            if json_match:
                search_plan = json.loads(json_match.group())
            else:
                # Fallback: use original query for semantic, extract keywords for hierarchical tree
                import re
                # Extract meaningful keywords (nouns/verbs, at least 3 chars)
                words = re.findall(r'\b\w+\b', query.lower())
                meaningful_keywords = [w for w in words if len(w) >= 3]
                search_plan = {
                    "semantic_queries": [query],
                    "hierarchical_keywords": meaningful_keywords[:5],  # Top 5 keywords
                    "object_classes": [],
                    "activity_name": "",
                    "activity_keywords": [],
                    "evidence_keywords": [],
                    "is_general_highlight_query": "highlight" in query.lower(),
                    "needs_clarification": False
                }
        
        if verbose:
            print("\n[SEARCH PLAN] Generated search queries:")
            print(f"  Semantic queries: {search_plan.get('semantic_queries', [])}")
            print(f"  Hierarchical keywords: {search_plan.get('hierarchical_keywords', [])}")
            print(f"  Object classes: {search_plan.get('object_classes', [])}")
            print(f"  Activity: {search_plan.get('activity_name', 'N/A')}")
            print(f"  Is general highlight query: {search_plan.get('is_general_highlight_query', False)}")
        
        needs_clarification = search_plan.get("needs_clarification", False)
        clarification_question = search_plan.get("clarification_question")
        
        # Check for audio query ambiguity
        query_lower = query.lower()
        audio_indicators = ["say", "said", "talk", "speak", "mention", "discuss", "conversation", "audio", "transcription"]
        is_audio_query = any(indicator in query_lower for indicator in audio_indicators)
        ambiguous_pronouns = ["they", "he", "she", "it", "that", "this", "those", "these"]
        has_ambiguous_refs = any(pronoun in query_lower.split() for pronoun in ambiguous_pronouns)
        
        if is_audio_query and has_ambiguous_refs and not needs_clarification:
            needs_clarification = True
            clarification_question = "I notice your query mentions audio/transcription but uses ambiguous references. Could you clarify who or what you're looking for?"
            if verbose:
                print("[WARNING] Ambiguous audio query detected - will ask for clarification")
        
        if needs_clarification and not is_audio_query:
            if verbose:
                print("[INFO] LLM suggested clarification, but query seems clear enough. Will try searching first.")
            needs_clarification = False
        
        if not needs_clarification:
            if verbose:
                print("\n[STEP 2] Executing search with hierarchical tree + semantic search...")
            
            # STEP 2: Execute search types and collect results
            all_search_results = []  # Store all results with metadata about search type
            all_time_ranges = []  # Collect all time ranges
            
            # Check if this is a general highlight query
            is_general_highlight = search_plan.get("is_general_highlight_query", False)
            if "highlight" in query_lower and ("all" in query_lower or "find" in query_lower):
                is_general_highlight = True
            
            # 0. General highlight detection (using hierarchical tree)
            if is_general_highlight and segment_tree.hierarchical_tree:
                if verbose:
                    print(f"\n  [SEARCH TYPE: General Highlights (Hierarchical Tree)]")
                # Extract action keywords from query if any
                hierarchical_keywords = search_plan.get("hierarchical_keywords", [])
                scored_leaves = segment_tree.hierarchical_score_leaves_for_highlights(
                    action_keywords=hierarchical_keywords if hierarchical_keywords else None,
                    max_results=20
                )
                for leaf in scored_leaves:
                    tr = leaf.get("time_range", [])
                    if tr and len(tr) >= 2:
                        all_time_ranges.append((tr[0], tr[1], "hierarchical_highlight", leaf.get("score", 0)))
                        all_search_results.append({
                            "search_type": "hierarchical_highlight",
                            "time_range": tr,
                            "score": leaf.get("score", 0),
                            "node_id": leaf.get("node_id"),
                            "keyword_count": leaf.get("keyword_count", 0)
                        })
                if verbose:
                    print(f"      Found {len(scored_leaves)} highlight candidates")
            
            # 1. Hierarchical tree keyword search (fast pre-filter)
            hierarchical_keywords = search_plan.get("hierarchical_keywords", [])
            if hierarchical_keywords and segment_tree.hierarchical_tree:
                if verbose:
                    print(f"\n  [SEARCH TYPE: Hierarchical Tree (Fast Keyword Lookup)]")
                for keyword in hierarchical_keywords:
                    if verbose:
                        print(f"    Keyword: '{keyword}'")
                    results = segment_tree.hierarchical_keyword_search(
                        [keyword],
                        match_mode="any",
                        max_results=20
                    )
                    for result in results:
                        # Prefer leaf nodes
                        if result.get("level", -1) == 0:
                            tr = result.get("time_range", [])
                            if tr and len(tr) >= 2:
                                all_time_ranges.append((tr[0], tr[1], "hierarchical", result.get("match_count", 1)))
                                all_search_results.append({
                                    "search_type": "hierarchical",
                                    "time_range": tr,
                                    "score": result.get("match_count", 1),
                                    "node_id": result.get("node_id"),
                                    "matched_keyword": keyword
                                })
                    if verbose:
                        print(f"      Found {len([r for r in results if r.get('level') == 0])} leaf node matches")
            
            # 2. Semantic search (replaces old visual/audio keyword search)
            semantic_queries = search_plan.get("semantic_queries", [query])
            if semantic_queries:
                if verbose:
                    print(f"\n  [SEARCH TYPE: Semantic Search (Visual + Audio)]")
                for sem_query in semantic_queries:
                    if verbose:
                        print(f"    Query: '{sem_query}'")
                    threshold = 0.45
                    results = segment_tree.semantic_search(
                        sem_query,
                        top_k=15,
                        threshold=threshold,
                        search_transcriptions=True,
                        search_unified=True,
                        verbose=False
                    )
                    if not results:
                        threshold = 0.35
                        results = segment_tree.semantic_search(
                            sem_query,
                            top_k=15,
                            threshold=threshold,
                            search_transcriptions=True,
                            search_unified=True,
                            verbose=False
                        )
                    if not results:
                        threshold = 0.3
                        results = segment_tree.semantic_search(
                            sem_query,
                            top_k=15,
                            threshold=threshold,
                            search_transcriptions=True,
                            search_unified=True,
                            verbose=False
                        )
                    
                    for result in results:
                        result["search_type"] = "semantic"
                        result["search_query"] = sem_query
                        all_search_results.append(result)
                        tr = result.get("time_range", [])
                        if tr and len(tr) >= 2:
                            all_time_ranges.append((tr[0], tr[1], "semantic", result.get("score", 0)))
                    if verbose:
                        print(f"      Found {len(results)} matches")
            
            # 3. Object search (keep - uses YOLO detection data)
            object_classes = search_plan.get("object_classes", [])
            if object_classes:
                if verbose:
                    print(f"\n  [SEARCH TYPE: Object Detection (YOLO)]")
                for obj_class in object_classes:
                    if verbose:
                        print(f"    Class: '{obj_class}'")
                    results = segment_tree.find_objects_by_class(obj_class)
                    for result in results:
                        result["search_type"] = "object"
                        result["object_class"] = obj_class
                        all_search_results.append(result)
                        tr = result.get("time_range", [])
                        if tr and len(tr) >= 2:
                            all_time_ranges.append((tr[0], tr[1], "object", 1.0))
                    if verbose:
                        print(f"      Found {len(results)} detections")
            
            # 4. Activity search (keep - pattern matching)
            activity_name = search_plan.get("activity_name", "")
            activity_keywords = search_plan.get("activity_keywords", [])
            evidence_keywords = search_plan.get("evidence_keywords", [])
            if activity_name or activity_keywords:
                if verbose:
                    print(f"\n  [SEARCH TYPE: Activity Pattern Matching]")
                if "fish" in query_lower and ("catch" in query_lower or "caught" in query_lower):
                    if verbose:
                        print(f"    Activity: Fish catching (specialized)")
                    result = segment_tree.check_fish_caught()
                    result["search_type"] = "activity"
                    all_search_results.append(result)
                    for evidence in result.get("evidence", []):
                        tr = evidence.get("time_range", [])
                        if tr and len(tr) >= 2:
                            all_time_ranges.append((tr[0], tr[1], "activity", 1.0))
                    if verbose:
                        print(f"      Evidence scenes: {result.get('fish_holding_count', 0)}")
                elif activity_keywords:
                    if verbose:
                        print(f"    Activity: {activity_name or 'general'}")
                    result = segment_tree.check_activity(
                        activity_keywords=activity_keywords,
                        evidence_keywords=evidence_keywords or activity_keywords,
                        activity_name=activity_name or "activity"
                    )
                    result["search_type"] = "activity"
                    all_search_results.append(result)
                    for evidence in result.get("evidence", []):
                        tr = evidence.get("time_range", [])
                        if tr and len(tr) >= 2:
                            all_time_ranges.append((tr[0], tr[1], "activity", 1.0))
                    if verbose:
                        print(f"      Evidence scenes: {result.get('evidence_count', 0)}")
            
            if verbose:
                print(f"\n[AGGREGATION] Collected results from all search types:")
                print(f"  Total results: {len(all_search_results)}")
                print(f"  Total time ranges: {len(all_time_ranges)}")
            
            # STEP 3: Deduplicate and shortlist time ranges
            # Group by time range (with small tolerance for near-duplicates)
            time_range_map = {}  # (start, end) -> (count, max_score, search_types)
            for start, end, search_type, score in all_time_ranges:
                # Round to nearest second for deduplication
                start_rounded = round(start)
                end_rounded = round(end)
                key = (start_rounded, end_rounded)
                
                if key not in time_range_map:
                    time_range_map[key] = {
                        "count": 0,
                        "max_score": score,
                        "search_types": set(),
                        "original_ranges": []
                    }
                
                time_range_map[key]["count"] += 1
                time_range_map[key]["max_score"] = max(time_range_map[key]["max_score"], score)
                time_range_map[key]["search_types"].add(search_type)
                time_range_map[key]["original_ranges"].append((start, end, search_type, score))
            
            # Convert to list of (start, end, metadata) for ranking
            shortlisted_ranges = []
            for (start, end), metadata in time_range_map.items():
                # Prefer hierarchical tree ranges (5 seconds) over semantic search (1 second)
                # Sort by: 1) hierarchical search types first, 2) longer duration
                def range_priority(r):
                    _, _, search_type, _ = r
                    is_hierarchical = search_type in ["hierarchical", "hierarchical_highlight"]
                    duration = abs(r[1] - r[0])
                    return (not is_hierarchical, -duration)  # False (hierarchical) sorts first
                
                best_range = min(metadata["original_ranges"], key=range_priority)
                shortlisted_ranges.append({
                    "time_range": (best_range[0], best_range[1]),
                    "count": metadata["count"],
                    "score": metadata["max_score"],
                    "search_types": list(metadata["search_types"]),
                    "match_count": metadata["count"]
                })
            
            # Sort by match count and score
            shortlisted_ranges.sort(key=lambda x: (x["count"], x["score"]), reverse=True)
            
            if verbose:
                print(f"\n[SHORTLISTING] Deduplicated to {len(shortlisted_ranges)} unique time ranges")
                print(f"  Top 10 ranges:")
                for i, r in enumerate(shortlisted_ranges[:10], 1):
                    start, end = r["time_range"]
                    print(f"    {i}. {start:.1f}s - {end:.1f}s | Matches: {r['count']} | Score: {r['score']:.3f} | Types: {', '.join(r['search_types'])}")
            
            # STEP 4: Use LLM to rank/filter results
            if verbose:
                print(f"\n[STEP 3] Using LLM to rank and filter results...")
            
            # Prepare results summary for LLM
            results_summary = []
            for i, r in enumerate(shortlisted_ranges[:20], 1):  # Top 20 for ranking
                start, end = r["time_range"]
                results_summary.append({
                    "index": i,
                    "time_range": f"{start:.1f}s - {end:.1f}s",
                    "duration": f"{end - start:.1f}s",
                    "match_count": r["count"],
                    "score": r["score"],
                    "search_types": ", ".join(r["search_types"])
                })
            
            system_prompt_step2 = """You are a video search result ranker. Your job is to:
1. Rank the search results by relevance to the user's query
2. Determine if the user query implies they want ONE specific moment or MULTIPLE moments
3. Filter out irrelevant or low-quality results
4. Return the best results based on the user's intent

Indicators that user wants ONE result:
- "find the moment", "show me when", "find when" (singular)
- "the best", "the most", "the first"
- Queries that describe a specific, unique event

Indicators that user wants MULTIPLE results:
- "find moments", "show all", "find all", "every time"
- "all instances", "all scenes"
- Queries that describe recurring events or multiple occurrences

Return JSON with:
{
    "user_wants_one": true/false,  // Does user query imply they want one result?
    "ranked_indices": [1, 3, 5, ...],  // Indices of results in order of relevance (1-based)
    "filtered_indices": [2, 4, ...],  // Indices to exclude (irrelevant results)
    "confidence": 0.0-1.0,  // Confidence in the ranking
    "reasoning": "brief explanation"  // Why these results were selected
}"""
            
            results_text = "\n".join([
                f"{r['index']}. Time: {r['time_range']} (duration: {r['duration']}) | "
                f"Matches: {r['match_count']} | Score: {r['score']:.3f} | Types: {r['search_types']}"
                for r in results_summary
            ])
            
            messages_step2 = [
                SystemMessage(content=system_prompt_step2),
                HumanMessage(content=f"User query: {query}\n\nSearch results:\n{results_text}\n\nRank and filter these results. Return JSON only.")
            ]
            
            if verbose:
                print("[LLM] Calling LLM to rank results...")
            
            response_step2 = llm.invoke(messages_step2)
            response_text_step2 = response_step2.content.strip()
            
            if verbose:
                print(f"[LLM RESPONSE] {response_text_step2[:300]}...")
            
            # Extract JSON
            if "```json" in response_text_step2:
                response_text_step2 = response_text_step2.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text_step2:
                response_text_step2 = response_text_step2.split("```")[1].split("```")[0].strip()
            
            try:
                ranking = json.loads(response_text_step2)
            except:
                import re
                json_match = re.search(r'\{[^}]+\}', response_text_step2)
                if json_match:
                    ranking = json.loads(json_match.group())
                else:
                    # Fallback: use top results
                    ranking = {
                        "user_wants_one": "moment" in query.lower() and "all" not in query.lower(),
                        "ranked_indices": list(range(1, min(6, len(shortlisted_ranges) + 1))),
                        "filtered_indices": [],
                        "confidence": 0.6,
                        "reasoning": "Fallback ranking"
                    }
            
            if verbose:
                print(f"\n[RANKING] LLM Analysis:")
                print(f"  User wants one result: {ranking.get('user_wants_one', False)}")
                print(f"  Ranked indices: {ranking.get('ranked_indices', [])}")
                print(f"  Filtered indices: {ranking.get('filtered_indices', [])}")
                print(f"  Confidence: {ranking.get('confidence', 0):.2f}")
                print(f"  Reasoning: {ranking.get('reasoning', 'N/A')}")
            
            # STEP 5: Select final time ranges based on ranking
            ranked_indices = ranking.get("ranked_indices", [])
            filtered_indices = set(ranking.get("filtered_indices", []))
            user_wants_one = ranking.get("user_wants_one", False)
            confidence = ranking.get("confidence", 0.7)
            
            # Get final time ranges
            final_time_ranges = []
            for idx in ranked_indices:
                if idx in filtered_indices:
                    continue
                if 1 <= idx <= len(shortlisted_ranges):
                    r = shortlisted_ranges[idx - 1]  # Convert to 0-based
                    final_time_ranges.append(r["time_range"])
            
            # If user wants one, take only the top result
            if user_wants_one and final_time_ranges:
                final_time_ranges = [final_time_ranges[0]]
                if verbose:
                    print(f"\n[SELECTION] User wants one result - selecting top match")
            elif not final_time_ranges and shortlisted_ranges:
                # Fallback: use top 5 if ranking failed
                final_time_ranges = [r["time_range"] for r in shortlisted_ranges[:5]]
                if verbose:
                    print(f"\n[SELECTION] Ranking failed - using top 5 results")
            
            # Store all search results for reference
            search_results = all_search_results
            time_ranges = final_time_ranges
            
            if verbose:
                print(f"\n[FINAL SELECTION]")
                print(f"  Selected {len(time_ranges)} time range(s)")
                if time_ranges:
                    print(f"  Time ranges:")
                    for i, (start, end) in enumerate(time_ranges, 1):
                        print(f"    {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
            
            if not time_ranges:
                confidence = 0.3
                needs_clarification = True
                clarification_question = "No relevant results found. Could you rephrase your query or provide more details?"
                if verbose:
                    print("  [WARNING] No results found - will ask for clarification")
            elif len(time_ranges) > 15:
                confidence = 0.6
                needs_clarification = True
                clarification_question = f"Found {len(time_ranges)} potential moments. Could you narrow down what you're looking for?"
                if verbose:
                    print(f"  [WARNING] Too many results ({len(time_ranges)}) - will ask for clarification")
            else:
                if verbose:
                    print(f"  [SUCCESS] Found {len(time_ranges)} time range(s) - proceeding with extraction")
        
        # Merge overlapping time ranges
        if time_ranges:
            original_count = len(time_ranges)
            if verbose:
                print(f"\n[MERGING] Merging {original_count} time ranges...")
                print(f"  Before merge: {original_count} ranges")
            time_ranges = merge_time_ranges(time_ranges)
            if verbose:
                print(f"  After merge: {len(time_ranges)} non-overlapping ranges")
                print(f"  Merged ranges with 2.0s padding:")
                for i, (start, end) in enumerate(time_ranges, 1):
                    print(f"    Range {i}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        else:
            if verbose:
                print("\n[MERGING] No time ranges to merge")
        
        return {
            **state,
            "query_type": "multi_modal",  # Indicate we used all search types
            "search_results": search_results,
            "time_ranges": time_ranges,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question
        }
    
    return planner_node


def merge_time_ranges(time_ranges: List[Tuple[float, float]], 
                     padding: float = 2.0) -> List[Tuple[float, float]]:
    """Merge overlapping time ranges and add padding."""
    if not time_ranges:
        return []
    
    # Sort by start time
    sorted_ranges = sorted(time_ranges, key=lambda x: x[0])
    merged = []
    
    current_start, current_end = sorted_ranges[0]
    
    for start, end in sorted_ranges[1:]:
        # Add padding
        if start - padding <= current_end:
            # Overlapping or close, merge
            current_end = max(current_end, end)
        else:
            # Not overlapping, save current and start new
            merged.append((max(0, current_start - padding), current_end + padding))
            current_start, current_end = start, end
    
    # Add last range
    merged.append((max(0, current_start - padding), current_end + padding))
    
    return merged


def create_execution_agent():
    """Create the execution agent that extracts and saves video clips."""
    
    def execution_node(state: AgentState) -> AgentState:
        """Execution agent: Extracts video clips and saves them."""
        video_path = state["video_path"]
        time_ranges = state.get("time_ranges", [])
        output_clips = state.get("output_clips", [])
        verbose = state.get("verbose", False)
        
        if verbose:
            print("\n" + "=" * 60)
            print("EXECUTION AGENT: Extracting Video Clips")
            print("=" * 60)
            print(f"Video: {video_path}")
            print(f"Time ranges to extract: {len(time_ranges)}")
        
        if not time_ranges:
            if verbose:
                print("[SKIP] No time ranges to extract")
            return {
                **state,
                "output_clips": output_clips
            }
        
        # Create output directory
        output_dir = "extracted_clips"
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Output directory: {output_dir}/")
        
        # Load video
        try:
            if verbose:
                print(f"\n[LOADING] Opening video file: {video_path}")
            video = VideoFileClip(video_path)
            if verbose:
                print(f"[LOADING] Video loaded successfully:")
                print(f"  Duration: {video.duration:.2f}s")
                print(f"  FPS: {video.fps}")
                print(f"  Resolution: {video.size}")
                print(f"  Codec: {video.codec if hasattr(video, 'codec') else 'N/A'}")
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to load video: {str(e)}")
            return {
                **state,
                "output_clips": [],
                "messages": state["messages"] + [
                    AIMessage(content=f"Error loading video: {str(e)}")
                ]
            }
        
        saved_clips = []
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            clip = None
            try:
                if verbose:
                    print(f"\n[EXTRACTING] Clip {i+1}/{len(time_ranges)}")
                    print(f"  Original range: {start_time:.2f}s - {end_time:.2f}s")
                
                # Close and reopen video file between clips to reset MoviePy's internal state
                # This ensures clean subprocess state for each clip on Windows
                if i > 0:  # Don't close on first clip
                    if verbose:
                        print(f"  [CLEANUP] Closing video file to reset state...")
                    video.close()
                    time.sleep(1.0)  # Give Windows time to fully cleanup subprocesses
                    if verbose:
                        print(f"  [CLEANUP] Reopening video file...")
                    video = VideoFileClip(video_path)
                
                # Ensure times are within video duration
                start_time = max(0, min(start_time, video.duration))
                end_time = max(start_time + 1, min(end_time, video.duration))
                
                if verbose:
                    print(f"  Adjusted range: {start_time:.2f}s - {end_time:.2f}s")
                    print(f"  Duration: {end_time - start_time:.2f}s")
                
                # Extract clip (moviepy 2.x uses subclipped instead of subclip)
                if verbose:
                    print(f"  [PROCESSING] Extracting subclip...")
                
                # Extract the subclip (no output suppression needed for subclipped)
                clip = video.subclipped(start_time, end_time)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clip_{i+1}_{int(start_time)}s_to_{int(end_time)}s_{timestamp}.mp4"
                output_path = os.path.join(output_dir, filename)
                
                if verbose:
                    print(f"  [SAVING] Writing to: {filename}")
                
                # Write clip without logger - MoviePy 2.x works fine without it
                # Use unique temp audio file per clip to avoid Windows subprocess conflicts
                temp_audio = f'temp-audio-{i+1}-{timestamp}.m4a'
                
                clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=temp_audio,
                    remove_temp=True
                )
                
                if clip:
                    clip.close()
                
                # Clean up temp audio file if it still exists
                if os.path.exists(temp_audio):
                    try:
                        os.remove(temp_audio)
                    except:
                        pass
                
                # Verify file was created and has content
                if not os.path.exists(output_path):
                    raise Exception(f"Output file was not created: {output_path}")
                
                file_size = os.path.getsize(output_path)
                if file_size < 1000:  # Less than 1KB is likely corrupted/empty
                    raise Exception(f"Output file is too small ({file_size} bytes), likely corrupted")
                
                saved_clips.append(output_path)
                
                if verbose:
                    print(f"  [SUCCESS] Clip saved: {output_path} ({file_size} bytes)")
                
            except Exception as e:
                error_msg = f"Error extracting clip {i+1}: {str(e)}"
                if verbose:
                    print(f"  [ERROR] {error_msg}")
                print(error_msg)
                if clip:
                    try:
                        clip.close()
                    except:
                        pass
                continue
        
        # Final cleanup
        if video:
            try:
                video.close()
            except:
                pass
        
        if verbose:
            print(f"\n[COMPLETE] Successfully extracted {len(saved_clips)} clip(s)")
            print(f"  Output directory: {output_dir}/")
            for clip in saved_clips:
                print(f"    - {os.path.basename(clip)}")
        
        return {
            **state,
            "output_clips": saved_clips,
            "messages": state["messages"] + [
                AIMessage(content=f"Successfully extracted {len(saved_clips)} clip(s) to {output_dir}/")
            ]
        }
    
    return execution_node


def create_clarification_node():
    """Node that asks user for clarification."""
    
    def clarification_node(state: AgentState) -> AgentState:
        """Ask user for clarification and get their response."""
        verbose = state.get("verbose", False)
        question = state.get("clarification_question", "Could you provide more details?")
        
        if verbose:
            print("\n" + "=" * 60)
            print("CLARIFIER: Asking for Clarification")
            print("=" * 60)
            print(f"Question: {question}")
            print("\n[CLARIFIER] Waiting for user response...")
        
        # Prompt user for input
        print(f"\n{question}")
        user_response = input("Your response: ").strip()
        
        if not user_response:
            # If user just presses enter, use original query
            user_response = state["user_query"]
            if verbose:
                print("[CLARIFIER] No response provided, using original query")
        else:
            if verbose:
                print(f"[CLARIFIER] User provided: {user_response}")
        
        # Update state with new query and continue to planner
        return {
            **state,
            "user_query": user_response,
            "messages": state["messages"] + [
                AIMessage(content=question),
                HumanMessage(content=user_response)
            ],
            "needs_clarification": False,  # Clear the flag so planner processes the new query
            "clarification_question": None
        }
    
    return clarification_node


def should_ask_clarification(state: AgentState) -> Literal["clarify", "execute"]:
    """Router: decide whether to ask for clarification or execute."""
    verbose = state.get("verbose", False)
    needs_clarification = state.get("needs_clarification", False)
    
    if verbose:
        print("\n" + "=" * 60)
        print("ROUTING DECISION")
        print("=" * 60)
        print(f"Needs clarification: {needs_clarification}")
        if needs_clarification:
            print(f"Clarification question: {state.get('clarification_question', 'N/A')}")
            print("[ROUTING] → Going to CLARIFIER")
        else:
            print(f"Confidence: {state.get('confidence', 0):.2f}")
            print(f"Time ranges found: {len(state.get('time_ranges', []))}")
            print("[ROUTING] → Going to EXECUTOR")
    
    if needs_clarification:
        return "clarify"
    return "execute"


def create_video_clip_agent(json_path: str, video_path: str, model_name: str = "gpt-4o-mini"):
    """Create the complete video clip extraction agent workflow."""
    
    # Load segment tree
    segment_tree = load_segment_tree(json_path)
    
    # Create nodes
    planner = create_planner_agent(model_name)
    executor = create_execution_agent()
    clarifier = create_clarification_node()
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("clarifier", clarifier)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add conditional edge after planner
    workflow.add_conditional_edges(
        "planner",
        should_ask_clarification,
        {
            "clarify": "clarifier",
            "execute": "executor"
        }
    )
    
    # From clarifier, go back to planner (user will provide new query)
    workflow.add_edge("clarifier", "planner")
    
    # From executor, end
    workflow.add_edge("executor", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app, segment_tree


def run_agent(query: str, json_path: str, video_path: str, model_name: str = "gpt-4o-mini", verbose: bool = True):
    """Run the video clip extraction agent with a user query."""
    
    if verbose:
        print("\n" + "=" * 60)
        print("VIDEO CLIP EXTRACTION AGENT")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Video: {video_path}")
        print(f"Segment Tree: {json_path}")
        print(f"Model: {model_name}")
        print("\n[INITIALIZATION] Starting agent...")
    
    # Create agent
    if verbose:
        print("[INITIALIZATION] Loading segment tree...")
    app, segment_tree = create_video_clip_agent(json_path, video_path, model_name)
    
    if verbose:
        video_info = segment_tree.get_video_info()
        print(f"[INITIALIZATION] Segment tree loaded:")
        print(f"  Video duration: {video_info.get('duration_seconds', 0)} seconds")
        print(f"  FPS: {video_info.get('fps', 0)}")
        print(f"  Total frames: {video_info.get('total_frames', 0)}")
        print(f"[INITIALIZATION] Workflow graph created")
        print(f"[INITIALIZATION] Ready to process query\n")
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "video_path": video_path,
        "json_path": json_path,
        "query_type": None,
        "search_results": None,
        "time_ranges": None,
        "confidence": None,
        "needs_clarification": False,
        "clarification_question": None,
        "output_clips": [],
        "segment_tree": segment_tree,
        "verbose": verbose
    }
    
    # Run agent
    if verbose:
        print("[WORKFLOW] Invoking agent workflow...")
    result = app.invoke(initial_state)
    
    if verbose:
        print("\n[WORKFLOW] Agent workflow completed")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video clip extraction agent")
    parser.add_argument("query", help="User query (e.g., 'find moments where they catch fish')")
    parser.add_argument("--json", default="camp_segment_tree.json", help="Path to segment tree JSON")
    parser.add_argument("--video", default="camp.mp4", help="Path to video file")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Set verbose based on flags
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print(f"Processing query: {args.query}")
        print(f"Video: {args.video}")
        print(f"JSON: {args.json}")
        print()
    
    result = run_agent(args.query, args.json, args.video, args.model, verbose=verbose)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if result.get("needs_clarification"):
        print(f"\nClarification needed: {result.get('clarification_question')}")
    else:
        print(f"\nConfidence: {result.get('confidence', 0):.2f}")
        print(f"Time ranges found: {len(result.get('time_ranges', []))}")
        print(f"Clips saved: {len(result.get('output_clips', []))}")
        
        if result.get("output_clips"):
            print("\nSaved clips:")
            for clip in result["output_clips"]:
                print(f"  - {clip}")

