"""
LangGraph-based video clip extraction agent with planner and execution agents.
Takes user queries, retrieves relevant moments from video, and saves clips as MP4.
"""

import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Tuple, Set
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import dotenv
dotenv.load_dotenv()

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
    # Memory/context fields for agentic behavior
    previous_time_ranges: Optional[List[Tuple[float, float]]]  # Previous search results
    previous_scored_seconds: Optional[List[Dict]]  # Previous scored seconds with scores
    previous_query: Optional[str]  # Original query before clarification
    previous_search_results: Optional[List[Dict]]  # Previous search results


class PerSecondFeatureExtractor:
    """Extract normalized features for each second of video."""
    
    def __init__(self, segment_tree: SegmentTreeQuery):
        self.segment_tree = segment_tree
        self.feature_cache = {}  # Cache per-second features
        
    def extract_features_for_second(self, second_idx: int) -> Dict[str, Any]:
        """Extract all features for a single second."""
        if second_idx in self.feature_cache:
            return self.feature_cache[second_idx]
            
        second_data = self.segment_tree.get_second_by_index(second_idx)
        if not second_data:
            return None
        
        features = {
            "second": second_idx,
            "time_range": second_data.get("time_range", []),
            "semantic_score": 0.0,  # Will be filled by semantic search
            "activity_score": 0.0,  # Will be filled by activity search
            "hierarchical_score": 0.0,  # Will be filled by hierarchical search
            "transcript_score": 0.0,  # Will be filled by transcript search
            "object_presence": {},  # {class_name: normalized_count}
            "object_tracks": {},  # {class_name: distinct_track_count}
            "event_flags": {}  # {event_name: bool}
        }
        
        # Extract object presence (normalized per second)
        for group in second_data.get("detection_groups", []):
            for detection in group.get("detections", []):
                class_name = detection.get("class_name")
                if class_name:
                    if class_name not in features["object_presence"]:
                        features["object_presence"][class_name] = 0
                        features["object_tracks"][class_name] = set()
                    features["object_presence"][class_name] += 1
                    track_id = detection.get("track_id")
                    if track_id is not None:
                        features["object_tracks"][class_name].add(track_id)
        
        # Normalize object counts (cap at 3 or use log)
        for class_name in list(features["object_presence"].keys()):
            count = features["object_presence"][class_name]
            features["object_presence"][class_name] = min(count, 3)  # Cap at 3
            features["object_tracks"][class_name] = len(features["object_tracks"][class_name])
        
        self.feature_cache[second_idx] = features
        return features
    
    def get_all_seconds_features(self):
        """Get features for all seconds (lazy load)."""
        for i in range(len(self.segment_tree.seconds)):
            yield self.extract_features_for_second(i)


def extract_json(text: str) -> str:
    """Extract JSON from LLM response."""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    else:
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
    return text.strip()


def analyze_query_intent(query: str, llm) -> Dict[str, Any]:
    """
    Analyze query to determine:
    1. Is it object-centric? (e.g., "find all clips where person is detected")
    2. What object classes are mentioned?
    3. What's the primary intent?
    """
    system_prompt = """Analyze the user query and determine the search intent.

CRITICAL DISTINCTION:
- OBJECT-CENTRIC: User wants to find clips where specific objects appear, regardless of what they're doing.
  Examples: "find all clips where person is detected", "show all cars", "find every time a boat appears"
  Key indicators: "detected", "appears", "is present", "find all clips where [object]"
  
- ACTIVITY-CENTRIC: User wants to find moments where specific activities/actions happen, even if objects are mentioned.
  Examples: "find moments where they catch fish", "show when someone runs", "find scenes of cooking"
  Key indicators: Action verbs (catch, run, jump, cook, throw, etc.) + objects, "moments where", "when they"
  
- HYBRID: Query mentions both objects and activities, or is ambiguous.
  Examples: "find person catching fish", "show car driving"

Return JSON:
{
    "is_object_centric": true/false,
    "mentioned_classes": ["person", "car"],
    "primary_intent": "object" | "activity" | "semantic" | "hybrid",
    "object_priority": {"person": 1.0, "car": 0.8},  // Priority scores for each class (0-1)
    "needs_semantic": true/false,
    "needs_activity": true/false,
    "needs_hierarchical": true/false,
    "confidence": 0.0-1.0  // Confidence in classification
}"""
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\n\nAnalyze this query carefully. Check for action verbs that indicate activities. Return JSON only.")
        ])
        
        response_text = response.content.strip()
        json_text = extract_json(response_text)
        result = json.loads(json_text)
        
        # Ensure confidence is set
        if "confidence" not in result:
            result["confidence"] = 0.7
        
        return result
    except Exception as e:
        # Fallback: try to detect using heuristics
        query_lower = query.lower()
        
        # Activity verb indicators
        activity_verbs = ["catch", "throw", "run", "jump", "cook", "drive", "walk", "swim", "climb", 
                         "kick", "hit", "throw", "pick", "drop", "lift", "carry", "push", "pull",
                         "eat", "drink", "play", "dance", "sing", "talk", "speak", "write", "read"]
        has_activity_verb = any(verb in query_lower for verb in activity_verbs)
        
        # Object-centric indicators (must be explicit)
        object_indicators = ["is detected", "appears", "is present", "find all clips where", 
                            "show all", "every time", "all instances of"]
        is_object_centric = any(indicator in query_lower for indicator in object_indicators) and not has_activity_verb
        
        # Activity-centric indicators
        activity_indicators = ["moments where", "when they", "when someone", "scenes of", "instances of"]
        is_activity_centric = (has_activity_verb or any(indicator in query_lower for indicator in activity_indicators)) and not is_object_centric
        
        # Extract potential object classes
        mentioned_classes = []
        common_classes = ["person", "car", "fish", "boat", "dog", "cat", "bird", "bike", "truck"]
        for cls in common_classes:
            if cls in query_lower:
                mentioned_classes.append(cls)
        
        # Determine primary intent
        if is_object_centric:
            primary_intent = "object"
        elif is_activity_centric:
            primary_intent = "activity"
        else:
            primary_intent = "hybrid"  # Default to hybrid when ambiguous
        
        return {
            "is_object_centric": is_object_centric,
            "mentioned_classes": mentioned_classes,
            "primary_intent": primary_intent,
            "object_priority": {cls: 1.0 if is_object_centric else 0.5 for cls in mentioned_classes},
            "needs_semantic": not is_object_centric,
            "needs_activity": not is_object_centric or is_activity_centric,
            "needs_hierarchical": True,
            "confidence": 0.6  # Lower confidence for fallback
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
        "object_weights": {},
        "threshold": 0.3
    }
    
    # Initialize all object classes with default low weight
    for class_name in all_object_classes:
        weights["object_weights"][class_name] = 0.1  # Default low
    
    # If object-centric query
    if query_intent.get("is_object_centric"):
        # Set high weights for mentioned classes
        for class_name, priority in query_intent.get("object_priority", {}).items():
            weights["object_weights"][class_name] = priority
        
        # Lower threshold for pure object queries
        weights["threshold"] = 0.1
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


def score_seconds(feature_extractor: PerSecondFeatureExtractor, 
                  query_intent: Dict,
                  weights: Dict,
                  semantic_results: List[Dict],
                  activity_results: List[Dict],
                  hierarchical_results: List[Dict],
                  verbose: bool = False) -> List[Dict]:
    """
    Score all seconds using weighted features.
    
    Returns list of {second_idx, score, features, time_range, ...}
    """
    scored_seconds = []
    
    # Helper function to map time to second index
    def time_to_second_idx(time_seconds: float) -> Optional[int]:
        """Map a time in seconds to the corresponding second index."""
        for idx, second_data in enumerate(feature_extractor.segment_tree.seconds):
            tr = second_data.get("time_range", [])
            if tr and len(tr) >= 2 and tr[0] <= time_seconds <= tr[1]:
                return idx
        # Fallback: approximate by rounding
        return int(time_seconds) if 0 <= int(time_seconds) < len(feature_extractor.segment_tree.seconds) else None
    
    # Build lookup maps for search results
    semantic_map = {}  # second_idx -> max_score
    for result in semantic_results:
        tr = result.get("time_range", [])
        if tr and len(tr) >= 2:
            second_idx = time_to_second_idx(tr[0])
            if second_idx is not None:
                score = result.get("score", 0)
                semantic_map[second_idx] = max(semantic_map.get(second_idx, 0), score)
    
    activity_map = {}  # second_idx -> score
    for result in activity_results:
        for evidence in result.get("evidence", []):
            tr = evidence.get("time_range", [])
            if tr and len(tr) >= 2:
                second_idx = time_to_second_idx(tr[0])
                if second_idx is not None:
                    activity_map[second_idx] = 1.0  # Activity is binary
    
    hierarchical_map = {}  # second_idx -> score
    for result in hierarchical_results:
        tr = result.get("time_range", [])
        if tr and len(tr) >= 2:
            second_idx = time_to_second_idx(tr[0])
            if second_idx is not None:
                hierarchical_map[second_idx] = result.get("score", 1.0)
    
    if verbose:
        print(f"\n[SCORING] Scoring {len(feature_extractor.segment_tree.seconds)} seconds...")
        print(f"  Semantic matches: {len(semantic_map)} seconds")
        print(f"  Activity matches: {len(activity_map)} seconds")
        print(f"  Hierarchical matches: {len(hierarchical_map)} seconds")
    
    # Score each second
    for second_idx in range(len(feature_extractor.segment_tree.seconds)):
        features = feature_extractor.extract_features_for_second(second_idx)
        if not features:
            continue
        
        # Get search result scores
        semantic_score = semantic_map.get(second_idx, 0.0)
        activity_score = activity_map.get(second_idx, 0.0)
        hierarchical_score = hierarchical_map.get(second_idx, 0.0)
        
        # Compute object score
        object_score = 0.0
        for class_name, normalized_count in features["object_presence"].items():
            class_weight = weights["object_weights"].get(class_name, 0.1)
            # Normalize count to [0, 1] (max 3 -> 1.0)
            normalized = normalized_count / 3.0
            object_score += class_weight * normalized
        
        # Final weighted score
        final_score = (
            weights["semantic_weight"] * semantic_score +
            weights["activity_weight"] * activity_score +
            weights["hierarchical_weight"] * hierarchical_score +
            object_score
        )
        
        # Contextual boost: if object-centric and object present, ensure minimum score
        if query_intent.get("is_object_centric") and object_score > 0:
            final_score = max(final_score, weights["threshold"])
        
        scored_seconds.append({
            "second": second_idx,
            "score": final_score,
            "time_range": features["time_range"],
            "semantic_score": semantic_score,
            "activity_score": activity_score,
            "hierarchical_score": hierarchical_score,
            "object_score": object_score,
            "features": features
        })
    
    if verbose:
        if scored_seconds:
            sorted_scores = sorted(scored_seconds, key=lambda x: x["score"], reverse=True)
            print(f"  Score range: {sorted_scores[-1]['score']:.3f} - {sorted_scores[0]['score']:.3f}")
            top_scores = [f"{s['second']}:{s['score']:.3f}" for s in sorted_scores[:5]]
            print(f"  Top 5 scores: {top_scores}")
    
    return scored_seconds


def group_contiguous_seconds(scored_seconds: List[Dict], 
                              min_duration: float = 1.0,
                              gap_tolerance: float = 2.0) -> List[Tuple[float, float]]:
    """
    Group contiguous high-scoring seconds into time ranges.
    """
    if not scored_seconds:
        return []
    
    # Sort by second index
    sorted_seconds = sorted(scored_seconds, key=lambda x: x["second"])
    
    time_ranges = []
    current_start = None
    current_end = None
    
    for sec in sorted_seconds:
        tr = sec.get("time_range", [])
        if not tr or len(tr) < 2:
            continue
        
        start, end = tr[0], tr[1]
        
        if current_start is None:
            current_start = start
            current_end = end
        elif start - current_end <= gap_tolerance:
            # Contiguous or close enough, merge
            current_end = end
        else:
            # Gap detected, save current range and start new
            if current_end - current_start >= min_duration:
                time_ranges.append((current_start, current_end))
            current_start = start
            current_end = end
    
    # Add final range
    if current_start is not None and current_end - current_start >= min_duration:
        time_ranges.append((current_start, current_end))
    
    return time_ranges


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

Context:
- Previous query: "{previous_query}"
- Previous results: Found {num_results} time ranges
- Current user response: "{current_query}"

Examples:
- User said "top 10" after you found 28 results → REFINE (select top 10 from 28)
- User said "best 5" after you found results → REFINE (select best 5)
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
    """
    previous_time_ranges = state.get("previous_time_ranges", [])
    previous_scored_seconds = state.get("previous_scored_seconds", [])
    extract_number = decision.get("extract_number")
    
    if verbose:
        print("\n[REFINEMENT] Refining existing results...")
        print(f"  Previous results: {len(previous_time_ranges)} time ranges")
        print(f"  User request: {decision.get('reason', 'N/A')}")
        if extract_number:
            print(f"  Extracting top {extract_number} results")
    
    # If we have scored seconds, use them to rank
    if previous_scored_seconds:
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
        
        # AGENTIC DECISION: Check if we should refine existing results or do new search
        if state.get("previous_time_ranges") and state.get("previous_query"):
            if verbose:
                print("\n" + "=" * 60)
                print("PLANNER AGENT: Context-Aware Decision")
                print("=" * 60)
                print(f"Previous query: {state.get('previous_query')}")
                print(f"Previous results: {len(state.get('previous_time_ranges', []))} time ranges")
                print(f"Current query: {query}")
                print("\n[DECISION] Analyzing user intent...")
            
            decision = decide_refine_or_research(state, llm)
            
            if verbose:
                print(f"  Decision: {decision.get('action')}")
                print(f"  Reason: {decision.get('reason')}")
                if decision.get('extract_number'):
                    print(f"  Extract number: {decision.get('extract_number')}")
            
            if decision.get("action") == "REFINE":
                # Refine existing results
                return refine_existing_results(state, decision, verbose=verbose)
            # Otherwise, continue with new search (fall through)
        
        # Initialize variables
        search_results = []
        time_ranges = []
        confidence = 0.5
        needs_clarification = False
        clarification_question = None
        filtered_seconds = []  # For preserving scored seconds
        
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
        
        # NEW: Analyze query intent
        if verbose:
            print("\n[QUERY ANALYSIS] Analyzing query intent...")
        query_intent = analyze_query_intent(query, llm)
        if verbose:
            print(f"  Is object-centric: {query_intent.get('is_object_centric', False)}")
            print(f"  Primary intent: {query_intent.get('primary_intent', 'hybrid')}")
            print(f"  Mentioned classes: {query_intent.get('mentioned_classes', [])}")
            print(f"  Confidence: {query_intent.get('confidence', 0.7):.2f}")
        
        # NEW: Cross-validate with search plan
        if verbose:
            print("\n[VALIDATION] Cross-validating intent with search plan...")
        query_intent = validate_and_adjust_intent(query_intent, search_plan, verbose=verbose)
        if verbose:
            print(f"  Final intent: {query_intent.get('primary_intent', 'hybrid')}")
            print(f"  Needs semantic: {query_intent.get('needs_semantic', True)}")
            print(f"  Needs activity: {query_intent.get('needs_activity', True)}")
        
        # NEW: Initialize feature extractor
        feature_extractor = PerSecondFeatureExtractor(segment_tree)
        
        # NEW: Get all object classes and configure weights
        all_object_classes = set(segment_tree.get_all_classes().keys())
        weights = configure_weights(query_intent, all_object_classes)
        if verbose:
            print(f"\n[WEIGHT CONFIGURATION]")
            print(f"  Semantic weight: {weights['semantic_weight']:.2f}")
            print(f"  Activity weight: {weights['activity_weight']:.2f}")
            print(f"  Hierarchical weight: {weights['hierarchical_weight']:.2f}")
            print(f"  Threshold: {weights['threshold']:.2f}")
            high_priority_objects = {k: v for k, v in weights['object_weights'].items() if v > 0.3}
            if high_priority_objects:
                print(f"  High priority objects: {high_priority_objects}")
        
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
                    if verbose:
                        print(f"      Evidence scenes: {result.get('evidence_count', 0)}")
            
            if verbose:
                print(f"\n[AGGREGATION] Collected results from all search types:")
                print(f"  Total results: {len(all_search_results)}")
            
            # NEW STEP 3: Score all seconds using weighted features
            if verbose:
                print(f"\n[STEP 3] Scoring all seconds with weighted features...")
            
            # Separate results by type for scoring
            semantic_results = [r for r in all_search_results if r.get("search_type") == "semantic"]
            activity_results = [r for r in all_search_results if r.get("search_type") == "activity"]
            hierarchical_results = [r for r in all_search_results if r.get("search_type") in ["hierarchical", "hierarchical_highlight"]]
            
            # Score all seconds
            scored_seconds = score_seconds(
                feature_extractor,
                query_intent,
                weights,
                semantic_results,
                activity_results,
                hierarchical_results,
                verbose=verbose
            )
            
            # Filter by threshold and sort
            threshold = weights["threshold"]
            filtered_seconds = [s for s in scored_seconds if s["score"] >= threshold]
            filtered_seconds.sort(key=lambda x: x["score"], reverse=True)
            
            if verbose:
                print(f"\n[FILTERING] Filtered to {len(filtered_seconds)} seconds above threshold {threshold:.2f}")
                if filtered_seconds:
                    print(f"  Top 10 scored seconds:")
                    for i, sec in enumerate(filtered_seconds[:10], 1):
                        print(f"    {i}. Second {sec['second']}: score={sec['score']:.3f} "
                              f"(sem={sec['semantic_score']:.2f}, act={sec['activity_score']:.2f}, "
                              f"obj={sec['object_score']:.2f})")
            
            # NEW STEP 4: Group contiguous seconds into time ranges
            if verbose:
                print(f"\n[STEP 4] Grouping contiguous seconds into time ranges...")
            
            time_ranges = group_contiguous_seconds(filtered_seconds, min_duration=1.0, gap_tolerance=2.0)
            
            if verbose:
                print(f"  Grouped into {len(time_ranges)} time range(s)")
                if time_ranges:
                    for i, (start, end) in enumerate(time_ranges, 1):
                        print(f"    {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
            
            # Determine if user wants one or multiple results
            query_lower = query.lower()
            
            # Indicators for multiple results (strong signals)
            multiple_indicators = [
                "all", "every", "each", "multiple", "several",
                "moments", "instances", "scenes", "clips", "times"
            ]
            has_multiple_indicator = any(indicator in query_lower for indicator in multiple_indicators)
            
            # Indicators for single result (strong signals)
            single_indicators = [
                "the moment", "the best", "the first", "the one",
                "a moment", "one moment", "single moment"
            ]
            has_single_indicator = any(indicator in query_lower for indicator in single_indicators)
            
            # Check for plural vs singular "moment"
            has_plural_moment = "moments" in query_lower
            has_singular_moment = "moment" in query_lower and "moments" not in query_lower
            
            # Determine intent
            if has_multiple_indicator:
                user_wants_one = False
                reason = "multiple indicator detected"
            elif has_single_indicator:
                user_wants_one = True
                reason = "single indicator detected"
            elif has_plural_moment:
                user_wants_one = False  # Plural suggests multiple
                reason = "plural 'moments' detected"
            elif has_singular_moment:
                user_wants_one = True  # Singular suggests one
                reason = "singular 'moment' detected"
            else:
                # Default: if we found many results, assume user wants multiple
                user_wants_one = len(time_ranges) <= 3
                reason = f"default (found {len(time_ranges)} ranges)"
            
            if verbose:
                print(f"\n[SELECTION] User intent detection:")
                print(f"  Multiple indicators: {has_multiple_indicator}")
                print(f"  Single indicators: {has_single_indicator}")
                print(f"  Plural 'moments': {has_plural_moment}")
                print(f"  Singular 'moment': {has_singular_moment}")
                print(f"  Decision: {'ONE result' if user_wants_one else 'MULTIPLE results'} ({reason})")
            
            if user_wants_one and time_ranges:
                time_ranges = [time_ranges[0]]
                if verbose:
                    print(f"  [ACTION] Selecting top match only")
            elif not user_wants_one and verbose:
                print(f"  [ACTION] Returning all {len(time_ranges)} time ranges")
            
            # Calculate confidence based on scores
            if filtered_seconds:
                top_score = filtered_seconds[0]["score"]
                confidence = min(0.95, max(0.3, top_score))  # Map score to confidence
            else:
                confidence = 0.3
            
            # Store all search results for reference
            search_results = all_search_results
            
            if verbose:
                print(f"\n[FINAL SELECTION]")
                print(f"  Selected {len(time_ranges)} time range(s)")
                print(f"  Confidence: {confidence:.2f}")
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
        
        # Preserve state for potential refinement (preserve even if clarification needed)
        # This allows user to refine results after clarification
        preserved_scored_seconds = filtered_seconds if filtered_seconds else state.get("previous_scored_seconds")
        
        return {
            **state,
            "query_type": "multi_modal",  # Indicate we used all search types
            "search_results": search_results,
            "time_ranges": time_ranges,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            # Preserve for future refinement (always preserve if we have new results)
            "previous_time_ranges": time_ranges if time_ranges else state.get("previous_time_ranges"),
            "previous_scored_seconds": preserved_scored_seconds,
            "previous_query": query if time_ranges else state.get("previous_query"),
            "previous_search_results": search_results if search_results else state.get("previous_search_results")
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
        "verbose": verbose,
        # Memory fields (initialized to None for first query)
        "previous_time_ranges": None,
        "previous_scored_seconds": None,
        "previous_query": None,
        "previous_search_results": None
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

