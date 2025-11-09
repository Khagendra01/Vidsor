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
        """Planner agent: Analyzes user query and retrieves relevant moments."""
        query = state["user_query"]
        segment_tree = state["segment_tree"]
        verbose = state.get("verbose", False)
        
        if verbose:
            print("\n" + "=" * 60)
            print("PLANNER AGENT: Analyzing Query")
            print("=" * 60)
            print(f"Query: {query}")
            print("\n[THINKING] Determining search strategy...")
        
        system_prompt = """You are a video analysis planner agent. Your job is to:
1. Analyze user queries about video content
2. Determine what type of search is needed (visual descriptions, audio transcriptions, object detection, or activities)
3. Use the available tools to find relevant moments
4. Assess confidence in results
5. Ask for clarification if the query is ambiguous, especially for audio-related queries

Available query types:
- "visual": Search visual descriptions (BLIP/unified)
- "audio": Search audio transcriptions
- "combined": Search both visual and audio
- "object": Find specific objects by class name
- "activity": Check for specific activities (e.g., fishing, catching fish)

For audio queries, be especially careful and ask for clarification if:
- The query uses ambiguous pronouns ("they", "he", "she", "it") without clear context
- The query mentions words that could have multiple meanings in audio context
- The query is vague or could match many different moments
- You're unsure if the user wants to search audio transcriptions or visual descriptions

Examples of queries that need clarification:
- "find when they talk about it" -> Who is "they"? What is "it"?
- "show me that moment" -> Which moment? Need more context
- "find the conversation" -> Too vague, which conversation?

Examples of good queries:
- "find moments where they catch fish" -> Clear activity
- "find when someone says 'Alaska'" -> Clear audio search
- "show me all boats in the video" -> Clear object search

Return your analysis as JSON with:
{
    "query_type": "visual|audio|combined|object|activity",
    "search_keywords": ["keyword1", "keyword2"],
    "object_class": "class_name" (if object search),
    "activity_name": "activity" (if activity search),
    "evidence_keywords": ["keyword1"] (if activity search, for stronger evidence),
    "confidence": 0.0-1.0,
    "needs_clarification": true/false,
    "clarification_question": "question to ask" (if needed)
}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User query: {query}\n\nAnalyze this query and determine how to search the video. Return JSON only.")
        ]
        
        if verbose:
            print("[THINKING] Calling LLM to analyze query...")
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        if verbose:
            print(f"[LLM RESPONSE] {response_text[:200]}...")
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            plan = json.loads(response_text)
        except:
            # Fallback: try to extract JSON from text
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                # Default plan
                plan = {
                    "query_type": "combined",
                    "search_keywords": query.split(),
                    "confidence": 0.5,
                    "needs_clarification": True,
                    "clarification_question": "Could you clarify what specific moment you're looking for?"
                }
        
        if verbose:
            print("\n[PLAN] Extracted plan:")
            print(f"  Query Type: {plan.get('query_type', 'N/A')}")
            print(f"  Search Keywords: {plan.get('search_keywords', [])}")
            print(f"  Confidence: {plan.get('confidence', 0):.2f}")
            print(f"  Needs Clarification: {plan.get('needs_clarification', False)}")
        
        # Execute the search based on plan
        search_results = []
        time_ranges = []
        confidence = plan.get("confidence", 0.5)
        needs_clarification = plan.get("needs_clarification", False)
        clarification_question = plan.get("clarification_question")
        
        # Check for audio query ambiguity
        query_lower = query.lower()
        audio_indicators = ["say", "said", "talk", "speak", "mention", "discuss", "conversation", "audio", "transcription"]
        is_audio_query = any(indicator in query_lower for indicator in audio_indicators)
        
        # Check for ambiguous references
        ambiguous_pronouns = ["they", "he", "she", "it", "that", "this", "those", "these"]
        has_ambiguous_refs = any(pronoun in query_lower.split() for pronoun in ambiguous_pronouns)
        
        if verbose:
            print(f"\n[ANALYSIS] Audio query detected: {is_audio_query}")
            print(f"  Ambiguous references found: {has_ambiguous_refs}")
        
        # Only ask for clarification for audio queries with ambiguous references
        # For visual/activity queries, "they" is usually clear from context
        if is_audio_query and has_ambiguous_refs and not needs_clarification:
            needs_clarification = True
            clarification_question = "I notice your query mentions audio/transcription but uses ambiguous references. Could you clarify who or what you're looking for? For example, 'find when they say [specific word]' or 'find conversation about [specific topic]'."
            if verbose:
                print("[WARNING] Ambiguous audio query detected - will ask for clarification")
        
        # If LLM suggested clarification but it's not an audio query, try searching anyway
        # We'll only ask for clarification if search returns no results
        if needs_clarification and not is_audio_query:
            if verbose:
                print("[INFO] LLM suggested clarification, but query seems clear enough. Will try searching first.")
            needs_clarification = False  # Try searching first
        
        if not needs_clarification:
            if verbose:
                print("\n[EXECUTING] Performing search...")
            query_type = plan.get("query_type", "combined")
            search_keywords = plan.get("search_keywords", [])
            
            if query_type == "visual":
                # Search visual descriptions
                if verbose:
                    print(f"  [SEARCH] Visual search for keywords: {search_keywords}")
                for keyword in search_keywords:
                    results = segment_tree.search_descriptions(keyword, search_type="any")
                    search_results.extend(results)
                    if verbose:
                        print(f"    Keyword '{keyword}': Found {len(results)} visual matches")
            
            elif query_type == "audio":
                # Search audio transcriptions
                if verbose:
                    print(f"  [SEARCH] Audio search for keywords: {search_keywords}")
                for keyword in search_keywords:
                    results = segment_tree.search_transcriptions(keyword)
                    search_results.extend(results)
                    if verbose:
                        print(f"    Keyword '{keyword}': Found {len(results)} audio matches")
                    # Extract time ranges
                    for result in results:
                        tr = result.get("time_range", [])
                        if tr and len(tr) >= 2:
                            time_ranges.append((tr[0], tr[1]))
                            if verbose:
                                print(f"      Time range: {tr[0]:.1f}s - {tr[1]:.1f}s")
            
            elif query_type == "combined":
                # Search all modalities
                if verbose:
                    print(f"  [SEARCH] Combined search (visual + audio) for keywords: {search_keywords}")
                for keyword in search_keywords:
                    result = segment_tree.search_all_modalities(keyword)
                    search_results.append(result)
                    if verbose:
                        print(f"    Keyword '{keyword}': {result.get('visual_count', 0)} visual, {result.get('audio_count', 0)} audio matches")
                    # Extract time ranges from both visual and audio
                    for match in result.get("all_matches", []):
                        tr = match.get("time_range", [])
                        if tr and len(tr) >= 2:
                            time_ranges.append((tr[0], tr[1]))
                            if verbose:
                                source = match.get("source", "unknown")
                                print(f"      [{source}] Time range: {tr[0]:.1f}s - {tr[1]:.1f}s")
            
            elif query_type == "object":
                # Find objects by class
                object_class = plan.get("object_class", search_keywords[0] if search_keywords else "")
                if object_class:
                    if verbose:
                        print(f"  [SEARCH] Object search for class: {object_class}")
                    results = segment_tree.find_objects_by_class(object_class)
                    search_results = results
                    if verbose:
                        print(f"    Found {len(results)} detections of '{object_class}'")
                    # Extract time ranges
                    for result in results:
                        tr = result.get("time_range", [])
                        if tr and len(tr) >= 2:
                            time_ranges.append((tr[0], tr[1]))
                            if verbose:
                                track_id = result.get("detection", {}).get("track_id", "N/A")
                                print(f"      Track {track_id} at {tr[0]:.1f}s - {tr[1]:.1f}s")
            
            elif query_type == "activity":
                # Check for activities - special handling for "fish caught"
                activity_name = plan.get("activity_name", "activity")
                if "fish" in query_lower and ("catch" in query_lower or "caught" in query_lower):
                    # Use the specialized fish catching function
                    if verbose:
                        print(f"  [SEARCH] Activity search: Fish catching (specialized function)")
                    result = segment_tree.check_fish_caught()
                    search_results.append(result)
                    if verbose:
                        print(f"    Fish caught: {result.get('fish_caught', False)}")
                        print(f"    Evidence scenes: {result.get('fish_holding_count', 0)}")
                    # Extract time ranges from evidence
                    for evidence in result.get("evidence", []):
                        tr = evidence.get("time_range", [])
                        if tr and len(tr) >= 2:
                            time_ranges.append((tr[0], tr[1]))
                            if verbose:
                                second = evidence.get("second", "N/A")
                                desc = evidence.get("description", "")[:50]
                                print(f"      Second {second}: {desc}...")
                else:
                    # Use general activity check
                    evidence_keywords = plan.get("evidence_keywords", search_keywords)
                    result = segment_tree.check_activity(
                        activity_keywords=search_keywords,
                        evidence_keywords=evidence_keywords,
                        activity_name=activity_name
                    )
                    search_results.append(result)
                    # Extract time ranges from evidence
                    for evidence in result.get("evidence", []):
                        tr = evidence.get("time_range", [])
                        if tr and len(tr) >= 2:
                            time_ranges.append((tr[0], tr[1]))
            
            # Assess confidence based on results
            if verbose:
                print(f"\n[RESULTS] Search completed:")
                print(f"  Total search results: {len(search_results)}")
                print(f"  Time ranges found: {len(time_ranges)}")
            
            if not search_results or not time_ranges:
                confidence = 0.3
                needs_clarification = True
                clarification_question = "No results found. Could you rephrase your query or provide more details? For example, try 'find moments where someone is cooking' or 'find scenes with food preparation'."
                if verbose:
                    print("  [WARNING] No results found - will ask for clarification")
            elif len(time_ranges) == 0:
                confidence = 0.4
                needs_clarification = True
                clarification_question = "Found some matches but couldn't extract time ranges. Could you be more specific?"
                if verbose:
                    print("  [WARNING] No time ranges extracted - will ask for clarification")
            elif len(time_ranges) > 15:
                # Only ask for clarification if there are way too many results
                confidence = 0.6
                needs_clarification = True
                clarification_question = f"Found {len(time_ranges)} potential moments. Could you narrow down what you're looking for?"
                if verbose:
                    print(f"  [WARNING] Too many results ({len(time_ranges)}) - will ask for clarification")
            else:
                # Good results found, proceed with extraction
                if verbose:
                    print(f"  [SUCCESS] Found {len(time_ranges)} time range(s) - proceeding with extraction")
        
        # Merge overlapping time ranges
        if time_ranges:
            original_count = len(time_ranges)
            time_ranges = merge_time_ranges(time_ranges)
            if verbose:
                print(f"\n[MERGING] Merged {original_count} time ranges into {len(time_ranges)} non-overlapping ranges")
                for i, (start, end) in enumerate(time_ranges, 1):
                    print(f"  Range {i}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
        
        return {
            **state,
            "query_type": plan.get("query_type"),
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
                print(f"\n[LOADING] Opening video file...")
            video = VideoFileClip(video_path)
            if verbose:
                print(f"  Video duration: {video.duration:.2f}s")
                print(f"  FPS: {video.fps}")
                print(f"  Resolution: {video.size}")
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
            try:
                if verbose:
                    print(f"\n[EXTRACTING] Clip {i+1}/{len(time_ranges)}")
                    print(f"  Original range: {start_time:.2f}s - {end_time:.2f}s")
                
                # Ensure times are within video duration
                start_time = max(0, min(start_time, video.duration))
                end_time = max(start_time + 1, min(end_time, video.duration))
                
                if verbose:
                    print(f"  Adjusted range: {start_time:.2f}s - {end_time:.2f}s")
                    print(f"  Duration: {end_time - start_time:.2f}s")
                
                # Extract clip (moviepy 2.x uses subclipped instead of subclip)
                if verbose:
                    print(f"  [PROCESSING] Extracting subclip...")
                clip = video.subclipped(start_time, end_time)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clip_{i+1}_{int(start_time)}s_to_{int(end_time)}s_{timestamp}.mp4"
                output_path = os.path.join(output_dir, filename)
                
                if verbose:
                    print(f"  [SAVING] Writing to: {filename}")
                
                # Write clip (moviepy 2.x uses logger instead of verbose)
                clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger=None  # None suppresses output
                )
                
                clip.close()
                saved_clips.append(output_path)
                
                if verbose:
                    print(f"  [SUCCESS] Clip saved: {output_path}")
                
            except Exception as e:
                error_msg = f"Error extracting clip {i+1}: {str(e)}"
                if verbose:
                    print(f"  [ERROR] {error_msg}")
                print(error_msg)
                continue
        
        video.close()
        
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
        """Ask user for clarification."""
        question = state.get("clarification_question", "Could you provide more details?")
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=question)
            ]
        }
    
    return clarification_node


def should_ask_clarification(state: AgentState) -> Literal["clarify", "execute"]:
    """Router: decide whether to ask for clarification or execute."""
    if state.get("needs_clarification", False):
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
    workflow.add_edge("clarifier", END)
    
    # From executor, end
    workflow.add_edge("executor", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app, segment_tree


def run_agent(query: str, json_path: str, video_path: str, model_name: str = "gpt-4o-mini", verbose: bool = True):
    """Run the video clip extraction agent with a user query."""
    
    # Create agent
    app, segment_tree = create_video_clip_agent(json_path, video_path, model_name)
    
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
    result = app.invoke(initial_state)
    
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

