"""
LangGraph-based video clip extraction agent with planner and execution agents.
Takes user queries, retrieves relevant moments from video, and saves clips as MP4.
"""

import dotenv
dotenv.load_dotenv()

from typing import Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from agent.planner import create_planner_agent
from agent.executor import create_execution_agent
from agent.clarifier import create_clarification_node, should_ask_clarification
from agent.utils.segment_tree_utils import load_segment_tree
from agent.utils.logging_utils import DualLogger, create_log_file


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


def run_agent(query: str, json_path: str, video_path: str, model_name: str = "gpt-4o-mini", 
              verbose: bool = True, log_file: Optional[str] = None):
    """
    Run the video clip extraction agent with a user query.
    
    Args:
        query: User query string
        json_path: Path to segment tree JSON file
        video_path: Path to video file
        model_name: LLM model name (default: "gpt-4o-mini")
        verbose: Print verbose output (default: True)
        log_file: Path to log file (if None, auto-generate based on query)
    """
    
    # Set up logging
    if log_file is None:
        log_file = create_log_file(query)
    
    logger = DualLogger(log_file=log_file, verbose=verbose)
    
    if verbose:
        logger.print("\n" + "=" * 60)
        logger.print("VIDEO CLIP EXTRACTION AGENT")
        logger.print("=" * 60)
        logger.print(f"Query: {query}")
        logger.print(f"Video: {video_path}")
        logger.print(f"Segment Tree: {json_path}")
        logger.print(f"Model: {model_name}")
        logger.print(f"Log file: {log_file}")
        logger.print("\n[INITIALIZATION] Starting agent...")
    else:
        logger.info(f"Starting agent - Query: {query}, Video: {video_path}")
    
    # Create agent
    if verbose:
        logger.print("[INITIALIZATION] Loading segment tree...")
    app, segment_tree = create_video_clip_agent(json_path, video_path, model_name)
    
    if verbose:
        video_info = segment_tree.get_video_info()
        logger.print(f"[INITIALIZATION] Segment tree loaded:")
        logger.print(f"  Video duration: {video_info.get('duration_seconds', 0)} seconds")
        logger.print(f"  FPS: {video_info.get('fps', 0)}")
        logger.print(f"  Total frames: {video_info.get('total_frames', 0)}")
        logger.print(f"[INITIALIZATION] Workflow graph created")
        logger.print(f"[INITIALIZATION] Ready to process query\n")
    
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
        "logger": logger,  # Pass logger to state
        # Memory fields (initialized to None for first query)
        "previous_time_ranges": None,
        "previous_scored_seconds": None,
        "previous_query": None,
        "previous_search_results": None
    }
    
    # Run agent
    if verbose:
        logger.print("[WORKFLOW] Invoking agent workflow...")
    result = app.invoke(initial_state)
    
    if verbose:
        logger.print("\n[WORKFLOW] Agent workflow completed")
    
    # Add log file path to result
    result["log_file"] = log_file
    
    return result


# Export main functions and classes
__all__ = [
    "AgentState",
    "create_video_clip_agent",
    "run_agent",
    "create_planner_agent",
    "create_execution_agent",
    "create_clarification_node",
    "should_ask_clarification",
    # Orchestrator exports
    "OrchestratorState",
    "create_orchestrator_agent",
    "run_orchestrator",
]

# Orchestrator imports
from agent.state import OrchestratorState
from agent.orchestrator import create_orchestrator_agent
from agent.orchestrator_runner import run_orchestrator


# Main entry point for command-line usage
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

