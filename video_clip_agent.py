"""
LangGraph-based video clip extraction agent with planner and execution agents.
Takes user queries, retrieves relevant moments from video, and saves clips as MP4.

This file is maintained for backward compatibility.
The code has been refactored into the agent/ module.
"""

# Import everything from the agent module for backward compatibility
from agent import (
    AgentState,
    create_video_clip_agent,
    run_agent,
    create_planner_agent,
    create_execution_agent,
    create_clarification_node,
    should_ask_clarification,
)

# Re-export for backward compatibility
__all__ = [
    "AgentState",
    "create_video_clip_agent",
    "run_agent",
    "create_planner_agent",
    "create_execution_agent",
    "create_clarification_node",
    "should_ask_clarification",
]

# Main entry point
if __name__ == "__main__":
    from agent import run_agent
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
