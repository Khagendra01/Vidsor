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
    parser.add_argument(
        "--openshot",
        action="store_true",
        help="Create OpenShot project file after extraction"
    )
    parser.add_argument(
        "--openshot-auto-open",
        action="store_true",
        help="Automatically open OpenShot project after creation (requires --openshot)"
    )
    
    args = parser.parse_args()
    
    # Set verbose based on flags
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print(f"Processing query: {args.query}")
        print(f"Video: {args.video}")
        print(f"JSON: {args.json}")
        if args.openshot:
            print(f"OpenShot: Enabled (auto-open: {args.openshot_auto_open})")
        print()
    
    result = run_agent(
        args.query, 
        args.json, 
        args.video, 
        args.model, 
        verbose=verbose,
        create_openshot_project=args.openshot,
        auto_open_openshot=args.openshot_auto_open
    )
    
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
        
        if result.get("openshot_project_path"):
            print(f"\nOpenShot project: {result.get('openshot_project_path')}")
            print("  You can open this file in OpenShot Video Editor for further editing.")
        
        if result.get("log_file"):
            print(f"\nLog file: {result.get('log_file')}")
            print("  Full execution log saved to file.")