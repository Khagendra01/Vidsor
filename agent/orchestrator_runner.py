"""Runner for orchestrator agent - complete workflow integration."""

from typing import Optional, Dict, Any
from agent.orchestrator_state import OrchestratorState
from agent.orchestrator import create_orchestrator_agent
from agent.timeline_manager import TimelineManager
from agent.utils.segment_tree_utils import load_segment_tree


def run_orchestrator(
    query: str,
    timeline_path: str,
    json_path: str,
    video_path: str,
    model_name: str = "gpt-4o-mini",
    verbose: bool = True,
    logger: Optional[Any] = None,
    preserved_state: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Run the orchestrator agent with a user query.
    
    This is the main entry point for using the orchestrator agent.
    It handles the complete workflow: loading timeline, segment tree,
    running orchestrator, and returning results.
    
    Args:
        query: User query (e.g., "find highlights", "cut timeline index 0")
        timeline_path: Path to timeline.json file
        json_path: Path to segment tree JSON file
        video_path: Path to source video file
        model_name: LLM model name (default: "gpt-4o-mini")
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with operation results and updated timeline state
    """
    
    # Use logger if provided, otherwise use print
    log = logger if logger else None
    
    if log:
        log.info("\n" + "=" * 80)
        log.info("ORCHESTRATOR AGENT RUNNER")
        log.info("=" * 80)
        log.info(f"Query: {query}")
        log.info(f"Timeline: {timeline_path}")
        log.info(f"Segment Tree: {json_path}")
        log.info(f"Video: {video_path}")
    elif verbose:
        print("\n" + "=" * 60)
        print("ORCHESTRATOR AGENT RUNNER")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Timeline: {timeline_path}")
        print(f"Segment Tree: {json_path}")
        print(f"Video: {video_path}")
        print()
    
    # Load segment tree
    if log:
        log.info("[LOADING] Loading segment tree...")
    elif verbose:
        print("[LOADING] Loading segment tree...")
    try:
        segment_tree = load_segment_tree(json_path)
        if log:
            log.info(f"  ✓ Segment tree loaded")
        elif verbose:
            print(f"  ✓ Segment tree loaded")
    except Exception as e:
        error_msg = f"  ✗ Failed to load segment tree: {e}"
        if log:
            log.error(error_msg)
        elif verbose:
            print(error_msg)
        return {
            "success": False,
            "error": f"Failed to load segment tree: {e}",
            "operation_result": None
        }
    
    # Create orchestrator agent
    orchestrator_node = create_orchestrator_agent(model_name)
    
    # Prepare initial state
    # If preserved_state is provided (from clarification), use it to continue from where we left off
    initial_state: OrchestratorState = {
        "messages": [],
        "user_query": query,
        "video_path": video_path,
        "json_path": json_path,
        "query_type": None,
        "search_results": preserved_state.get("search_results") if preserved_state else None,
        "time_ranges": preserved_state.get("time_ranges") if preserved_state else None,
        "confidence": None,
        "needs_clarification": False,
        "clarification_question": None,
        "output_clips": [],
        "segment_tree": segment_tree,
        "verbose": verbose,
        "logger": logger,
        "previous_time_ranges": preserved_state.get("previous_time_ranges") if preserved_state else None,
        "previous_scored_seconds": preserved_state.get("previous_scored_seconds") if preserved_state else None,
        "previous_query": preserved_state.get("previous_query") if preserved_state else None,
        "previous_search_results": preserved_state.get("previous_search_results") if preserved_state else None,
        # Orchestrator-specific fields
        "timeline_path": timeline_path,
        "timeline_chunks": None,
        "timeline_version": None,
        "current_operation": None,
        "operation_params": None,
        "selected_timeline_indices": None,
        "broll_time_range": None,
        "editing_history": None,
        "narrative_notes": None,
    }
    
    # Run orchestrator
    try:
        if log:
            log.info("\n[EXECUTING] Running orchestrator agent...")
        elif verbose:
            print("\n[EXECUTING] Running orchestrator agent...")
        
        result_state = orchestrator_node(initial_state)
        
        # Extract results
        operation_result = result_state.get("operation_result")
        current_operation = result_state.get("current_operation")
        timeline_chunks = result_state.get("timeline_chunks")
        
        if log:
            log.info("\n" + "=" * 80)
            log.info("ORCHESTRATOR RESULTS")
            log.info("=" * 80)
            log.info(f"Operation: {current_operation}")
            
            if operation_result:
                if operation_result.get("success"):
                    log.info(f"Status: ✓ Success")
                    if "chunks_created" in operation_result:
                        log.info(f"Chunks created: {len(operation_result['chunks_created'])}")
                    if "chunks_removed" in operation_result:
                        log.info(f"Chunks removed: {len(operation_result['chunks_removed'])}")
                    if "chunks_inserted" in operation_result:
                        log.info(f"Chunks inserted: {len(operation_result['chunks_inserted'])}")
                    if "chunks_added" in operation_result:
                        log.info(f"Chunks added: {len(operation_result['chunks_added'])}")
                else:
                    log.error(f"Status: ✗ Failed")
                    log.error(f"Error: {operation_result.get('error', 'Unknown error')}")
            
            if timeline_chunks:
                log.info(f"Timeline now has {len(timeline_chunks)} chunks")
                total_duration = max(chunk.get("end_time", 0) for chunk in timeline_chunks) if timeline_chunks else 0
                log.info(f"Total duration: {total_duration:.2f}s")
        elif verbose:
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Operation: {current_operation}")
            
            if operation_result:
                if operation_result.get("success"):
                    print(f"Status: ✓ Success")
                    if "chunks_created" in operation_result:
                        print(f"Chunks created: {len(operation_result['chunks_created'])}")
                    if "chunks_removed" in operation_result:
                        print(f"Chunks removed: {len(operation_result['chunks_removed'])}")
                    if "chunks_inserted" in operation_result:
                        print(f"Chunks inserted: {len(operation_result['chunks_inserted'])}")
                else:
                    print(f"Status: ✗ Failed")
                    print(f"Error: {operation_result.get('error', 'Unknown error')}")
            
            if timeline_chunks:
                print(f"\nTimeline now has {len(timeline_chunks)} chunks")
                total_duration = max(chunk.get("end_time", 0) for chunk in timeline_chunks) if timeline_chunks else 0
                print(f"Total duration: {total_duration:.2f}s")
        
        return {
            "success": operation_result.get("success", False) if operation_result else False,
            "operation": current_operation,
            "operation_result": operation_result,
            "timeline_chunks": timeline_chunks,
            "timeline_path": timeline_path,
            "state": result_state
        }
        
    except Exception as e:
        error_msg = f"Orchestrator execution failed: {e}"
        if log:
            log.error(f"\n[ERROR] {error_msg}")
            import traceback
            log.error(traceback.format_exc())
        elif verbose:
            print(f"\n[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "operation_result": None
        }


def run_orchestrator_interactive(
    timeline_path: str,
    json_path: str,
    video_path: str,
    model_name: str = "gpt-4o-mini",
    verbose: bool = True
):
    """
    Run orchestrator in interactive mode - accepts multiple queries.
    
    Args:
        timeline_path: Path to timeline.json file
        json_path: Path to segment tree JSON file
        video_path: Path to source video file
        model_name: LLM model name
        verbose: Whether to print verbose output
    """
    
    print("\n" + "=" * 60)
    print("ORCHESTRATOR INTERACTIVE MODE")
    print("=" * 60)
    print("Enter queries to edit the timeline.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'show' to display current timeline.")
    print()
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break
            
            if query.lower() == "show":
                # Show current timeline
                manager = TimelineManager(timeline_path, verbose=False)
                try:
                    manager.load()
                    print(f"\nTimeline: {len(manager.chunks)} chunks")
                    for i, chunk in enumerate(manager.chunks):
                        print(f"  {i}. {chunk.get('start_time', 0):.1f}s - {chunk.get('end_time', 0):.1f}s "
                              f"({chunk.get('description', 'N/A')[:50]}...)")
                except Exception as e:
                    print(f"Error loading timeline: {e}")
                continue
            
            # Run orchestrator
            result = run_orchestrator(
                query=query,
                timeline_path=timeline_path,
                json_path=json_path,
                video_path=video_path,
                model_name=model_name,
                verbose=verbose
            )
            
            if not result.get("success"):
                print(f"\n⚠️  Operation failed: {result.get('error', 'Unknown error')}")
            
            print()  # Blank line for readability
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Orchestrator agent for timeline editing")
    parser.add_argument("query", nargs="?", help="User query (optional, if not provided runs in interactive mode)")
    parser.add_argument("--timeline", default="projects/asdf/timeline.json", help="Path to timeline.json")
    parser.add_argument("--json", default="projects/asdf/segment_tree.json", help="Path to segment tree JSON")
    parser.add_argument("--video", default="projects/asdf/video/camp_5min.mp4", help="Path to video file")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    if args.query:
        # Single query mode
        result = run_orchestrator(
            query=args.query,
            timeline_path=args.timeline,
            json_path=args.json,
            video_path=args.video,
            model_name=args.model,
            verbose=verbose
        )
        
        if not result.get("success"):
            exit(1)
    else:
        # Interactive mode
        run_orchestrator_interactive(
            timeline_path=args.timeline,
            json_path=args.json,
            video_path=args.video,
            model_name=args.model,
            verbose=verbose
        )

