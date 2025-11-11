"""Operation execution handlers for orchestrator agent."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from agent.timeline_manager import TimelineManager
from agent.orchestrator_state import OrchestratorState


def create_timeline_chunk(
    original_start: float,
    original_end: float,
    timeline_start: float,
    chunk_type: str = "highlight",
    description: str = "",
    unified_description: str = "",
    audio_description: str = "",
    score: float = 1.0,
    clip_path: Optional[str] = None
) -> Dict:
    """
    Create a timeline chunk dictionary.
    
    Args:
        original_start: Start time in source video
        original_end: End time in source video
        timeline_start: Start time in timeline
        chunk_type: Type of chunk (highlight, broll, etc.)
        description: Visual description
        unified_description: Unified visual description
        audio_description: Audio description
        score: Relevance score
        clip_path: Path to extracted clip file
        
    Returns:
        Chunk dictionary
    """
    duration = original_end - original_start
    return {
        "start_time": timeline_start,
        "end_time": timeline_start + duration,
        "chunk_type": chunk_type,
        "speed": 1.0,
        "description": description or f"Clip from {original_start:.1f}s to {original_end:.1f}s",
        "score": score,
        "original_start_time": original_start,
        "original_end_time": original_end,
        "unified_description": unified_description or description,
        "audio_description": audio_description or "",
        "clip_path": clip_path
    }


def handle_find_highlights(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    planner_agent,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle FIND_HIGHLIGHTS operation.
    Calls planner agent to find highlights, then creates timeline chunks.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        planner_agent: Planner agent function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and created chunks
    """
    if verbose:
        print("\n[OPERATION] FIND_HIGHLIGHTS")
        print("  Calling planner agent to find highlights...")
    
    # Prepare state for planner (ensure all required fields)
    planner_state = {
        "user_query": state.get("user_query", "find highlights"),
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "time_ranges": None,  # Will be set by planner
        "needs_clarification": False,
        "messages": state.get("messages", []),
    }
    
    # Call planner agent
    try:
        planner_result = planner_agent(planner_state)
        
        if planner_result.get("needs_clarification"):
            return {
                "success": False,
                "error": planner_result.get("clarification_question", "Clarification needed"),
                "chunks_created": []
            }
        
        time_ranges = planner_result.get("time_ranges", [])
        if not time_ranges:
            return {
                "success": False,
                "error": "No highlights found",
                "chunks_created": []
            }
        
        if verbose:
            print(f"  Found {len(time_ranges)} highlight time ranges")
        
        # Create timeline chunks from time ranges
        chunks_created = []
        current_timeline_time = timeline_manager.calculate_timeline_duration()
        
        # Get search results for descriptions
        search_results = planner_result.get("search_results", [])
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            # Try to get description from search results
            description = f"Highlight {i+1}: {start_time:.1f}s - {end_time:.1f}s"
            unified_description = description
            audio_description = ""
            
            # Look for matching search result
            for result in search_results:
                result_tr = result.get("time_range", [])
                if result_tr and len(result_tr) >= 2:
                    if abs(result_tr[0] - start_time) < 1.0:  # Close match
                        description = result.get("description", description)
                        unified_description = result.get("unified_description", description)
                        audio_description = result.get("audio_description", "")
                        break
            
            chunk = create_timeline_chunk(
                original_start=start_time,
                original_end=end_time,
                timeline_start=current_timeline_time,
                chunk_type="highlight",
                description=description,
                unified_description=unified_description,
                audio_description=audio_description,
                score=planner_result.get("confidence", 0.7)
            )
            
            chunks_created.append(chunk)
            current_timeline_time = chunk["end_time"]
            
            if verbose:
                print(f"    Created chunk {i+1}: timeline {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s "
                      f"(source: {start_time:.1f}s - {end_time:.1f}s)")
        
        # Add chunks to timeline
        timeline_manager.chunks.extend(chunks_created)
        
        if verbose:
            print(f"  ✓ Created {len(chunks_created)} highlight chunks")
        
        return {
            "success": True,
            "chunks_created": chunks_created,
            "time_ranges": time_ranges
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error in FIND_HIGHLIGHTS: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_created": []
        }


def handle_cut(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle CUT operation.
    Removes chunks at specified timeline indices.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with timeline_indices
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and removed chunks
    """
    if verbose:
        print("\n[OPERATION] CUT")
    
    indices = params.get("timeline_indices", [])
    if not indices:
        return {
            "success": False,
            "error": "No timeline indices provided",
            "chunks_removed": []
        }
    
    # Validate indices
    is_valid, error = timeline_manager.validate_indices(indices)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "chunks_removed": []
        }
    
    # Sort indices in descending order to remove from end to start
    # This prevents index shifting issues
    sorted_indices = sorted(indices, reverse=True)
    
    removed_chunks = []
    for idx in sorted_indices:
        if 0 <= idx < len(timeline_manager.chunks):
            removed_chunks.append(timeline_manager.chunks.pop(idx))
            if verbose:
                print(f"  Removed chunk at index {idx}")
    
    # Recalculate timeline start_times for remaining chunks
    current_time = 0.0
    for chunk in timeline_manager.chunks:
        duration = chunk["end_time"] - chunk["start_time"]
        chunk["start_time"] = current_time
        chunk["end_time"] = current_time + duration
        current_time = chunk["end_time"]
    
    if verbose:
        print(f"  ✓ Removed {len(removed_chunks)} chunk(s)")
        print(f"  Timeline now has {len(timeline_manager.chunks)} chunks")
    
    return {
        "success": True,
        "chunks_removed": removed_chunks,
        "remaining_chunks": len(timeline_manager.chunks)
    }


def handle_replace(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    planner_agent,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle REPLACE operation.
    Replaces chunks at specified indices with new content from planner.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with timeline_indices and search_query
        planner_agent: Planner agent function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and replaced chunks
    """
    if verbose:
        print("\n[OPERATION] REPLACE")
    
    indices = params.get("timeline_indices", [])
    search_query = params.get("search_query")
    
    if not indices:
        return {
            "success": False,
            "error": "No timeline indices provided",
            "chunks_replaced": []
        }
    
    if not search_query:
        return {
            "success": False,
            "error": "No search query provided for replacement",
            "chunks_replaced": []
        }
    
    # Validate indices
    is_valid, error = timeline_manager.validate_indices(indices)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "chunks_replaced": []
        }
    
    # Get chunks to be replaced (for reference)
    chunks_to_replace = timeline_manager.get_chunks(indices)
    
    if verbose:
        print(f"  Replacing {len(indices)} chunk(s) at indices {indices}")
        print(f"  Search query: '{search_query}'")
    
    # Call planner to find replacement content
    planner_state = {
        "user_query": search_query,
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "time_ranges": None,
        "needs_clarification": False,
        "messages": state.get("messages", []),
    }
    
    try:
        planner_result = planner_agent(planner_state)
        
        if planner_result.get("needs_clarification"):
            return {
                "success": False,
                "error": planner_result.get("clarification_question", "Clarification needed"),
                "chunks_replaced": []
            }
        
        time_ranges = planner_result.get("time_ranges", [])
        if not time_ranges:
            return {
                "success": False,
                "error": "No replacement content found",
                "chunks_replaced": []
            }
        
        # Limit replacement to same number of chunks (or fewer)
        if len(time_ranges) > len(indices):
            time_ranges = time_ranges[:len(indices)]
            if verbose:
                print(f"  Limiting replacement to {len(indices)} chunk(s)")
        
        # Sort indices in descending order for replacement
        sorted_indices = sorted(indices, reverse=True)
        
        # Remove old chunks
        removed_chunks = []
        for idx in sorted_indices:
            if 0 <= idx < len(timeline_manager.chunks):
                removed_chunks.append(timeline_manager.chunks.pop(idx))
        
        # Create new chunks from planner results
        # Insert at the position of the first removed chunk
        insert_position = min(indices) if indices else 0
        
        new_chunks = []
        current_timeline_time = timeline_manager.chunks[insert_position - 1]["end_time"] if insert_position > 0 else 0.0
        
        search_results = planner_result.get("search_results", [])
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            # Get description from search results
            description = f"Replacement clip {i+1}: {start_time:.1f}s - {end_time:.1f}s"
            unified_description = description
            audio_description = ""
            
            for result in search_results:
                result_tr = result.get("time_range", [])
                if result_tr and len(result_tr) >= 2:
                    if abs(result_tr[0] - start_time) < 1.0:
                        description = result.get("description", description)
                        unified_description = result.get("unified_description", description)
                        audio_description = result.get("audio_description", "")
                        break
            
            chunk = create_timeline_chunk(
                original_start=start_time,
                original_end=end_time,
                timeline_start=current_timeline_time,
                chunk_type="highlight",
                description=description,
                unified_description=unified_description,
                audio_description=audio_description,
                score=planner_result.get("confidence", 0.7)
            )
            
            new_chunks.append(chunk)
            current_timeline_time = chunk["end_time"]
        
        # Insert new chunks at the position
        for i, chunk in enumerate(new_chunks):
            timeline_manager.chunks.insert(insert_position + i, chunk)
        
        # Recalculate timeline start_times
        current_time = 0.0
        for chunk in timeline_manager.chunks:
            duration = chunk["end_time"] - chunk["start_time"]
            chunk["start_time"] = current_time
            chunk["end_time"] = current_time + duration
            current_time = chunk["end_time"]
        
        if verbose:
            print(f"  ✓ Replaced {len(removed_chunks)} chunk(s) with {len(new_chunks)} new chunk(s)")
        
        return {
            "success": True,
            "chunks_removed": removed_chunks,
            "chunks_added": new_chunks,
            "chunks_replaced": len(new_chunks)
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error in REPLACE: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_replaced": []
        }


def handle_insert(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    planner_agent,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle INSERT operation.
    Inserts new clips at specified position in timeline.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with insert position and search_query
        planner_agent: Planner agent function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and inserted chunks
    """
    if verbose:
        print("\n[OPERATION] INSERT")
    
    search_query = params.get("search_query")
    insert_before_index = params.get("insert_before_index")
    insert_after_index = params.get("insert_after_index")
    insert_between_indices = params.get("insert_between_indices")
    
    # Determine insert position
    insert_position = None
    
    if insert_between_indices:
        # Insert between two indices
        idx1, idx2 = insert_between_indices[0], insert_between_indices[1]
        if 0 <= idx1 < len(timeline_manager.chunks) and 0 <= idx2 < len(timeline_manager.chunks):
            insert_position = idx2  # Insert after first index, before second
    elif insert_after_index is not None:
        # Insert after specified index
        if 0 <= insert_after_index < len(timeline_manager.chunks):
            insert_position = insert_after_index + 1
    elif insert_before_index is not None:
        # Insert before specified index
        if 0 <= insert_before_index < len(timeline_manager.chunks):
            insert_position = insert_before_index
    
    if insert_position is None:
        return {
            "success": False,
            "error": "Invalid insert position",
            "chunks_inserted": []
        }
    
    if not search_query:
        # Default query if none provided
        search_query = "interesting moments"
    
    if verbose:
        print(f"  Inserting at position {insert_position}")
        print(f"  Search query: '{search_query}'")
    
    # Call planner to find content
    planner_state = {
        "user_query": search_query,
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "time_ranges": None,
        "needs_clarification": False,
        "messages": state.get("messages", []),
    }
    
    try:
        planner_result = planner_agent(planner_state)
        
        if planner_result.get("needs_clarification"):
            return {
                "success": False,
                "error": planner_result.get("clarification_question", "Clarification needed"),
                "chunks_inserted": []
            }
        
        time_ranges = planner_result.get("time_ranges", [])
        if not time_ranges:
            return {
                "success": False,
                "error": "No content found to insert",
                "chunks_inserted": []
            }
        
        # Calculate timeline start time for insertion
        if insert_position > 0:
            timeline_start = timeline_manager.chunks[insert_position - 1]["end_time"]
        else:
            timeline_start = 0.0
        
        # Create chunks from planner results
        new_chunks = []
        current_timeline_time = timeline_start
        
        search_results = planner_result.get("search_results", [])
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            # Get description from search results
            description = f"Inserted clip {i+1}: {start_time:.1f}s - {end_time:.1f}s"
            unified_description = description
            audio_description = ""
            
            for result in search_results:
                result_tr = result.get("time_range", [])
                if result_tr and len(result_tr) >= 2:
                    if abs(result_tr[0] - start_time) < 1.0:
                        description = result.get("description", description)
                        unified_description = result.get("unified_description", description)
                        audio_description = result.get("audio_description", "")
                        break
            
            chunk = create_timeline_chunk(
                original_start=start_time,
                original_end=end_time,
                timeline_start=current_timeline_time,
                chunk_type="highlight",
                description=description,
                unified_description=unified_description,
                audio_description=audio_description,
                score=planner_result.get("confidence", 0.7)
            )
            
            new_chunks.append(chunk)
            current_timeline_time = chunk["end_time"]
        
        # Insert chunks at position
        for i, chunk in enumerate(new_chunks):
            timeline_manager.chunks.insert(insert_position + i, chunk)
        
        # Recalculate timeline start_times for all chunks after insertion
        current_time = 0.0
        for chunk in timeline_manager.chunks:
            duration = chunk["end_time"] - chunk["start_time"]
            chunk["start_time"] = current_time
            chunk["end_time"] = current_time + duration
            current_time = chunk["end_time"]
        
        if verbose:
            print(f"  ✓ Inserted {len(new_chunks)} chunk(s) at position {insert_position}")
        
        return {
            "success": True,
            "chunks_inserted": new_chunks,
            "insert_position": insert_position
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error in INSERT: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_inserted": []
        }


def handle_find_broll(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    planner_agent,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle FIND_BROLL operation.
    Finds complementary B-roll for selected timeline segments.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with timeline_indices
        planner_agent: Planner agent function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and B-roll chunks
    """
    if verbose:
        print("\n[OPERATION] FIND_BROLL")
    
    indices = params.get("timeline_indices", [])
    if not indices:
        return {
            "success": False,
            "error": "No timeline indices provided",
            "chunks_created": []
        }
    
    # Validate indices
    is_valid, error = timeline_manager.validate_indices(indices)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "chunks_created": []
        }
    
    # Get time range from selected chunks
    time_range = timeline_manager.get_timeline_range(indices)
    if not time_range:
        return {
            "success": False,
            "error": "Could not determine time range from selected chunks",
            "chunks_created": []
        }
    
    start_time, end_time = time_range
    
    # Analyze main action from selected chunks
    selected_chunks = timeline_manager.get_chunks(indices)
    main_keywords = []
    for chunk in selected_chunks:
        desc = chunk.get("unified_description", chunk.get("description", ""))
        # Extract key nouns/verbs (simple heuristic)
        words = desc.lower().split()
        main_keywords.extend([w for w in words if len(w) > 4])  # Longer words are likely keywords
    
    # Build B-roll search query
    broll_query = f"find B-roll between {start_time:.1f}s and {end_time:.1f}s, show nature, scenery, wide shots, different from main action"
    if main_keywords:
        # Exclude main action keywords
        exclude_terms = ", ".join(set(main_keywords[:5]))  # Top 5 unique keywords
        broll_query += f", not {exclude_terms}"
    
    if verbose:
        print(f"  Selected chunks: indices {indices}")
        print(f"  Source time range: {start_time:.1f}s - {end_time:.1f}s")
        print(f"  B-roll query: '{broll_query}'")
    
    # Call planner to find B-roll
    planner_state = {
        "user_query": broll_query,
        "video_path": state.get("video_path", ""),
        "json_path": state.get("json_path", ""),
        "segment_tree": state.get("segment_tree"),
        "verbose": verbose,
        "time_ranges": None,
        "needs_clarification": False,
        "messages": state.get("messages", []),
    }
    
    try:
        planner_result = planner_agent(planner_state)
        
        if planner_result.get("needs_clarification"):
            return {
                "success": False,
                "error": planner_result.get("clarification_question", "Clarification needed"),
                "chunks_created": []
            }
        
        time_ranges = planner_result.get("time_ranges", [])
        if not time_ranges:
            return {
                "success": False,
                "error": "No B-roll found",
                "chunks_created": []
            }
        
        # Create B-roll chunks
        chunks_created = []
        # Insert B-roll after the last selected chunk
        insert_position = max(indices) + 1
        if insert_position > len(timeline_manager.chunks):
            insert_position = len(timeline_manager.chunks)
        
        # Calculate timeline start time
        if insert_position > 0:
            timeline_start = timeline_manager.chunks[insert_position - 1]["end_time"]
        else:
            timeline_start = 0.0
        
        current_timeline_time = timeline_start
        search_results = planner_result.get("search_results", [])
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            # Get description from search results
            description = f"B-roll {i+1}: {start_time:.1f}s - {end_time:.1f}s"
            unified_description = description
            audio_description = ""
            
            for result in search_results:
                result_tr = result.get("time_range", [])
                if result_tr and len(result_tr) >= 2:
                    if abs(result_tr[0] - start_time) < 1.0:
                        description = result.get("description", description)
                        unified_description = result.get("unified_description", description)
                        audio_description = result.get("audio_description", "")
                        break
            
            chunk = create_timeline_chunk(
                original_start=start_time,
                original_end=end_time,
                timeline_start=current_timeline_time,
                chunk_type="broll",
                description=description,
                unified_description=unified_description,
                audio_description=audio_description,
                score=planner_result.get("confidence", 0.7)
            )
            
            chunks_created.append(chunk)
            current_timeline_time = chunk["end_time"]
        
        # Insert B-roll chunks
        for i, chunk in enumerate(chunks_created):
            timeline_manager.chunks.insert(insert_position + i, chunk)
        
        # Recalculate timeline start_times
        current_time = 0.0
        for chunk in timeline_manager.chunks:
            duration = chunk["end_time"] - chunk["start_time"]
            chunk["start_time"] = current_time
            chunk["end_time"] = current_time + duration
            current_time = chunk["end_time"]
        
        if verbose:
            print(f"  ✓ Created {len(chunks_created)} B-roll chunk(s) at position {insert_position}")
        
        return {
            "success": True,
            "chunks_created": chunks_created,
            "insert_position": insert_position
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error in FIND_BROLL: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_created": []
        }

