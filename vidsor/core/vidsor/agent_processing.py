"""
Agent processing functionality for orchestrator queries.
"""
import os
import json
import tkinter as tk
from typing import Dict
from agent.orchestrator_runner import run_orchestrator
from agent.utils.logging_utils import DualLogger, create_log_file


def run_agent_thread_with_clarification(self, clarification_response: str, segment_tree_path: str,
                                        operation: str, preserved_state: Dict, original_query: str):
    """Run orchestrator with clarification response, using preserved state to continue."""
    # Create logger for this clarification response
    log_file = create_log_file(f"{original_query}_clarification_{clarification_response}")
    logger = DualLogger(log_file=log_file, verbose=True)
    
    logger.info("=" * 80)
    logger.info("VIDSOR: Continuing with Clarification Response")
    logger.info("=" * 80)
    logger.info(f"Original query: {original_query}")
    logger.info(f"Clarification response: {clarification_response}")
    logger.info(f"Operation: {operation}")
    logger.info(f"Preserved state keys: {list(preserved_state.keys()) if preserved_state else []}")
    
    try:
        timeline_path = os.path.join(self.current_project_path, "timeline.json")
        
        # Use preserved state to continue - planner will refine results instead of re-searching
        # Combine original query with clarification response
        combined_query = f"{original_query} ({clarification_response})"
        
        logger.info(f"Combined query: {combined_query}")
        logger.info("Calling orchestrator with preserved state (planner will refine instead of re-searching)")
        logger.info(f"Preserved time_ranges: {len(preserved_state.get('time_ranges', []))}")
        logger.info(f"Preserved search_results: {len(preserved_state.get('search_results', []))}")
        
        result = run_orchestrator(
            query=combined_query,
            timeline_path=timeline_path,
            json_path=segment_tree_path,
            video_path=self.video_path,
            model_name="gpt-4o-mini",
            verbose=False,
            logger=logger,
            preserved_state=preserved_state  # Pass preserved state so planner can refine
        )
        
        # Process result same as normal flow
        operation_result = result.get("operation_result", {})
        success = result.get("success", False)
        timeline_chunks = result.get("timeline_chunks", [])
        
        # Generate response
        if not success:
            error_msg = operation_result.get("error", "Operation failed")
            logger.error(f"Operation failed: {error_msg}")
            response = f"Error: {error_msg}"
        elif operation == "FIND_HIGHLIGHTS":
            chunks_created = operation_result.get("chunks_created", [])
            if chunks_created:
                response = f"Found and added {len(chunks_created)} clip(s) to timeline based on your clarification."
            else:
                response = "No matching clips found based on your clarification."
        else:
            response = f"Operation completed based on your clarification."
        
        # Update timeline UI if succeeded
        if success and timeline_chunks is not None:
            if self.root:
                def update_timeline_ui():
                    try:
                        self._load_timeline()
                        if self.timeline_canvas:
                            self._draw_timeline()
                        # Update UI button states (especially play button)
                        self._update_ui_state()
                    except Exception as e:
                        logger.error(f"Error updating timeline UI: {e}")
                
                self.root.after(0, update_timeline_ui)
        
        # Update UI
        if self.root:
            self.root.after(0, lambda: self._add_chat_message("assistant", response))
            self.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
            self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
        
    except Exception as e:
        error_msg = f"Error processing clarification: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        if self.root:
            self.root.after(0, lambda: self._add_chat_message("assistant", error_msg))
            self.root.after(0, lambda: self.chat_status_label.config(text="Error occurred", foreground="red"))
            self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
    finally:
        self.agent_integration.is_agent_running = False
        logger.info("Clarification processing completed")


def run_agent_thread(self, query: str, segment_tree_path: str):
    """Run orchestrator agent in background thread."""
    # Create logger for this query
    log_file = create_log_file(query)
    logger = DualLogger(log_file=log_file, verbose=True)
    
    logger.info("=" * 80)
    logger.info("VIDSOR: Orchestrator Query Processing")
    logger.info("=" * 80)
    logger.info(f"Query: {query}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Project: {self.current_project_path}")
    logger.info(f"Video: {self.video_path}")
    logger.info(f"Segment tree: {segment_tree_path}")
    
    try:
        # Get timeline path from current project
        if not self.current_project_path:
            error_msg = "No project selected. Please select a project first."
            logger.error(error_msg)
            if self.root:
                self.root.after(0, lambda: self._add_chat_message("assistant", error_msg))
                self.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
            return
        
        timeline_path = os.path.join(self.current_project_path, "timeline.json")
        logger.info(f"Timeline path: {timeline_path}")
        
        # Check if there's a pending clarification and if this query looks like a follow-up
        if self.agent_integration.pending_clarification:
            preserved = self.agent_integration.pending_clarification
            previous_query = preserved.get("original_query", "")
            preserved_state = preserved.get("preserved_state", {})
            previous_time_ranges = preserved_state.get("time_ranges", [])
            
            # Simple heuristic: if query contains refinement keywords or numbers, treat as follow-up
            query_lower = query.lower()
            refinement_keywords = ["top", "best", "first", "most", "select", "give me", "show me", "only"]
            has_refinement = any(kw in query_lower for kw in refinement_keywords)
            has_number = any(char.isdigit() for char in query)
            
            if has_refinement or has_number:
                logger.info(f"Detected follow-up query to clarification")
                logger.info(f"  Previous query: {previous_query}")
                logger.info(f"  Current query: {query}")
                logger.info(f"  Previous results: {len(previous_time_ranges)} time ranges")
                logger.info("  Treating as clarification response - using preserved_state")
                
                # Use the clarification handler instead
                operation = preserved.get("operation")
                original_query = preserved.get("original_query")
                
                # Clear pending clarification
                self.agent_integration.pending_clarification = None
                
                # Run with preserved state
                self._run_agent_thread_with_clarification(
                    query,  # Use current query as clarification response
                    segment_tree_path,
                    operation,
                    preserved_state,
                    original_query
                )
                return
        
        # Check if timeline exists and log current state
        if os.path.exists(timeline_path):
            try:
                # Check if file is empty or whitespace only
                with open(timeline_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        logger.info("Timeline.json is empty - starting with empty timeline")
                    else:
                        # Parse JSON
                        with open(timeline_path, 'r') as f2:
                            timeline_data = json.load(f2)
                            chunks_count = len(timeline_data.get("chunks", []))
                            logger.info(f"Existing timeline loaded: {chunks_count} chunks")
            except json.JSONDecodeError as e:
                logger.warning(f"Timeline.json contains invalid JSON: {e}")
                logger.info("Starting with empty timeline")
            except Exception as e:
                logger.warning(f"Could not read existing timeline: {e}")
                logger.info("Starting with empty timeline")
        else:
            logger.info("Timeline.json does not exist - will be created")
        
        # Run orchestrator (it handles timeline.json loading, operations, and saving)
        logger.info("\n" + "-" * 80)
        logger.info("CALLING ORCHESTRATOR")
        logger.info("-" * 80)
        result = run_orchestrator(
            query=query,
            timeline_path=timeline_path,
            json_path=segment_tree_path,
            video_path=self.video_path,
            model_name="gpt-4o-mini",
            verbose=False,  # Use logger instead
            logger=logger  # Pass logger to orchestrator
        )
        
        logger.info("\n" + "-" * 80)
        logger.info("ORCHESTRATOR RESULTS")
        logger.info("-" * 80)
        
        # Extract orchestrator results
        operation = result.get("operation", "UNKNOWN")
        operation_result = result.get("operation_result", {})
        success = result.get("success", False)
        timeline_chunks = result.get("timeline_chunks", [])
        
        logger.info(f"Operation: {operation}")
        logger.info(f"Success: {success}")
        logger.info(f"Timeline chunks after operation: {len(timeline_chunks) if timeline_chunks else 0}")
        
        if operation_result:
            logger.info(f"Operation result keys: {list(operation_result.keys())}")
            if "chunks_created" in operation_result:
                logger.info(f"  Chunks created: {len(operation_result['chunks_created'])}")
            if "chunks_removed" in operation_result:
                logger.info(f"  Chunks removed: {len(operation_result['chunks_removed'])}")
            if "chunks_added" in operation_result:
                logger.info(f"  Chunks added: {len(operation_result['chunks_added'])}")
            if "chunks_inserted" in operation_result:
                logger.info(f"  Chunks inserted: {len(operation_result['chunks_inserted'])}")
        
        # Check if this is a clarification request (not a real error)
        needs_clarification = operation_result.get("needs_clarification", False)
        clarification_question = operation_result.get("clarification_question")
        preserved_state = operation_result.get("preserved_state")
        
        # If clarification_question is in error field but needs_clarification flag is missing, extract it
        if not clarification_question and not success:
            error_msg = operation_result.get("error", "")
            # Check if error message looks like a clarification question
            if error_msg and ("Could you narrow down" in error_msg or "potential moments" in error_msg.lower() or "Found" in error_msg and "moment" in error_msg.lower()):
                clarification_question = error_msg
                needs_clarification = True
                logger.info(f"Detected clarification question in error field: {clarification_question}")
                # If we detected clarification but preserved_state wasn't set, try to get it from operation_result
                if not preserved_state:
                    preserved_state = operation_result.get("preserved_state")
                    if preserved_state:
                        logger.info(f"Found preserved_state in operation_result: {len(preserved_state.get('time_ranges', []))} time ranges")
                    else:
                        logger.warning("Clarification detected but no preserved_state found - follow-up queries may not work correctly")
                
                # Ensure previous_time_ranges is set for refinement logic to work
                if preserved_state and "previous_time_ranges" not in preserved_state:
                    # If time_ranges exists but previous_time_ranges doesn't, copy it
                    if "time_ranges" in preserved_state:
                        preserved_state["previous_time_ranges"] = preserved_state["time_ranges"]
                        logger.info(f"Set previous_time_ranges from time_ranges: {len(preserved_state['time_ranges'])} ranges")
                    # Also set previous_query if not set
                    if "previous_query" not in preserved_state:
                        preserved_state["previous_query"] = query
                        logger.info(f"Set previous_query: {query}")
        
        # Generate response based on operation type
        response_parts = []
        
        if needs_clarification and clarification_question:
            # This is a clarification request, not an error
            logger.info(f"Clarification needed: {clarification_question}")
            logger.info("Preserving state for continuation")
            
            # Store preserved state for when user responds
            if self.root:
                def store_clarification_state():
                    # Ensure previous_time_ranges is set for refinement logic
                    if preserved_state:
                        if "previous_time_ranges" not in preserved_state and "time_ranges" in preserved_state:
                            preserved_state["previous_time_ranges"] = preserved_state["time_ranges"]
                        if "previous_query" not in preserved_state:
                            preserved_state["previous_query"] = query
                    
                    self.agent_integration.pending_clarification = {
                        "operation": operation,
                        "preserved_state": preserved_state,
                        "original_query": query,
                        "segment_tree_path": segment_tree_path,
                        "timeline_path": timeline_path
                    }
                    logger.info("Clarification state stored")
                    if preserved_state:
                        logger.info(f"  Preserved {len(preserved_state.get('time_ranges', preserved_state.get('previous_time_ranges', [])))} time ranges")
                
                self.root.after(0, store_clarification_state)
            
            # Show clarification question in chat (without "Error:" prefix)
            response = f"❓ {clarification_question}\n\nPlease respond to continue."
        elif not success:
            error_msg = operation_result.get("error", "Operation failed")
            logger.error(f"Operation failed: {error_msg}")
            response = f"Error: {error_msg}"
        elif operation == "FIND_HIGHLIGHTS":
            chunks_created = operation_result.get("chunks_created", [])
            if chunks_created:
                logger.info(f"FIND_HIGHLIGHTS: Created {len(chunks_created)} chunks")
                response_parts.append(f"Found and added {len(chunks_created)} clip(s) to timeline:")
                for i, chunk in enumerate(chunks_created, 1):
                    start = chunk.get("original_start_time", 0)
                    end = chunk.get("original_end_time", 0)
                    timeline_start = chunk.get("start_time", 0)
                    timeline_end = chunk.get("end_time", 0)
                    response_parts.append(f"  {i}. {start:.1f}s - {end:.1f}s")
                    logger.debug(f"  Chunk {i}: source={start:.1f}s-{end:.1f}s, timeline={timeline_start:.1f}s-{timeline_end:.1f}s")
            else:
                logger.warning("FIND_HIGHLIGHTS: No chunks created")
                response_parts.append("No matching clips found.")
            response = "\n".join(response_parts)
        elif operation == "CUT":
            chunks_removed = operation_result.get("chunks_removed", [])
            logger.info(f"CUT: Removed {len(chunks_removed)} chunks")
            response = f"Removed {len(chunks_removed)} clip(s) from timeline."
        elif operation == "REPLACE":
            chunks_added = operation_result.get("chunks_added", [])
            chunks_removed = operation_result.get("chunks_removed", [])
            logger.info(f"REPLACE: Removed {len(chunks_removed)}, Added {len(chunks_added)} chunks")
            response = f"Replaced {len(chunks_removed)} clip(s) with {len(chunks_added)} new clip(s)."
        elif operation == "INSERT":
            chunks_inserted = operation_result.get("chunks_inserted", [])
            logger.info(f"INSERT: Inserted {len(chunks_inserted)} chunks")
            response = f"Inserted {len(chunks_inserted)} clip(s) into timeline."
        elif operation == "FIND_BROLL":
            chunks_created = operation_result.get("chunks_created", [])
            logger.info(f"FIND_BROLL: Created {len(chunks_created)} B-roll chunks")
            response = f"Found and added {len(chunks_created)} B-roll clip(s) to timeline."
        elif operation == "UNKNOWN":
            logger.warning("Operation classification: UNKNOWN")
            response = "I couldn't understand what you want to do. Please try rephrasing your query."
        else:
            logger.info(f"Operation '{operation}' completed")
            response = f"Operation '{operation}' completed."
        
        logger.info(f"Response message: {response[:100]}...")
        
        # Update timeline UI if operation succeeded and timeline changed
        if success and timeline_chunks is not None:
            logger.info("\n" + "-" * 80)
            logger.info("UPDATING TIMELINE UI")
            logger.info("-" * 80)
            logger.info(f"Timeline chunks to display: {len(timeline_chunks)}")
            
            if self.root:
                def update_timeline_ui():
                    try:
                        logger.info("Reloading timeline from file...")
                        # Reload timeline from file (orchestrator already saved it)
                        self._load_timeline()
                        logger.info(f"Timeline loaded: {len(self.edit_state.chunks)} chunks")
                        
                        # Update timeline display
                        if self.timeline_canvas:
                            logger.info("Drawing timeline on canvas...")
                            self._draw_timeline()
                            logger.info("Timeline canvas updated")
                        
                        # Update UI button states (especially play button)
                        logger.info("Updating UI button states...")
                        self._update_ui_state()
                        
                        logger.info(f"[VIDSOR] Timeline updated: {len(timeline_chunks)} chunks")
                    except Exception as e:
                        logger.error(f"Error updating timeline UI: {e}")
                        import traceback
                        traceback.print_exc()
                
                self.root.after(0, update_timeline_ui)
        else:
            logger.info("Skipping timeline UI update (success=False or no timeline_chunks)")
        
        # Update UI in main thread
        logger.info("\n" + "-" * 80)
        logger.info("UPDATING CHAT UI")
        logger.info("-" * 80)
        if self.root:
            self.root.after(0, lambda: self._add_chat_message("assistant", response))
            self.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
            self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
            logger.info("Chat UI update scheduled")
        
        logger.info("\n" + "=" * 80)
        logger.info("QUERY PROCESSING COMPLETED")
        logger.info("=" * 80)
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error("\n" + "=" * 80)
        logger.error("ERROR OCCURRED")
        logger.error("=" * 80)
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        print(f"[VIDSOR] {error_msg}")
        traceback.print_exc()
        if self.root:
            self.root.after(0, lambda: self._add_chat_message("assistant", error_msg))
            self.root.after(0, lambda: self.chat_status_label.config(text="Error occurred", foreground="red"))
            self.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
    finally:
        self.agent_integration.is_agent_running = False
        logger.info("Agent thread completed")

