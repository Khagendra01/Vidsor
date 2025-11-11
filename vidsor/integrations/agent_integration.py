"""
Agent integration functionality for Vidsor.
Handles chat UI, agent interaction, and clarification handling.
"""

import os
import json
import threading
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox

from agent.orchestrator_runner import run_orchestrator
from agent.utils.logging_utils import DualLogger, create_log_file
from ..managers.chat_manager import ChatManager
from ..managers.timeline_manager import TimelineManager
from ..models import Chunk


class AgentIntegration:
    """Handles chat UI and agent interaction."""
    
    def __init__(self, vidsor):
        """
        Initialize agent integration.
        
        Args:
            vidsor: Reference to main Vidsor instance
        """
        self.vidsor = vidsor
        self.chat_history: List[Dict[str, str]] = []
        self.chat_text: Optional[tk.Text] = None
        self.chat_input: Optional[tk.Text] = None
        self.chat_send_btn: Optional[ttk.Button] = None
        self.chat_status_label: Optional[ttk.Label] = None
        self.agent_thread: Optional[threading.Thread] = None
        self.is_agent_running = False
        self.pending_clarification: Optional[Dict] = None
    
    def create_chat_ui(self, parent_frame):
        """Create chat interface UI components."""
        # Chat frame
        chat_frame = ttk.LabelFrame(parent_frame, text="Chat Assistant", padding="10")
        chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat history display (scrollable text widget)
        chat_history_frame = ttk.Frame(chat_frame)
        chat_history_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_history_frame.columnconfigure(0, weight=1)
        chat_history_frame.rowconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(chat_history_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.chat_text = tk.Text(
            chat_history_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            height=30,
            font=("Arial", 10),
            bg="#f5f5f5"
        )
        self.chat_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.chat_text.yview)
        
        # Chat input frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        input_frame.columnconfigure(0, weight=1)
        
        # Input text widget (multi-line)
        self.chat_input = tk.Text(
            input_frame,
            wrap=tk.WORD,
            height=3,
            font=("Arial", 10)
        )
        self.chat_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Send button
        self.chat_send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self.on_send_message,
            state=tk.DISABLED
        )
        self.chat_send_btn.grid(row=0, column=1, sticky=tk.E)
        
        # Bind Enter key (with Shift for new line)
        self.chat_input.bind("<Return>", self.on_chat_input_return)
        self.chat_input.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for new line
        
        # Chat status label
        self.chat_status_label = ttk.Label(chat_frame, text="Ready", foreground="gray")
        self.chat_status_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Display existing chat history if any
        self.display_chat_history()
    
    def on_chat_input_return(self, event):
        """Handle Enter key in chat input (send message, Shift+Enter for new line)."""
        if event.state & 0x1:  # Shift key is pressed
            return  # Allow default behavior (new line)
        else:
            self.on_send_message()
            return "break"  # Prevent default behavior
    
    def on_send_message(self):
        """Handle send message button click."""
        if self.is_agent_running:
            messagebox.showwarning("Warning", "Agent is already processing a query. Please wait.")
            return
        
        # Get message from input
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Check if this is a response to a clarification
        if self.pending_clarification:
            # User is responding to clarification - continue with preserved state
            self.continue_with_clarification(message)
            return
        
        # Check if project and video are loaded
        if not self.vidsor.current_project_path:
            messagebox.showwarning("Warning", "Please select a project first.")
            return
        
        if not self.vidsor.video_path:
            messagebox.showwarning("Warning", "Please upload a video first.")
            return
        
        segment_tree_path = os.path.join(self.vidsor.current_project_path, "segment_tree.json")
        if not os.path.exists(segment_tree_path):
            messagebox.showwarning(
                "Warning",
                "Segment tree not found. Please extract video features first using 'Upload Video'."
            )
            return
        
        # Clear input
        self.chat_input.delete("1.0", tk.END)
        
        # Add user message to history
        self.add_chat_message("user", message)
        
        # Run agent in background thread
        self.is_agent_running = True
        self.chat_send_btn.config(state=tk.DISABLED)
        self.chat_status_label.config(text="Processing query...", foreground="blue")
        
        self.agent_thread = threading.Thread(
            target=self.run_agent_thread,
            args=(message, segment_tree_path),
            daemon=True
        )
        self.agent_thread.start()
    
    def continue_with_clarification(self, user_response: str):
        """Continue operation with user's clarification response using preserved state."""
        if not self.pending_clarification:
            return
        
        # Clear input
        self.chat_input.delete("1.0", tk.END)
        
        # Add user response to history
        self.add_chat_message("user", user_response)
        
        # Get preserved state
        preserved = self.pending_clarification
        operation = preserved["operation"]
        preserved_state = preserved["preserved_state"]
        original_query = preserved["original_query"]
        segment_tree_path = preserved["segment_tree_path"]
        
        # Clear pending clarification
        self.pending_clarification = None
        
        # Run agent thread with clarification response
        self.is_agent_running = True
        self.chat_send_btn.config(state=tk.DISABLED)
        self.chat_status_label.config(text="Processing clarification...", foreground="blue")
        
        self.agent_thread = threading.Thread(
            target=self.run_agent_thread_with_clarification,
            args=(user_response, segment_tree_path, operation, preserved_state, original_query),
            daemon=True
        )
        self.agent_thread.start()
    
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
            timeline_path = os.path.join(self.vidsor.current_project_path, "timeline.json")
            
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
                video_path=self.vidsor.video_path,
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
                if self.vidsor.root:
                    def update_timeline_ui():
                        try:
                            self.vidsor._load_timeline()
                            if self.vidsor.timeline_canvas:
                                self.vidsor.timeline_controller.draw_timeline()
                            # Update UI button states (especially play button)
                            self.vidsor._update_ui_state()
                        except Exception as e:
                            logger.error(f"Error updating timeline UI: {e}")
                    
                    self.vidsor.root.after(0, update_timeline_ui)
            
            # Update UI
            if self.vidsor.root:
                self.vidsor.root.after(0, lambda: self.add_chat_message("assistant", response))
                self.vidsor.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                self.vidsor.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            error_msg = f"Error processing clarification: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            if self.vidsor.root:
                self.vidsor.root.after(0, lambda: self.add_chat_message("assistant", error_msg))
                self.vidsor.root.after(0, lambda: self.chat_status_label.config(text="Error occurred", foreground="red"))
                self.vidsor.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
        finally:
            self.is_agent_running = False
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
        logger.info(f"Project: {self.vidsor.current_project_path}")
        logger.info(f"Video: {self.vidsor.video_path}")
        logger.info(f"Segment tree: {segment_tree_path}")
        
        try:
            # Get timeline path from current project
            if not self.vidsor.current_project_path:
                error_msg = "No project selected. Please select a project first."
                logger.error(error_msg)
                if self.vidsor.root:
                    self.vidsor.root.after(0, lambda: self.add_chat_message("assistant", error_msg))
                    self.vidsor.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                    self.vidsor.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
                return
            
            timeline_path = os.path.join(self.vidsor.current_project_path, "timeline.json")
            logger.info(f"Timeline path: {timeline_path}")
            
            # Check if there's a pending clarification and if this query looks like a follow-up
            if self.pending_clarification:
                preserved = self.pending_clarification
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
                    self.pending_clarification = None
                    
                    # Run with preserved state
                    self.run_agent_thread_with_clarification(
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
                video_path=self.vidsor.video_path,
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
                if self.vidsor.root:
                    def store_clarification_state():
                        # Ensure previous_time_ranges is set for refinement logic
                        if preserved_state:
                            if "previous_time_ranges" not in preserved_state and "time_ranges" in preserved_state:
                                preserved_state["previous_time_ranges"] = preserved_state["time_ranges"]
                            if "previous_query" not in preserved_state:
                                preserved_state["previous_query"] = query
                        
                        self.pending_clarification = {
                            "operation": operation,
                            "preserved_state": preserved_state,
                            "original_query": query,
                            "segment_tree_path": segment_tree_path,
                            "timeline_path": timeline_path
                        }
                        logger.info("Clarification state stored")
                        if preserved_state:
                            logger.info(f"  Preserved {len(preserved_state.get('time_ranges', preserved_state.get('previous_time_ranges', [])))} time ranges")
                    
                    self.vidsor.root.after(0, store_clarification_state)
                
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
                
                if self.vidsor.root:
                    def update_timeline_ui():
                        try:
                            logger.info("Reloading timeline from file...")
                            # Reload timeline from file (orchestrator already saved it)
                            self.vidsor._load_timeline()
                            logger.info(f"Timeline loaded: {len(self.vidsor.edit_state.chunks)} chunks")
                            
                            # Update timeline display
                            if self.vidsor.timeline_canvas:
                                logger.info("Drawing timeline on canvas...")
                                self.vidsor.timeline_controller.draw_timeline()
                                logger.info("Timeline canvas updated")
                            
                            # Update UI button states (especially play button)
                            logger.info("Updating UI button states...")
                            self.vidsor._update_ui_state()
                            
                            logger.info(f"[VIDSOR] Timeline updated: {len(timeline_chunks)} chunks")
                        except Exception as e:
                            logger.error(f"Error updating timeline UI: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    self.vidsor.root.after(0, update_timeline_ui)
            else:
                logger.info("Skipping timeline UI update (success=False or no timeline_chunks)")
            
            # Update UI in main thread
            logger.info("\n" + "-" * 80)
            logger.info("UPDATING CHAT UI")
            logger.info("-" * 80)
            if self.vidsor.root:
                self.vidsor.root.after(0, lambda: self.add_chat_message("assistant", response))
                self.vidsor.root.after(0, lambda: self.chat_status_label.config(text="Ready", foreground="gray"))
                self.vidsor.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
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
            if self.vidsor.root:
                self.vidsor.root.after(0, lambda: self.add_chat_message("assistant", error_msg))
                self.vidsor.root.after(0, lambda: self.chat_status_label.config(text="Error occurred", foreground="red"))
                self.vidsor.root.after(0, lambda: self.chat_send_btn.config(state=tk.NORMAL))
        finally:
            self.is_agent_running = False
            logger.info("Agent thread completed")
    
    def add_chat_message(self, role: str, content: str):
        """Add a message to chat history and display it."""
        # Add to history
        self.chat_history.append({"role": role, "content": content})
        
        # Save to file
        self.save_chat_history()
        
        # Display in chat text widget
        self.chat_text.config(state=tk.NORMAL)
        
        # Format message
        if role == "user":
            prefix = "You: "
            tag = "user"
        else:
            prefix = "Assistant: "
            tag = "assistant"
        
        # Get start position before inserting
        start_pos = self.chat_text.index(tk.END)
        self.chat_text.insert(tk.END, f"{prefix}{content}\n\n")
        # Get end position after inserting (before the two newlines)
        end_pos = self.chat_text.index(f"{start_pos}+{len(prefix)+len(content)}c")
        
        # Apply tags for styling
        self.chat_text.tag_add(tag, start_pos, end_pos)
        
        # Configure tag styles
        self.chat_text.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("assistant", foreground="green", font=("Arial", 10))
        
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)
    
    def save_chat_history(self):
        """Save chat history to project folder."""
        ChatManager.save_chat_history(self.vidsor.current_project_path, self.chat_history)
    
    def load_chat_history(self):
        """Load chat history from project folder."""
        self.chat_history = ChatManager.load_chat_history(self.vidsor.current_project_path)
    
    def display_chat_history(self):
        """Display all chat history in the chat text widget."""
        if not self.chat_text:
            return
        
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        
        for msg in self.chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                prefix = "You: "
                tag = "user"
            else:
                prefix = "Assistant: "
                tag = "assistant"
            
            start_pos = self.chat_text.index(tk.END)
            self.chat_text.insert(tk.END, f"{prefix}{content}\n\n")
            end_pos = self.chat_text.index(tk.END)
            
            # Apply tags
            self.chat_text.tag_add(tag, start_pos, f"{end_pos}-2c")
        
        # Configure tag styles
        self.chat_text.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("assistant", foreground="green", font=("Arial", 10))
        
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)
        
        # Update send button state
        if self.chat_send_btn:
            has_project = self.vidsor.current_project_path is not None
            has_video = self.vidsor.video_path is not None
            has_segment_tree = self.vidsor.segment_tree_path is not None and os.path.exists(self.vidsor.segment_tree_path)
            self.chat_send_btn.config(state=tk.NORMAL if (has_project and has_video and has_segment_tree) else tk.DISABLED)

