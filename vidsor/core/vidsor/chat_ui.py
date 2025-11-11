"""
Chat UI functionality.
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox


def create_chat_ui(self, parent_frame):
    """Create chat interface UI components."""
    # Use agent_integration's chat UI (it handles all the UI creation)
    self.agent_integration.create_chat_ui(parent_frame)
    # Sync UI references so Vidsor can access them
    self.chat_text = self.agent_integration.chat_text
    self.chat_input = self.agent_integration.chat_input
    self.chat_send_btn = self.agent_integration.chat_send_btn
    self.chat_status_label = self.agent_integration.chat_status_label
    
    # Update the input bindings to use Vidsor's handlers (agent_integration already handles button click)
    if self.chat_input:
        # Create a wrapper that calls Vidsor's handler, which delegates to agent_integration
        def on_return_wrapper(event):
            return on_chat_input_return(self, event)
        self.chat_input.bind("<Return>", on_return_wrapper)
        self.chat_input.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for new line
    
    # Display existing chat history if any
    self._display_chat_history()


def on_chat_input_return(self, event):
    """Handle Enter key in chat input (send message, Shift+Enter for new line)."""
    if event.state & 0x1:  # Shift key is pressed
        return  # Allow default behavior (new line)
    else:
        self._on_send_message()
        return "break"  # Prevent default behavior


def on_send_message(self):
    """Handle send message button click."""
    # Delegate to agent_integration which handles all the logic
    self.agent_integration.on_send_message()


def continue_with_clarification(self, user_response: str):
    """Continue operation with user's clarification response using preserved state."""
    if not self.agent_integration.pending_clarification:
        return
    
    # Clear input
    self.chat_input.delete("1.0", tk.END)
    
    # Add user response to history
    self._add_chat_message("user", user_response)
    
    # Get preserved state
    preserved = self.agent_integration.pending_clarification
    operation = preserved["operation"]
    preserved_state = preserved["preserved_state"]
    original_query = preserved["original_query"]
    segment_tree_path = preserved["segment_tree_path"]
    timeline_path = preserved["timeline_path"]
    
    # Clear pending clarification
    self.agent_integration.pending_clarification = None
    
    # Run agent thread with clarification response
    self.agent_integration.is_agent_running = True
    self.chat_send_btn.config(state=tk.DISABLED)
    self.chat_status_label.config(text="Processing clarification...", foreground="blue")
    
    self.agent_integration.agent_thread = threading.Thread(
        target=self._run_agent_thread_with_clarification,
        args=(user_response, segment_tree_path, operation, preserved_state, original_query),
        daemon=True
    )
    self.agent_integration.agent_thread.start()

