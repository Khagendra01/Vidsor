"""
Chat UI functionality.
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox


def create_chat_ui(self, parent_frame):
    """Create chat interface UI components."""
    self.agent_integration.create_chat_ui(parent_frame)
    # Sync UI references
    self.chat_text = self.agent_integration.chat_text
    self.chat_input = self.agent_integration.chat_input
    self.chat_send_btn = self.agent_integration.chat_send_btn
    self.chat_status_label = self.agent_integration.chat_status_label
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
        command=self._on_send_message,
        state=tk.DISABLED
    )
    self.chat_send_btn.grid(row=0, column=1, sticky=tk.E)
    
    # Bind Enter key (with Shift for new line)
    self.chat_input.bind("<Return>", self._on_chat_input_return)
    self.chat_input.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for new line
    
    # Chat status label
    self.chat_status_label = ttk.Label(chat_frame, text="Ready", foreground="gray")
    self.chat_status_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
    
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
    self.agent_integration.on_send_message()
    if self.agent_integration.is_agent_running:
        messagebox.showwarning("Warning", "Agent is already processing a query. Please wait.")
        return
    
    # Get message from input
    message = self.chat_input.get("1.0", tk.END).strip()
    if not message:
        return
    
    # Check if this is a response to a clarification
    if self.agent_integration.pending_clarification:
        # User is responding to clarification - continue with preserved state
        self._continue_with_clarification(message)
        return
    
    # Check if project and video are loaded
    if not self.current_project_path:
        messagebox.showwarning("Warning", "Please select a project first.")
        return
    
    if not self.video_path:
        messagebox.showwarning("Warning", "Please upload a video first.")
        return
    
    segment_tree_path = os.path.join(self.current_project_path, "segment_tree.json")
    if not os.path.exists(segment_tree_path):
        messagebox.showwarning(
            "Warning",
            "Segment tree not found. Please extract video features first using 'Upload Video'."
        )
        return
    
    # Clear input
    self.chat_input.delete("1.0", tk.END)
    
    # Add user message to history
    self._add_chat_message("user", message)
    
    # Run agent in background thread
    self.agent_integration.is_agent_running = True
    self.chat_send_btn.config(state=tk.DISABLED)
    self.chat_status_label.config(text="Processing query...", foreground="blue")
    
    self.agent_integration.agent_thread = threading.Thread(
        target=self._run_agent_thread,
        args=(message, segment_tree_path),
        daemon=True
    )
    self.agent_integration.agent_thread.start()


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

