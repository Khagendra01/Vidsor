"""
Main UI creation and layout for Vidsor.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.vidsor_app import Vidsor


def configure_dark_theme(root: tk.Tk):
    """Configure modern dark theme with custom styles."""
    style = ttk.Style()
    
    # Try to use a modern theme, fallback to default
    try:
        style.theme_use('clam')
    except:
        pass
    
    # Color scheme - Modern dark theme
    colors = {
        'bg': '#1e1e1e',           # Main background (dark gray)
        'fg': '#e0e0e0',           # Main foreground (light gray)
        'frame_bg': '#252525',     # Frame background (slightly lighter)
        'entry_bg': '#2d2d2d',     # Entry/input background
        'entry_fg': '#ffffff',     # Entry text color
        'select_bg': '#3d3d3d',    # Selection background
        'select_fg': '#ffffff',    # Selection foreground
        'button_bg': '#0078d4',    # Primary button (blue)
        'button_hover': '#106ebe', # Button hover (darker blue)
        'button_active': '#005a9e', # Button active (darkest blue)
        'button_disabled': '#404040', # Disabled button
        'accent': '#00bcf2',       # Accent color (cyan)
        'border': '#3d3d3d',       # Border color
        'text_secondary': '#a0a0a0', # Secondary text
        'success': '#4caf50',      # Success color (green)
        'warning': '#ff9800',      # Warning color (orange)
        'error': '#f44336',        # Error color (red)
    }
    
    # Configure root window
    root.configure(bg=colors['bg'])
    
    # Modern font
    font_family = "Segoe UI"
    font_size = 10
    font_bold = (font_family, font_size, "bold")
    font_normal = (font_family, font_size)
    font_small = (font_family, 9)
    
    # Frame styles
    style.configure('TFrame', background=colors['bg'], borderwidth=0)
    style.configure('TLabelframe', background=colors['frame_bg'], foreground=colors['fg'],
                    borderwidth=1, relief='flat', bordercolor=colors['border'])
    style.configure('TLabelframe.Label', background=colors['frame_bg'], foreground=colors['fg'],
                    font=font_bold)
    
    # Label styles
    style.configure('TLabel', background=colors['bg'], foreground=colors['fg'], font=font_normal)
    style.map('TLabel', background=[('active', colors['bg'])])
    
    # Button styles with hover effects
    style.configure('TButton', 
                    background=colors['button_bg'],
                    foreground='white',
                    borderwidth=0,
                    focuscolor='none',
                    padding=(15, 8),
                    font=font_normal,
                    relief='flat')
    style.map('TButton',
              background=[('active', colors['button_hover']),
                         ('pressed', colors['button_active']),
                         ('disabled', colors['button_disabled'])],
              foreground=[('disabled', colors['text_secondary'])])
    
    # Primary button style (for important actions)
    style.configure('Primary.TButton',
                    background=colors['button_bg'],
                    foreground='white',
                    borderwidth=0,
                    focuscolor='none',
                    padding=(20, 10),
                    font=font_bold,
                    relief='flat')
    style.map('Primary.TButton',
              background=[('active', colors['button_hover']),
                         ('pressed', colors['button_active']),
                         ('disabled', colors['button_disabled'])],
              foreground=[('disabled', colors['text_secondary'])])
    
    # Entry/Combobox styles
    style.configure('TEntry',
                    fieldbackground=colors['entry_bg'],
                    foreground=colors['entry_fg'],
                    borderwidth=1,
                    relief='flat',
                    bordercolor=colors['border'],
                    padding=8,
                    font=font_normal)
    style.map('TEntry',
              fieldbackground=[('focus', colors['entry_bg'])],
              bordercolor=[('focus', colors['accent'])])
    
    style.configure('TCombobox',
                    fieldbackground=colors['entry_bg'],
                    foreground=colors['entry_fg'],
                    borderwidth=1,
                    relief='flat',
                    padding=8,
                    font=font_normal)
    style.map('TCombobox',
              fieldbackground=[('readonly', colors['entry_bg'])],
              selectbackground=[('readonly', colors['select_bg'])],
              selectforeground=[('readonly', colors['select_fg'])],
              bordercolor=[('focus', colors['accent'])])
    
    # Progressbar style
    style.configure('TProgressbar',
                    background=colors['accent'],
                    troughcolor=colors['frame_bg'],
                    borderwidth=0,
                    relief='flat',
                    thickness=8)
    
    # Scrollbar style
    style.configure('TScrollbar',
                    background=colors['frame_bg'],
                    troughcolor=colors['bg'],
                    borderwidth=0,
                    arrowcolor=colors['fg'],
                    darkcolor=colors['frame_bg'],
                    lightcolor=colors['frame_bg'],
                    relief='flat',
                    width=12)
    style.map('TScrollbar',
              background=[('active', colors['border'])],
              arrowcolor=[('active', colors['accent'])])
    
    # PanedWindow style
    style.configure('TPanedwindow', background=colors['bg'])
    style.map('TPanedwindow', background=[('active', colors['bg'])])
    
    return colors, font_normal, font_bold, font_small


def create_main_ui(vidsor: 'Vidsor'):
    """Create the main Tkinter UI for video editing."""
    vidsor.root = tk.Tk()
    vidsor.root.title("Vidsor - Video Editor")
    vidsor.root.geometry("1600x900")
    
    # Configure dark theme
    colors, font_normal, font_bold, font_small = configure_dark_theme(vidsor.root)
    
    # Main container with paned window for resizable split
    main_paned = ttk.PanedWindow(vidsor.root, orient=tk.HORIZONTAL)
    main_paned.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    vidsor.root.columnconfigure(0, weight=1)
    vidsor.root.rowconfigure(0, weight=1)
    
    # Left panel - Video editor
    left_frame = ttk.Frame(main_paned, padding="15")
    main_paned.add(left_frame, weight=2)
    
    # Right panel - Chat interface
    right_frame = ttk.Frame(main_paned, padding="15")
    main_paned.add(right_frame, weight=1)
    
    # Main container (for left panel)
    main_frame = ttk.Frame(left_frame, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    left_frame.columnconfigure(0, weight=1)
    left_frame.rowconfigure(0, weight=1)
    
    # Project management frame
    project_frame = ttk.LabelFrame(main_frame, text="Project", padding="20")
    project_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
    project_frame.columnconfigure(0, weight=1)
    
    # New Project button - first row with generous spacing
    new_project_btn = ttk.Button(project_frame, text="New Project", 
                                 command=vidsor._on_new_project, style='Primary.TButton')
    new_project_btn.grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5, 15))
    
    # Project selection row - second row
    ttk.Label(project_frame, text="Project:", font=font_bold).grid(row=1, column=0, padx=5, pady=8, sticky=tk.W)
    vidsor.project_combo = ttk.Combobox(project_frame, state="readonly", width=30)
    vidsor.project_combo.grid(row=1, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))
    vidsor.project_combo.bind("<<ComboboxSelected>>", vidsor._on_project_selected)
    
    vidsor.project_label = ttk.Label(project_frame, text="No project selected", 
                                     foreground=colors['text_secondary'])
    vidsor.project_label.grid(row=1, column=2, padx=5, pady=8, sticky=tk.W)
    
    project_frame.columnconfigure(1, weight=1)
    
    # Update project list
    vidsor._update_project_list()
    
    # Preview area
    preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
    preview_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    
    # Preview label with fixed minimum size to prevent collapsing
    # Use a Canvas for better image display control
    vidsor.preview_canvas = tk.Canvas(
        preview_frame,
        bg="#000000",
        highlightthickness=0,
        relief='flat'
    )
    # Canvas initially hidden, will be shown when playing
    
    # Also keep label for text display
    vidsor.preview_label = tk.Label(
        preview_frame,
        text="No project selected\n\nCreate a new project and upload a video to get started",
        bg="#000000",
        fg=colors['text_secondary'],
        font=font_normal,
        anchor="center",
        justify="center"
    )
    vidsor.preview_label.pack(fill=tk.BOTH, expand=True)
    
    # Set minimum size for preview to prevent collapsing
    preview_frame.grid_rowconfigure(0, weight=1, minsize=450)
    preview_frame.grid_columnconfigure(0, weight=1, minsize=800)
    
    # Timeline
    timeline_frame = ttk.LabelFrame(main_frame, text="Timeline", padding="10")
    timeline_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
    timeline_frame.columnconfigure(0, weight=1)
    
    # Canvas for timeline
    canvas_frame = ttk.Frame(timeline_frame)
    canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
    canvas_frame.columnconfigure(0, weight=1)
    
    vidsor.timeline_canvas = tk.Canvas(
        canvas_frame,
        height=220,
        bg="#1a1a1a",
        scrollregion=(0, 0, 1000, 220),
        highlightthickness=0,
        relief='flat'
    )
    vidsor.timeline_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    # Scrollbar for timeline
    timeline_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=vidsor.timeline_canvas.xview)
    timeline_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
    vidsor.timeline_canvas.configure(xscrollcommand=timeline_scroll.set)
    
    # Controls
    controls_frame = ttk.Frame(main_frame)
    controls_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))
    
    # Buttons with consistent spacing
    vidsor.load_video_btn = ttk.Button(controls_frame, text="Upload Video", 
                                      command=vidsor._on_load_video)
    vidsor.load_video_btn.pack(side=tk.LEFT, padx=8)
    
    vidsor.play_btn = ttk.Button(controls_frame, text="Play Preview", 
                                command=vidsor._on_play, state=tk.DISABLED)
    vidsor.play_btn.pack(side=tk.LEFT, padx=8)
    
    vidsor.pause_btn = ttk.Button(controls_frame, text="Pause", 
                                 command=vidsor._on_pause, state=tk.DISABLED)
    vidsor.pause_btn.pack(side=tk.LEFT, padx=8)
    
    ttk.Button(controls_frame, text="Stop", command=vidsor._on_stop).pack(side=tk.LEFT, padx=8)
    
    vidsor.export_btn = ttk.Button(controls_frame, text="Export", 
                                  command=vidsor._on_export, state=tk.DISABLED, style='Primary.TButton')
    vidsor.export_btn.pack(side=tk.LEFT, padx=8)
    
    # Progress bar
    progress_frame = ttk.Frame(main_frame)
    progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
    progress_frame.columnconfigure(0, weight=1)
    
    vidsor.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
    vidsor.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    # Status
    vidsor.status_label = ttk.Label(main_frame, text="Ready - No project selected",
                                   foreground=colors['text_secondary'], font=font_small)
    vidsor.status_label.grid(row=5, column=0, columnspan=2, pady=5)
    
    # Configure grid weights
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    
    # Bind timeline interactions
    vidsor.timeline_canvas.bind("<Button-1>", vidsor._on_timeline_click)
    vidsor.timeline_canvas.bind("<B1-Motion>", vidsor._on_timeline_drag)
    vidsor.timeline_canvas.bind("<ButtonRelease-1>", vidsor._on_timeline_release)
    vidsor.timeline_canvas.bind("<Motion>", vidsor._on_timeline_motion)
    vidsor.timeline_canvas.bind("<Leave>", vidsor._on_timeline_leave)
    
    # Track hover state and dragging
    vidsor.timeline_hover_chunk = None
    vidsor.is_dragging_playhead = False
    vidsor.timeline_update_counter = 0  # Counter for throttling timeline updates
    vidsor.audio_needs_restart = False  # Flag to indicate audio needs to restart from new position
    
    # Initialize UI state
    vidsor._update_ui_state()
    
    # Create chat interface
    vidsor._create_chat_ui(right_frame)


def update_project_list(vidsor: 'Vidsor'):
    """Update the project dropdown list."""
    if vidsor.project_combo:
        projects = vidsor.get_available_projects()
        vidsor.project_combo['values'] = projects
        
        # Select current project if any
        if vidsor.current_project_path:
            project_name = os.path.basename(vidsor.current_project_path)
            if project_name in projects:
                vidsor.project_combo.set(project_name)
                vidsor.project_label.config(text=f"Active: {project_name}")
            else:
                vidsor.project_label.config(text="No project selected")
        else:
            vidsor.project_combo.set("")
            vidsor.project_label.config(text="No project selected")


def on_new_project(vidsor: 'Vidsor'):
    """Handle new project button click."""
    project_name = simpledialog.askstring(
        "New Project",
        "Enter project name:",
        parent=vidsor.root
    )
    
    if not project_name:
        return
    
    try:
        project_path = vidsor.create_new_project(project_name)
        vidsor.set_current_project(project_path)
        update_project_list(vidsor)
        messagebox.showinfo("Success", f"Project '{project_name}' created successfully!")
    except ValueError as e:
        messagebox.showerror("Error", str(e))


def on_project_selected(vidsor: 'Vidsor', event=None):
    """Handle project selection from dropdown."""
    selected = vidsor.project_combo.get()
    if not selected:
        return
    
    project_path = os.path.join(vidsor.projects_dir, selected)
    if os.path.exists(project_path):
        vidsor.set_current_project(project_path)
        update_project_list(vidsor)
        vidsor._update_ui_state()
        # Update status
        if vidsor.status_label:
            project_name = os.path.basename(project_path)
            vidsor.status_label.config(text=f"Project '{project_name}' selected")
        # Reload chat history for the new project
        vidsor._load_chat_history()
        vidsor._display_chat_history()


def update_ui_state(vidsor: 'Vidsor'):
    """Update UI button states based on current state."""
    has_project = vidsor.current_project_path is not None
    has_video = vidsor.video_path is not None
    has_chunks = len(vidsor.edit_state.chunks) > 0
    
    # Update button states
    if vidsor.load_video_btn:
        vidsor.load_video_btn.config(state=tk.NORMAL if has_project else tk.DISABLED)
    
    if vidsor.play_btn:
        vidsor.play_btn.config(state=tk.NORMAL if (has_video and has_chunks) else tk.DISABLED)
    
    if vidsor.export_btn:
        vidsor.export_btn.config(state=tk.NORMAL if (has_video and has_chunks) else tk.DISABLED)


def update_playback_controls(vidsor: 'Vidsor'):
    """Update playback control button states."""
    if not vidsor.play_btn or not vidsor.pause_btn:
        return
    
    if vidsor.edit_state.is_playing:
        # Playing - show Pause button
        vidsor.play_btn.pack_forget()
        vidsor.pause_btn.pack(side=tk.LEFT, padx=5)
        vidsor.pause_btn.config(text="Pause", state=tk.NORMAL)
    elif vidsor.edit_state.has_started_playback:
        # Paused or stopped but playback has started - show Resume button
        vidsor.pause_btn.pack_forget()
        vidsor.play_btn.pack(side=tk.LEFT, padx=5)
        vidsor.play_btn.config(text="Resume", state=tk.NORMAL)
    else:
        # Not started - show Play button
        vidsor.pause_btn.pack_forget()
        vidsor.play_btn.pack(side=tk.LEFT, padx=5)
        vidsor.play_btn.config(text="Play Preview", state=tk.NORMAL if (vidsor.video_path and vidsor.edit_state.chunks) else tk.DISABLED)

