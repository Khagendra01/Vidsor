"""
Timeline control functionality for Vidsor.
Handles timeline drawing, interaction, and chunk visualization.
"""

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

import re
import tkinter as tk
from ..utils import format_time, get_chunk_color


class TimelineController:
    """Handles timeline drawing and interaction."""
    
    def __init__(self, vidsor):
        """
        Initialize timeline controller.
        
        Args:
            vidsor: Reference to main Vidsor instance for accessing video, state, and UI
        """
        self.vidsor = vidsor
        self.is_dragging_playhead = False
        self.timeline_hover_chunk = None
    
    def on_timeline_click(self, event):
        """Handle timeline click for chunk selection or playhead dragging."""
        if not self.vidsor.timeline_canvas:
            return
        
        # Pause playback if it's currently running for better drag handling
        if self.vidsor.edit_state.is_playing:
            self.vidsor.edit_state.is_playing = False
            self.vidsor._update_playback_controls()
        
        # Get canvas coordinates
        canvas_x = self.vidsor.timeline_canvas.canvasx(event.x)
        canvas_y = event.y
        
        # Calculate timeline duration and scale
        if not self.vidsor.edit_state.chunks:
            if not self.vidsor.video_clip:
                return
            timeline_duration = self.vidsor.video_clip.duration
        else:
            timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks)
        
        canvas_width = self.vidsor.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
        
        # Timeline dimensions (matching draw_timeline)
        RULER_HEIGHT = 35
        CHUNK_AREA_TOP = RULER_HEIGHT
        CHUNK_AREA_BOTTOM = CHUNK_AREA_TOP + 140
        
        # Check if clicking on a chunk (within chunk area)
        clicked_chunk_idx = None
        if self.vidsor.edit_state.chunks and CHUNK_AREA_TOP <= canvas_y <= CHUNK_AREA_BOTTOM:
            for i, chunk in enumerate(self.vidsor.edit_state.chunks):
                x_start = chunk.start_time * scale
                x_end = chunk.end_time * scale
                if x_start <= canvas_x <= x_end:
                    clicked_chunk_idx = i
                    break
        
        # Check if clicking on/near the playhead (within 10 pixels)
        playhead_x = self.vidsor.edit_state.preview_time * scale
        playhead_tolerance = 10
        
        if abs(canvas_x - playhead_x) <= playhead_tolerance:
            # Clicked on playhead - start dragging
            self.is_dragging_playhead = True
            # Stop playback and audio if playing
            if self.vidsor.edit_state.is_playing:
                self.vidsor.edit_state.is_playing = False
            # Stop audio completely
            if HAS_PYGAME:
                try:
                    pygame.mixer.music.stop()
                except:
                    pass
            # Mark that audio needs restart from new position
            self.vidsor.playback_controller.audio_needs_restart = True
            # Mark that playback has started so Resume button appears
            self.vidsor.edit_state.has_started_playback = True
            self.vidsor._update_playback_controls()
        elif clicked_chunk_idx is not None:
            # Clicked on a chunk - handle selection
            # Single click: toggle selection
            if clicked_chunk_idx in self.vidsor.edit_state.selected_chunks:
                # Already selected - unselect it
                self.vidsor.edit_state.selected_chunks.discard(clicked_chunk_idx)
            else:
                # Not selected - add to selection
                self.vidsor.edit_state.selected_chunks.add(clicked_chunk_idx)
            
            # Update timeline display
            self.draw_timeline()
            
            # Display selected chunks in chat
            self._display_selected_chunks_in_chat()
        else:
            # Clicked elsewhere on timeline - seek to that position
            new_time = canvas_x / scale
            new_time = max(0, min(new_time, timeline_duration))
            self.vidsor.playback_controller.seek_to_time(new_time)
            # Start dragging from this position
            self.is_dragging_playhead = True
    
    def on_timeline_drag(self, event):
        """Handle timeline drag for playhead scrubbing."""
        if not self.is_dragging_playhead or not self.vidsor.timeline_canvas:
            return
        
        # Stop playback and audio if it's currently running for better drag handling
        if self.vidsor.edit_state.is_playing:
            self.vidsor.edit_state.is_playing = False
        # Stop audio completely
        if HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        # Mark that audio needs restart from new position
        self.vidsor.playback_controller.audio_needs_restart = True
        # Mark that playback has started so Resume button appears
        self.vidsor.edit_state.has_started_playback = True
        self.vidsor._update_playback_controls()
        
        # Get canvas coordinates
        canvas_x = self.vidsor.timeline_canvas.canvasx(event.x)
        
        # Calculate timeline duration and scale
        if not self.vidsor.edit_state.chunks:
            if not self.vidsor.video_clip:
                return
            timeline_duration = self.vidsor.video_clip.duration
        else:
            timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks)
        
        canvas_width = self.vidsor.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
        
        # Calculate new time from mouse position
        new_time = canvas_x / scale
        new_time = max(0, min(new_time, timeline_duration))
        
        # Seek to new time
        self.vidsor.playback_controller.seek_to_time(new_time)
    
    def on_timeline_release(self, event):
        """Handle mouse release after timeline drag."""
        self.is_dragging_playhead = False
    
    def on_timeline_motion(self, event):
        """Handle mouse motion over timeline for hover effects."""
        # Don't update hover if dragging playhead
        if self.is_dragging_playhead:
            return
        
        if not self.vidsor.edit_state.chunks or not self.vidsor.timeline_canvas:
            return
        
        # Get canvas coordinates
        canvas_x = self.vidsor.timeline_canvas.canvasx(event.x)
        
        # Calculate timeline duration and scale
        timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks)
        canvas_width = self.vidsor.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / timeline_duration
        
        # Find which chunk is being hovered
        hovered_chunk_idx = None
        for i, chunk in enumerate(self.vidsor.edit_state.chunks):
            x_start = chunk.start_time * scale
            x_end = chunk.end_time * scale
            if x_start <= canvas_x <= x_end:
                hovered_chunk_idx = i
                break
        
        # Update hover state and redraw if changed
        if hovered_chunk_idx != self.timeline_hover_chunk:
            self.timeline_hover_chunk = hovered_chunk_idx
            self.draw_timeline()
    
    def on_timeline_leave(self, event):
        """Handle mouse leaving timeline."""
        if self.timeline_hover_chunk is not None:
            self.timeline_hover_chunk = None
            self.draw_timeline()
    
    def draw_timeline(self):
        """Draw professional timeline with chunks, playhead, and modern styling."""
        if not self.vidsor.timeline_canvas:
            return
        
        self.vidsor.timeline_canvas.delete("all")
        
        # Timeline dimensions
        TIMELINE_HEIGHT = 220
        RULER_HEIGHT = 35
        CHUNK_AREA_TOP = RULER_HEIGHT
        CHUNK_AREA_HEIGHT = 140
        CHUNK_AREA_BOTTOM = CHUNK_AREA_TOP + CHUNK_AREA_HEIGHT
        TIME_MARKER_HEIGHT = 12
        
        # Calculate timeline duration
        if not self.vidsor.edit_state.chunks:
            if not self.vidsor.video_clip:
                return
            timeline_duration = self.vidsor.video_clip.duration
        else:
            timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks)
        
        canvas_width = self.vidsor.timeline_canvas.winfo_width() or 1000
        if canvas_width <= 1:
            canvas_width = 1000
        scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
        
        # Draw background
        self.vidsor.timeline_canvas.create_rectangle(
            0, 0, canvas_width, TIMELINE_HEIGHT,
            fill="#1a1a1a", outline="", tags="background"
        )
        
        # Draw ruler background (darker)
        self.vidsor.timeline_canvas.create_rectangle(
            0, 0, canvas_width, RULER_HEIGHT,
            fill="#0f0f0f", outline="", tags="ruler_bg"
        )
        
        # Draw time markers with professional styling
        # Major markers every 10 seconds
        for t in range(0, int(timeline_duration) + 1, 10):
            x = t * scale
            if x > canvas_width:
                break
            
            # Major tick line
            self.vidsor.timeline_canvas.create_line(
                x, RULER_HEIGHT - TIME_MARKER_HEIGHT, x, RULER_HEIGHT,
                fill="#666666", width=2, tags="time_marker"
            )
            
            # Time label
            time_str = format_time(t)
            self.vidsor.timeline_canvas.create_text(
                x, RULER_HEIGHT - TIME_MARKER_HEIGHT - 8,
                text=time_str, font=("Segoe UI", 9, "normal"),
                fill="#cccccc", anchor="s", tags="time_label"
            )
        
        # Minor markers every 5 seconds (between major markers)
        for t in range(5, int(timeline_duration) + 1, 10):
            x = t * scale
            if x > canvas_width:
                break
            self.vidsor.timeline_canvas.create_line(
                x, RULER_HEIGHT - TIME_MARKER_HEIGHT // 2, x, RULER_HEIGHT,
                fill="#444444", width=1, tags="time_marker_minor"
            )
        
        # Draw chunks with professional styling
        if self.vidsor.edit_state.chunks:
            for i, chunk in enumerate(self.vidsor.edit_state.chunks):
                x_start = chunk.start_time * scale
                x_end = chunk.end_time * scale
                width = max(x_end - x_start, 2)  # Minimum width
                
                if x_end < 0 or x_start > canvas_width:
                    continue  # Skip chunks outside visible area
                
                # Determine if this chunk is hovered or selected
                is_hovered = (self.timeline_hover_chunk == i)
                is_selected = (i in self.vidsor.edit_state.selected_chunks)
                
                # Get colors
                fill_color, outline_color, gradient_color = get_chunk_color(
                    chunk.chunk_type, is_hovered, is_selected
                )
                
                # Draw chunk with gradient effect (simulated with multiple rectangles)
                chunk_y_top = CHUNK_AREA_TOP + 5
                chunk_y_bottom = CHUNK_AREA_BOTTOM - 5
                chunk_height = chunk_y_bottom - chunk_y_top
                
                # Main chunk rectangle with rounded corners effect (using polygon)
                # Draw shadow first (darker rectangle for depth effect)
                shadow_offset = 2
                self.vidsor.timeline_canvas.create_rectangle(
                    x_start + shadow_offset, chunk_y_top + shadow_offset,
                    x_end + shadow_offset, chunk_y_bottom + shadow_offset,
                    fill="#0a0a0a", outline="", tags=f"chunk_shadow_{i}"
                )
                
                # Main chunk body
                self.vidsor.timeline_canvas.create_rectangle(
                    x_start, chunk_y_top, x_end, chunk_y_bottom,
                    fill=fill_color, outline=outline_color, width=2,
                    tags=f"chunk_{i}"
                )
                
                # Gradient effect (top lighter, bottom darker)
                gradient_height = chunk_height // 3
                self.vidsor.timeline_canvas.create_rectangle(
                    x_start, chunk_y_top, x_end, chunk_y_top + gradient_height,
                    fill=gradient_color, outline="", tags=f"chunk_gradient_{i}"
                )
                
                # Draw chunk label with better typography
                if width > 60:  # Only draw label if chunk is wide enough
                    # Chunk type label
                    type_label = chunk.chunk_type.replace("_", " ").title()
                    if chunk.chunk_type == "highlight":
                        type_label = "★ Highlight"
                    
                    self.vidsor.timeline_canvas.create_text(
                        x_start + width/2, chunk_y_top + 20,
                        text=type_label, font=("Segoe UI", 10, "bold"),
                        fill="#ffffff", tags=f"chunk_label_{i}"
                    )
                    
                    # Time range
                    time_range = f"{format_time(chunk.start_time)} - {format_time(chunk.end_time)}"
                    self.vidsor.timeline_canvas.create_text(
                        x_start + width/2, chunk_y_top + 40,
                        text=time_range, font=("Segoe UI", 8),
                        fill="#ffffff", tags=f"chunk_time_{i}"
                    )
                    
                    # Duration
                    duration = chunk.end_time - chunk.start_time
                    duration_str = f"{duration:.1f}s"
                    if chunk.speed != 1.0:
                        duration_str += f" @ {chunk.speed}x"
                    self.vidsor.timeline_canvas.create_text(
                        x_start + width/2, chunk_y_top + 55,
                        text=duration_str, font=("Segoe UI", 8),
                        fill="#dddddd", tags=f"chunk_duration_{i}"
                    )
                    
                    # Description (truncated if too long)
                    if chunk.description and width > 100:
                        desc = chunk.description
                        max_chars = int(width / 6)  # Approximate chars that fit
                        if len(desc) > max_chars:
                            desc = desc[:max_chars-3] + "..."
                        self.vidsor.timeline_canvas.create_text(
                            x_start + width/2, chunk_y_top + 75,
                            text=desc, font=("Segoe UI", 7),
                            fill="#cccccc", width=int(width - 10),
                            tags=f"chunk_desc_{i}"
                        )
                else:
                    # Very narrow chunk - just show type icon
                    if chunk.chunk_type == "highlight":
                        self.vidsor.timeline_canvas.create_text(
                            x_start + width/2, chunk_y_top + chunk_height/2,
                            text="★", font=("Segoe UI", 14),
                            fill="#ffffff", tags=f"chunk_icon_{i}"
                        )
        
        # Draw playhead indicator (red line showing current playback position)
        if self.vidsor.edit_state.preview_time >= 0:
            playhead_x = self.vidsor.edit_state.preview_time * scale
            if 0 <= playhead_x <= canvas_width:
                # Playhead line
                self.vidsor.timeline_canvas.create_line(
                    playhead_x, 0, playhead_x, TIMELINE_HEIGHT,
                    fill="#ff0000", width=2, tags="playhead"
                )
                
                # Playhead triangle at top
                triangle_size = 8
                self.vidsor.timeline_canvas.create_polygon(
                    playhead_x, 0,
                    playhead_x - triangle_size, triangle_size,
                    playhead_x + triangle_size, triangle_size,
                    fill="#ff0000", outline="#cc0000", width=1, tags="playhead_triangle"
                )
                
                # Current time label above playhead
                current_time_str = format_time(self.vidsor.edit_state.preview_time)
                self.vidsor.timeline_canvas.create_text(
                    playhead_x, triangle_size + 5,
                    text=current_time_str, font=("Segoe UI", 9, "bold"),
                    fill="#ff0000", anchor="n", tags="playhead_time"
                )
        
        # Draw separator line between ruler and chunks
        self.vidsor.timeline_canvas.create_line(
            0, RULER_HEIGHT, canvas_width, RULER_HEIGHT,
            fill="#333333", width=1, tags="separator"
        )
        
        # Update scroll region
        timeline_width = max(canvas_width, timeline_duration * scale)
        self.vidsor.timeline_canvas.configure(scrollregion=(0, 0, timeline_width, TIMELINE_HEIGHT))
    
    def on_timeline_double_click(self, event):
        """Handle double-click on timeline to unselect chunks."""
        if not self.vidsor.timeline_canvas:
            return
        
        # Get canvas coordinates
        canvas_x = self.vidsor.timeline_canvas.canvasx(event.x)
        canvas_y = event.y
        
        # Calculate timeline duration and scale
        if not self.vidsor.edit_state.chunks:
            return
        
        timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks)
        canvas_width = self.vidsor.timeline_canvas.winfo_width() or 1000
        scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
        
        # Timeline dimensions (matching draw_timeline)
        RULER_HEIGHT = 35
        CHUNK_AREA_TOP = RULER_HEIGHT
        CHUNK_AREA_BOTTOM = CHUNK_AREA_TOP + 140
        
        # Check if double-clicking on a chunk
        if CHUNK_AREA_TOP <= canvas_y <= CHUNK_AREA_BOTTOM:
            for i, chunk in enumerate(self.vidsor.edit_state.chunks):
                x_start = chunk.start_time * scale
                x_end = chunk.end_time * scale
                if x_start <= canvas_x <= x_end:
                    # Double-clicked on chunk - unselect it
                    self.vidsor.edit_state.selected_chunks.discard(i)
                    # Update timeline display
                    self.draw_timeline()
                    # Update chat display
                    self._display_selected_chunks_in_chat()
                    break
    
    def _display_selected_chunks_in_chat(self):
        """Insert selected clip references into the input box, or clear them if none selected."""
        # Insert into input box
        if hasattr(self.vidsor, 'agent_integration') and self.vidsor.agent_integration:
            if self.vidsor.agent_integration.chat_input:
                # Get current content
                current_text = self.vidsor.agent_integration.chat_input.get("1.0", tk.END).strip()
                
                # Remove any existing clip references (like @clip1, @clip2, etc.)
                # Pattern to match @clip followed by digits
                current_text = re.sub(r'@clip\d+\s*', '', current_text).strip()
                
                # If there are selected chunks, add their references
                if self.vidsor.edit_state.selected_chunks:
                    # Get selected chunks sorted by start time
                    selected_chunks = sorted(
                        [(i, self.vidsor.edit_state.chunks[i]) for i in self.vidsor.edit_state.selected_chunks],
                        key=lambda x: x[1].start_time
                    )
                    
                    # Build clip references like @clip1 @clip2
                    clip_refs = []
                    for idx, chunk in selected_chunks:
                        clip_refs.append(f"@clip{idx + 1}")
                    
                    # Add new clip references
                    new_clip_refs = ' '.join(clip_refs)
                    # Combine with existing text (if any) and new clip references
                    new_text = f"{current_text} {new_clip_refs}".strip() if current_text else new_clip_refs
                else:
                    # No selected chunks - just use the text without clip references
                    new_text = current_text
                
                # Update input box
                self.vidsor.agent_integration.chat_input.delete("1.0", tk.END)
                self.vidsor.agent_integration.chat_input.insert("1.0", new_text)
                # Focus on input box
                self.vidsor.agent_integration.chat_input.focus_set()

