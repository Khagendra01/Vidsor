"""
Timeline drawing functionality.
"""

def draw_timeline(self):
    """Draw professional timeline with chunks, playhead, and modern styling."""
    self.timeline_controller.draw_timeline()
    if not self.timeline_canvas:
        return
    
    self.timeline_canvas.delete("all")
    
    # Timeline dimensions
    TIMELINE_HEIGHT = 220
    RULER_HEIGHT = 35
    CHUNK_AREA_TOP = RULER_HEIGHT
    CHUNK_AREA_HEIGHT = 140
    CHUNK_AREA_BOTTOM = CHUNK_AREA_TOP + CHUNK_AREA_HEIGHT
    TIME_MARKER_HEIGHT = 12
    
    # Calculate timeline duration
    if not self.edit_state.chunks:
        if not self.video_clip:
            return
        timeline_duration = self.video_clip.duration
    else:
        timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
    
    canvas_width = self.timeline_canvas.winfo_width() or 1000
    if canvas_width <= 1:
        canvas_width = 1000
    scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
    
    # Draw background
    self.timeline_canvas.create_rectangle(
        0, 0, canvas_width, TIMELINE_HEIGHT,
        fill="#1a1a1a", outline="", tags="background"
    )
    
    # Draw ruler background (darker)
    self.timeline_canvas.create_rectangle(
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
        self.timeline_canvas.create_line(
            x, RULER_HEIGHT - TIME_MARKER_HEIGHT, x, RULER_HEIGHT,
            fill="#666666", width=2, tags="time_marker"
        )
        
        # Time label
        time_str = self._format_time(t)
        self.timeline_canvas.create_text(
            x, RULER_HEIGHT - TIME_MARKER_HEIGHT - 8,
            text=time_str, font=("Segoe UI", 9, "normal"),
            fill="#cccccc", anchor="s", tags="time_label"
        )
    
    # Minor markers every 5 seconds (between major markers)
    for t in range(5, int(timeline_duration) + 1, 10):
        x = t * scale
        if x > canvas_width:
            break
        self.timeline_canvas.create_line(
            x, RULER_HEIGHT - TIME_MARKER_HEIGHT // 2, x, RULER_HEIGHT,
            fill="#444444", width=1, tags="time_marker_minor"
        )
    
    # Draw chunks with professional styling
    if self.edit_state.chunks:
        for i, chunk in enumerate(self.edit_state.chunks):
            x_start = chunk.start_time * scale
            x_end = chunk.end_time * scale
            width = max(x_end - x_start, 2)  # Minimum width
            
            if x_end < 0 or x_start > canvas_width:
                continue  # Skip chunks outside visible area
            
            # Determine if this chunk is hovered or selected
            is_hovered = (self.timeline_hover_chunk == i)
            is_selected = (self.edit_state.selected_chunk == i)
            
            # Get colors
            fill_color, outline_color, gradient_color = self._get_chunk_color(
                chunk.chunk_type, is_hovered, is_selected
            )
            
            # Draw chunk with gradient effect (simulated with multiple rectangles)
            chunk_y_top = CHUNK_AREA_TOP + 5
            chunk_y_bottom = CHUNK_AREA_BOTTOM - 5
            chunk_height = chunk_y_bottom - chunk_y_top
            
            # Main chunk rectangle with rounded corners effect (using polygon)
            # Draw shadow first (darker rectangle for depth effect)
            shadow_offset = 2
            self.timeline_canvas.create_rectangle(
                x_start + shadow_offset, chunk_y_top + shadow_offset,
                x_end + shadow_offset, chunk_y_bottom + shadow_offset,
                fill="#0a0a0a", outline="", tags=f"chunk_shadow_{i}"
            )
            
            # Main chunk body
            self.timeline_canvas.create_rectangle(
                x_start, chunk_y_top, x_end, chunk_y_bottom,
                fill=fill_color, outline=outline_color, width=2,
                tags=f"chunk_{i}"
            )
            
            # Gradient effect (top lighter, bottom darker)
            gradient_height = chunk_height // 3
            self.timeline_canvas.create_rectangle(
                x_start, chunk_y_top, x_end, chunk_y_top + gradient_height,
                fill=gradient_color, outline="", tags=f"chunk_gradient_{i}"
            )
            
            # Draw chunk label with better typography
            if width > 60:  # Only draw label if chunk is wide enough
                # Chunk type label
                type_label = chunk.chunk_type.replace("_", " ").title()
                if chunk.chunk_type == "highlight":
                    type_label = "★ Highlight"
                
                self.timeline_canvas.create_text(
                    x_start + width/2, chunk_y_top + 20,
                    text=type_label, font=("Segoe UI", 10, "bold"),
                    fill="#ffffff", tags=f"chunk_label_{i}"
                )
                
                # Time range
                time_range = f"{self._format_time(chunk.start_time)} - {self._format_time(chunk.end_time)}"
                self.timeline_canvas.create_text(
                    x_start + width/2, chunk_y_top + 40,
                    text=time_range, font=("Segoe UI", 8),
                    fill="#ffffff", tags=f"chunk_time_{i}"
                )
                
                # Duration
                duration = chunk.end_time - chunk.start_time
                duration_str = f"{duration:.1f}s"
                if chunk.speed != 1.0:
                    duration_str += f" @ {chunk.speed}x"
                self.timeline_canvas.create_text(
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
                    self.timeline_canvas.create_text(
                        x_start + width/2, chunk_y_top + 75,
                        text=desc, font=("Segoe UI", 7),
                        fill="#cccccc", width=int(width - 10),
                        tags=f"chunk_desc_{i}"
                    )
            else:
                # Very narrow chunk - just show type icon
                if chunk.chunk_type == "highlight":
                    self.timeline_canvas.create_text(
                        x_start + width/2, chunk_y_top + chunk_height/2,
                        text="★", font=("Segoe UI", 14),
                        fill="#ffffff", tags=f"chunk_icon_{i}"
                    )
    
    # Draw playhead indicator (red line showing current playback position)
    if self.edit_state.preview_time >= 0:
        playhead_x = self.edit_state.preview_time * scale
        if 0 <= playhead_x <= canvas_width:
            # Playhead line
            self.timeline_canvas.create_line(
                playhead_x, 0, playhead_x, TIMELINE_HEIGHT,
                fill="#ff0000", width=2, tags="playhead"
            )
            
            # Playhead triangle at top
            triangle_size = 8
            self.timeline_canvas.create_polygon(
                playhead_x, 0,
                playhead_x - triangle_size, triangle_size,
                playhead_x + triangle_size, triangle_size,
                fill="#ff0000", outline="#cc0000", width=1, tags="playhead_triangle"
            )
            
            # Current time label above playhead
            current_time_str = self._format_time(self.edit_state.preview_time)
            self.timeline_canvas.create_text(
                playhead_x, triangle_size + 5,
                text=current_time_str, font=("Segoe UI", 9, "bold"),
                fill="#ff0000", anchor="n", tags="playhead_time"
            )
    
    # Draw separator line between ruler and chunks
    self.timeline_canvas.create_line(
        0, RULER_HEIGHT, canvas_width, RULER_HEIGHT,
        fill="#333333", width=1, tags="separator"
    )
    
    # Update scroll region
    timeline_width = max(canvas_width, timeline_duration * scale)
    self.timeline_canvas.configure(scrollregion=(0, 0, timeline_width, TIMELINE_HEIGHT))

