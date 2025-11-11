"""
Timeline event handlers.
"""
import tkinter as tk

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def on_timeline_click(self, event):
    """Handle timeline click for chunk selection or playhead dragging."""
    self.timeline_controller.on_timeline_click(event)
    if not self.timeline_canvas:
        return
    
    # Pause playback if it's currently running for better drag handling
    if self.edit_state.is_playing:
        self.edit_state.is_playing = False
        self._update_playback_controls()
    
    # Get canvas coordinates
    canvas_x = self.timeline_canvas.canvasx(event.x)
    
    # Calculate timeline duration and scale
    if not self.edit_state.chunks:
        if not self.video_clip:
            return
        timeline_duration = self.video_clip.duration
    else:
        timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
    
    canvas_width = self.timeline_canvas.winfo_width() or 1000
    scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
    
    # Check if clicking on/near the playhead (within 10 pixels)
    playhead_x = self.edit_state.preview_time * scale
    playhead_tolerance = 10
    
    if abs(canvas_x - playhead_x) <= playhead_tolerance:
        # Clicked on playhead - start dragging
        self.is_dragging_playhead = True
        # Stop playback and audio if playing
        if self.edit_state.is_playing:
            self.edit_state.is_playing = False
        # Stop audio completely
        if HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        # Mark that audio needs restart from new position
        self.audio_needs_restart = True
        # Mark that playback has started so Resume button appears
        self.edit_state.has_started_playback = True
        self._update_playback_controls()
    else:
        # Clicked elsewhere on timeline - seek to that position
        new_time = canvas_x / scale
        new_time = max(0, min(new_time, timeline_duration))
        self._seek_to_time(new_time)
        # Start dragging from this position
        self.is_dragging_playhead = True


def on_timeline_drag(self, event):
    """Handle timeline drag for playhead scrubbing."""
    self.timeline_controller.on_timeline_drag(event)
    if not self.is_dragging_playhead or not self.timeline_canvas:
        return
    
    # Stop playback and audio if it's currently running for better drag handling
    if self.edit_state.is_playing:
        self.edit_state.is_playing = False
    # Stop audio completely
    if HAS_PYGAME:
        try:
            pygame.mixer.music.stop()
        except:
            pass
    # Mark that audio needs restart from new position
    self.audio_needs_restart = True
    # Mark that playback has started so Resume button appears
    self.edit_state.has_started_playback = True
    self._update_playback_controls()
    
    # Get canvas coordinates
    canvas_x = self.timeline_canvas.canvasx(event.x)
    
    # Calculate timeline duration and scale
    if not self.edit_state.chunks:
        if not self.video_clip:
            return
        timeline_duration = self.video_clip.duration
    else:
        timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
    
    canvas_width = self.timeline_canvas.winfo_width() or 1000
    scale = canvas_width / timeline_duration if timeline_duration > 0 else 1
    
    # Calculate new time from mouse position
    new_time = canvas_x / scale
    new_time = max(0, min(new_time, timeline_duration))
    
    # Seek to new time
    self._seek_to_time(new_time)


def on_timeline_release(self, event):
    """Handle mouse release after timeline drag."""
    self.timeline_controller.on_timeline_release(event)


def on_timeline_motion(self, event):
    """Handle mouse motion over timeline for hover effects."""
    self.timeline_controller.on_timeline_motion(event)
    # Don't update hover if dragging playhead
    if self.is_dragging_playhead:
        return
    
    if not self.edit_state.chunks or not self.timeline_canvas:
        return
    
    # Get canvas coordinates
    canvas_x = self.timeline_canvas.canvasx(event.x)
    
    # Calculate timeline duration and scale
    timeline_duration = max(chunk.end_time for chunk in self.edit_state.chunks)
    canvas_width = self.timeline_canvas.winfo_width() or 1000
    scale = canvas_width / timeline_duration
    
    # Find which chunk is being hovered
    hovered_chunk_idx = None
    for i, chunk in enumerate(self.edit_state.chunks):
        x_start = chunk.start_time * scale
        x_end = chunk.end_time * scale
        if x_start <= canvas_x <= x_end:
            hovered_chunk_idx = i
            break
    
    # Update hover state and redraw if changed
    if hovered_chunk_idx != self.timeline_hover_chunk:
        self.timeline_hover_chunk = hovered_chunk_idx
        self._draw_timeline()


def on_timeline_leave(self, event):
    """Handle mouse leaving timeline."""
    self.timeline_controller.on_timeline_leave(event)

