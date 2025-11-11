"""
Preview functionality for video frames.
"""
import tkinter as tk
try:
    from moviepy import concatenate_videoclips
except ImportError:
    from moviepy.editor import concatenate_videoclips


def update_preview_frame(self, photo):
    """Update preview canvas with new frame (called from main thread)."""
    if self.preview_canvas:
        # Get canvas size (update first to get accurate size)
        self.preview_canvas.update_idletasks()
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Use actual canvas size or fallback
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 500
        
        # Get image size
        img_width = photo.width()
        img_height = photo.height()
        
        # Calculate position to center image
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2
        
        # Clear canvas and draw image (centered)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self.preview_canvas.image = photo  # Keep a reference to prevent garbage collection
        
        # Hide label, show canvas
        if self.preview_label:
            self.preview_label.pack_forget()
        if not self.preview_canvas.winfo_viewable():
            self.preview_canvas.pack(fill=tk.BOTH, expand=True)


def render_preview_from_timeline(self):
    """Render video preview from timeline.json chunks."""
    if not self.video_clip or not self.edit_state.chunks:
        return
    
    clips = []
    
    for chunk in self.edit_state.chunks:
        # Use original timing to extract from source video
        extract_start = chunk.original_start_time if chunk.original_start_time is not None else chunk.start_time
        extract_end = chunk.original_end_time if chunk.original_end_time is not None else chunk.end_time
        
        try:
            # Extract subclip from source video
            subclip = self.video_clip.subclipped(extract_start, extract_end)
            
            # Apply speed if needed
            if chunk.speed != 1.0:
                original_fps = subclip.fps
                original_duration = subclip.duration
                subclip = subclip.set_fps(original_fps * chunk.speed)
                subclip = subclip.set_duration(original_duration / chunk.speed)
            
            clips.append(subclip)
        except Exception as e:
            print(f"[VIDSOR] Warning: Failed to extract clip {extract_start}-{extract_end}s: {e}")
            continue
    
    if not clips:
        raise Exception("No clips could be extracted from timeline")
    
    # Concatenate all clips
    preview_clip = concatenate_videoclips(clips)
    
    # Store preview clip (will be used for playback)
    if hasattr(self, 'preview_clip') and self.preview_clip:
        self.preview_clip.close()
    self.preview_clip = preview_clip
    
    print(f"[VIDSOR] Preview rendered: {len(clips)} clips, total duration: {preview_clip.duration:.2f}s")
    
    # Clean up individual clips (preview_clip has its own copy)
    for clip in clips:
        clip.close()

