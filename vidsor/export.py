"""
Video export functionality for Vidsor.
"""

from moviepy import VideoFileClip, concatenate_videoclips
from .models import Chunk, EditState


class VideoExporter:
    """Handles video export functionality."""
    
    @staticmethod
    def export_video(video_clip: VideoFileClip, edit_state: EditState, output_path: str) -> None:
        """
        Export edited video to file.
        
        Args:
            video_clip: Source video clip
            edit_state: Current editing state with chunks
            output_path: Path to save output video
        """
        if not video_clip or not edit_state.chunks:
            raise Exception("No video or chunks to export")
        
        clips = []
        
        for chunk in edit_state.chunks:
            # Use original timing to extract from source video
            # If original timing is not available, fall back to start_time/end_time
            extract_start = chunk.original_start_time if chunk.original_start_time is not None else chunk.start_time
            extract_end = chunk.original_end_time if chunk.original_end_time is not None else chunk.end_time
            
            # Extract subclip from source video using original timing
            subclip = video_clip.subclipped(extract_start, extract_end)
            
            # Apply speed if needed
            if chunk.speed != 1.0:
                # Adjust speed by changing FPS and duration
                original_fps = subclip.fps
                original_duration = subclip.duration
                subclip = subclip.set_fps(original_fps * chunk.speed)
                subclip = subclip.set_duration(original_duration / chunk.speed)
            
            clips.append(subclip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)
        
        # Write to file
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            remove_temp=True
        )
        
        # Cleanup
        final_clip.close()
        for clip in clips:
            clip.close()
        
        print(f"[VIDSOR] Video exported to: {output_path}")

