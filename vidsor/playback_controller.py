"""
Playback control functionality for Vidsor.
Handles video playback, audio synchronization, seeking, and preview rendering.
"""

import os
import time
import threading
import tempfile
import tkinter as tk
from typing import Optional
from tkinter import messagebox

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    from moviepy import concatenate_videoclips, concatenate_audioclips
except ImportError:
    from moviepy.editor import concatenate_videoclips, concatenate_audioclips


class PlaybackController:
    """Handles video and audio playback control."""
    
    def __init__(self, vidsor):
        """
        Initialize playback controller.
        
        Args:
            vidsor: Reference to main Vidsor instance for accessing video, state, and UI
        """
        self.vidsor = vidsor
        self.playback_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_clip = None
        self.preview_clip = None
        self.timeline_update_counter = 0
        self.audio_needs_restart = False
    
    def on_play(self):
        """Play preview button handler - plays video directly from timeline.json chunks."""
        if not self.vidsor.video_clip:
            messagebox.showwarning("Warning", "No video loaded")
            return
        
        if not self.vidsor.edit_state.chunks:
            messagebox.showwarning("Warning", "No clips in timeline. Load timeline.json or generate highlights first.")
            return
        
        # If already playing, do nothing
        if self.vidsor.edit_state.is_playing:
            return
        
        # Start playback directly from source video using timeline chunks
        self.start_playback_from_timeline()
    
    def start_playback_from_timeline(self):
        """Start video playback directly from timeline chunks (no pre-rendering)."""
        if not self.vidsor.video_clip or not self.vidsor.edit_state.chunks:
            return
        
        if not HAS_PIL:
            messagebox.showerror(
                "Error",
                "PIL/Pillow is required for video preview.\n\nInstall with: pip install pillow"
            )
            return
        
        # Calculate total duration from timeline
        timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks) if self.vidsor.edit_state.chunks else 0
        
        self.vidsor.edit_state.is_playing = True
        self.vidsor.edit_state.preview_time = 0.0
        self.vidsor.edit_state.has_started_playback = True  # Mark that playback has started
        self.vidsor.status_label.config(text=f"Playing preview... ({timeline_duration:.1f}s)")
        
        # Update playback controls
        self.vidsor._update_playback_controls()
        
        # Show canvas, hide label
        if self.vidsor.preview_canvas:
            self.vidsor.preview_canvas.pack(fill=tk.BOTH, expand=True)
        if self.vidsor.preview_label:
            self.vidsor.preview_label.pack_forget()
        
        # Initialize audio playback if available
        if HAS_PYGAME and self.vidsor.video_clip.audio is not None:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                # Extract audio from video
                self.audio_clip = self.vidsor.video_clip.audio
            except Exception as e:
                print(f"[VIDSOR] Warning: Could not initialize audio: {e}")
                self.audio_clip = None
        else:
            self.audio_clip = None
        
        # Start video playback thread
        if self.playback_thread and self.playback_thread.is_alive():
            return
        
        self.playback_thread = threading.Thread(
            target=self.playback_loop_from_timeline,
            daemon=True
        )
        self.playback_thread.start()
        
        # Start audio playback thread if audio is available
        if self.audio_clip and HAS_PYGAME:
            if self.audio_thread and self.audio_thread.is_alive():
                return
            
            self.audio_thread = threading.Thread(
                target=self.audio_playback_loop,
                daemon=True
            )
            self.audio_thread.start()
    
    def playback_loop_from_timeline(self):
        """Main playback loop - plays directly from source video using timeline chunks."""
        if not self.vidsor.video_clip or not self.vidsor.edit_state.chunks:
            return
        
        if not HAS_PIL:
            return
        
        # Get video FPS for frame timing
        video_fps = self.vidsor.video_clip.fps
        frame_duration = 1.0 / video_fps
        
        # Calculate total timeline duration
        timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks)
        
        try:
            while self.vidsor.edit_state.preview_time < timeline_duration:
                # Check if paused - wait for resume
                if not self.vidsor.edit_state.is_playing:
                    # Paused - wait in a loop until resumed
                    while not self.vidsor.edit_state.is_playing:
                        time.sleep(0.1)  # Check every 100ms
                        # If we're no longer supposed to be playing (e.g., stopped), exit
                        if not self.vidsor.edit_state.has_started_playback:
                            return
                    # Resumed - continue playback
                    continue
                
                start_time = time.time()
                
                # Find which chunk we're currently in
                current_chunk = None
                for chunk in self.vidsor.edit_state.chunks:
                    if chunk.start_time <= self.vidsor.edit_state.preview_time < chunk.end_time:
                        current_chunk = chunk
                        break
                
                if not current_chunk:
                    # Between chunks or past end - advance time and continue
                    self.vidsor.edit_state.preview_time += frame_duration
                    time.sleep(frame_duration)
                    continue
                
                # Calculate position within this chunk (0.0 to 1.0)
                chunk_position = (self.vidsor.edit_state.preview_time - current_chunk.start_time) / (current_chunk.end_time - current_chunk.start_time)
                
                # Map to original video time
                if current_chunk.original_start_time is not None and current_chunk.original_end_time is not None:
                    original_time = current_chunk.original_start_time + chunk_position * (current_chunk.original_end_time - current_chunk.original_start_time)
                else:
                    # Fallback to sequential timing
                    original_time = current_chunk.start_time + chunk_position * (current_chunk.end_time - current_chunk.start_time)
                
                # Ensure we're within video bounds
                original_time = max(0, min(original_time, self.vidsor.video_clip.duration - 0.1))
                
                # Get frame from source video at original time
                frame = self.vidsor.video_clip.get_frame(original_time)
                
                # Convert frame to PIL Image
                frame_pil = Image.fromarray(frame)
                
                # Resize to fit preview canvas (maintain aspect ratio)
                max_width = 800
                max_height = 500
                
                # Calculate resize dimensions maintaining aspect ratio
                img_width, img_height = frame_pil.size
                aspect = img_width / img_height
                max_aspect = max_width / max_height
                
                if max_aspect > aspect:
                    # Max area is wider - fit to height
                    new_height = max_height
                    new_width = int(new_height * aspect)
                else:
                    # Max area is taller - fit to width
                    new_width = max_width
                    new_height = int(new_width / aspect)
                
                # Always resize to fit preview area
                frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image=frame_pil)
                
                # Update preview label in main thread
                if self.vidsor.root:
                    self.vidsor.root.after(0, lambda p=photo: self.vidsor._update_preview_frame(p))
                
                # Advance timeline time
                self.vidsor.edit_state.preview_time += frame_duration
                
                # Update timeline playhead in main thread (throttled to ~10fps for smooth animation)
                if self.vidsor.root:
                    self.timeline_update_counter += 1
                    if self.timeline_update_counter >= 3:
                        self.timeline_update_counter = 0
                        self.vidsor.root.after(0, self.vidsor._draw_timeline)
                
                # Sleep to maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_duration - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Playback finished
            if self.vidsor.root:
                self.vidsor.edit_state.is_playing = False
                self.vidsor.root.after(0, lambda: self.vidsor.status_label.config(
                    text=f"Preview finished ({timeline_duration:.1f}s)"
                ))
                self.vidsor.root.after(0, self.vidsor._update_playback_controls)
                # Update timeline to show final playhead position
                self.vidsor.root.after(0, self.vidsor._draw_timeline)
                
        except Exception as e:
            print(f"[VIDSOR] Playback error: {e}")
            import traceback
            traceback.print_exc()
            if self.vidsor.root:
                self.vidsor.root.after(0, lambda: self.vidsor.status_label.config(text=f"Playback error: {str(e)}"))
                self.vidsor.edit_state.is_playing = False
    
    def audio_playback_loop(self):
        """Audio playback loop - plays audio synchronized with video timeline."""
        if not self.audio_clip or not HAS_PYGAME or not self.vidsor.edit_state.chunks:
            return
        
        try:
            # Determine start time for audio playback
            start_timeline_time = self.vidsor.edit_state.preview_time
            self.audio_needs_restart = False  # Reset flag after using it
            
            # Build composite audio from timeline chunks starting from start_timeline_time
            audio_segments = []
            for chunk in self.vidsor.edit_state.chunks:
                # Skip chunks that end before the start time
                if chunk.end_time <= start_timeline_time:
                    continue
                
                # Use original timing to extract from source audio
                extract_start = chunk.original_start_time if chunk.original_start_time is not None else chunk.start_time
                extract_end = chunk.original_end_time if chunk.original_end_time is not None else chunk.end_time
                
                # If this chunk contains the start time, trim it
                if chunk.start_time <= start_timeline_time < chunk.end_time:
                    # Calculate position within this chunk
                    chunk_position = (start_timeline_time - chunk.start_time) / (chunk.end_time - chunk.start_time)
                    # Map to original video time
                    if chunk.original_start_time is not None and chunk.original_end_time is not None:
                        extract_start = chunk.original_start_time + chunk_position * (chunk.original_end_time - chunk.original_start_time)
                    else:
                        extract_start = chunk.start_time + chunk_position * (chunk.end_time - chunk.start_time)
                
                try:
                    # Extract audio segment
                    audio_segment = self.audio_clip.subclipped(extract_start, extract_end)
                    audio_segments.append(audio_segment)
                except Exception as e:
                    print(f"[VIDSOR] Warning: Could not extract audio segment {extract_start}-{extract_end}s: {e}")
                    continue
            
            if not audio_segments:
                print("[VIDSOR] No audio segments to play")
                return
            
            # Concatenate audio segments
            try:
                composite_audio = concatenate_audioclips(audio_segments)
            except ImportError:
                # Fallback for older MoviePy versions
                from moviepy.editor import concatenate_audioclips
                composite_audio = concatenate_audioclips(audio_segments)
            
            # Create a temporary audio file for playback
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Write composite audio to temporary file
            try:
                # Try with verbose parameter (MoviePy 2.x)
                try:
                    composite_audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                except TypeError:
                    # Fallback for MoviePy 1.x which doesn't support verbose parameter
                    composite_audio.write_audiofile(temp_audio_path)
            except Exception as e:
                print(f"[VIDSOR] Warning: Could not write audio file: {e}")
                # Cleanup
                composite_audio.close()
                for seg in audio_segments:
                    seg.close()
                return
            
            # Cleanup audio segments
            composite_audio.close()
            for seg in audio_segments:
                seg.close()
            
            # Load and play audio
            pygame.mixer.music.load(temp_audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish or stop
            audio_playing = True
            while audio_playing:
                # Check if music is still playing
                is_busy = pygame.mixer.music.get_busy()
                
                # Check if paused or stopped
                if not self.vidsor.edit_state.is_playing:
                    # If paused, wait for resume
                    while not self.vidsor.edit_state.is_playing:
                        time.sleep(0.1)
                        # Check if we should stop
                        if not self.vidsor.edit_state.has_started_playback:
                            audio_playing = False
                            break
                    
                    # If resumed, try to unpause
                    if self.vidsor.edit_state.is_playing and audio_playing:
                        try:
                            if pygame.mixer.music.get_busy():
                                pygame.mixer.music.unpause()
                            else:
                                print("[VIDSOR] Warning: Audio stopped while paused, cannot resume")
                                audio_playing = False
                        except Exception as e:
                            print(f"[VIDSOR] Audio unpause error: {e}")
                            audio_playing = False
                else:
                    # Playing normally
                    if not is_busy:
                        # Music finished playing
                        audio_playing = False
                    else:
                        time.sleep(0.1)
            
            # Cleanup
            try:
                pygame.mixer.music.stop()
                os.unlink(temp_audio_path)
            except:
                pass
                
        except Exception as e:
            print(f"[VIDSOR] Audio playback error: {e}")
            import traceback
            traceback.print_exc()
            try:
                pygame.mixer.music.stop()
            except:
                pass
    
    def seek_to_time(self, timeline_time: float):
        """
        Seek to a specific time in the timeline and update the video preview.
        
        Args:
            timeline_time: Time in the timeline (in seconds)
        """
        if not self.vidsor.video_clip:
            return
        
        if not HAS_PIL:
            return
        
        # Update preview time
        self.vidsor.edit_state.preview_time = timeline_time
        
        # Stop audio if playing (audio will restart from new position when resumed)
        if self.vidsor.edit_state.is_playing and HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except:
                pass
            self.audio_needs_restart = True
        
        # Mark that playback has started so Resume button appears
        self.vidsor.edit_state.has_started_playback = True
        if self.vidsor.root:
            self.vidsor.root.after(0, self.vidsor._update_playback_controls)
        
        # Calculate timeline duration
        if not self.vidsor.edit_state.chunks:
            timeline_duration = self.vidsor.video_clip.duration
            original_time = max(0, min(timeline_time, self.vidsor.video_clip.duration - 0.1))
        else:
            timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks)
            
            # Find which chunk we're in
            current_chunk = None
            for chunk in self.vidsor.edit_state.chunks:
                if chunk.start_time <= timeline_time < chunk.end_time:
                    current_chunk = chunk
                    break
            
            if current_chunk:
                # Calculate position within this chunk (0.0 to 1.0)
                chunk_position = (timeline_time - current_chunk.start_time) / (current_chunk.end_time - current_chunk.start_time)
                
                # Map to original video time
                if current_chunk.original_start_time is not None and current_chunk.original_end_time is not None:
                    original_time = current_chunk.original_start_time + chunk_position * (current_chunk.original_end_time - current_chunk.original_start_time)
                else:
                    # Fallback to sequential timing
                    original_time = current_chunk.start_time + chunk_position * (current_chunk.end_time - current_chunk.start_time)
            else:
                # Between chunks or past end - use last chunk's end or video duration
                if self.vidsor.edit_state.chunks:
                    last_chunk = self.vidsor.edit_state.chunks[-1]
                    if last_chunk.original_end_time is not None:
                        original_time = last_chunk.original_end_time
                    else:
                        original_time = last_chunk.end_time
                else:
                    original_time = timeline_time
            
            # Ensure we're within video bounds
            original_time = max(0, min(original_time, self.vidsor.video_clip.duration - 0.1))
        
        try:
            # Get frame from source video at original time
            frame = self.vidsor.video_clip.get_frame(original_time)
            
            # Convert frame to PIL Image
            frame_pil = Image.fromarray(frame)
            
            # Resize to fit preview canvas (maintain aspect ratio)
            max_width = 800
            max_height = 500
            
            # Calculate resize dimensions maintaining aspect ratio
            img_width, img_height = frame_pil.size
            aspect = img_width / img_height
            max_aspect = max_width / max_height
            
            if max_aspect > aspect:
                # Max area is wider - fit to height
                new_height = max_height
                new_width = int(new_height * aspect)
            else:
                # Max area is taller - fit to width
                new_width = max_width
                new_height = int(new_width / aspect)
            
            # Resize frame
            frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=frame_pil)
            
            # Update preview frame in main thread
            if self.vidsor.root:
                self.vidsor.root.after(0, lambda p=photo: self.vidsor._update_preview_frame(p))
            
            # Update timeline to show new playhead position
            if self.vidsor.timeline_canvas:
                self.vidsor._draw_timeline()
                
        except Exception as e:
            print(f"[VIDSOR] Error seeking to time {timeline_time}: {e}")
    
    def on_pause(self):
        """Pause/Resume preview button handler."""
        if self.vidsor.edit_state.is_playing:
            # Pause
            self.vidsor.edit_state.is_playing = False
            self.vidsor.status_label.config(text="Paused")
            
            # Pause audio
            if HAS_PYGAME:
                try:
                    pygame.mixer.music.pause()
                except:
                    pass
            
            # Update playback controls
            self.vidsor._update_playback_controls()
        else:
            # Resume
            self.vidsor.edit_state.is_playing = True
            timeline_duration = max(chunk.end_time for chunk in self.vidsor.edit_state.chunks) if self.vidsor.edit_state.chunks else 0
            self.vidsor.status_label.config(text=f"Playing preview... ({timeline_duration:.1f}s)")
            
            # Check if playback thread is still alive - if not, restart it
            if not self.playback_thread or not self.playback_thread.is_alive():
                if self.vidsor.video_clip and self.vidsor.edit_state.chunks:
                    self.playback_thread = threading.Thread(
                        target=self.playback_loop_from_timeline,
                        daemon=True
                    )
                    self.playback_thread.start()
            
            # Check if audio thread is still alive - if not, restart it
            if self.audio_clip and HAS_PYGAME:
                if self.audio_needs_restart:
                    try:
                        pygame.mixer.music.stop()
                    except:
                        pass
                
                if not self.audio_thread or not self.audio_thread.is_alive() or self.audio_needs_restart:
                    self.audio_thread = threading.Thread(
                        target=self.audio_playback_loop,
                        daemon=True
                    )
                    self.audio_thread.start()
                else:
                    # Audio thread is alive and doesn't need restart - just unpause
                    try:
                        pygame.mixer.music.unpause()
                    except Exception as e:
                        print(f"[VIDSOR] Audio unpause error on resume: {e}")
                        # If unpause fails, restart audio thread
                        if self.audio_thread:
                            self.audio_thread = threading.Thread(
                                target=self.audio_playback_loop,
                                daemon=True
                            )
                            self.audio_thread.start()
            
            # Update playback controls
            self.vidsor._update_playback_controls()
    
    def on_stop(self):
        """Stop preview button handler."""
        self.vidsor.edit_state.is_playing = False
        self.vidsor.edit_state.preview_time = 0.0
        self.vidsor.edit_state.has_started_playback = False  # Reset playback flag
        self.vidsor.status_label.config(text="Stopped")
        
        # Stop audio
        if HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        
        # Update playback controls
        self.vidsor._update_playback_controls()
        
        # Reset preview display
        if self.vidsor.preview_canvas:
            self.vidsor.preview_canvas.delete("all")
            self.vidsor.preview_canvas.pack_forget()
        
        # Update timeline to show playhead at start
        self.vidsor._draw_timeline()
        
        if self.vidsor.preview_label:
            if has_video := self.vidsor.video_clip is not None:
                timeline_info = ""
                if self.vidsor.edit_state.chunks:
                    highlight_count = sum(1 for c in self.vidsor.edit_state.chunks if c.chunk_type == "highlight")
                    total_chunks = len(self.vidsor.edit_state.chunks)
                    timeline_info = f"\nTimeline: {total_chunks} chunks ({highlight_count} highlights)"
                
                self.vidsor.preview_label.config(
                    text=f"Video: {os.path.basename(self.vidsor.video_path)}\n\n"
                         f"Duration: {self.vidsor.video_clip.duration:.1f}s\n"
                         f"FPS: {self.vidsor.video_clip.fps}{timeline_info}\n\n"
                         f"Click Play Preview to view timeline"
                )
                self.vidsor.preview_label.pack(fill=tk.BOTH, expand=True)
            else:
                self.vidsor.preview_label.config(text="No video loaded")
                self.vidsor.preview_label.pack(fill=tk.BOTH, expand=True)
    
    def render_preview_from_timeline(self):
        """Render video preview from timeline.json chunks."""
        if not self.vidsor.video_clip or not self.vidsor.edit_state.chunks:
            return
        
        clips = []
        
        for chunk in self.vidsor.edit_state.chunks:
            # Use original timing to extract from source video
            extract_start = chunk.original_start_time if chunk.original_start_time is not None else chunk.start_time
            extract_end = chunk.original_end_time if chunk.original_end_time is not None else chunk.end_time
            
            try:
                # Extract subclip from source video
                subclip = self.vidsor.video_clip.subclipped(extract_start, extract_end)
                
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
        if self.preview_clip:
            self.preview_clip.close()
        self.preview_clip = preview_clip
        
        print(f"[VIDSOR] Preview rendered: {len(clips)} clips, total duration: {preview_clip.duration:.2f}s")
        
        # Clean up individual clips (preview_clip has its own copy)
        for clip in clips:
            clip.close()

