"""Execution agent for extracting and saving video clips."""

import os
import time
from datetime import datetime
from langchain_core.messages import AIMessage
from agent.state import AgentState

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # MoviePy 2.x uses different import path
    from moviepy.video.io.VideoFileClip import VideoFileClip


def create_execution_agent():
    """Create the execution agent that extracts and saves video clips."""
    
    def execution_node(state: AgentState) -> AgentState:
        """Execution agent: Extracts video clips and saves them."""
        video_path = state["video_path"]
        time_ranges = state.get("time_ranges", [])
        output_clips = state.get("output_clips", [])
        verbose = state.get("verbose", False)
        
        if verbose:
            print("\n" + "=" * 60)
            print("EXECUTION AGENT: Extracting Video Clips")
            print("=" * 60)
            print(f"Video: {video_path}")
            print(f"Time ranges to extract: {len(time_ranges)}")
        
        if not time_ranges:
            if verbose:
                print("[SKIP] No time ranges to extract")
            return {
                **state,
                "output_clips": output_clips
            }
        
        # Create output directory
        output_dir = "extracted_clips"
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Output directory: {output_dir}/")
        
        # Load video
        try:
            if verbose:
                print(f"\n[LOADING] Opening video file: {video_path}")
            video = VideoFileClip(video_path)
            if verbose:
                print(f"[LOADING] Video loaded successfully:")
                print(f"  Duration: {video.duration:.2f}s")
                print(f"  FPS: {video.fps}")
                print(f"  Resolution: {video.size}")
                print(f"  Codec: {video.codec if hasattr(video, 'codec') else 'N/A'}")
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to load video: {str(e)}")
            return {
                **state,
                "output_clips": [],
                "messages": state["messages"] + [
                    AIMessage(content=f"Error loading video: {str(e)}")
                ]
            }
        
        saved_clips = []
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            clip = None
            try:
                if verbose:
                    print(f"\n[EXTRACTING] Clip {i+1}/{len(time_ranges)}")
                    print(f"  Original range: {start_time:.2f}s - {end_time:.2f}s")
                
                # Close and reopen video file between clips to reset MoviePy's internal state
                # This ensures clean subprocess state for each clip on Windows
                if i > 0:  # Don't close on first clip
                    if verbose:
                        print(f"  [CLEANUP] Closing video file to reset state...")
                    video.close()
                    time.sleep(1.0)  # Give Windows time to fully cleanup subprocesses
                    if verbose:
                        print(f"  [CLEANUP] Reopening video file...")
                    video = VideoFileClip(video_path)
                
                # Ensure times are within video duration
                start_time = max(0, min(start_time, video.duration))
                end_time = max(start_time + 1, min(end_time, video.duration))
                
                if verbose:
                    print(f"  Adjusted range: {start_time:.2f}s - {end_time:.2f}s")
                    print(f"  Duration: {end_time - start_time:.2f}s")
                
                # Extract clip (moviepy 2.x uses subclipped instead of subclip)
                if verbose:
                    print(f"  [PROCESSING] Extracting subclip...")
                
                # Extract the subclip (no output suppression needed for subclipped)
                clip = video.subclipped(start_time, end_time)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clip_{i+1}_{int(start_time)}s_to_{int(end_time)}s_{timestamp}.mp4"
                output_path = os.path.join(output_dir, filename)
                
                if verbose:
                    print(f"  [SAVING] Writing to: {filename}")
                
                # Write clip without logger - MoviePy 2.x works fine without it
                # Use unique temp audio file per clip to avoid Windows subprocess conflicts
                temp_audio = f'temp-audio-{i+1}-{timestamp}.m4a'
                
                clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=temp_audio,
                    remove_temp=True
                )
                
                if clip:
                    clip.close()
                
                # Clean up temp audio file if it still exists
                if os.path.exists(temp_audio):
                    try:
                        os.remove(temp_audio)
                    except:
                        pass
                
                # Verify file was created and has content
                if not os.path.exists(output_path):
                    raise Exception(f"Output file was not created: {output_path}")
                
                file_size = os.path.getsize(output_path)
                if file_size < 1000:  # Less than 1KB is likely corrupted/empty
                    raise Exception(f"Output file is too small ({file_size} bytes), likely corrupted")
                
                saved_clips.append(output_path)
                
                if verbose:
                    print(f"  [SUCCESS] Clip saved: {output_path} ({file_size} bytes)")
                
            except Exception as e:
                error_msg = f"Error extracting clip {i+1}: {str(e)}"
                if verbose:
                    print(f"  [ERROR] {error_msg}")
                print(error_msg)
                if clip:
                    try:
                        clip.close()
                    except:
                        pass
                continue
        
        # Final cleanup
        if video:
            try:
                video.close()
            except:
                pass
        
        if verbose:
            print(f"\n[COMPLETE] Successfully extracted {len(saved_clips)} clip(s)")
            print(f"  Output directory: {output_dir}/")
            for clip in saved_clips:
                print(f"    - {os.path.basename(clip)}")
        
        # Optional: Create OpenShot project if requested
        create_openshot_project = state.get("create_openshot_project", False)
        openshot_project_path = None
        
        if create_openshot_project and saved_clips:
            try:
                from openshot_integration import create_openshot_project_from_clips
                query = state.get("user_query", "clips")[:30].replace(" ", "_")
                openshot_project_path = create_openshot_project_from_clips(
                    saved_clips,
                    project_name=f"agent_{query}",
                    auto_open=state.get("auto_open_openshot", False),
                    verbose=verbose
                )
                if verbose:
                    print(f"\n[OPENSHOT] Project created: {openshot_project_path}")
            except ImportError:
                if verbose:
                    print("\n[OPENSHOT] openshot_integration module not found. Skipping project creation.")
            except Exception as e:
                if verbose:
                    print(f"\n[OPENSHOT] Error creating project: {str(e)}")
        
        return {
            **state,
            "output_clips": saved_clips,
            "openshot_project_path": openshot_project_path,
            "messages": state["messages"] + [
                AIMessage(content=f"Successfully extracted {len(saved_clips)} clip(s) to {output_dir}/")
            ]
        }
    
    return execution_node

