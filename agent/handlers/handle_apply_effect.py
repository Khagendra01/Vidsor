"""Handler for APPLY_EFFECT operation."""

import os
from typing import Dict, Any, Optional
from moviepy import VideoFileClip
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState
from agent.utils.effect_utils import (
    find_object_in_time_range,
    get_available_objects_in_time_range
)


def handle_apply_effect(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle APPLY_EFFECT operation.
    Applies visual effects (like zoom) to specified timeline chunks.
    
    Args:
        state: Orchestrator state
        timeline_manager: Timeline manager instance
        params: Operation parameters with timeline_indices, effect_type, effect_object, effect_duration
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with success status and affected chunks
    """
    logger = state.get("logger")
    log = logger if logger else None
    
    if log:
        log.info("\n[OPERATION] APPLY_EFFECT")
    elif verbose:
        print("\n[OPERATION] APPLY_EFFECT")
    
    indices = params.get("timeline_indices", [])
    effect_type = params.get("effect_type")
    effect_object = params.get("effect_object")
    effect_duration = params.get("effect_duration", 1.0)
    
    if not indices:
        return {
            "success": False,
            "error": "No timeline indices provided",
            "chunks_affected": []
        }
    
    if not effect_type:
        return {
            "success": False,
            "error": "No effect type provided",
            "chunks_affected": []
        }
    
    if not effect_object:
        return {
            "success": False,
            "error": "No target object provided",
            "chunks_affected": []
        }
    
    # Validate indices
    is_valid, error = timeline_manager.validate_indices(indices)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "chunks_affected": []
        }
    
    # Get segment tree
    segment_tree = state.get("segment_tree")
    if not segment_tree:
        return {
            "success": False,
            "error": "Segment tree not available",
            "chunks_affected": []
        }
    
    # Get video path
    video_path = state.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return {
            "success": False,
            "error": f"Video file not found: {video_path}",
            "chunks_affected": []
        }
    
    if log:
        log.info(f"  Applying {effect_type} effect to {len(indices)} chunk(s)")
        log.info(f"  Target object: {effect_object}")
        log.info(f"  Effect duration: {effect_duration:.2f}s")
    elif verbose:
        print(f"  Applying {effect_type} effect to {len(indices)} chunk(s)")
        print(f"  Target object: {effect_object}")
        print(f"  Effect duration: {effect_duration:.2f}s")
    
    affected_chunks = []
    errors = []
    
    # Load video clip
    try:
        video = VideoFileClip(video_path)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load video: {e}",
            "chunks_affected": []
        }
    
    try:
        for idx in indices:
            if idx < 0 or idx >= len(timeline_manager.chunks):
                errors.append(f"Invalid index: {idx}")
                continue
            
            chunk = timeline_manager.chunks[idx]
            
            # Get original time range from chunk
            original_start = chunk.get("original_start_time")
            original_end = chunk.get("original_end_time")
            
            if original_start is None or original_end is None:
                errors.append(f"Chunk {idx} missing original timing")
                continue
            
            if log:
                log.info(f"\n  Processing chunk {idx}: {original_start:.2f}s - {original_end:.2f}s")
            elif verbose:
                print(f"\n  Processing chunk {idx}: {original_start:.2f}s - {original_end:.2f}s")
            
            # Find object in this time range
            object_info = find_object_in_time_range(
                segment_tree=segment_tree,
                time_start=original_start,
                time_end=original_end,
                object_name=effect_object,
                verbose=verbose,
                logger=logger
            )
            
            if not object_info:
                # Object not found - get available objects
                available = get_available_objects_in_time_range(
                    segment_tree=segment_tree,
                    time_start=original_start,
                    time_end=original_end,
                    verbose=verbose
                )
                
                available_names = list(available.keys())
                if available_names:
                    error_msg = (
                        f"Object '{effect_object}' not found in chunk {idx} "
                        f"({original_start:.2f}s - {original_end:.2f}s). "
                        f"Available objects: {', '.join(available_names)}"
                    )
                else:
                    error_msg = (
                        f"No objects detected in chunk {idx} "
                        f"({original_start:.2f}s - {original_end:.2f}s)"
                    )
                
                errors.append(error_msg)
                if log:
                    log.warning(f"  ✗ {error_msg}")
                elif verbose:
                    print(f"  ✗ {error_msg}")
                continue
            
            # Store effect metadata in chunk
            # Note: Actual effect will be applied during video export/preview rendering
            try:
                bbox = object_info["bbox"]
                
                # Initialize effects list if not present
                if "effects" not in chunk:
                    chunk["effects"] = []
                
                # Add effect metadata
                # Store when object appears relative to clip start
                object_appearance_offset = object_info.get("object_appearance_offset", 0.0)
                
                chunk["effects"].append({
                    "type": effect_type,
                    "object": effect_object,
                    "duration": effect_duration,
                    "bbox": bbox,
                    "confidence": object_info.get("confidence", 0.0),
                    "center_x": (bbox[0] + bbox[2]) / 2,
                    "center_y": (bbox[1] + bbox[3]) / 2,
                    "object_appearance_offset": object_appearance_offset,  # When object appears in clip
                    "detection_time": object_info.get("detection_time")
                })
                
                affected_chunks.append(idx)
                
                if log:
                    log.info(f"  ✓ Stored {effect_type} effect metadata for chunk {idx}")
                    log.info(f"    Object: {effect_object}, BBox: {bbox}, Duration: {effect_duration:.2f}s")
                    log.info(f"    Object appears at offset: {object_appearance_offset:.2f}s")
                elif verbose:
                    print(f"  ✓ Stored {effect_type} effect metadata for chunk {idx}")
                    print(f"    Object: {effect_object}, BBox: {bbox}, Duration: {effect_duration:.2f}s")
                    print(f"    Object appears at offset: {object_appearance_offset:.2f}s")
                    
            except Exception as e:
                error_msg = f"Failed to store effect metadata for chunk {idx}: {e}"
                errors.append(error_msg)
                if log:
                    log.error(f"  ✗ {error_msg}")
                    import traceback
                    log.error(traceback.format_exc())
                elif verbose:
                    print(f"  ✗ {error_msg}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # Close video
        video.close()
        
        if errors and not affected_chunks:
            # All chunks failed
            return {
                "success": False,
                "error": "; ".join(errors),
                "chunks_affected": []
            }
        
        result = {
            "success": True,
            "chunks_affected": affected_chunks,
            "chunks_count": len(affected_chunks)
        }
        
        if errors:
            result["warnings"] = errors
            result["partial_success"] = True
        
        if log:
            log.info(f"\n  ✓ Successfully applied effect to {len(affected_chunks)} chunk(s)")
            if errors:
                log.warning(f"  ⚠ {len(errors)} chunk(s) failed")
                for error in errors:
                    log.warning(f"    - {error}")
        elif verbose:
            print(f"\n  ✓ Successfully applied effect to {len(affected_chunks)} chunk(s)")
            if errors:
                print(f"  ⚠ {len(errors)} chunk(s) failed")
                for error in errors:
                    print(f"    - {error}")
        
        return result
        
    except Exception as e:
        video.close()
        error_msg = f"Error in APPLY_EFFECT: {e}"
        if log:
            log.error(f"  ✗ {error_msg}")
            import traceback
            log.error(traceback.format_exc())
        elif verbose:
            print(f"  ✗ {error_msg}")
            import traceback
            traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "chunks_affected": affected_chunks if affected_chunks else []
        }

