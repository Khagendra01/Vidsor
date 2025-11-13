"""Utility functions for applying video effects like zoom."""

from typing import Dict, List, Optional, Tuple, Any
from moviepy import VideoFileClip
import numpy as np


def find_object_in_time_range(
    segment_tree,
    time_start: float,
    time_end: float,
    object_name: str,
    verbose: bool = False,
    logger=None
) -> Optional[Dict[str, Any]]:
    """
    Find a specific object in a time range using YOLO detection data.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        time_start: Start time in seconds
        time_end: End time in seconds
        object_name: Name of object to find (e.g., "man", "person", "plane")
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with detection info (bbox, confidence, etc.) or None if not found
    """
    # Map common object names to YOLO class names
    object_mapping = {
        "man": "person",
        "woman": "person",
        "people": "person",
        "airplane": "airplane",
        "plane": "airplane",
        "car": "car",
        "truck": "truck",
        "boat": "boat",
        "dog": "dog",
        "cat": "cat",
        "bird": "bird",
    }
    
    # Normalize object name
    object_name_lower = object_name.lower()
    yolo_class = object_mapping.get(object_name_lower, object_name_lower)
    
    log = logger if logger else None
    
    if log:
        log.info(f"  [OBJECT SEARCH] Looking for '{object_name}' (YOLO class: '{yolo_class}')")
        log.info(f"  [OBJECT SEARCH] Time range: {time_start:.2f}s - {time_end:.2f}s")
    elif verbose:
        print(f"  [OBJECT SEARCH] Looking for '{object_name}' (YOLO class: '{yolo_class}')")
        print(f"  [OBJECT SEARCH] Time range: {time_start:.2f}s - {time_end:.2f}s")
    
    # Query segment tree for objects in time range
    objects_data = segment_tree.find_objects_in_time_range(time_start, time_end)
    
    # Find matching object class
    if yolo_class not in objects_data.get("objects", {}):
        if log:
            log.warning(f"  [OBJECT SEARCH] Object '{yolo_class}' not found in time range")
            available = list(objects_data.get("objects", {}).keys())
            if available:
                log.info(f"  [OBJECT SEARCH] Available objects: {available}")
        elif verbose:
            print(f"  [OBJECT SEARCH] Object '{yolo_class}' not found in time range")
            available = list(objects_data.get("objects", {}).keys())
            if available:
                print(f"  [OBJECT SEARCH] Available objects: {available}")
        return None
    
    # Get all detections for this object class
    detections = objects_data["objects"][yolo_class]
    
    if not detections:
        if log:
            log.warning(f"  [OBJECT SEARCH] No detections found for '{yolo_class}'")
        elif verbose:
            print(f"  [OBJECT SEARCH] No detections found for '{yolo_class}'")
        return None
    
    # Strategy: Find the earliest detection (closest to clip start) with good confidence
    # This ensures we start zooming when the object first appears
    # Filter detections with confidence > 0.5
    valid_detections = [
        d for d in detections 
        if d["detection"].get("confidence", 0.0) > 0.5
    ]
    
    if not valid_detections:
        # Fall back to all detections if none meet confidence threshold
        valid_detections = detections
    
    # Find earliest detection (closest to time_start)
    # Use the start of the time_range as the key
    earliest_detection = min(
        valid_detections,
        key=lambda d: d.get("time_range", [time_start, time_end])[0]
    )
    
    # Also get highest confidence detection for reference
    best_confidence_detection = max(
        valid_detections,
        key=lambda d: d["detection"].get("confidence", 0.0)
    )
    
    # Use earliest detection's bbox, but prefer higher confidence if they're close in time
    earliest_time = earliest_detection.get("time_range", [time_start, time_end])[0]
    best_time = best_confidence_detection.get("time_range", [time_start, time_end])[0]
    
    # If best confidence detection is within 2 seconds of earliest, use it
    if abs(best_time - earliest_time) <= 2.0:
        selected_detection = best_confidence_detection
        selection_reason = "highest confidence (within 2s of first appearance)"
    else:
        selected_detection = earliest_detection
        selection_reason = "first appearance"
    
    detection_info = selected_detection["detection"]
    bbox = detection_info.get("bbox")
    confidence = detection_info.get("confidence", 0.0)
    detection_time = selected_detection.get("time_range", [time_start, time_end])[0]
    
    # Calculate offset from clip start (when object appears)
    object_appearance_offset = max(0.0, detection_time - time_start)
    
    if log:
        log.info(f"  [OBJECT SEARCH] Found '{yolo_class}' with confidence {confidence:.2f}")
        log.info(f"  [OBJECT SEARCH] Selection: {selection_reason}")
        log.info(f"  [OBJECT SEARCH] Object appears at {detection_time:.2f}s (offset: {object_appearance_offset:.2f}s from clip start)")
        log.info(f"  [OBJECT SEARCH] Bounding box: {bbox}")
    elif verbose:
        print(f"  [OBJECT SEARCH] Found '{yolo_class}' with confidence {confidence:.2f}")
        print(f"  [OBJECT SEARCH] Selection: {selection_reason}")
        print(f"  [OBJECT SEARCH] Object appears at {detection_time:.2f}s (offset: {object_appearance_offset:.2f}s from clip start)")
        print(f"  [OBJECT SEARCH] Bounding box: {bbox}")
    
    return {
        "bbox": bbox,
        "confidence": confidence,
        "class_name": yolo_class,
        "track_id": detection_info.get("track_id"),
        "time_range": selected_detection.get("time_range"),
        "second": selected_detection.get("second"),
        "detection_time": detection_time,
        "object_appearance_offset": object_appearance_offset  # Time offset from clip start when object appears
    }


def get_available_objects_in_time_range(
    segment_tree,
    time_start: float,
    time_end: float,
    verbose: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all available objects in a time range.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        time_start: Start time in seconds
        time_end: End time in seconds
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary mapping object class names to lists of detection info
    """
    objects_data = segment_tree.find_objects_in_time_range(time_start, time_end)
    
    available_objects = {}
    for class_name, detections in objects_data.get("objects", {}).items():
        if detections:
            # Get best detection for each class (highest confidence)
            best_detection = max(detections, key=lambda d: d["detection"].get("confidence", 0.0))
            available_objects[class_name] = {
                "bbox": best_detection["detection"].get("bbox"),
                "confidence": best_detection["detection"].get("confidence", 0.0),
                "count": len(detections)
            }
    
    if verbose:
        print(f"  [AVAILABLE OBJECTS] Found {len(available_objects)} object classes:")
        for class_name, info in available_objects.items():
            print(f"    - {class_name}: confidence {info['confidence']:.2f}, {info['count']} detections")
    
    return available_objects


def apply_zoom_effect(
    clip: VideoFileClip,
    bbox: List[float],
    effect_type: str = "zoom_in_to_out",
    duration: float = 1.0,
    padding: float = 0.1,
    object_appearance_offset: float = 0.0,
    verbose: bool = False
) -> VideoFileClip:
    """
    Apply zoom effect to a video clip.
    
    Args:
        clip: Input video clip
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        effect_type: Type of zoom effect ("zoom_in_to_out", "zoom_in", "zoom_out")
        duration: Duration of the zoom effect in seconds
        padding: Padding around bounding box (0.1 = 10% padding)
        object_appearance_offset: Time offset from clip start when object appears (default 0.0)
                                  If object appears after clip start, zoom will wait until it appears
        verbose: Whether to print verbose output
        
    Returns:
        Video clip with zoom effect applied
    """
    from PIL import Image
    
    if verbose:
        print(f"  [ZOOM EFFECT] Applying {effect_type} effect")
        print(f"  [ZOOM EFFECT] Duration: {duration:.2f}s")
        print(f"  [ZOOM EFFECT] Object appears at offset: {object_appearance_offset:.2f}s")
        print(f"  [ZOOM EFFECT] Bounding box: {bbox}")
    
    # Get video dimensions
    w, h = clip.size
    
    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox
    
    # Add padding
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    pad_x = bbox_width * padding
    pad_y = bbox_height * padding
    
    # Calculate crop region with padding
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_y)
    
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    # Calculate center of bounding box
    center_x = (crop_x1 + crop_x2) / 2
    center_y = (crop_y1 + crop_y2) / 2
    
    # Calculate zoom factor to fill bounding box
    # We want to zoom so the bbox fills the frame
    zoom_factor = max(w / crop_width, h / crop_height)
    
    if verbose:
        print(f"  [ZOOM EFFECT] Crop region: ({crop_x1:.0f}, {crop_y1:.0f}) to ({crop_x2:.0f}, {crop_y2:.0f})")
        print(f"  [ZOOM EFFECT] Center: ({center_x:.0f}, {center_y:.0f})")
        print(f"  [ZOOM EFFECT] Zoom factor: {zoom_factor:.2f}x")
    
    # Clip duration
    clip_duration = clip.duration
    effect_duration = min(duration, clip_duration)
    
    # object_appearance_offset is passed as parameter
    # This allows zoom to start when object appears, not at clip start
    
    def process_frame(frame, t):
        """Process frame with zoom effect."""
        # Adjust time relative to when object appears
        # If object appears after clip start, wait until it appears
        if t < object_appearance_offset:
            # Object hasn't appeared yet, show full frame
            return frame
        
        # Time relative to when object appears
        effect_time = t - object_appearance_offset
        
        if effect_type == "zoom_in_to_out":
            # Start zoomed in, zoom out to full frame
            if effect_time < effect_duration:
                # Interpolate zoom from zoom_factor to 1.0
                progress = effect_time / effect_duration
                # Use easing function (ease-out)
                eased_progress = 1 - (1 - progress) ** 3
                current_zoom = zoom_factor - (zoom_factor - 1.0) * eased_progress
            else:
                # Show full frame
                current_zoom = 1.0
        elif effect_type == "zoom_in":
            # Zoom in and stay zoomed
            if effect_time < effect_duration:
                progress = effect_time / effect_duration
                eased_progress = progress ** 2  # Ease-in
                current_zoom = 1.0 + (zoom_factor - 1.0) * eased_progress
            else:
                current_zoom = zoom_factor
        elif effect_type == "zoom_out":
            # Start zoomed in, zoom out
            if effect_time < effect_duration:
                progress = effect_time / effect_duration
                eased_progress = 1 - (1 - progress) ** 3  # Ease-out
                current_zoom = zoom_factor - (zoom_factor - 1.0) * eased_progress
            else:
                current_zoom = 1.0
        else:
            # Unknown effect, return original
            return frame
        
        if current_zoom <= 1.0:
            # No zoom needed, return original frame
            return frame
        
        # Calculate current crop size
        current_crop_w = w / current_zoom
        current_crop_h = h / current_zoom
        
        # Calculate crop coordinates centered on bounding box center
        current_x1 = max(0, center_x - current_crop_w / 2)
        current_y1 = max(0, center_y - current_crop_h / 2)
        current_x2 = min(w, current_x1 + current_crop_w)
        current_y2 = min(h, current_y1 + current_crop_h)
        
        # Adjust if we hit boundaries
        if current_x2 - current_x1 < current_crop_w:
            current_x1 = max(0, current_x2 - current_crop_w)
        if current_y2 - current_y1 < current_crop_h:
            current_y1 = max(0, current_y2 - current_crop_h)
        
        # Ensure integer coordinates
        x1_int = int(max(0, current_x1))
        y1_int = int(max(0, current_y1))
        x2_int = int(min(w, current_x1 + current_crop_w))
        y2_int = int(min(h, current_y1 + current_crop_h))
        
        # Crop and resize
        cropped = frame[y1_int:y2_int, x1_int:x2_int]
        
        if cropped.size == 0:
            return frame
        
        # Resize to original dimensions
        from PIL import Image
        pil_img = Image.fromarray(cropped)
        resized = pil_img.resize((w, h), Image.Resampling.LANCZOS)
        return np.array(resized)
    
    # Apply effect using fl_image for frame-by-frame processing
    return clip.fl_image(process_frame)

