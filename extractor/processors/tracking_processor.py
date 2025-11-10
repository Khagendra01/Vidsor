"""
Tracking data processing and grouping module.
"""

from typing import Dict, List, Optional
from extractor.config import FPS, FRAMES_PER_GROUP, GROUPS_PER_SECOND


class TrackingProcessor:
    """Processes and groups tracking data."""
    
    def __init__(self, yolo_stride: int = 10):
        """
        Initialize tracking processor.
        
        Args:
            yolo_stride: Stride used in YOLO processing
        """
        self.yolo_stride = yolo_stride
    
    def find_frame_by_number(self, frames_data: List[Dict], frame_number: int) -> Optional[Dict]:
        """
        Find frame data by frame_number (handles stride in tracking data).
        
        Args:
            frames_data: List of frame data dictionaries
            frame_number: Frame number to find
            
        Returns:
            Frame data dictionary or None
        """
        # Binary search for efficiency since frames are sorted by frame_number
        left, right = 0, len(frames_data) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_frame_num = frames_data[mid].get('frame_number', 0)
            if mid_frame_num == frame_number:
                return frames_data[mid]
            elif mid_frame_num < frame_number:
                left = mid + 1
            else:
                right = mid - 1
        return None
    
    def group_detections(self, frames_data: List[Dict], start_frame: int, end_frame: int) -> List[Dict]:
        """
        Group frames of detections together, deduplicate by track_id.
        
        Args:
            frames_data: All frame detection data
            start_frame: Start frame number
            end_frame: End frame number
            
        Returns:
            List of detection groups
        """
        groups = []
        
        for group_idx in range(GROUPS_PER_SECOND):
            group_start = start_frame + (group_idx * FRAMES_PER_GROUP)
            group_end = min(group_start + FRAMES_PER_GROUP - 1, end_frame)
            frame_range = list(range(group_start, group_end + 1))
            
            # Collect all detections from this group
            all_detections = {}
            for frame_num in frame_range:
                frame_data = self.find_frame_by_number(frames_data, frame_num)
                if frame_data:
                    for det in frame_data.get('detections', []):
                        track_id = det.get('track_id')
                        if track_id is not None:
                            # Deduplicate by track_id, keep highest confidence
                            if track_id not in all_detections or det['confidence'] > all_detections[track_id]['confidence']:
                                all_detections[track_id] = det.copy()
            
            # Convert to list
            group_detections = list(all_detections.values())
            unique_tracks = list(all_detections.keys())
            
            groups.append({
                "group_index": group_idx,
                "frame_range": [group_start, group_end],
                "representative_frame": group_start if group_idx in [0, 2, 5] else None,
                "detections": group_detections,
                "unique_tracks": unique_tracks,
                "total_detections": len(group_detections)
            })
        
        return groups
    
    def create_detection_summary(self, detection_groups: List[Dict]) -> str:
        """
        Create summary string from detection groups.
        
        Args:
            detection_groups: List of detection groups
            
        Returns:
            Summary string
        """
        all_tracks = set()
        for group in detection_groups:
            all_tracks.update(group["unique_tracks"])
        return f"Total unique tracks: {len(all_tracks)}, Groups: {len(detection_groups)}"

