"""Feature extraction for video segments."""

from typing import Dict, Any
from segment_tree_utils import SegmentTreeQuery


class PerSecondFeatureExtractor:
    """Extract normalized features for each second of video."""
    
    def __init__(self, segment_tree: SegmentTreeQuery):
        self.segment_tree = segment_tree
        self.feature_cache = {}  # Cache per-second features
        
    def extract_features_for_second(self, second_idx: int) -> Dict[str, Any]:
        """Extract all features for a single second."""
        if second_idx in self.feature_cache:
            return self.feature_cache[second_idx]
            
        second_data = self.segment_tree.get_second_by_index(second_idx)
        if not second_data:
            return None
        
        features = {
            "second": second_idx,
            "time_range": second_data.get("time_range", []),
            "semantic_score": 0.0,  # Will be filled by semantic search
            "activity_score": 0.0,  # Will be filled by activity search
            "hierarchical_score": 0.0,  # Will be filled by hierarchical search
            "transcript_score": 0.0,  # Will be filled by transcript search
            "object_presence": {},  # {class_name: normalized_count}
            "object_tracks": {},  # {class_name: distinct_track_count}
            "event_flags": {}  # {event_name: bool}
        }
        
        # Extract object presence (normalized per second)
        for group in second_data.get("detection_groups", []):
            for detection in group.get("detections", []):
                class_name = detection.get("class_name")
                if class_name:
                    if class_name not in features["object_presence"]:
                        features["object_presence"][class_name] = 0
                        features["object_tracks"][class_name] = set()
                    features["object_presence"][class_name] += 1
                    track_id = detection.get("track_id")
                    if track_id is not None:
                        features["object_tracks"][class_name].add(track_id)
        
        # Normalize object counts (cap at 3 or use log)
        for class_name in list(features["object_presence"].keys()):
            count = features["object_presence"][class_name]
            features["object_presence"][class_name] = min(count, 3)  # Cap at 3
            features["object_tracks"][class_name] = len(features["object_tracks"][class_name])
        
        self.feature_cache[second_idx] = features
        return features
    
    def get_all_seconds_features(self):
        """Get features for all seconds (lazy load)."""
        for i in range(len(self.segment_tree.seconds)):
            yield self.extract_features_for_second(i)

