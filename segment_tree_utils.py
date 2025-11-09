"""
Utility functions for querying and analyzing segment tree JSON files.
Provides efficient querying capabilities for AI agents.
"""

import json
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict


class SegmentTreeQuery:
    """Query interface for segment tree JSON files."""
    
    def __init__(self, json_path: str):
        """Load and initialize the segment tree from JSON file."""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.video = self.data.get("video", "")
        self.fps = self.data.get("fps", 30)
        self.tracker = self.data.get("tracker", "")
        self.seconds = self.data.get("seconds", [])
        
    def get_video_info(self) -> Dict[str, Any]:
        """Get basic video information."""
        total_seconds = len(self.seconds)
        total_frames = self.seconds[-1]["frame_range"][1] if self.seconds else 0
        return {
            "video": self.video,
            "fps": self.fps,
            "tracker": self.tracker,
            "duration_seconds": total_seconds,
            "total_frames": total_frames
        }
    
    def get_second_by_time(self, time_seconds: float) -> Optional[Dict]:
        """Get second data for a specific time."""
        for second_data in self.seconds:
            time_range = second_data.get("time_range", [])
            if time_range[0] <= time_seconds <= time_range[1]:
                return second_data
        return None
    
    def get_second_by_index(self, second_idx: int) -> Optional[Dict]:
        """Get second data by index."""
        if 0 <= second_idx < len(self.seconds):
            return self.seconds[second_idx]
        return None
    
    def find_objects_by_class(self, class_name: str, 
                             time_start: Optional[float] = None,
                             time_end: Optional[float] = None) -> List[Dict]:
        """
        Find all occurrences of a specific object class.
        
        Args:
            class_name: Name of the class to search for (e.g., "person", "boat")
            time_start: Optional start time in seconds
            time_end: Optional end time in seconds
            
        Returns:
            List of detections with metadata about when/where they appear
        """
        results = []
        
        for second_data in self.seconds:
            time_range = second_data.get("time_range", [])
            
            # Filter by time range if specified
            if time_start is not None and time_range[1] < time_start:
                continue
            if time_end is not None and time_range[0] > time_end:
                continue
            
            second_idx = second_data.get("second", 0)
            
            for group in second_data.get("detection_groups", []):
                for detection in group.get("detections", []):
                    if detection.get("class_name") == class_name:
                        results.append({
                            "detection": detection,
                            "second": second_idx,
                            "time_range": time_range,
                            "frame_range": group.get("frame_range"),
                            "group_index": group.get("group_index")
                        })
        
        return results
    
    def find_track_timeline(self, track_id: int) -> List[Dict]:
        """
        Get the complete timeline for a specific track_id.
        
        Args:
            track_id: The track ID to follow
            
        Returns:
            List of all detections for this track across time
        """
        timeline = []
        
        for second_data in self.seconds:
            second_idx = second_data.get("second", 0)
            time_range = second_data.get("time_range", [])
            
            for group in second_data.get("detection_groups", []):
                for detection in group.get("detections", []):
                    if detection.get("track_id") == track_id:
                        timeline.append({
                            "detection": detection,
                            "second": second_idx,
                            "time_range": time_range,
                            "frame_range": group.get("frame_range"),
                            "group_index": group.get("group_index")
                        })
        
        return sorted(timeline, key=lambda x: x["frame_range"][0] if x["frame_range"] else 0)
    
    def search_descriptions(self, keyword: str, 
                          search_type: str = "any",
                          time_start: Optional[float] = None,
                          time_end: Optional[float] = None) -> List[Dict]:
        """
        Search for keywords in descriptions (BLIP or unified).
        
        Args:
            keyword: Keyword to search for
            search_type: "any" (BLIP or unified), "blip", "unified", or "both"
            time_start: Optional start time in seconds
            time_end: Optional end time in seconds
            
        Returns:
            List of matching seconds with description context
        """
        keyword_lower = keyword.lower()
        results = []
        
        for second_data in self.seconds:
            time_range = second_data.get("time_range", [])
            
            # Filter by time range if specified
            if time_start is not None and time_range[1] < time_start:
                continue
            if time_end is not None and time_range[0] > time_end:
                continue
            
            matches = []
            
            # Search unified description
            unified_desc = second_data.get("unified_description", "")
            if search_type in ["any", "unified", "both"]:
                if unified_desc and unified_desc.lower() != "0" and keyword_lower in unified_desc.lower():
                    matches.append({
                        "type": "unified",
                        "description": unified_desc
                    })
            
            # Search BLIP descriptions
            if search_type in ["any", "blip", "both"]:
                for blip_desc in second_data.get("blip_descriptions", []):
                    desc_text = blip_desc.get("description", "")
                    if keyword_lower in desc_text.lower():
                        matches.append({
                            "type": "blip",
                            "description": desc_text,
                            "frame_number": blip_desc.get("frame_number"),
                            "frame_range": blip_desc.get("frame_range")
                        })
            
            if matches:
                results.append({
                    "second": second_data.get("second", 0),
                    "time_range": time_range,
                    "matches": matches
                })
        
        return results
    
    def get_all_classes(self) -> Dict[str, int]:
        """
        Get all unique object classes and their occurrence counts.
        
        Returns:
            Dictionary mapping class_name to count
        """
        class_counts = defaultdict(int)
        
        for second_data in self.seconds:
            for group in second_data.get("detection_groups", []):
                for detection in group.get("detections", []):
                    class_name = detection.get("class_name")
                    if class_name:
                        class_counts[class_name] += 1
        
        return dict(class_counts)
    
    def get_all_tracks(self) -> Set[int]:
        """
        Get all unique track IDs in the video.
        
        Returns:
            Set of track IDs
        """
        track_ids = set()
        
        for second_data in self.seconds:
            for group in second_data.get("detection_groups", []):
                for detection in group.get("detections", []):
                    track_id = detection.get("track_id")
                    if track_id is not None:
                        track_ids.add(track_id)
        
        return track_ids
    
    def get_scene_summary(self, time_start: Optional[float] = None,
                         time_end: Optional[float] = None) -> Dict:
        """
        Get a summary of what's happening in a time range.
        
        Args:
            time_start: Optional start time in seconds
            time_end: Optional end time in seconds
            
        Returns:
            Summary dictionary with objects, descriptions, and statistics
        """
        relevant_seconds = []
        
        for second_data in self.seconds:
            time_range = second_data.get("time_range", [])
            
            if time_start is not None and time_range[1] < time_start:
                continue
            if time_end is not None and time_range[0] > time_end:
                continue
            
            relevant_seconds.append(second_data)
        
        if not relevant_seconds:
            return {
                "time_range": [time_start, time_end] if time_start and time_end else None,
                "seconds_count": 0,
                "objects": {},
                "descriptions": [],
                "total_detections": 0
            }
        
        # Collect objects
        class_counts = defaultdict(int)
        all_descriptions = []
        total_detections = 0
        
        for second_data in relevant_seconds:
            # Count objects
            for group in second_data.get("detection_groups", []):
                for detection in group.get("detections", []):
                    class_name = detection.get("class_name")
                    if class_name:
                        class_counts[class_name] += 1
                        total_detections += 1
            
            # Collect descriptions
            unified_desc = second_data.get("unified_description", "")
            if unified_desc and unified_desc.lower() != "0":
                all_descriptions.append({
                    "type": "unified",
                    "second": second_data.get("second", 0),
                    "text": unified_desc
                })
            
            for blip_desc in second_data.get("blip_descriptions", []):
                all_descriptions.append({
                    "type": "blip",
                    "second": second_data.get("second", 0),
                    "text": blip_desc.get("description", "")
                })
        
        return {
            "time_range": [
                relevant_seconds[0].get("time_range", [])[0],
                relevant_seconds[-1].get("time_range", [])[1]
            ],
            "seconds_count": len(relevant_seconds),
            "objects": dict(class_counts),
            "descriptions": all_descriptions,
            "total_detections": total_detections
        }
    
    def find_objects_in_time_range(self, time_start: float, time_end: float) -> Dict:
        """
        Get all objects detected in a specific time range.
        
        Args:
            time_start: Start time in seconds
            time_end: End time in seconds
            
        Returns:
            Dictionary with objects, tracks, and detections
        """
        objects = defaultdict(list)
        tracks = set()
        all_detections = []
        
        for second_data in self.seconds:
            time_range = second_data.get("time_range", [])
            
            if time_range[1] < time_start or time_range[0] > time_end:
                continue
            
            for group in second_data.get("detection_groups", []):
                for detection in group.get("detections", []):
                    class_name = detection.get("class_name")
                    track_id = detection.get("track_id")
                    
                    if class_name:
                        objects[class_name].append({
                            "detection": detection,
                            "second": second_data.get("second", 0),
                            "time_range": time_range,
                            "frame_range": group.get("frame_range")
                        })
                    
                    if track_id is not None:
                        tracks.add(track_id)
                    
                    all_detections.append(detection)
        
        return {
            "time_range": [time_start, time_end],
            "objects": dict(objects),
            "unique_tracks": list(tracks),
            "total_detections": len(all_detections),
            "detections": all_detections
        }
    
    def get_narrative_description(self, start_second: int, end_second: int,
                                 include_timestamps: bool = False,
                                 separator: str = " ") -> Dict[str, Any]:
        """
        Get a concatenated narrative description of what happens in a time range.
        Combines all unified and BLIP descriptions in chronological order.
        
        Args:
            start_second: Start second (inclusive)
            end_second: End second (inclusive)
            include_timestamps: Whether to include timestamps in the narrative
            separator: String to use between descriptions (default: space)
            
        Returns:
            Dictionary with:
            - narrative: Concatenated description string
            - descriptions: List of individual descriptions with metadata
            - time_range: [start_second, end_second]
            - seconds_covered: Number of seconds in range
        """
        descriptions_list = []
        
        # Collect all descriptions in the range
        for second_data in self.seconds:
            second_idx = second_data.get("second", 0)
            
            if second_idx < start_second or second_idx > end_second:
                continue
            
            time_range = second_data.get("time_range", [])
            
            # Add unified description if valid
            unified_desc = second_data.get("unified_description", "")
            if unified_desc and unified_desc.lower() != "0":
                descriptions_list.append({
                    "type": "unified",
                    "second": second_idx,
                    "time_range": time_range,
                    "text": unified_desc.strip()
                })
            
            # Add BLIP descriptions
            for blip_desc in second_data.get("blip_descriptions", []):
                desc_text = blip_desc.get("description", "").strip()
                if desc_text:
                    descriptions_list.append({
                        "type": "blip",
                        "second": second_idx,
                        "time_range": time_range,
                        "frame_number": blip_desc.get("frame_number"),
                        "frame_range": blip_desc.get("frame_range"),
                        "text": desc_text
                    })
        
        # Sort by second (and frame if available for BLIP)
        descriptions_list.sort(key=lambda x: (
            x["second"],
            x.get("frame_number", 0) if x.get("frame_number") is not None else 0
        ))
        
        # Build narrative string
        narrative_parts = []
        for desc in descriptions_list:
            text = desc["text"]
            
            if include_timestamps:
                time_str = f"[{desc['second']}s] "
                text = time_str + text
            
            narrative_parts.append(text)
        
        narrative = separator.join(narrative_parts)
        
        return {
            "narrative": narrative,
            "descriptions": descriptions_list,
            "time_range": [start_second, end_second],
            "seconds_covered": end_second - start_second + 1,
            "description_count": len(descriptions_list)
        }


def load_segment_tree(json_path: str) -> SegmentTreeQuery:
    """
    Convenience function to load a segment tree JSON file.
    
    Args:
        json_path: Path to the segment tree JSON file
        
    Returns:
        SegmentTreeQuery instance
    """
    return SegmentTreeQuery(json_path)

