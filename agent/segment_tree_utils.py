"""
Utility functions for querying and analyzing segment tree JSON files.
Provides efficient querying capabilities for AI agents.
"""

import json
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
import os
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SegmentTreeQuery:
    """Query interface for segment tree JSON files."""
    
    def __init__(self, json_path: str):
        """Load and initialize the segment tree from JSON file."""
        self.json_path = json_path
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.video = self.data.get("video", "")
        self.fps = self.data.get("fps", 30)
        self.tracker = self.data.get("tracker", "")
        self.seconds = self.data.get("seconds", [])
        self.transcriptions = self.data.get("transcriptions", [])
        # Create a mapping from transcription_id to transcription for quick lookup
        self._transcription_map = {t.get("id"): t for t in self.transcriptions}
        
        # Semantic search attributes (lazy loaded)
        self._embedding_model = None
        self._transcription_embeddings = None
        self._unified_description_embeddings = None
        self._embedding_metadata = None
        
        # Cache file path for embeddings
        json_path_obj = Path(json_path)
        self._cache_path = json_path_obj.parent / f"{json_path_obj.stem}_embeddings.npz"
        
        # Hierarchical tree (if available)
        self.hierarchical_tree = self.data.get("hierarchical_tree")
        self._hierarchical_nodes = None
        self._hierarchical_indexes = None
        if self.hierarchical_tree:
            self._hierarchical_nodes = self.hierarchical_tree.get("nodes", {})
            self._hierarchical_indexes = self.hierarchical_tree.get("indexes", {})
        
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
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
            time_range = second_data.get("time_range", [])
            if time_range and len(time_range) >= 2 and time_range[0] <= time_seconds <= time_range[1]:
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
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
            time_range = second_data.get("time_range", [])
            
            # Filter by time range if specified
            if not time_range or len(time_range) < 2:
                continue
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
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
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
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
            time_range = second_data.get("time_range", [])
            
            # Filter by time range if specified
            if not time_range or len(time_range) < 2:
                continue
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
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
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
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
            for group in second_data.get("detection_groups", []):
                # Skip None values in detection_groups
                if group is None:
                    continue
                for detection in group.get("detections", []):
                    # Skip None values in detections
                    if detection is None:
                        continue
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
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
            time_range = second_data.get("time_range", [])
            
            if not time_range or len(time_range) < 2:
                continue
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
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
            time_range = second_data.get("time_range", [])
            
            if not time_range or len(time_range) < 2:
                continue
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
    
    def check_activity(self, 
                      activity_keywords: List[str],
                      evidence_keywords: Optional[List[str]] = None,
                      activity_name: str = "activity",
                      evidence_name: str = "evidence") -> Dict[str, Any]:
        """
        General function to check for any activity/event in the video.
        Searches descriptions for activity keywords and evidence keywords.
        
        Args:
            activity_keywords: List of keywords that indicate the activity (e.g., ["fishing", "fish"])
            evidence_keywords: Optional list of keywords that indicate strong evidence 
                            (e.g., ["holding a fish", "caught"]). If None, uses activity_keywords.
            activity_name: Name of the activity for summary (e.g., "fishing")
            evidence_name: Name of the evidence for summary (e.g., "fish caught")
        
        Returns:
            Dictionary with:
            - detected: Boolean indicating if evidence was found
            - evidence: List of evidence (descriptions, timestamps)
            - activity_scenes: List of all activity-related scenes
            - evidence_scenes: List of scenes with strong evidence
            - activity_count: Number of activity scenes
            - evidence_count: Number of evidence scenes
            - summary: Text summary of findings
        """
        evidence = []
        activity_scenes = []
        evidence_scenes = []
        
        if evidence_keywords is None:
            evidence_keywords = activity_keywords
        
        # Search for activity-related descriptions
        for second_data in self.seconds:
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
            second_idx = second_data.get("second", 0)
            time_range = second_data.get("time_range", [])
            
            # Check unified description
            unified_desc = second_data.get("unified_description", "")
            if unified_desc and unified_desc.lower() != "0":
                desc_lower = unified_desc.lower()
                
                # Check for activity
                if any(keyword.lower() in desc_lower for keyword in activity_keywords):
                    has_evidence = any(keyword.lower() in desc_lower for keyword in evidence_keywords)
                    
                    activity_scenes.append({
                        "second": second_idx,
                        "time_range": time_range,
                        "type": "unified",
                        "description": unified_desc,
                        "has_evidence": has_evidence
                    })
                    
                    # Check if it has strong evidence
                    if has_evidence:
                        evidence_scenes.append({
                            "second": second_idx,
                            "time_range": time_range,
                            "type": "unified",
                            "description": unified_desc
                        })
                        evidence.append({
                            "second": second_idx,
                            "time_range": time_range,
                            "type": "unified",
                            "description": unified_desc,
                            "evidence_type": evidence_name
                        })
            
            # Check BLIP descriptions
            for blip_desc in second_data.get("blip_descriptions", []):
                # Skip None values in blip_descriptions
                if blip_desc is None:
                    continue
                desc_text = blip_desc.get("description", "").strip()
                if desc_text:
                    desc_lower = desc_text.lower()
                    
                    # Check for activity
                    if any(keyword.lower() in desc_lower for keyword in activity_keywords):
                        has_evidence = any(keyword.lower() in desc_lower for keyword in evidence_keywords)
                        
                        activity_scenes.append({
                            "second": second_idx,
                            "time_range": time_range,
                            "type": "blip",
                            "description": desc_text,
                            "frame_number": blip_desc.get("frame_number"),
                            "has_evidence": has_evidence
                        })
                        
                        # Check if it has strong evidence
                        if has_evidence:
                            evidence_scenes.append({
                                "second": second_idx,
                                "time_range": time_range,
                                "type": "blip",
                                "description": desc_text,
                                "frame_number": blip_desc.get("frame_number")
                            })
                            evidence.append({
                                "second": second_idx,
                                "time_range": time_range,
                                "type": "blip",
                                "description": desc_text,
                                "frame_number": blip_desc.get("frame_number"),
                                "evidence_type": evidence_name
                            })
        
        # Determine if activity/evidence was found
        detected = len(evidence_scenes) > 0
        
        # Build summary
        if detected:
            summary = f"YES - {evidence_name.capitalize()} detected! Found {len(evidence_scenes)} scene(s) with evidence."
            if evidence:
                summary += f" Evidence found at seconds: {[e['second'] for e in evidence]}"
        elif len(activity_scenes) > 0:
            summary = f"UNCERTAIN - {activity_name.capitalize()} activity detected ({len(activity_scenes)} scenes) but no clear evidence found."
        else:
            summary = f"NO - No {activity_name} activity or {evidence_name} detected in the video."
        
        return {
            "detected": detected,
            "evidence": evidence,
            "activity_scenes": activity_scenes,
            "evidence_scenes": evidence_scenes,
            "activity_count": len(activity_scenes),
            "evidence_count": len(evidence_scenes),
            "summary": summary
        }
    
    def check_fish_caught(self) -> Dict[str, Any]:
        """
        Convenience wrapper for checking if fish are caught.
        Uses the general check_activity function with fishing-specific keywords.
        
        Returns:
            Dictionary with fishing-specific results (backward compatible)
        """
        result = self.check_activity(
            activity_keywords=["fishing", "fish", "caught", "catch"],
            evidence_keywords=["holding a fish", "holding fish", "caught", "with a fish", "fish in", "fish on"],
            activity_name="fishing",
            evidence_name="fish caught"
        )
        
        # Return in backward-compatible format
        return {
            "fish_caught": result["detected"],
            "evidence": result["evidence"],
            "fishing_scenes": result["activity_scenes"],
            "fish_holding_scenes": result["evidence_scenes"],
            "fishing_scene_count": result["activity_count"],
            "fish_holding_count": result["evidence_count"],
            "summary": result["summary"]
        }
    
    def get_transcription_by_id(self, transcription_id: int) -> Optional[Dict]:
        """
        Get a transcription by its ID.
        
        Args:
            transcription_id: The transcription ID
            
        Returns:
            Transcription dictionary or None if not found
        """
        return self._transcription_map.get(transcription_id)
    
    def get_transcription_for_second(self, second_idx: int) -> Optional[Dict]:
        """
        Get the transcription associated with a specific second.
        
        Args:
            second_idx: The second index
            
        Returns:
            Transcription dictionary or None if not found
        """
        second_data = self.get_second_by_index(second_idx)
        if not second_data:
            return None
        
        transcription_id = second_data.get("transcription_id")
        if transcription_id is None:
            return None
        
        return self.get_transcription_by_id(transcription_id)
    
    def get_transcriptions_for_time_range(self, time_start: float, 
                                         time_end: float) -> List[Dict]:
        """
        Get all transcriptions that overlap with a time range.
        
        Args:
            time_start: Start time in seconds
            time_end: End time in seconds
            
        Returns:
            List of transcriptions that overlap with the time range
        """
        results = []
        
        for transcription in self.transcriptions:
            tr_time_range = transcription.get("time_range", [])
            if not tr_time_range or len(tr_time_range) < 2:
                continue
            
            # Check if transcription overlaps with the requested range
            tr_start, tr_end = tr_time_range[0], tr_time_range[1]
            if tr_end >= time_start and tr_start <= time_end:
                results.append(transcription)
        
        return sorted(results, key=lambda x: x.get("time_range", [0])[0])
    
    def search_transcriptions(self, keyword: str,
                             time_start: Optional[float] = None,
                             time_end: Optional[float] = None) -> List[Dict]:
        """
        Search for keywords in transcriptions.
        
        Args:
            keyword: Keyword to search for
            time_start: Optional start time in seconds
            time_end: Optional end time in seconds
            
        Returns:
            List of matching transcriptions with context
        """
        keyword_lower = keyword.lower()
        results = []
        
        for transcription in self.transcriptions:
            tr_time_range = transcription.get("time_range", [])
            if not tr_time_range or len(tr_time_range) < 2:
                continue
            
            # Filter by time range if specified
            if time_start is not None and tr_time_range[1] < time_start:
                continue
            if time_end is not None and tr_time_range[0] > time_end:
                continue
            
            transcription_text = transcription.get("transcription", "")
            if keyword_lower in transcription_text.lower():
                results.append({
                    "transcription": transcription,
                    "time_range": tr_time_range,
                    "id": transcription.get("id"),
                    "metadata": transcription.get("metadata", {})
                })
        
        return sorted(results, key=lambda x: x["time_range"][0])
    
    def get_combined_narrative(self, start_second: int, end_second: int,
                             include_timestamps: bool = False,
                             include_audio: bool = True,
                             separator: str = " ") -> Dict[str, Any]:
        """
        Get a combined narrative that includes both visual descriptions and audio transcriptions.
        
        Args:
            start_second: Start second (inclusive)
            end_second: End second (inclusive)
            include_timestamps: Whether to include timestamps in the narrative
            include_audio: Whether to include audio transcriptions
            separator: String to use between descriptions (default: space)
            
        Returns:
            Dictionary with:
            - narrative: Combined narrative string
            - visual_descriptions: List of visual descriptions
            - audio_transcriptions: List of audio transcriptions
            - time_range: [start_second, end_second]
            - seconds_covered: Number of seconds in range
        """
        visual_descriptions = []
        audio_transcriptions = []
        
        # Get time range
        start_time = start_second
        end_time = end_second + 0.999
        
        # Collect visual descriptions
        for second_data in self.seconds:
            second_idx = second_data.get("second", 0)
            
            if second_idx < start_second or second_idx > end_second:
                continue
            
            time_range = second_data.get("time_range", [])
            
            # Add unified description if valid
            unified_desc = second_data.get("unified_description", "")
            if unified_desc and unified_desc.lower() != "0":
                visual_descriptions.append({
                    "type": "unified",
                    "second": second_idx,
                    "time_range": time_range,
                    "text": unified_desc.strip(),
                    "source": "visual"
                })
            
            # Add BLIP descriptions
            for blip_desc in second_data.get("blip_descriptions", []):
                desc_text = blip_desc.get("description", "").strip()
                if desc_text:
                    visual_descriptions.append({
                        "type": "blip",
                        "second": second_idx,
                        "time_range": time_range,
                        "frame_number": blip_desc.get("frame_number"),
                        "frame_range": blip_desc.get("frame_range"),
                        "text": desc_text,
                        "source": "visual"
                    })
        
        # Collect audio transcriptions if requested
        if include_audio:
            transcriptions = self.get_transcriptions_for_time_range(start_time, end_time)
            for transcription in transcriptions:
                tr_text = transcription.get("transcription", "").strip()
                if tr_text:
                    tr_time_range = transcription.get("time_range", [])
                    audio_transcriptions.append({
                        "type": "audio",
                        "time_range": tr_time_range,
                        "id": transcription.get("id"),
                        "text": tr_text,
                        "source": "audio",
                        "metadata": transcription.get("metadata", {})
                    })
        
        # Combine and sort all descriptions by time
        all_descriptions = visual_descriptions + audio_transcriptions
        all_descriptions.sort(key=lambda x: (
            x.get("time_range", [0])[0] if x.get("time_range") else 0,
            x.get("second", 0),
            x.get("frame_number", 0) if x.get("frame_number") is not None else 0
        ))
        
        # Build narrative string
        narrative_parts = []
        for desc in all_descriptions:
            text = desc["text"]
            source_tag = f"[{desc['source']}]" if include_timestamps else ""
            
            if include_timestamps:
                if desc.get("second") is not None:
                    time_str = f"[{desc['second']}s{source_tag}] "
                elif desc.get("time_range"):
                    time_str = f"[{desc['time_range'][0]:.1f}s{source_tag}] "
                else:
                    time_str = f"[{source_tag}] "
                text = time_str + text
            
            narrative_parts.append(text)
        
        narrative = separator.join(narrative_parts)
        
        return {
            "narrative": narrative,
            "visual_descriptions": visual_descriptions,
            "audio_transcriptions": audio_transcriptions,
            "all_descriptions": all_descriptions,
            "time_range": [start_second, end_second],
            "seconds_covered": end_second - start_second + 1,
            "visual_count": len(visual_descriptions),
            "audio_count": len(audio_transcriptions),
            "total_count": len(all_descriptions)
        }
    
    def find_audio_mentions(self, keywords: List[str],
                           object_classes: Optional[List[str]] = None,
                           time_start: Optional[float] = None,
                           time_end: Optional[float] = None) -> Dict[str, Any]:
        """
        Find moments where audio mentions specific keywords or objects.
        Can also cross-reference with detected objects.
        
        Args:
            keywords: List of keywords to search for in audio
            object_classes: Optional list of object class names to cross-reference
            time_start: Optional start time in seconds
            time_end: Optional end time in seconds
            
        Returns:
            Dictionary with:
            - mentions: List of transcriptions containing keywords
            - cross_references: List of moments where audio mentions objects that were detected
            - summary: Summary of findings
        """
        mentions = []
        cross_references = []
        
        # Search transcriptions
        for transcription in self.transcriptions:
            tr_time_range = transcription.get("time_range", [])
            if not tr_time_range or len(tr_time_range) < 2:
                continue
            
            # Filter by time range if specified
            if time_start is not None and tr_time_range[1] < time_start:
                continue
            if time_end is not None and tr_time_range[0] > time_end:
                continue
            
            transcription_text = transcription.get("transcription", "").lower()
            if not transcription_text:
                continue
            
            # Check for keyword matches
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in transcription_text:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                mentions.append({
                    "transcription": transcription,
                    "time_range": tr_time_range,
                    "id": transcription.get("id"),
                    "matched_keywords": matched_keywords,
                    "text": transcription.get("transcription", "")
                })
        
        # Cross-reference with detected objects if requested
        if object_classes:
            for mention in mentions:
                tr_time_range = mention["time_range"]
                tr_start, tr_end = tr_time_range[0], tr_time_range[1]
                
                # Find objects detected in this time range
                detected_objects = self.find_objects_in_time_range(tr_start, tr_end)
                
                # Check if any mentioned objects were actually detected
                mentioned_objects = []
                for obj_class in object_classes:
                    if obj_class.lower() in mention["text"].lower():
                        if obj_class in detected_objects.get("objects", {}):
                            mentioned_objects.append({
                                "class": obj_class,
                                "detections": detected_objects["objects"][obj_class],
                                "count": len(detected_objects["objects"][obj_class])
                            })
                
                if mentioned_objects:
                    cross_references.append({
                        "transcription": mention["transcription"],
                        "time_range": tr_time_range,
                        "mentioned_objects": mentioned_objects,
                        "text": mention["text"]
                    })
        
        # Build summary
        summary = f"Found {len(mentions)} audio mention(s) of keywords: {', '.join(keywords)}"
        if object_classes:
            summary += f"\nCross-referenced with {len(cross_references)} object detection(s)"
        
        return {
            "mentions": mentions,
            "cross_references": cross_references,
            "keyword_count": len(mentions),
            "cross_reference_count": len(cross_references),
            "summary": summary
        }
    
    def search_all_modalities(self, keyword: str,
                             time_start: Optional[float] = None,
                             time_end: Optional[float] = None) -> Dict[str, Any]:
        """
        Search across all modalities: visual descriptions (BLIP/unified) and audio transcriptions.
        
        Args:
            keyword: Keyword to search for
            time_start: Optional start time in seconds
            time_end: Optional end time in seconds
            
        Returns:
            Dictionary with results from all modalities:
            - visual_matches: Matches in visual descriptions
            - audio_matches: Matches in audio transcriptions
            - all_matches: Combined and sorted matches
            - summary: Summary of findings
        """
        # Search visual descriptions
        visual_matches = self.search_descriptions(
            keyword, 
            search_type="any",
            time_start=time_start,
            time_end=time_end
        )
        
        # Search audio transcriptions
        audio_matches = self.search_transcriptions(
            keyword,
            time_start=time_start,
            time_end=time_end
        )
        
        # Combine and sort all matches
        all_matches = []
        
        for match in visual_matches:
            all_matches.append({
                "type": "visual",
                "second": match.get("second", 0),
                "time_range": match.get("time_range", []),
                "matches": match.get("matches", []),
                "source": "visual"
            })
        
        for match in audio_matches:
            all_matches.append({
                "type": "audio",
                "time_range": match.get("time_range", []),
                "transcription": match.get("transcription", {}),
                "text": match.get("transcription", {}).get("transcription", ""),
                "source": "audio"
            })
        
        # Sort by time
        all_matches.sort(key=lambda x: (
            x.get("time_range", [0])[0] if x.get("time_range") else 0,
            x.get("second", 0)
        ))
        
        summary = f"Found '{keyword}' in {len(visual_matches)} visual scene(s) and {len(audio_matches)} audio segment(s)"
        
        return {
            "visual_matches": visual_matches,
            "audio_matches": audio_matches,
            "all_matches": all_matches,
            "visual_count": len(visual_matches),
            "audio_count": len(audio_matches),
            "total_count": len(all_matches),
            "summary": summary
        }
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install it with: pip install sentence-transformers"
            )
        
        if self._embedding_model is None:
            # Use a fast, lightweight model
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return self._embedding_model
    
    def _load_embeddings_from_cache(self, verbose: bool = True) -> bool:
        """Try to load embeddings from cache file. Returns True if successful."""
        if not self._cache_path.exists():
            if verbose:
                print(f"[CACHE] Cache file not found: {self._cache_path}")
            return False
        
        try:
            # Check if cache is newer than JSON file
            json_mtime = os.path.getmtime(self.json_path)
            cache_mtime = os.path.getmtime(self._cache_path)
            
            if cache_mtime < json_mtime:
                # JSON file was modified after cache, cache is stale
                if verbose:
                    print(f"[CACHE] Cache file is older than JSON file (cache: {cache_mtime}, json: {json_mtime})")
                    print("[CACHE] Will recompute embeddings.")
                return False
            
            # Load from cache
            if verbose:
                print(f"[CACHE] Loading embeddings from cache: {self._cache_path}")
            cache_data = np.load(self._cache_path, allow_pickle=True)
            
            self._transcription_embeddings = cache_data.get("transcription_embeddings")
            self._unified_description_embeddings = cache_data.get("unified_description_embeddings")
            self._embedding_metadata = cache_data["embedding_metadata"].item()
            
            if verbose:
                trans_count = len(self._embedding_metadata.get("transcriptions", [])) if self._embedding_metadata else 0
                unified_count = len(self._embedding_metadata.get("unified", [])) if self._embedding_metadata else 0
                print(f"[CACHE] Embeddings loaded successfully:")
                print(f"  Transcriptions: {trans_count}")
                print(f"  Unified descriptions: {unified_count}")
            return True
        except Exception as e:
            if verbose:
                print(f"[CACHE] Error loading cache: {e}. Will recompute embeddings.")
            return False
    
    def _save_embeddings_to_cache(self, verbose: bool = True):
        """Save computed embeddings to cache file."""
        try:
            if verbose:
                print(f"[CACHE] Saving embeddings to cache: {self._cache_path}")
            np.savez(
                self._cache_path,
                transcription_embeddings=self._transcription_embeddings,
                unified_description_embeddings=self._unified_description_embeddings,
                embedding_metadata=self._embedding_metadata
            )
            if verbose:
                import os
                file_size = os.path.getsize(self._cache_path) / 1024  # KB
                print(f"[CACHE] Embeddings saved successfully ({file_size:.1f} KB)")
        except Exception as e:
            if verbose:
                print(f"[CACHE] Warning: Could not save embeddings to cache: {e}")
    
    def _compute_embeddings(self, verbose: bool = True):
        """Compute embeddings for all transcriptions and unified descriptions."""
        if self._transcription_embeddings is not None:
            if verbose:
                print("[EMBEDDINGS] Embeddings already computed, skipping.")
            return  # Already computed
        
        # Try to load from cache first
        if self._load_embeddings_from_cache(verbose=verbose):
            return
        
        model = self._get_embedding_model()
        
        # Collect all texts to embed
        transcription_texts = []
        transcription_metadata = []
        unified_texts = []
        unified_metadata = []
        
        # Process transcriptions
        for transcription in self.transcriptions:
            text = transcription.get("transcription", "").strip()
            if text:
                transcription_texts.append(text)
                transcription_metadata.append({
                    "id": transcription.get("id"),
                    "time_range": transcription.get("time_range", []),
                    "type": "transcription"
                })
        
        # Process unified descriptions
        for second_data in self.seconds:
            unified_desc = second_data.get("unified_description", "")
            if unified_desc and unified_desc.lower() != "0":
                unified_texts.append(unified_desc.strip())
                unified_metadata.append({
                    "second": second_data.get("second", 0),
                    "time_range": second_data.get("time_range", []),
                    "type": "unified",
                    "text": unified_desc.strip()  # Store text directly to avoid lookup issues
                })
        
        # Compute embeddings in batches
        if verbose:
            print(f"[EMBEDDINGS] Computing embeddings for {len(transcription_texts)} transcriptions and {len(unified_texts)} unified descriptions...")
            print(f"[EMBEDDINGS] Using model: all-MiniLM-L6-v2 (384 dimensions)")
        
        if transcription_texts:
            if verbose:
                print(f"[EMBEDDINGS] Encoding {len(transcription_texts)} transcriptions...")
            self._transcription_embeddings = model.encode(
                transcription_texts,
                show_progress_bar=verbose,
                batch_size=32,
                convert_to_numpy=True
            )
            if verbose:
                print(f"[EMBEDDINGS] Transcription embeddings shape: {self._transcription_embeddings.shape}")
        
        if unified_texts:
            if verbose:
                print(f"[EMBEDDINGS] Encoding {len(unified_texts)} unified descriptions...")
            self._unified_description_embeddings = model.encode(
                unified_texts,
                show_progress_bar=verbose,
                batch_size=32,
                convert_to_numpy=True
            )
            if verbose:
                print(f"[EMBEDDINGS] Unified description embeddings shape: {self._unified_description_embeddings.shape}")
        
        # Store metadata
        self._embedding_metadata = {
            "transcriptions": transcription_metadata,
            "unified": unified_metadata
        }
        
        if verbose:
            print("[EMBEDDINGS] Embeddings computed successfully.")
        
        # Save to cache
        self._save_embeddings_to_cache(verbose=verbose)
    
    def semantic_search(self, query: str, top_k: int = 5, 
                       threshold: float = 0.3,
                       search_transcriptions: bool = True,
                       search_unified: bool = True,
                       verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search across transcriptions and unified descriptions.
        
        Args:
            query: User query string to search for
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0-1) to include results
            search_transcriptions: Whether to search in transcriptions
            search_unified: Whether to search in unified descriptions
            
        Returns:
            List of dictionaries with:
            - time_range: [start, end] in seconds
            - score: Similarity score (0-1)
            - type: "transcription" or "unified"
            - text: The matched text snippet
            - metadata: Additional metadata (id for transcriptions, second for unified)
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install it with: pip install sentence-transformers"
            )
        
        # Compute embeddings if not already done
        self._compute_embeddings(verbose=verbose)
        
        model = self._get_embedding_model()
        
        # Embed the query
        if verbose:
            print(f"[SEMANTIC SEARCH] Embedding query: '{query}'")
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        if verbose:
            print(f"[SEMANTIC SEARCH] Query embedding shape: {query_embedding.shape}")
        
        results = []
        
        # Search transcriptions
        if search_transcriptions and self._transcription_embeddings is not None:
            if verbose:
                print(f"[SEMANTIC SEARCH] Searching {len(self._embedding_metadata['transcriptions'])} transcriptions...")
            # Compute cosine similarity
            if verbose:
                print(f"[SEMANTIC SEARCH] Computing cosine similarity for transcriptions...")
            similarities = np.dot(
                self._transcription_embeddings,
                query_embedding
            ) / (
                np.linalg.norm(self._transcription_embeddings, axis=1) *
                np.linalg.norm(query_embedding)
            )
            
            if verbose:
                above_threshold = np.sum(similarities >= threshold)
                print(f"[SEMANTIC SEARCH] Found {above_threshold} transcription matches above threshold {threshold}")
            
            # Get top matches
            for idx, similarity in enumerate(similarities):
                if similarity >= threshold:
                    metadata = self._embedding_metadata["transcriptions"][idx]
                    transcription = self._transcription_map.get(metadata["id"])
                    if transcription:
                        results.append({
                            "time_range": metadata["time_range"],
                            "score": float(similarity),
                            "type": "transcription",
                            "text": transcription.get("transcription", ""),
                            "metadata": {
                                "id": metadata["id"],
                                "transcription_id": metadata["id"]
                            }
                        })
        
        # Search unified descriptions
        if search_unified and self._unified_description_embeddings is not None:
            if verbose:
                print(f"[SEMANTIC SEARCH] Searching {len(self._embedding_metadata['unified'])} unified descriptions...")
            # Compute cosine similarity
            if verbose:
                print(f"[SEMANTIC SEARCH] Computing cosine similarity for unified descriptions...")
            similarities = np.dot(
                self._unified_description_embeddings,
                query_embedding
            ) / (
                np.linalg.norm(self._unified_description_embeddings, axis=1) *
                np.linalg.norm(query_embedding)
            )
            
            if verbose:
                above_threshold = np.sum(similarities >= threshold)
                print(f"[SEMANTIC SEARCH] Found {above_threshold} unified description matches above threshold {threshold}")
            
            # Get top matches
            for idx, similarity in enumerate(similarities):
                if similarity >= threshold:
                    metadata = self._embedding_metadata["unified"][idx]
                    # Get text from metadata (stored during embedding creation) or fallback to lookup
                    unified_text = metadata.get("text", "")
                    if not unified_text:
                        # Fallback: try to get from seconds list
                        second_num = metadata.get("second", -1)
                        if 0 <= second_num < len(self.seconds):
                            unified_text = self.seconds[second_num].get("unified_description", "")
                    
                    results.append({
                        "time_range": metadata["time_range"],
                        "score": float(similarity),
                        "type": "unified",
                        "text": unified_text,
                        "metadata": {
                            "second": metadata["second"]
                        }
                    })
        
        # Sort by score (descending) and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        
        if verbose:
            print(f"[SEMANTIC SEARCH] Total matches found: {len(results)}")
            print(f"[SEMANTIC SEARCH] Returning top {min(top_k, len(results))} results")
            if results:
                print(f"[SEMANTIC SEARCH] Score range: {results[-1]['score']:.3f} - {results[0]['score']:.3f}")
        
        return results[:top_k]
    
    def get_transcription_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the audio transcriptions.
        
        Returns:
            Dictionary with:
            - total_transcriptions: Total number of transcriptions
            - total_words: Total word count
            - total_characters: Total character count
            - speaking_time: Total time with non-empty transcriptions (seconds)
            - average_words_per_transcription: Average words per transcription
            - transcription_coverage: Percentage of video with transcriptions
            - language: Detected language (if available)
        """
        if not self.transcriptions:
            return {
                "total_transcriptions": 0,
                "total_words": 0,
                "total_characters": 0,
                "speaking_time": 0.0,
                "average_words_per_transcription": 0.0,
                "transcription_coverage": 0.0,
                "language": None
            }
        
        total_words = 0
        total_characters = 0
        speaking_time = 0.0
        languages = set()
        
        for transcription in self.transcriptions:
            text = transcription.get("transcription", "").strip()
            if text:
                words = text.split()
                total_words += len(words)
                total_characters += len(text)
                
                # Calculate speaking time from time_range
                tr_time_range = transcription.get("time_range", [])
                if tr_time_range and len(tr_time_range) >= 2:
                    duration = tr_time_range[1] - tr_time_range[0]
                    speaking_time += duration
            
            # Collect language info
            metadata = transcription.get("metadata", {})
            lang = metadata.get("language")
            if lang:
                languages.add(lang)
        
        # Get video duration
        video_info = self.get_video_info()
        video_duration = video_info.get("duration_seconds", 0)
        
        # Calculate coverage
        transcription_coverage = (speaking_time / video_duration * 100) if video_duration > 0 else 0.0
        
        # Calculate average
        non_empty_count = sum(1 for t in self.transcriptions if t.get("transcription", "").strip())
        average_words = total_words / non_empty_count if non_empty_count > 0 else 0.0
        
        return {
            "total_transcriptions": len(self.transcriptions),
            "total_words": total_words,
            "total_characters": total_characters,
            "speaking_time": round(speaking_time, 2),
            "average_words_per_transcription": round(average_words, 2),
            "transcription_coverage": round(transcription_coverage, 2),
            "language": list(languages) if languages else None,
            "non_empty_transcriptions": non_empty_count,
            "video_duration": video_duration
        }
    
    def hierarchical_keyword_search(self, keywords: List[str], 
                                    match_mode: str = "any",
                                    max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fast keyword search using hierarchical tree index.
        
        Args:
            keywords: List of keywords to search for
            match_mode: "any" (OR) or "all" (AND) - whether to match any or all keywords
            max_results: Maximum number of results to return (None = all)
            
        Returns:
            List of nodes that contain the keywords, sorted by relevance
        """
        if not self.hierarchical_tree or not self._hierarchical_indexes:
            return []
        
        by_keyword = self._hierarchical_indexes.get("by_keyword", {})
        keywords_lower = [kw.lower() for kw in keywords]
        
        # Find nodes containing keywords
        if match_mode == "all":
            # AND: nodes must contain ALL keywords
            matching_nodes = None
            for keyword in keywords_lower:
                nodes_with_keyword = set(by_keyword.get(keyword, []))
                if matching_nodes is None:
                    matching_nodes = nodes_with_keyword
                else:
                    matching_nodes = matching_nodes.intersection(nodes_with_keyword)
            
            if matching_nodes is None:
                matching_nodes = set()
        else:
            # OR: nodes containing ANY keyword
            matching_nodes = set()
            for keyword in keywords_lower:
                matching_nodes.update(by_keyword.get(keyword, []))
        
        # Get node details and score by keyword count
        results = []
        for node_id in matching_nodes:
            node = self._hierarchical_nodes.get(node_id)
            if not node:
                continue
            
            # Count how many keywords this node contains
            node_keywords = set(kw.lower() for kw in node.get("keywords", []))
            matched_keywords = [kw for kw in keywords_lower if kw in node_keywords]
            match_count = len(matched_keywords)
            
            results.append({
                "node_id": node_id,
                "time_range": node.get("time_range", []),
                "duration": node.get("duration", 0),
                "level": node.get("level", -1),
                "keyword_count": node.get("keyword_count", 0),
                "matched_keywords": matched_keywords,
                "match_count": match_count,
                "keywords": node.get("keywords", []),
                "node": node
            })
        
        # Sort by match count (more keywords = better), then by level (leaves first)
        results.sort(key=lambda x: (-x["match_count"], x["level"]))
        
        if max_results:
            results = results[:max_results]
        
        return results
    
    def hierarchical_get_leaf_nodes(self, time_start: Optional[float] = None,
                                   time_end: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get all leaf nodes, optionally filtered by time range.
        
        Args:
            time_start: Optional start time filter
            time_end: Optional end time filter
            
        Returns:
            List of leaf nodes
        """
        if not self.hierarchical_tree or not self._hierarchical_nodes:
            return []
        
        leaves = []
        for node_id, node in self._hierarchical_nodes.items():
            if node.get("level", -1) != 0:  # Only leaf nodes (level 0)
                continue
            
            time_range = node.get("time_range", [])
            if not time_range or len(time_range) < 2:
                continue
            
            # Filter by time range if specified
            if time_start is not None and time_range[1] < time_start:
                continue
            if time_end is not None and time_range[0] > time_end:
                continue
            
            leaves.append({
                "node_id": node_id,
                "time_range": time_range,
                "duration": node.get("duration", 0),
                "keywords": node.get("keywords", []),
                "keyword_count": node.get("keyword_count", 0),
                "visual_text": node.get("visual_text", ""),
                "audio_text": node.get("audio_text", ""),
                "combined_text": node.get("combined_text", ""),
                "node": node
            })
        
        # Sort by time
        leaves.sort(key=lambda x: x["time_range"][0])
        return leaves
    
    def hierarchical_score_leaves_for_highlights(self,
                                                action_keywords: Optional[List[str]] = None,
                                                max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Score all leaf nodes to find highlights based on keyword density and activity.
        
        Args:
            action_keywords: Optional list of action keywords to boost (e.g., ["fishing", "cooking", "laughing"])
            max_results: Maximum number of results to return
            
        Returns:
            List of leaf nodes scored and ranked for highlight potential
        """
        leaves = self.hierarchical_get_leaf_nodes()
        
        if action_keywords is None:
            # Default action keywords
            action_keywords = ["fishing", "cooking", "catching", "laughing", "cheering", 
                             "running", "jumping", "excited", "surprised", "achievement"]
        
        action_keywords_lower = [kw.lower() for kw in action_keywords]
        
        # Score each leaf
        scored_leaves = []
        for leaf in leaves:
            keywords = [kw.lower() for kw in leaf.get("keywords", [])]
            keyword_count = leaf.get("keyword_count", 0)
            
            # Score components
            base_score = keyword_count  # More keywords = more activity
            
            # Boost for action keywords
            action_matches = sum(1 for kw in keywords if kw in action_keywords_lower)
            action_score = action_matches * 2  # Action keywords worth 2x
            
            # Boost for unique/rare keywords (if keyword count is high but not too high)
            uniqueness_score = 0
            if 20 <= keyword_count <= 80:  # Sweet spot for interesting moments
                uniqueness_score = 10
            
            total_score = base_score + action_score + uniqueness_score
            
            scored_leaves.append({
                **leaf,
                "score": total_score,
                "base_score": base_score,
                "action_score": action_score,
                "uniqueness_score": uniqueness_score,
                "action_matches": action_matches
            })
        
        # Sort by score (descending)
        scored_leaves.sort(key=lambda x: x["score"], reverse=True)
        
        if max_results:
            scored_leaves = scored_leaves[:max_results]
        
        return scored_leaves
    
    def inspect_content(self, max_keywords: int = 100, max_sample_descriptions: int = 20) -> Dict[str, Any]:
        """
        Inspect and summarize the content available in the video.
        Returns keywords, objects, and sample descriptions to help understand what's in the video.
        
        Args:
            max_keywords: Maximum number of unique keywords to return
            max_sample_descriptions: Maximum number of sample descriptions to return
            
        Returns:
            Dictionary with:
            - all_keywords: List of unique keywords found in the video
            - object_classes: Dictionary of object classes and their counts
            - sample_descriptions: Sample unified descriptions from the video
            - hierarchical_available: Whether hierarchical tree is available
            - total_seconds: Total video duration in seconds
            - summary: Text summary of available content
        """
        all_keywords = set()
        object_classes = self.get_all_classes()
        sample_descriptions = []
        
        # Collect keywords from hierarchical tree if available
        if self.hierarchical_tree and self._hierarchical_indexes:
            by_keyword = self._hierarchical_indexes.get("by_keyword", {})
            all_keywords.update(by_keyword.keys())
        
        # Also collect keywords from descriptions (fallback if no hierarchical tree)
        if not all_keywords:
            # Extract keywords from unified descriptions
            for second_data in self.seconds[:max_sample_descriptions * 2]:  # Check more to get good samples
                unified_desc = second_data.get("unified_description", "")
                if unified_desc and unified_desc.lower() != "0":
                    # Simple keyword extraction (words of 3+ chars)
                    words = unified_desc.lower().split()
                    keywords = [w.strip('.,!?;:()[]{}') for w in words if len(w.strip('.,!?;:()[]{}')) >= 3]
                    all_keywords.update(keywords)
        
        # Collect sample descriptions
        descriptions_collected = 0
        for second_data in self.seconds:
            # Skip None values that might be in the seconds list
            if second_data is None:
                continue
            if descriptions_collected >= max_sample_descriptions:
                break
            unified_desc = second_data.get("unified_description", "")
            if unified_desc and unified_desc.lower() != "0":
                sample_descriptions.append({
                    "second": second_data.get("second", 0),
                    "time_range": second_data.get("time_range", []),
                    "description": unified_desc
                })
                descriptions_collected += 1
        
        # Sort keywords (most common first if we have frequency data, otherwise alphabetically)
        all_keywords_list = sorted(list(all_keywords))[:max_keywords]
        
        # Build summary
        total_seconds = len(self.seconds)
        summary_parts = [
            f"Video duration: {total_seconds} seconds",
            f"Object classes detected: {len(object_classes)} ({', '.join(sorted(object_classes.keys())[:10])}{'...' if len(object_classes) > 10 else ''})",
            f"Unique keywords: {len(all_keywords_list)}",
        ]
        if self.hierarchical_tree:
            summary_parts.append("Hierarchical tree: Available")
        else:
            summary_parts.append("Hierarchical tree: Not available")
        
        summary = "\n".join(summary_parts)
        
        return {
            "all_keywords": all_keywords_list,
            "object_classes": object_classes,
            "sample_descriptions": sample_descriptions,
            "hierarchical_available": self.hierarchical_tree is not None,
            "total_seconds": total_seconds,
            "summary": summary,
            "keyword_count": len(all_keywords_list),
            "object_class_count": len(object_classes),
            "sample_count": len(sample_descriptions)
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


def main():
    """Command-line interface for segment tree utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query and analyze segment tree JSON files"
    )
    parser.add_argument(
        "json_file",
        help="Path to the segment tree JSON file"
    )
    parser.add_argument(
        "--check-fish",
        action="store_true",
        help="Check if fish are caught in the video"
    )
    parser.add_argument(
        "--check-activity",
        nargs="+",
        metavar="KEYWORD",
        help="Check for any activity using keywords (e.g., --check-activity boat sailing water)"
    )
    parser.add_argument(
        "--activity-name",
        default="activity",
        help="Name of the activity for --check-activity (default: 'activity')"
    )
    parser.add_argument(
        "--evidence-keywords",
        nargs="+",
        metavar="KEYWORD",
        help="Evidence keywords for --check-activity (stronger indicators)"
    )
    parser.add_argument(
        "--narrative",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Get narrative description for a time range (seconds)"
    )
    parser.add_argument(
        "--find-class",
        metavar="CLASS_NAME",
        help="Find all occurrences of an object class (e.g., 'boat', 'person')"
    )
    parser.add_argument(
        "--search",
        metavar="KEYWORD",
        help="Search for keyword in descriptions"
    )
    parser.add_argument(
        "--summary",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Get scene summary for a time range (seconds)"
    )
    parser.add_argument(
        "--search-audio",
        metavar="KEYWORD",
        help="Search for keyword in audio transcriptions"
    )
    parser.add_argument(
        "--combined-narrative",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Get combined narrative with visual and audio (seconds)"
    )
    parser.add_argument(
        "--search-all",
        metavar="KEYWORD",
        help="Search across all modalities (visual and audio)"
    )
    parser.add_argument(
        "--audio-mentions",
        nargs="+",
        metavar="KEYWORD",
        help="Find audio mentions of keywords (e.g., --audio-mentions boat water)"
    )
    parser.add_argument(
        "--cross-reference",
        nargs="+",
        metavar="CLASS",
        help="Cross-reference audio mentions with object classes (use with --audio-mentions)"
    )
    parser.add_argument(
        "--transcription-stats",
        action="store_true",
        help="Get statistics about audio transcriptions"
    )
    parser.add_argument(
        "--transcriptions",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Get transcriptions for a time range (seconds)"
    )
    
    args = parser.parse_args()
    
    # Load the segment tree
    try:
        query = load_segment_tree(args.json_file)
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        return
    
    # Execute requested command
    if args.check_fish:
        result = query.check_fish_caught()
        print("\n" + "=" * 60)
        print("FISH CATCHING ANALYSIS")
        print("=" * 60)
        print(f"\nResult: {result['summary']}")
        print(f"\nFish Caught: {'YES' if result['fish_caught'] else 'NO'}")
        print(f"Fishing Scenes Found: {result['fishing_scene_count']}")
        print(f"Fish Holding Scenes: {result['fish_holding_count']}")
        
        if result['evidence']:
            print(f"\nEvidence ({len(result['evidence'])}):")
            for i, ev in enumerate(result['evidence'], 1):
                print(f"  {i}. Second {ev['second']} ({ev['time_range']}):")
                print(f"     {ev['description'][:100]}...")
        
        if result['fishing_scenes'] and not result['fish_caught']:
            print(f"\nFishing Activity Detected ({len(result['fishing_scenes'])} scenes):")
            for i, scene in enumerate(result['fishing_scenes'][:5], 1):
                print(f"  {i}. Second {scene['second']}: {scene['description'][:80]}...")
    
    elif args.check_activity:
        result = query.check_activity(
            activity_keywords=args.check_activity,
            evidence_keywords=args.evidence_keywords,
            activity_name=args.activity_name,
            evidence_name=args.activity_name
        )
        print("\n" + "=" * 60)
        print(f"ACTIVITY ANALYSIS: {args.activity_name.upper()}")
        print("=" * 60)
        print(f"\nResult: {result['summary']}")
        print(f"\nDetected: {'YES' if result['detected'] else 'NO'}")
        print(f"Activity Scenes Found: {result['activity_count']}")
        print(f"Evidence Scenes: {result['evidence_count']}")
        print(f"\nKeywords used: {', '.join(args.check_activity)}")
        if args.evidence_keywords:
            print(f"Evidence keywords: {', '.join(args.evidence_keywords)}")
        
        if result['evidence']:
            print(f"\nEvidence ({len(result['evidence'])}):")
            for i, ev in enumerate(result['evidence'], 1):
                print(f"  {i}. Second {ev['second']} ({ev['time_range']}):")
                print(f"     {ev['description'][:100]}...")
        
        if result['activity_scenes'] and not result['detected']:
            print(f"\nActivity Detected ({len(result['activity_scenes'])} scenes):")
            for i, scene in enumerate(result['activity_scenes'][:5], 1):
                print(f"  {i}. Second {scene['second']}: {scene['description'][:80]}...")
    
    elif args.narrative:
        start, end = args.narrative
        result = query.get_narrative_description(start, end, include_timestamps=True)
        print("\n" + "=" * 60)
        print(f"NARRATIVE DESCRIPTION (Seconds {start}-{end})")
        print("=" * 60)
        print(f"\nDescriptions found: {result['description_count']}")
        print(f"Seconds covered: {result['seconds_covered']}")
        print(f"\nNarrative:\n{result['narrative']}")
    
    elif args.find_class:
        results = query.find_objects_by_class(args.find_class)
        print("\n" + "=" * 60)
        print(f"FIND OBJECTS: {args.find_class.upper()}")
        print("=" * 60)
        print(f"\nFound {len(results)} detections")
        if results:
            print(f"\nFirst 10 occurrences:")
            for i, obj in enumerate(results[:10], 1):
                print(f"  {i}. Second {obj['second']}, Track {obj['detection']['track_id']}, "
                      f"Confidence: {obj['detection']['confidence']:.2f}")
    
    elif args.search:
        results = query.search_descriptions(args.search)
        print("\n" + "=" * 60)
        print(f"SEARCH RESULTS: '{args.search}'")
        print("=" * 60)
        print(f"\nFound in {len(results)} seconds")
        for i, result in enumerate(results[:10], 1):
            print(f"\n  {i}. Second {result['second']} ({result['time_range']}):")
            for match in result['matches']:
                print(f"     [{match['type']}] {match['description'][:80]}...")
    
    elif args.summary:
        start, end = args.summary
        result = query.get_scene_summary(time_start=start, time_end=end)
        print("\n" + "=" * 60)
        print(f"SCENE SUMMARY (Seconds {start}-{end})")
        print("=" * 60)
        print(f"\nSeconds covered: {result['seconds_count']}")
        print(f"Total detections: {result['total_detections']}")
        print(f"\nObjects detected:")
        for class_name, count in sorted(result['objects'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {class_name}: {count}")
        if result['descriptions']:
            print(f"\nDescriptions ({len(result['descriptions'])}):")
            for desc in result['descriptions'][:5]:
                print(f"  [{desc['type']}] Second {desc['second']}: {desc['text'][:70]}...")
    
    elif args.search_audio:
        results = query.search_transcriptions(args.search_audio)
        print("\n" + "=" * 60)
        print(f"AUDIO SEARCH RESULTS: '{args.search_audio}'")
        print("=" * 60)
        print(f"\nFound in {len(results)} transcription(s)")
        for i, result in enumerate(results[:10], 1):
            tr = result['transcription']
            print(f"\n  {i}. Time {result['time_range']} (ID: {result['id']}):")
            print(f"     {tr.get('transcription', '')[:100]}...")
    
    elif args.combined_narrative:
        start, end = args.combined_narrative
        result = query.get_combined_narrative(start, end, include_timestamps=True)
        print("\n" + "=" * 60)
        print(f"COMBINED NARRATIVE (Seconds {start}-{end})")
        print("=" * 60)
        print(f"\nVisual descriptions: {result['visual_count']}")
        print(f"Audio transcriptions: {result['audio_count']}")
        print(f"Total: {result['total_count']}")
        print(f"\nNarrative:\n{result['narrative']}")
    
    elif args.search_all:
        result = query.search_all_modalities(args.search_all)
        print("\n" + "=" * 60)
        print(f"MULTI-MODAL SEARCH: '{args.search_all}'")
        print("=" * 60)
        print(f"\n{result['summary']}")
        print(f"\nVisual matches: {result['visual_count']}")
        print(f"Audio matches: {result['audio_count']}")
        print(f"Total matches: {result['total_count']}")
        
        if result['all_matches']:
            print(f"\nFirst 10 matches:")
            for i, match in enumerate(result['all_matches'][:10], 1):
                time_str = f"Second {match.get('second', 'N/A')}" if match.get('second') is not None else f"Time {match.get('time_range', [])}"
                source = match.get('source', 'unknown')
                text = match.get('text', match.get('transcription', {}).get('transcription', 'N/A'))
                print(f"  {i}. [{source}] {time_str}: {text[:80]}...")
    
    elif args.audio_mentions:
        result = query.find_audio_mentions(
            keywords=args.audio_mentions,
            object_classes=args.cross_reference
        )
        print("\n" + "=" * 60)
        print(f"AUDIO MENTIONS: {', '.join(args.audio_mentions)}")
        print("=" * 60)
        print(f"\n{result['summary']}")
        print(f"\nMentions found: {result['keyword_count']}")
        if args.cross_reference:
            print(f"Cross-references: {result['cross_reference_count']}")
        
        if result['mentions']:
            print(f"\nMentions ({len(result['mentions'])}):")
            for i, mention in enumerate(result['mentions'][:10], 1):
                print(f"  {i}. Time {mention['time_range']} (Keywords: {', '.join(mention['matched_keywords'])}):")
                print(f"     {mention['text'][:100]}...")
        
        if result['cross_references']:
            print(f"\nCross-references ({len(result['cross_references'])}):")
            for i, ref in enumerate(result['cross_references'][:5], 1):
                print(f"  {i}. Time {ref['time_range']}:")
                print(f"     Audio: {ref['text'][:80]}...")
                for obj in ref['mentioned_objects']:
                    print(f"     Detected: {obj['class']} ({obj['count']} detection(s))")
    
    elif args.transcription_stats:
        stats = query.get_transcription_statistics()
        print("\n" + "=" * 60)
        print("TRANSCRIPTION STATISTICS")
        print("=" * 60)
        print(f"\nTotal transcriptions: {stats['total_transcriptions']}")
        print(f"Non-empty transcriptions: {stats['non_empty_transcriptions']}")
        print(f"Total words: {stats['total_words']}")
        print(f"Total characters: {stats['total_characters']}")
        print(f"Speaking time: {stats['speaking_time']} seconds")
        print(f"Video duration: {stats['video_duration']} seconds")
        print(f"Transcription coverage: {stats['transcription_coverage']}%")
        print(f"Average words per transcription: {stats['average_words_per_transcription']}")
        if stats['language']:
            print(f"Language(s): {', '.join(stats['language'])}")
    
    elif args.transcriptions:
        start, end = args.transcriptions
        transcriptions = query.get_transcriptions_for_time_range(start, end)
        print("\n" + "=" * 60)
        print(f"TRANSCRIPTIONS (Seconds {start}-{end})")
        print("=" * 60)
        print(f"\nFound {len(transcriptions)} transcription(s)")
        for i, tr in enumerate(transcriptions, 1):
            print(f"\n  {i}. Time {tr.get('time_range', [])} (ID: {tr.get('id', 'N/A')}):")
            text = tr.get('transcription', '').strip()
            if text:
                print(f"     {text}")
            else:
                print(f"     (empty)")
            metadata = tr.get('metadata', {})
            if metadata:
                print(f"     Model: {metadata.get('model', 'N/A')}, Language: {metadata.get('language', 'N/A')}")
    
    else:
        # Default: show video info
        info = query.get_video_info()
        print("\n" + "=" * 60)
        print("VIDEO INFORMATION")
        print("=" * 60)
        print(f"Video: {info['video']}")
        print(f"Duration: {info['duration_seconds']} seconds")
        print(f"Total Frames: {info['total_frames']}")
        print(f"FPS: {info['fps']}")
        print(f"Tracker: {info['tracker']}")
        
        all_classes = query.get_all_classes()
        print(f"\nObject Classes ({len(all_classes)}):")
        for class_name, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {class_name}: {count}")
        
        print("\nUse --help to see available commands.")


if __name__ == "__main__":
    main()

