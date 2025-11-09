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

