"""
Generate an enhanced version of segment tree JSON with indexes for faster AI agent queries.
Adds reverse indexes and precomputed statistics without modifying the original structure.
"""

import json
from typing import Dict, List, Any, Set
from collections import defaultdict


def generate_indexed_segment_tree(input_json_path: str, output_json_path: str):
    """
    Generate an enhanced segment tree JSON with indexes.
    
    The enhanced structure includes:
    - Class index: Maps class names to all occurrences with time/frame info
    - Track index: Maps track IDs to their complete timelines
    - Description index: Keyword-based index for text search
    - Statistics: Aggregate counts and summaries
    
    Args:
        input_json_path: Path to the original segment tree JSON
        output_json_path: Path to save the enhanced JSON
    """
    print(f"Loading segment tree from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    seconds = data.get("seconds", [])
    print(f"Processing {len(seconds)} seconds...")
    
    # Initialize indexes
    class_index = defaultdict(list)  # class_name -> list of occurrences
    track_index = defaultdict(list)   # track_id -> list of occurrences
    description_keywords = defaultdict(list)  # keyword -> list of seconds
    all_classes = set()
    all_tracks = set()
    class_counts = defaultdict(int)
    
    # Process each second
    for second_data in seconds:
        second_idx = second_data.get("second", 0)
        time_range = second_data.get("time_range", [])
        frame_range = second_data.get("frame_range", [])
        
        # Index detections by class
        for group in second_data.get("detection_groups", []):
            group_frame_range = group.get("frame_range", [])
            
            for detection in group.get("detections", []):
                class_name = detection.get("class_name")
                track_id = detection.get("track_id")
                
                if class_name:
                    all_classes.add(class_name)
                    class_counts[class_name] += 1
                    
                    class_index[class_name].append({
                        "second": second_idx,
                        "time_range": time_range,
                        "frame_range": group_frame_range,
                        "track_id": track_id,
                        "confidence": detection.get("confidence"),
                        "bbox": detection.get("bbox")
                    })
                
                if track_id is not None:
                    all_tracks.add(track_id)
                    
                    track_index[track_id].append({
                        "second": second_idx,
                        "time_range": time_range,
                        "frame_range": group_frame_range,
                        "class_name": class_name,
                        "confidence": detection.get("confidence"),
                        "bbox": detection.get("bbox")
                    })
        
        # Index descriptions by keywords
        unified_desc = second_data.get("unified_description", "")
        if unified_desc and unified_desc.lower() != "0":
            keywords = extract_keywords(unified_desc)
            for keyword in keywords:
                description_keywords[keyword].append({
                    "second": second_idx,
                    "time_range": time_range,
                    "type": "unified",
                    "description": unified_desc
                })
        
        for blip_desc in second_data.get("blip_descriptions", []):
            desc_text = blip_desc.get("description", "")
            if desc_text:
                keywords = extract_keywords(desc_text)
                for keyword in keywords:
                    description_keywords[keyword].append({
                        "second": second_idx,
                        "time_range": time_range,
                        "type": "blip",
                        "description": desc_text,
                        "frame_number": blip_desc.get("frame_number"),
                        "frame_range": blip_desc.get("frame_range")
                    })
    
    # Sort track timelines by frame
    for track_id in track_index:
        track_index[track_id].sort(key=lambda x: x["frame_range"][0] if x["frame_range"] else 0)
    
    # Sort class occurrences by time
    for class_name in class_index:
        class_index[class_name].sort(key=lambda x: x["time_range"][0])
    
    # Build statistics
    stats = {
        "total_seconds": len(seconds),
        "total_frames": seconds[-1]["frame_range"][1] if seconds else 0,
        "unique_classes": len(all_classes),
        "unique_tracks": len(all_tracks),
        "class_counts": dict(class_counts),
        "most_common_classes": sorted(
            class_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Top 10
    }
    
    # Build enhanced structure
    enhanced_data = {
        # Original data (unchanged)
        "video": data.get("video", ""),
        "fps": data.get("fps", 30),
        "tracker": data.get("tracker", ""),
        "seconds": seconds,
        
        # New index layer
        "indexes": {
            "class_index": dict(class_index),
            "track_index": dict(track_index),
            "description_keywords": dict(description_keywords)
        },
        
        # Statistics
        "statistics": stats
    }
    
    # Save enhanced JSON
    print(f"Saving enhanced segment tree to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"Enhanced segment tree generated!")
    print(f"  - {len(all_classes)} unique object classes")
    print(f"  - {len(all_tracks)} unique tracks")
    print(f"  - {len(description_keywords)} unique keywords indexed")
    print(f"  - Top class: {stats['most_common_classes'][0][0] if stats['most_common_classes'] else 'N/A'}")


def extract_keywords(text: str, min_length: int = 3) -> Set[str]:
    """
    Extract keywords from text for indexing.
    
    Args:
        text: Text to extract keywords from
        min_length: Minimum keyword length
        
    Returns:
        Set of lowercase keywords
    """
    # Simple keyword extraction - split on whitespace and punctuation
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out common stop words and short words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    keywords = {w for w in words if len(w) >= min_length and w not in stop_words}
    return keywords


class IndexedSegmentTreeQuery:
    """Query interface for indexed segment tree JSON files (enhanced version)."""
    
    def __init__(self, json_path: str):
        """Load and initialize the indexed segment tree from JSON file."""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.video = self.data.get("video", "")
        self.fps = self.data.get("fps", 30)
        self.tracker = self.data.get("tracker", "")
        self.seconds = self.data.get("seconds", [])
        self.indexes = self.data.get("indexes", {})
        self.statistics = self.data.get("statistics", {})
    
    def get_statistics(self) -> Dict:
        """Get precomputed statistics."""
        return self.statistics
    
    def find_objects_by_class_fast(self, class_name: str) -> List[Dict]:
        """
        Fast lookup of objects by class using the index.
        
        Args:
            class_name: Name of the class to search for
            
        Returns:
            List of occurrences with time/frame info
        """
        class_index = self.indexes.get("class_index", {})
        return class_index.get(class_name, [])
    
    def find_track_timeline_fast(self, track_id: int) -> List[Dict]:
        """
        Fast lookup of track timeline using the index.
        
        Args:
            track_id: The track ID to follow
            
        Returns:
            Complete timeline for this track
        """
        track_index = self.indexes.get("track_index", {})
        return track_index.get(track_id, [])
    
    def search_descriptions_fast(self, keyword: str) -> List[Dict]:
        """
        Fast keyword search in descriptions using the index.
        
        Args:
            keyword: Keyword to search for (case-insensitive)
            
        Returns:
            List of matching seconds with descriptions
        """
        description_keywords = self.indexes.get("description_keywords", {})
        keyword_lower = keyword.lower()
        return description_keywords.get(keyword_lower, [])
    
    def get_all_classes_fast(self) -> Dict[str, int]:
        """Get all classes and their counts from statistics."""
        return self.statistics.get("class_counts", {})
    
    def get_most_common_classes(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most common object classes."""
        most_common = self.statistics.get("most_common_classes", [])
        return most_common[:top_n]


def main():
    """Command-line interface for generating indexed segment tree."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate enhanced segment tree JSON with indexes"
    )
    parser.add_argument(
        "--input",
        default="camp_segment_tree.json",
        help="Input segment tree JSON file"
    )
    parser.add_argument(
        "--output",
        default="camp_segment_tree_indexed.json",
        help="Output indexed segment tree JSON file"
    )
    
    args = parser.parse_args()
    
    generate_indexed_segment_tree(args.input, args.output)


if __name__ == "__main__":
    main()

