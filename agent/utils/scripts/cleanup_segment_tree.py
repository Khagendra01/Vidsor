"""
Script to clean up segment_tree.json by removing unused fields.

Removes:
- bakllava_descriptions and bakllava_metadata (unified_description is sufficient)
- class_id from detections (only class_name is used)
- representative_frame, unique_tracks, and total_detections from detection_groups
  (can be computed on-the-fly)
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def cleanup_segment_tree(input_path: str, output_path: str = None, create_backup: bool = True) -> Dict[str, Any]:
    """
    Clean up segment_tree.json by removing unused fields.
    
    Args:
        input_path: Path to input segment_tree.json
        output_path: Path to output file (if None, overwrites input)
        create_backup: Whether to create a backup before cleaning
        
    Returns:
        Dictionary with statistics about what was removed
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create backup if requested
    if create_backup:
        backup_path = input_path.parent / f"{input_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print(f"Creating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)
        print(f"  ✓ Backup created: {backup_path}")
    
    # Load the segment tree
    print(f"\nLoading segment tree from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_size = os.path.getsize(input_path)
    print(f"  Original file size: {original_size / (1024*1024):.2f} MB")
    
    # Statistics
    stats = {
        "seconds_processed": 0,
        "bakllava_descriptions_removed": 0,
        "bakllava_metadata_removed": 0,
        "class_id_removed": 0,
        "representative_frame_removed": 0,
        "unique_tracks_removed": 0,
        "total_detections_removed": 0,
        "detection_groups_processed": 0,
        "detections_processed": 0
    }
    
    # Process each second
    seconds = data.get("seconds", [])
    print(f"\nProcessing {len(seconds)} seconds...")
    
    for second_data in seconds:
        if second_data is None:
            continue
        
        stats["seconds_processed"] += 1
        
        # Remove bakllava_descriptions
        if "bakllava_descriptions" in second_data:
            bakllava_descriptions = second_data.pop("bakllava_descriptions")
            if bakllava_descriptions:
                stats["bakllava_descriptions_removed"] += len(bakllava_descriptions)
        
        # Remove bakllava_metadata
        if "bakllava_metadata" in second_data:
            second_data.pop("bakllava_metadata")
            stats["bakllava_metadata_removed"] += 1
        
        # Process detection_groups
        detection_groups = second_data.get("detection_groups", [])
        for group in detection_groups:
            if group is None:
                continue
            
            stats["detection_groups_processed"] += 1
            
            # Remove representative_frame
            if "representative_frame" in group:
                group.pop("representative_frame")
                stats["representative_frame_removed"] += 1
            
            # Remove unique_tracks
            if "unique_tracks" in group:
                group.pop("unique_tracks")
                stats["unique_tracks_removed"] += 1
            
            # Remove total_detections
            if "total_detections" in group:
                group.pop("total_detections")
                stats["total_detections_removed"] += 1
            
            # Process detections
            detections = group.get("detections", [])
            for detection in detections:
                if detection is None:
                    continue
                
                stats["detections_processed"] += 1
                
                # Remove class_id
                if "class_id" in detection:
                    detection.pop("class_id")
                    stats["class_id_removed"] += 1
    
    # Determine output path
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    # Save cleaned data
    print(f"\nSaving cleaned segment tree to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    new_size = os.path.getsize(output_path)
    size_reduction = original_size - new_size
    size_reduction_percent = (size_reduction / original_size * 100) if original_size > 0 else 0
    
    stats["original_size_mb"] = original_size / (1024 * 1024)
    stats["new_size_mb"] = new_size / (1024 * 1024)
    stats["size_reduction_mb"] = size_reduction / (1024 * 1024)
    stats["size_reduction_percent"] = size_reduction_percent
    
    print(f"  ✓ Cleaned file size: {new_size / (1024*1024):.2f} MB")
    print(f"  ✓ Size reduction: {size_reduction / (1024*1024):.2f} MB ({size_reduction_percent:.1f}%)")
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print cleanup statistics."""
    print("\n" + "=" * 60)
    print("CLEANUP STATISTICS")
    print("=" * 60)
    print(f"\nSeconds processed: {stats['seconds_processed']}")
    print(f"\nFields removed:")
    print(f"  - bakllava_descriptions: {stats['bakllava_descriptions_removed']} entries")
    print(f"  - bakllava_metadata: {stats['bakllava_metadata_removed']} entries")
    print(f"  - class_id: {stats['class_id_removed']} detections")
    print(f"  - representative_frame: {stats['representative_frame_removed']} groups")
    print(f"  - unique_tracks: {stats['unique_tracks_removed']} groups")
    print(f"  - total_detections: {stats['total_detections_removed']} groups")
    print(f"\nDetection groups processed: {stats['detection_groups_processed']}")
    print(f"Detections processed: {stats['detections_processed']}")
    print(f"\nFile size:")
    print(f"  Original: {stats['original_size_mb']:.2f} MB")
    print(f"  Cleaned: {stats['new_size_mb']:.2f} MB")
    print(f"  Reduction: {stats['size_reduction_mb']:.2f} MB ({stats['size_reduction_percent']:.1f}%)")
    print("=" * 60)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean up segment_tree.json by removing unused fields"
    )
    parser.add_argument(
        "input_file",
        help="Path to input segment_tree.json file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output file (default: overwrites input)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup before cleaning"
    )
    
    args = parser.parse_args()
    
    try:
        stats = cleanup_segment_tree(
            input_path=args.input_file,
            output_path=args.output,
            create_backup=not args.no_backup
        )
        print_statistics(stats)
        print("\n✓ Cleanup completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

