import json
from collections import defaultdict

def load_tracking_results(json_path):
    """Load tracking results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_person_tracking(data, tracker_name):
    """Analyze person-specific tracking"""
    person_tracks = defaultdict(lambda: {
        "first_frame": None,
        "last_frame": None,
        "frame_count": 0,
        "total_detections": 0,
        "frames": []
    })
    
    all_person_detections = []
    
    for frame_data in data["frames"]:
        frame_num = frame_data["frame_number"]
        for det in frame_data["detections"]:
            if det["class_name"] == "person" and det["track_id"] is not None:
                track_id = det["track_id"]
                all_person_detections.append({
                    "frame": frame_num,
                    "track_id": track_id,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"]
                })
                
                # Update track info
                if person_tracks[track_id]["first_frame"] is None:
                    person_tracks[track_id]["first_frame"] = frame_num
                person_tracks[track_id]["last_frame"] = frame_num
                person_tracks[track_id]["frame_count"] += 1
                person_tracks[track_id]["total_detections"] += 1
                person_tracks[track_id]["frames"].append(frame_num)
    
    # Calculate track statistics
    unique_tracks = len(person_tracks)
    total_frames = len(data["frames"])
    
    # Calculate coverage (how many frames have person detections)
    frames_with_persons = set()
    for det in all_person_detections:
        frames_with_persons.add(det["frame"])
    coverage = len(frames_with_persons) / total_frames * 100 if total_frames > 0 else 0
    
    # Calculate track lifetimes
    track_lifetimes = []
    for track_id, info in person_tracks.items():
        lifetime = info["last_frame"] - info["first_frame"] + 1
        track_lifetimes.append({
            "track_id": track_id,
            "lifetime": lifetime,
            "frame_count": info["frame_count"],
            "first_frame": info["first_frame"],
            "last_frame": info["last_frame"],
            "coverage": (info["frame_count"] / lifetime * 100) if lifetime > 0 else 0
        })
    
    # Sort by lifetime (longest first)
    track_lifetimes.sort(key=lambda x: x["lifetime"], reverse=True)
    
    return {
        "tracker_name": tracker_name,
        "unique_person_tracks": unique_tracks,
        "total_person_detections": len(all_person_detections),
        "frames_with_persons": len(frames_with_persons),
        "total_frames": total_frames,
        "coverage_percentage": coverage,
        "track_lifetimes": track_lifetimes,
        "person_tracks": dict(person_tracks),
        "all_detections": all_person_detections
    }

def compare_person_tracking(bytetrack_json, botsort_json):
    """Compare person tracking between ByteTrack and Bot-SORT"""
    print("=" * 80)
    print("PERSON TRACKING ANALYSIS")
    print("=" * 80)
    print("\nExpected: 2 unique persons in the video")
    print("=" * 80)
    
    # Load and analyze
    print("\nLoading tracking results...")
    bytetrack_data = load_tracking_results(bytetrack_json)
    botsort_data = load_tracking_results(botsort_json)
    
    print("Analyzing person tracking...")
    bt_analysis = analyze_person_tracking(bytetrack_data, "ByteTrack")
    bs_analysis = analyze_person_tracking(botsort_data, "Bot-SORT")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("PERSON TRACKING COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<40} {'ByteTrack':<20} {'Bot-SORT':<20}")
    print("-" * 80)
    print(f"{'Unique Person Tracks':<40} {bt_analysis['unique_person_tracks']:<20} {bs_analysis['unique_person_tracks']:<20}")
    print(f"{'Total Person Detections':<40} {bt_analysis['total_person_detections']:<20} {bs_analysis['total_person_detections']:<20}")
    print(f"{'Frames with Person Detections':<40} {bt_analysis['frames_with_persons']:<20} {bs_analysis['frames_with_persons']:<20}")
    print(f"{'Total Frames':<40} {bt_analysis['total_frames']:<20} {bs_analysis['total_frames']:<20}")
    print(f"{'Coverage (%)':<40} {bt_analysis['coverage_percentage']:<20.1f} {bs_analysis['coverage_percentage']:<20.1f}")
    
    # Track quality analysis
    print("\n" + "=" * 80)
    print("TRACK QUALITY ANALYSIS")
    print("=" * 80)
    
    print("\nByteTrack Person Tracks:")
    print("-" * 80)
    print(f"{'Track ID':<12} {'Lifetime':<12} {'Frames':<12} {'First':<12} {'Last':<12} {'Coverage %':<12}")
    print("-" * 80)
    for track_info in bt_analysis['track_lifetimes']:
        print(f"{track_info['track_id']:<12} {track_info['lifetime']:<12} {track_info['frame_count']:<12} "
              f"{track_info['first_frame']:<12} {track_info['last_frame']:<12} {track_info['coverage']:<12.1f}")
    
    print("\nBot-SORT Person Tracks:")
    print("-" * 80)
    print(f"{'Track ID':<12} {'Lifetime':<12} {'Frames':<12} {'First':<12} {'Last':<12} {'Coverage %':<12}")
    print("-" * 80)
    for track_info in bs_analysis['track_lifetimes']:
        print(f"{track_info['track_id']:<12} {track_info['lifetime']:<12} {track_info['frame_count']:<12} "
              f"{track_info['first_frame']:<12} {track_info['last_frame']:<12} {track_info['coverage']:<12.1f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Ideal: 2 tracks (one per person)
    expected_tracks = 2
    
    bt_tracks = bt_analysis['unique_person_tracks']
    bs_tracks = bs_analysis['unique_person_tracks']
    
    print(f"\nExpected: {expected_tracks} unique person tracks (one per person)")
    print(f"ByteTrack found: {bt_tracks} tracks")
    print(f"Bot-SORT found: {bs_tracks} tracks")
    
    if bt_tracks == expected_tracks:
        print("✓ ByteTrack: Perfect! Found exactly 2 person tracks")
    elif bt_tracks < expected_tracks:
        print(f"⚠ ByteTrack: Found fewer tracks than expected (might have missed a person)")
    else:
        print(f"⚠ ByteTrack: Found {bt_tracks - expected_tracks} extra tracks (person re-identified as new track)")
        print("  This suggests track loss and re-acquisition (occlusions, similar appearances)")
    
    if bs_tracks == expected_tracks:
        print("✓ Bot-SORT: Perfect! Found exactly 2 person tracks")
    elif bs_tracks < expected_tracks:
        print(f"⚠ Bot-SORT: Found fewer tracks than expected (might have missed a person)")
    else:
        print(f"⚠ Bot-SORT: Found {bs_tracks - expected_tracks} extra tracks (person re-identified as new track)")
        print("  This suggests track loss and re-acquisition (occlusions, similar appearances)")
    
    # Find longest tracks (likely the main person tracks)
    print("\n" + "=" * 80)
    print("LONGEST TRACKS (Likely the 2 main persons)")
    print("=" * 80)
    
    print("\nByteTrack Top 2 Tracks:")
    for i, track_info in enumerate(bt_analysis['track_lifetimes'][:2], 1):
        print(f"  Person {i} (Track ID {track_info['track_id']}): "
              f"{track_info['lifetime']} frames ({track_info['frame_count']} detections, "
              f"{track_info['coverage']:.1f}% coverage)")
    
    print("\nBot-SORT Top 2 Tracks:")
    for i, track_info in enumerate(bs_analysis['track_lifetimes'][:2], 1):
        print(f"  Person {i} (Track ID {track_info['track_id']}): "
              f"{track_info['lifetime']} frames ({track_info['frame_count']} detections, "
              f"{track_info['coverage']:.1f}% coverage)")
    
    # Calculate how well the top 2 tracks cover the video
    print("\n" + "=" * 80)
    print("COVERAGE BY TOP 2 TRACKS")
    print("=" * 80)
    
    bt_top2_coverage = sum(t['frame_count'] for t in bt_analysis['track_lifetimes'][:2])
    bs_top2_coverage = sum(t['frame_count'] for t in bs_analysis['track_lifetimes'][:2])
    
    print(f"\nByteTrack top 2 tracks: {bt_top2_coverage} detections out of {bt_analysis['total_person_detections']} "
          f"({bt_top2_coverage/bt_analysis['total_person_detections']*100:.1f}%)")
    print(f"Bot-SORT top 2 tracks: {bs_top2_coverage} detections out of {bs_analysis['total_person_detections']} "
          f"({bs_top2_coverage/bs_analysis['total_person_detections']*100:.1f}%)")
    
    # Winner
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    bt_score = 0
    bs_score = 0
    
    # Score based on number of extra tracks (fewer is better)
    bt_extra = bt_tracks - expected_tracks
    bs_extra = bs_tracks - expected_tracks
    
    if bt_extra < bs_extra:
        bt_score += 1
        print(f"\n✓ ByteTrack: Fewer extra tracks ({bt_extra} vs {bs_extra} extra tracks)")
    elif bs_extra < bt_extra:
        bs_score += 1
        print(f"\n✓ Bot-SORT: Fewer extra tracks ({bs_extra} vs {bt_extra} extra tracks)")
    else:
        print(f"\n= Both trackers have same number of extra tracks ({bt_extra})")
    
    # Note: Both are far from ideal (expected 2 tracks)
    if bt_extra > 0 and bs_extra > 0:
        print(f"  Note: Both trackers create multiple tracks per person (fragmentation)")
    
    # Score based on top 2 track coverage
    bt_top2_pct = bt_top2_coverage/bt_analysis['total_person_detections']*100 if bt_analysis['total_person_detections'] > 0 else 0
    bs_top2_pct = bs_top2_coverage/bs_analysis['total_person_detections']*100 if bs_analysis['total_person_detections'] > 0 else 0
    
    if bt_top2_pct > bs_top2_pct:
        bt_score += 1
        print(f"✓ ByteTrack: Better coverage by main tracks ({bt_top2_pct:.1f}% vs {bs_top2_pct:.1f}%)")
    elif bs_top2_pct > bt_top2_pct:
        bs_score += 1
        print(f"✓ Bot-SORT: Better coverage by main tracks ({bs_top2_pct:.1f}% vs {bt_top2_pct:.1f}%)")
    else:
        print(f"= Both trackers have similar main track coverage")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    bytetrack_file = "kbc_yolo_bytetrack.json"
    botsort_file = "kbc_yolo_deepsort.json"
    
    try:
        compare_person_tracking(bytetrack_file, botsort_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run both tracking scripts first:")
        print("  1. python yolo_bytetrack.py")
        print("  2. python yolo_deepsort.py")
        print("Then run this analysis script again.")

