import json
import time

def load_tracking_results(json_path):
    """Load tracking results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_tracker_performance(data):
    """Analyze tracking performance metrics"""
    total_frames = len(data["frames"])
    total_detections = 0
    total_tracks = set()
    track_lifetimes = {}  # Track how long each track persists
    
    for frame_data in data["frames"]:
        for det in frame_data["detections"]:
            total_detections += 1
            if det["track_id"] is not None:
                track_id = det["track_id"]
                total_tracks.add(track_id)
                
                if track_id not in track_lifetimes:
                    track_lifetimes[track_id] = {"first_frame": frame_data["frame_number"], 
                                                "last_frame": frame_data["frame_number"],
                                                "frame_count": 0}
                track_lifetimes[track_id]["last_frame"] = frame_data["frame_number"]
                track_lifetimes[track_id]["frame_count"] += 1
    
    # Calculate average track lifetime
    if track_lifetimes:
        avg_lifetime = sum(t["frame_count"] for t in track_lifetimes.values()) / len(track_lifetimes)
        max_lifetime = max(t["frame_count"] for t in track_lifetimes.values())
        min_lifetime = min(t["frame_count"] for t in track_lifetimes.values())
    else:
        avg_lifetime = max_lifetime = min_lifetime = 0
    
    return {
        "tracker_name": data.get("tracker", "Unknown"),
        "total_frames": total_frames,
        "total_detections": total_detections,
        "unique_tracks": len(total_tracks),
        "avg_detections_per_frame": total_detections / total_frames if total_frames > 0 else 0,
        "time_taken_seconds": data.get("time_taken_seconds", 0),
        "fps": total_frames / data.get("time_taken_seconds", 1) if data.get("time_taken_seconds", 0) > 0 else 0,
        "avg_track_lifetime": round(avg_lifetime, 2),
        "max_track_lifetime": max_lifetime,
        "min_track_lifetime": min_lifetime,
        "track_lifetimes": track_lifetimes
    }

def compare_trackers(bytetrack_json, botsort_json):
    """Compare ByteTrack and Bot-SORT performance"""
    print("=" * 70)
    print("TRACKER COMPARISON: ByteTrack vs Bot-SORT (DeepSORT)")
    print("=" * 70)
    
    # Load results
    print("\nLoading tracking results...")
    bytetrack_data = load_tracking_results(bytetrack_json)
    botsort_data = load_tracking_results(botsort_json)
    
    # Analyze
    print("Analyzing performance...")
    bt_stats = analyze_tracker_performance(bytetrack_data)
    bs_stats = analyze_tracker_performance(botsort_data)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'ByteTrack':<20} {'Bot-SORT':<20}")
    print("-" * 70)
    print(f"{'Processing Time (s)':<30} {bt_stats['time_taken_seconds']:<20} {bs_stats['time_taken_seconds']:<20}")
    print(f"{'FPS':<30} {bt_stats['fps']:<20.2f} {bs_stats['fps']:<20.2f}")
    print(f"{'Total Frames':<30} {bt_stats['total_frames']:<20} {bs_stats['total_frames']:<20}")
    print(f"{'Total Detections':<30} {bt_stats['total_detections']:<20} {bs_stats['total_detections']:<20}")
    print(f"{'Unique Tracks':<30} {bt_stats['unique_tracks']:<20} {bs_stats['unique_tracks']:<20}")
    print(f"{'Avg Detections/Frame':<30} {bt_stats['avg_detections_per_frame']:<20.2f} {bs_stats['avg_detections_per_frame']:<20.2f}")
    print(f"{'Avg Track Lifetime':<30} {bt_stats['avg_track_lifetime']:<20} {bs_stats['avg_track_lifetime']:<20}")
    print(f"{'Max Track Lifetime':<30} {bt_stats['max_track_lifetime']:<20} {bs_stats['max_track_lifetime']:<20}")
    print(f"{'Min Track Lifetime':<30} {bt_stats['min_track_lifetime']:<20} {bs_stats['min_track_lifetime']:<20}")
    
    # Speed comparison
    print("\n" + "=" * 70)
    print("SPEED COMPARISON")
    print("=" * 70)
    speed_diff = ((bs_stats['time_taken_seconds'] - bt_stats['time_taken_seconds']) / bt_stats['time_taken_seconds']) * 100
    fps_diff = ((bt_stats['fps'] - bs_stats['fps']) / bs_stats['fps']) * 100 if bs_stats['fps'] > 0 else 0
    
    print(f"\nByteTrack is {abs(speed_diff):.1f}% {'faster' if speed_diff > 0 else 'slower'}")
    print(f"FPS difference: {fps_diff:.1f}% {'faster' if fps_diff > 0 else 'slower'} (ByteTrack vs Bot-SORT)")
    
    # Accuracy indicators (longer track lifetimes = better tracking)
    print("\n" + "=" * 70)
    print("TRACKING QUALITY INDICATORS")
    print("=" * 70)
    print(f"\nAverage track lifetime:")
    print(f"  ByteTrack: {bt_stats['avg_track_lifetime']} frames")
    print(f"  Bot-SORT:  {bs_stats['avg_track_lifetime']} frames")
    
    if bs_stats['avg_track_lifetime'] > bt_stats['avg_track_lifetime']:
        print(f"\n✓ Bot-SORT maintains tracks longer (better for occlusions)")
    else:
        print(f"\n✓ ByteTrack maintains tracks longer")
    
    print(f"\nLongest track:")
    print(f"  ByteTrack: {bt_stats['max_track_lifetime']} frames")
    print(f"  Bot-SORT:  {bs_stats['max_track_lifetime']} frames")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nByteTrack:")
    print("  • Faster processing")
    print("  • Uses motion prediction (Kalman filter) + IoU matching")
    print("  • No appearance features (lighter computation)")
    print("  • Good for real-time applications")
    
    print("\nBot-SORT (DeepSORT):")
    print("  • More accurate tracking")
    print("  • Uses motion prediction + appearance features + camera compensation")
    print("  • Better at handling occlusions and similar-looking objects")
    print("  • Slightly slower due to appearance feature extraction")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Compare the two trackers
    bytetrack_file = "kbc_yolo_bytetrack.json"
    botsort_file = "kbc_yolo_deepsort.json"
    
    try:
        compare_trackers(bytetrack_file, botsort_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run both tracking scripts first:")
        print("  1. python yolo_bytetrack.py")
        print("  2. python yolo_deepsort.py")
        print("Then run this comparison script again.")

