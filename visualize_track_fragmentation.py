import json
import cv2
import numpy as np
from collections import defaultdict

def get_color_for_track_id(track_id):
    """Generate a consistent color for each track ID"""
    np.random.seed(track_id)
    color = np.random.randint(0, 255, 3).tolist()
    return tuple(int(c) for c in color)

def analyze_track_continuity(json_path):
    """Analyze track continuity and fragmentation"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Group person detections by track ID
    track_info = defaultdict(lambda: {
        "frames": [],
        "first_frame": float('inf'),
        "last_frame": 0,
        "gaps": []
    })
    
    for frame_data in data["frames"]:
        frame_num = frame_data["frame_number"]
        for det in frame_data["detections"]:
            if det["class_name"] == "person" and det["track_id"] is not None:
                track_id = det["track_id"]
                track_info[track_id]["frames"].append(frame_num)
                track_info[track_id]["first_frame"] = min(track_info[track_id]["first_frame"], frame_num)
                track_info[track_id]["last_frame"] = max(track_info[track_id]["last_frame"], frame_num)
    
    # Find gaps in tracks
    for track_id, info in track_info.items():
        frames = sorted(info["frames"])
        for i in range(len(frames) - 1):
            gap = frames[i+1] - frames[i]
            if gap > 1:  # Gap of more than 1 frame
                info["gaps"].append((frames[i], frames[i+1], gap))
    
    return track_info

def visualize_track_fragmentation(video_path, json_path, output_path=None):
    """
    Visualize tracking with emphasis on track fragmentation
    Shows different colors for different track IDs, highlighting fragmentation
    """
    # Load tracking results
    print(f"Loading tracking results from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Analyze track continuity
    track_info = analyze_track_continuity(json_path)
    
    # Get top 2 longest tracks (likely the 2 main persons)
    sorted_tracks = sorted(track_info.items(), 
                          key=lambda x: len(x[1]["frames"]), 
                          reverse=True)
    top2_tracks = [t[0] for t in sorted_tracks[:2]]
    
    print(f"\nTop 2 longest tracks (likely the 2 main persons): {top2_tracks}")
    print(f"Total unique person tracks: {len(track_info)}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    print("\nVisualizing track fragmentation...")
    print("Top 2 tracks (main persons) shown in GREEN and BLUE")
    print("Other tracks (fragments) shown in other colors")
    print("Press 'q' to quit")
    
    # Track statistics per frame
    stats_text_y = 30
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame_copy = frame.copy()
        
        # Get detections for this frame
        if frame_idx <= len(data["frames"]):
            frame_data = data["frames"][frame_idx - 1]
            
            # Separate top 2 tracks from fragments
            top2_dets = []
            fragment_dets = []
            
            for det in frame_data["detections"]:
                if det["class_name"] == "person" and det["track_id"] is not None:
                    if det["track_id"] in top2_tracks:
                        top2_dets.append(det)
                    else:
                        fragment_dets.append(det)
            
            # Draw top 2 tracks first (main persons) - use bright colors
            top2_colors = [(0, 255, 0), (255, 0, 0)]  # Green, Blue
            for i, det in enumerate(top2_dets):
                x1, y1, x2, y2 = map(int, det["bbox"])
                track_id = det["track_id"]
                class_name = det["class_name"]
                confidence = det["confidence"]
                
                color = top2_colors[i % len(top2_colors)]
                
                # Draw bounding box (thicker for main tracks)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"Person {i+1} ID:{track_id} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = max(y1, label_size[1] + 10)
                
                # Draw label background
                cv2.rectangle(frame_copy, (x1, label_y - label_size[1] - 5), 
                            (x1 + label_size[0], label_y + 5), color, -1)
                
                # Draw label text
                cv2.putText(frame_copy, label, (x1, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw fragment tracks (thinner, different colors)
            for det in fragment_dets:
                x1, y1, x2, y2 = map(int, det["bbox"])
                track_id = det["track_id"]
                class_name = det["class_name"]
                confidence = det["confidence"]
                
                # Use random but consistent color
                color = get_color_for_track_id(track_id)
                
                # Draw bounding box (thinner for fragments)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Draw label (smaller)
                label = f"Frag ID:{track_id} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1, label_size[1] + 10)
                
                # Draw label background
                cv2.rectangle(frame_copy, (x1, label_y - label_size[1] - 5), 
                            (x1 + label_size[0], label_y + 5), color, -1)
                
                # Draw label text
                cv2.putText(frame_copy, label, (x1, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add statistics text
        stats = f"Frame: {frame_idx} | Main tracks: {len(top2_dets)} | Fragments: {len(fragment_dets)}"
        cv2.putText(frame_copy, stats, (10, stats_text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_copy, stats, (10, stats_text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Write frame if output video specified
        if writer:
            writer.write(frame_copy)
        
        # Display frame
        cv2.imshow('Track Fragmentation Visualization', frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            break
        
        # Print progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nVisualization complete! Processed {frame_idx} frames")
    print(f"\nTrack Fragmentation Summary:")
    print(f"  Total unique person tracks: {len(track_info)}")
    print(f"  Expected: 2 (one per person)")
    print(f"  Extra tracks (fragments): {len(track_info) - 2}")
    
    # Show gaps in top 2 tracks
    print(f"\nTop 2 Track Analysis:")
    for i, (track_id, info) in enumerate([(t[0], t[1]) for t in sorted_tracks[:2]], 1):
        print(f"  Person {i} (Track ID {track_id}):")
        print(f"    Frames: {info['first_frame']} to {info['last_frame']} ({len(info['frames'])} detections)")
        if info['gaps']:
            print(f"    Gaps: {len(info['gaps'])} gaps found")
            for gap_start, gap_end, gap_size in info['gaps'][:3]:  # Show first 3 gaps
                print(f"      Gap at frame {gap_start}-{gap_end} (size: {gap_size} frames)")
        else:
            print(f"    Gaps: None (continuous track)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        tracker = sys.argv[1].lower()
        if tracker == "bytetrack":
            json_path = "kbc_yolo_bytetrack.json"
            output_video = "kbc_bytetrack_fragmentation.mp4"
        elif tracker == "botsort" or tracker == "deepsort":
            json_path = "kbc_yolo_deepsort.json"
            output_video = "kbc_botsort_fragmentation.mp4"
        else:
            print("Usage: python visualize_track_fragmentation.py [bytetrack|botsort]")
            sys.exit(1)
    else:
        # Default to ByteTrack
        json_path = "kbc_yolo_bytetrack.json"
        output_video = "kbc_bytetrack_fragmentation.mp4"
    
    video_path = "kbc.mp4"
    
    print(f"Visualizing: {json_path}")
    visualize_track_fragmentation(video_path, json_path, output_video)

