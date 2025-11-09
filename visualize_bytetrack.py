import json
import cv2
import numpy as np

def get_color_for_track_id(track_id):
    """Generate a consistent color for each track ID"""
    np.random.seed(track_id)
    color = np.random.randint(0, 255, 3).tolist()
    return tuple(int(c) for c in color)

def visualize_tracking(video_path, json_path, output_path=None):
    """
    Visualize tracking results from JSON file on video
    
    Args:
        video_path: Path to input video
        json_path: Path to JSON file with tracking results
        output_path: Optional path to save output video (if None, just displays)
    """
    # Load tracking results
    print(f"Loading tracking results from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
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
    print("Visualizing tracking results... (Press 'q' to quit)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Get detections for this frame
        if frame_idx <= len(data["frames"]):
            frame_data = data["frames"][frame_idx - 1]
            
            # Draw bounding boxes with track IDs
            for det in frame_data["detections"]:
                if det["track_id"] is not None:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    track_id = det["track_id"]
                    class_name = det["class_name"]
                    confidence = det["confidence"]
                    
                    # Get color for this track ID
                    color = get_color_for_track_id(track_id)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with track ID
                    label = f"ID:{track_id} {class_name} {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = max(y1, label_size[1] + 10)
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                                (x1 + label_size[0], label_y + 5), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, label_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame if output video specified
        if writer:
            writer.write(frame)
        
        # Display frame
        cv2.imshow('ByteTrack Visualization', frame)
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
    
    print(f"Visualization complete! Processed {frame_idx} frames")

if __name__ == "__main__":
    # Example usage
    video_path = "kbc.mp4"
    json_path = "kbc_yolo_bytetrack.json"
    output_video = "kbc_bytetrack_visualization.mp4"  # Optional: set to None to just display
    
    visualize_tracking(video_path, json_path, output_video)

