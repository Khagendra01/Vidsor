from ultralytics import YOLO
import json
import time
import cv2
import numpy as np

# Load the YOLOv11m model
print("Loading YOLO model...")
model = YOLO("yolo11m.pt")

# Start timing
start_time = time.time()

# Run tracking on video with ByteTrack (default tracker for YOLO)
# tracker="bytetrack.yaml" explicitly uses ByteTrack
print("Running YOLO with ByteTrack tracking...")
results = model.track(
    source="kbc.mp4", 
    tracker="bytetrack.yaml",  # Use ByteTrack tracker
    show=False,  # Set to True if you want to see visualization
    device=0,  # GPU 0, use 'cpu' for CPU mode
    persist=True  # Persist tracks across frames
)

# End timing
end_time = time.time()
time_taken = end_time - start_time

# Collect results in JSON format with tracking IDs
output_data = {
    "video": "kbc.mp4",
    "tracker": "ByteTrack",
    "time_taken_seconds": round(time_taken, 2),
    "frames": []
}

# Process results with frame numbers and track IDs
frame_number = 0
for result in results:
    frame_number += 1
    boxes = result.boxes
    
    frame_detections = []
    if boxes is not None and boxes.id is not None:
        # Tracking is active - boxes have IDs
        for i in range(len(boxes)):
            track_id = int(boxes.id[i]) if boxes.id[i] is not None else None
            
            detection = {
                "track_id": track_id,
                "class_id": int(boxes.cls[i]),
                "class_name": result.names[int(boxes.cls[i])],
                "confidence": float(boxes.conf[i]),
                "bbox": boxes.xyxy[i].tolist()
            }
            frame_detections.append(detection)
    elif boxes is not None:
        # No tracking IDs available (shouldn't happen with track(), but handle it)
        for i in range(len(boxes)):
            detection = {
                "track_id": None,
                "class_id": int(boxes.cls[i]),
                "class_name": result.names[int(boxes.cls[i])],
                "confidence": float(boxes.conf[i]),
                "bbox": boxes.xyxy[i].tolist()
            }
            frame_detections.append(detection)
    
    # Store frame data
    output_data["frames"].append({
        "frame_number": frame_number,
        "detections": frame_detections
    })
    
    # Print progress every 100 frames
    if frame_number % 100 == 0:
        print(f"Processed {frame_number} frames...")

# Save JSON to file
output_file = "kbc_yolo_bytetrack.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nResults saved to {output_file}")
print(f"Total processing time: {time_taken:.2f} seconds")
print(f"Average FPS: {frame_number / time_taken:.2f}")

# Print some statistics
total_tracks = set()
for frame_data in output_data["frames"]:
    for det in frame_data["detections"]:
        if det["track_id"] is not None:
            total_tracks.add(det["track_id"])

print(f"Total unique tracks detected: {len(total_tracks)}")

