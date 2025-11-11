import cv2
import os
import json
import time
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES

# Path to the video file
video_path = "kbc.mp4"

if not os.path.exists(video_path):
    print(f"Error: {video_path} not found!")
    exit(1)

print(f"Loading RF-DETR model...")
model = RFDETRMedium()

print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

# Initialize output data structure
output_data = {
    "video": video_path,
    "video_properties": {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames
    },
    "time_taken_seconds": 0,
    "frames": []
}

frame_count = 0
print("\nProcessing video frames...")

# Start timing
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Convert frame to RGB (RF-DETR expects RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    detections = model.predict(rgb_frame)
    
    # Draw bounding boxes and labels on the frame
    annotated_frame = frame.copy()
    
    # Collect detections for this frame
    frame_detections = []
    if len(detections) > 0:
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].tolist()
            class_id = int(detections.class_id[i])
            confidence = float(detections.confidence[i])
            class_name = COCO_CLASSES[class_id]
            
            # Draw bounding box (convert to int for drawing)
            x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
            cv2.rectangle(annotated_frame, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1_int, y1_int - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            }
            frame_detections.append(detection)
    
    # Store frame data
    output_data["frames"].append({
        "frame_number": frame_count,
        "detections": frame_detections
    })
    
    # Display frame with detections
    cv2.imshow('RF-DETR Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user (press 'q' to quit)")
        break
    
    # Print progress
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

# End timing
end_time = time.time()
time_taken = end_time - start_time
output_data["time_taken_seconds"] = round(time_taken, 2)

cap.release()
cv2.destroyAllWindows()

# Save JSON to file
output_file = "kbc_rfdetr.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"Results saved to {output_file}")

