from ultralytics import YOLO
import json
import time

# Load the YOLOv11m model
model = YOLO("yolo11m.pt")

# Start timing
start_time = time.time()

# Run inference on video with visualization
results = model.predict(source="kbc.mp4", show=True, device=0)  
# 'device=0' → use GPU 0; use 'device=cpu' for CPU mode

# End timing
end_time = time.time()
time_taken = end_time - start_time

# Collect results in JSON format
output_data = {
    "video": "kbc.mp4",
    "time_taken_seconds": round(time_taken, 2),
    "frames": []
}

# Process results with frame numbers
frame_number = 0
for result in results:
    frame_number += 1
    boxes = result.boxes
    
    frame_detections = []
    if boxes is not None:
        for i in range(len(boxes)):
            detection = {
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

# Save JSON to file
output_file = "kbc_yolo.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"Results saved to {output_file}")
