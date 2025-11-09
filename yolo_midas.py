from ultralytics import YOLO
import cv2
import numpy as np
import json
import time
import torch

# Load models
print("Loading YOLO model...")
yolo_model = YOLO("yolo11m.pt")

print("Loading MiDaS depth estimation model...")
# Load MiDaS model using torch.hub
# Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
model_type = "DPT_Hybrid"  # Balanced accuracy and speed
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

video_path = "kbc.mp4"

# Open video
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
    
    # Convert BGR to RGB for both models
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO detection
    yolo_results = yolo_model.predict(rgb_frame, verbose=False, device=0)
    result = yolo_results[0]
    
    # Run MiDaS depth estimation
    input_batch = transform(rgb_frame).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map to 0-255 for visualization
    depth_normalized = (depth_map / depth_map.max() * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    
    # Create annotated frame
    annotated_frame = frame.copy()
    
    # Process YOLO detections
    frame_detections = []
    boxes = result.boxes
    
    if boxes is not None:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            class_id = int(boxes.cls[i])
            class_name = result.names[class_id]
            confidence = float(boxes.conf[i])
            
            # Convert to int for drawing
            x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
            
            # Extract depth information for this detection
            # Get depth at center of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Ensure coordinates are within bounds for depth map
            depth_h, depth_w = depth_map.shape
            center_x_depth = max(0, min(center_x, depth_w - 1))
            center_y_depth = max(0, min(center_y, depth_h - 1))
            y1_depth = max(0, min(y1_int, depth_h - 1))
            y2_depth = max(0, min(y2_int, depth_h - 1))
            x1_depth = max(0, min(x1_int, depth_w - 1))
            x2_depth = max(0, min(x2_int, depth_w - 1))
            
            # Get depth value at center (higher values = closer)
            depth_value = float(depth_map[center_y_depth, center_x_depth])
            
            # Also calculate average depth within bounding box
            bbox_region = depth_map[y1_depth:y2_depth, x1_depth:x2_depth]
            avg_depth = float(np.mean(bbox_region)) if bbox_region.size > 0 else depth_value
            min_depth = float(np.min(bbox_region)) if bbox_region.size > 0 else depth_value
            max_depth = float(np.max(bbox_region)) if bbox_region.size > 0 else depth_value
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2)
            
            # Draw label with depth info
            label = f"{class_name}: {confidence:.2f} | Depth: {depth_value:.3f}"
            cv2.putText(annotated_frame, label, (x1_int, y1_int - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw depth at center point
            cv2.circle(annotated_frame, (center_x, center_y), 3, (255, 0, 0), -1)
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "depth": {
                    "center_depth": depth_value,
                    "avg_depth": avg_depth,
                    "min_depth": min_depth,
                    "max_depth": max_depth
                }
            }
            frame_detections.append(detection)
    
    # Store frame data
    output_data["frames"].append({
        "frame_number": frame_count,
        "detections": frame_detections
    })
    
    # Create side-by-side visualization (original + depth map)
    # depth_vis is already in BGR format from applyColorMap
    combined = np.hstack([annotated_frame, depth_vis])
    
    # Display combined view
    cv2.imshow('YOLO + MiDaS: Detection (left) | Depth Map (right)', combined)
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
output_file = "kbc_yolo_midas.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to {output_file}")
print(f"Total processing time: {time_taken:.2f} seconds")

