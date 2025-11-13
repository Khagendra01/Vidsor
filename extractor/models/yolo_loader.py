"""
YOLO model loader for object detection and tracking.
"""

import os
from pathlib import Path
from ultralytics import YOLO
from typing import Optional
from extractor.config import YOLO_MODEL


class YOLOLoader:
    """Manages YOLO model loading and tracking generation."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLO loader.
        
        Args:
            model_path: Optional model path override
        """
        self.model_path = model_path or YOLO_MODEL
        self.model = None
        self._loaded = False
    
    def _resolve_model_path(self) -> str:
        """
        Resolve model path to absolute path if relative.
        
        Returns:
            Resolved model path
        """
        model_path = Path(self.model_path)
        
        # If relative path, resolve from current working directory
        if not model_path.is_absolute():
            resolved = Path.cwd() / model_path
            if resolved.exists():
                return str(resolved.resolve())
        
        # If absolute or doesn't exist, return as-is (YOLO will handle download if needed)
        return str(model_path)
    
    def load(self):
        """Load YOLO model."""
        if self._loaded:
            return
        
        # Resolve model path
        resolved_path = self._resolve_model_path()
        
        # Check if file exists
        if not os.path.exists(resolved_path):
            print(f"Warning: YOLO model not found at {resolved_path}")
            print("YOLO will attempt to download the model if it's a standard model name...")
        else:
            print(f"Loading YOLO model from: {resolved_path}")
        
        self.model = YOLO(resolved_path)
        self._loaded = True
    
    def generate_tracking(
        self,
        video_path: str,
        output_path: str,
        tracker: str = "bytetrack",
        stride: int = 10
    ) -> None:
        """
        Generate tracking data for a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save tracking JSON
            tracker: Tracker name ("bytetrack" or "deepsort")
            stride: Process every Nth frame
        """
        if not self._loaded:
            self.load()
        
        if os.path.exists(output_path):
            print(f"Tracking file already exists: {output_path}")
            return
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Generating tracking data from {video_path}...")
        
        # Determine tracker config
        if tracker == "bytetrack":
            tracker_config = "bytetrack.yaml"
            tracker_name = "ByteTrack"
        elif tracker == "deepsort":
            tracker_config = "botsort.yaml"
            tracker_name = "Bot-SORT"
        else:
            tracker_config = "bytetrack.yaml"
            tracker_name = "ByteTrack"
        
        print(f"Running YOLO with {tracker_name} tracking (processing every {stride}th frame)...")
        
        # Process results and save to JSON
        import json
        import time
        import traceback
        
        # Determine device (try GPU first, fallback to CPU)
        try:
            import torch
            device = 0 if torch.cuda.is_available() else 'cpu'
            if device == 'cpu':
                print("Warning: CUDA not available, using CPU (this will be slower)")
        except ImportError:
            device = 'cpu'
            print("Warning: PyTorch not available, using CPU")
        
        try:
            # Run tracking
            results = self.model.track(
                source=video_path,
                tracker=tracker_config,
                show=False,
                device=device,
                persist=True,
                stream=True,
                imgsz=640,
                verbose=False,
                vid_stride=stride
            )
            
            start_time = time.time()
            output_data = {
                "video": video_path,
                "tracker": tracker_name,
                "time_taken_seconds": 0,  # Will be updated
                "frames": []
            }
            
            result_index = 0
            try:
                for result in results:
                    result_index += 1
                    frame_number = (result_index - 1) * stride + 1
                    boxes = result.boxes
                    
                    frame_detections = []
                    if boxes is not None and boxes.id is not None:
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
                        for i in range(len(boxes)):
                            detection = {
                                "track_id": None,
                                "class_id": int(boxes.cls[i]),
                                "class_name": result.names[int(boxes.cls[i])],
                                "confidence": float(boxes.conf[i]),
                                "bbox": boxes.xyxy[i].tolist()
                            }
                            frame_detections.append(detection)
                    
                    output_data["frames"].append({
                        "frame_number": frame_number,
                        "detections": frame_detections
                    })
                    
                    if result_index % 25 == 0:
                        print(f"Processed {result_index} frames (video frames up to {frame_number})...")
            except Exception as e:
                print(f"\nERROR: Failed to process tracking results: {e}")
                print("Full traceback:")
                traceback.print_exc()
                raise
            
            time_taken = time.time() - start_time
            output_data["time_taken_seconds"] = round(time_taken, 2)
            
            # Save JSON
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Tracking data saved to {output_path}")
            print(f"Total processing time: {time_taken:.2f}s")
            if time_taken > 0:
                print(f"Average FPS: {result_index / time_taken:.2f}")
            else:
                print(f"Processed {result_index} frames")
                
        except Exception as e:
            print(f"\nERROR: Failed to generate tracking data: {e}")
            print("Full traceback:")
            traceback.print_exc()
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

