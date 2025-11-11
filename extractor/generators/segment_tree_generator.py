"""
Segment tree generation module.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from PIL import Image

from extractor.config import FPS, AUDIO_SEGMENT_DURATION, AUDIO_OVERLAP
from extractor.models.blip_loader import BLIPLoader
from extractor.models.whisper_loader import WhisperLoader
from extractor.models.yolo_loader import YOLOLoader
from extractor.processors.visual_processor import VisualProcessor
from extractor.processors.audio_processor import AudioProcessor
from extractor.processors.llava_processor import LLaVAProcessor
from extractor.processors.tracking_processor import TrackingProcessor
from extractor.utils.video_utils import VideoUtils


class SegmentTreeGenerator:
    """Generates segment tree from video."""
    
    def __init__(
        self,
        video_path: str,
        tracking_json_path: str,
        config,
        blip_loader: BLIPLoader,
        whisper_loader: WhisperLoader,
        yolo_loader: YOLOLoader,
        video_utils: VideoUtils
    ):
        """
        Initialize segment tree generator.
        
        Args:
            video_path: Path to video file
            tracking_json_path: Path to tracking JSON file
            config: ExtractorConfig instance
            blip_loader: BLIP model loader
            whisper_loader: Whisper model loader
            yolo_loader: YOLO model loader
            video_utils: Video utilities instance
        """
        self.video_path = video_path
        self.tracking_json_path = tracking_json_path
        self.config = config
        self.blip_loader = blip_loader
        self.whisper_loader = whisper_loader
        self.yolo_loader = yolo_loader
        self.video_utils = video_utils
        
        # Initialize processors
        self.visual_processor = VisualProcessor(blip_loader)
        self.audio_processor = AudioProcessor(whisper_loader, video_path)
        self.llava_processor = LLaVAProcessor(
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model,
            use_images=config.use_images
        )
        self.tracking_processor = TrackingProcessor(config.yolo_stride)
    
    def _load_tracking_data(self) -> Dict:
        """Load or generate tracking data."""
        import os
        
        # Generate tracking if needed
        if not os.path.exists(self.tracking_json_path):
            self.yolo_loader.generate_tracking(
                self.video_path,
                self.tracking_json_path,
                tracker=self.config.tracker,
                stride=self.config.yolo_stride
            )
        
        with open(self.tracking_json_path, 'r') as f:
            return json.load(f)
    
    def _get_blip_frame_numbers(self, start_frame: int, end_frame: int) -> List[int]:
        """Get frame numbers for BLIP processing based on blip_split config."""
        middle_frame = start_frame + (end_frame - start_frame) // 2
        
        if self.config.blip_split == 1:
            return [middle_frame]
        elif self.config.blip_split == 2:
            return [start_frame, end_frame]
        elif self.config.blip_split == 3:
            return [start_frame, middle_frame, end_frame]
        else:
            return [middle_frame]
    
    def _process_second(self, second_idx: int, frames_data: List[Dict], max_frame: Optional[int] = None) -> Dict:
        """Process one second of video."""
        start_frame = (second_idx * FPS) + 1
        if max_frame is not None:
            end_frame = min(start_frame + FPS - 1, max_frame)
        else:
            max_tracked_frame = max((f.get('frame_number', 0) for f in frames_data), default=0)
            end_frame = min(start_frame + FPS - 1, max_tracked_frame)
        
        # Group detections
        detection_groups = self.tracking_processor.group_detections(frames_data, start_frame, end_frame)
        detection_summary = self.tracking_processor.create_detection_summary(detection_groups)
        
        # Get BLIP frame numbers
        blip_frame_numbers = self._get_blip_frame_numbers(start_frame, end_frame)
        
        # Process BLIP frames in parallel
        blip_results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            for frame_num in blip_frame_numbers:
                frame_image = self.video_utils.get_frame(frame_num)
                if frame_image:
                    future = executor.submit(self.visual_processor.process_frame, frame_num, frame_image)
                    futures[future] = frame_num
            
            for future in as_completed(futures):
                frame_num = futures[future]
                try:
                    result = future.result()
                    blip_results.append((frame_num, result))
                except Exception as e:
                    print(f"Error processing BLIP for frame {frame_num}: {e}")
                    blip_results.append((frame_num, {
                        "description": f"Error: {str(e)}",
                        "blip_metadata": {"model": "blip", "processing_time": 0}
                    }))
        
        # Sort and create blip_descriptions
        blip_results.sort(key=lambda x: x[0])
        blip_descriptions = []
        for frame_num, result in blip_results:
            # Find group for this frame
            group_idx = -1
            for i, group in enumerate(detection_groups):
                frame_range = group["frame_range"]
                if frame_range[0] <= frame_num <= frame_range[1]:
                    group_idx = i
                    break
            if group_idx == -1:
                group_idx = min(len(detection_groups) - 1, max(0, (frame_num - start_frame) // 5))
            
            blip_descriptions.append({
                "group_index": group_idx,
                "frame_number": frame_num,
                "frame_range": detection_groups[group_idx]["frame_range"] if group_idx < len(detection_groups) else [end_frame, end_frame],
                "description": result["description"],
                "blip_metadata": result["blip_metadata"]
            })
        
        # Process LLaVA if enabled
        if self.config.use_llava:
            blip_texts = [desc["description"] for desc in blip_descriptions]
            if not blip_texts:
                blip_texts = ["No image descriptions available"]
            
            images_to_send = None
            if self.config.use_images:
                images_to_send = []
                for desc in blip_descriptions:
                    frame_num = desc.get("frame_number")
                    if frame_num:
                        frame_image = self.video_utils.get_frame(frame_num)
                        if frame_image:
                            images_to_send.append(frame_image)
                if not images_to_send:
                    images_to_send = None
            
            llava_result = self.llava_processor.process(blip_texts, detection_summary, images=images_to_send)
            unified_description = llava_result["unified_description"]
            llava_metadata = llava_result["llava_metadata"]
        else:
            blip_texts = [desc["description"] for desc in blip_descriptions]
            if blip_texts:
                unified_description = " | ".join(blip_texts)
            else:
                unified_description = detection_summary
            llava_metadata = {
                "model": "none",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "processing_time": 0,
                "note": "LLaVA processing disabled, using BLIP descriptions only"
            }
        
        return {
            "second": second_idx,
            "time_range": [round(second_idx, 3), round(second_idx + 0.999, 3)],
            "frame_range": [start_frame, end_frame],
            "unified_description": unified_description,
            "llava_metadata": llava_metadata,
            "blip_descriptions": blip_descriptions,
            "detection_groups": detection_groups
        }
    
    def generate(self) -> str:
        """
        Generate segment tree.
        
        Returns:
            Path to output JSON file
        """
        print("Starting segment tree generation...")
        start_time = time.time()
        
        # Load tracking data
        tracking_data = self._load_tracking_data()
        frames_data = tracking_data["frames"]
        video_path = tracking_data.get("video", self.video_path)
        
        # Load models
        self.blip_loader.load()
        self.whisper_loader.load()
        
        # Calculate number of seconds
        video_duration = self.video_utils.get_duration()
        if video_duration:
            num_seconds = int(video_duration) + (1 if video_duration % 1 >= 0.5 else 0)
            max_frame = int(video_duration * FPS)
            print(f"Processing {num_seconds} seconds (video duration: {video_duration:.2f}s, max frame: {max_frame})...")
        else:
            max_tracked_frame = max((f.get('frame_number', 0) for f in frames_data), default=0)
            num_seconds = (max_tracked_frame + FPS - 1) // FPS
            max_frame = max_tracked_frame
            print(f"Processing {num_seconds} seconds (estimated from tracking data, max frame: {max_frame})...")
        
        # Process seconds in parallel
        seconds_data = [None] * num_seconds  # Pre-allocate array to maintain structure
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_second, i, frames_data, max_frame): i
                for i in range(num_seconds)
            }
            
            for future in as_completed(futures):
                second_idx = futures[future]
                try:
                    result = future.result()
                    seconds_data[second_idx] = result
                    print(f"Completed second {second_idx + 1}/{num_seconds}")
                except Exception as e:
                    print(f"Error processing second {second_idx}: {e}")
                    # Create a minimal fallback entry instead of leaving None
                    seconds_data[second_idx] = {
                        "second": second_idx,
                        "time_range": [round(second_idx, 3), round(second_idx + 0.999, 3)],
                        "frame_range": [second_idx * FPS + 1, min((second_idx + 1) * FPS, max_frame)],
                        "unified_description": f"Error processing second {second_idx}: {str(e)}",
                        "llava_metadata": {
                            "model": "error",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "processing_time": 0,
                            "error": str(e)
                        },
                        "blip_descriptions": [],
                        "detection_groups": []
                    }
        
        # Convert to sparse array (remove None placeholders, keep only valid entries)
        # This maintains compatibility with code that expects sparse arrays
        seconds_data = [s for s in seconds_data if s is not None]
        # Sort by second index to ensure proper ordering
        seconds_data.sort(key=lambda x: x["second"])
        
        # Generate transcriptions
        print("\nGenerating audio transcriptions with 1-second overlap...")
        transcriptions = []
        if video_duration and self.whisper_loader.is_loaded():
            step = AUDIO_SEGMENT_DURATION - AUDIO_OVERLAP
            transcription_start = 0
            
            while transcription_start < video_duration:
                transcription_end = min(transcription_start + AUDIO_SEGMENT_DURATION, video_duration)
                
                transcription_result = self.audio_processor.transcribe_segment(transcription_start, transcription_end)
                
                transcriptions.append({
                    "id": len(transcriptions),
                    "time_range": [round(transcription_start, 2), round(transcription_end, 2)],
                    "transcription": transcription_result["transcription"],
                    "metadata": transcription_result["transcription_metadata"]
                })
                
                print(f"Transcribed audio segment {len(transcriptions)}: {transcription_start:.1f}s - {transcription_end:.1f}s")
                transcription_start += step
        
        # Map transcriptions to seconds
        for second_data in seconds_data:
            second_idx = second_data["second"]
            second_time = second_idx + 0.5
            
            best_transcription_id = None
            best_distance = float('inf')
            
            for trans in transcriptions:
                time_range = trans["time_range"]
                if time_range[0] <= second_time <= time_range[1]:
                    distance = abs(time_range[0] - second_time)
                    if distance < best_distance:
                        best_distance = distance
                        best_transcription_id = trans["id"]
            
            if best_transcription_id is None:
                for trans in transcriptions:
                    time_range = trans["time_range"]
                    if abs(time_range[0] - second_time) < 3:
                        distance = abs(time_range[0] - second_time)
                        if distance < best_distance:
                            best_distance = distance
                            best_transcription_id = trans["id"]
            
            second_data["transcription_id"] = best_transcription_id
        
        # Build output structure
        output_data = {
            "video": video_path,
            "fps": FPS,
            "tracker": self.config.tracker,
            "seconds": seconds_data,
            "transcriptions": transcriptions
        }
        
        # Save output
        with open(self.config.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\nSegment tree generated in {total_time:.2f}s")
        print(f"Output saved to {self.config.output_path}")
        
        return self.config.output_path

