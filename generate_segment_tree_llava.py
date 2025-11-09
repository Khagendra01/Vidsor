import json
import cv2
import time
import threading
import io
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import requests
import base64
from typing import Dict, List, Any
import os
from ultralytics import YOLO
from prompt import get_llava_prompt
import whisper

# Configuration
FPS = 30
FRAMES_PER_GROUP = 5
GROUPS_PER_SECOND = 6  # 30 frames / 5 = 6 groups
BLIP_FRAMES = [1, 11, 26]  # 1st frame of groups 1, 3, 6 (0-indexed: 0, 2, 5)
TRACKER = "bytetrack"  # or "deepsort" - configurable
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "bakllava"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
WHISPER_MODEL = "base"  # Fast and accurate balance
AUDIO_SEGMENT_DURATION = 5  # Transcribe 5-second segments for better accuracy (optimal for Whisper)
AUDIO_OVERLAP = 1  # 1 second overlap between segments
MAX_WORKERS = 3


class SegmentTreeGenerator:
    def __init__(self, video_path: str, tracking_json_path: str, tracker: str = "bytetrack", use_llava: bool = True, use_images: bool = False, yolo_stride: int = 10, blip_split: int = 1):
        self.video_path = video_path
        self.tracking_json_path = tracking_json_path
        self.tracker = tracker
        self.use_llava = use_llava
        self.use_images = use_images
        self.yolo_stride = yolo_stride
        self.blip_split = blip_split
        self.cap = None
        self.cap_lock = threading.Lock()  # Lock for thread-safe VideoCapture access
        self.blip_processor = None
        self.blip_model = None
        self.whisper_model = None
        self.video_duration = None  # Video duration in seconds
        
    def load_models(self):
        """Load BLIP model once (shared across threads)"""
        print("Loading BLIP model...")
        start = time.time()
        self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL, use_fast=True)
        
        # Configure 8-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,  # optional, needs bitsandbytes
            use_safetensors=True       # use safetensors to avoid torch.load vulnerability
        )
        print(f"BLIP model loaded in {time.time() - start:.2f}s")
        
        # Load Whisper model
        print("Loading Whisper model...")
        start = time.time()
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        print(f"Whisper model loaded in {time.time() - start:.2f}s")
        
        # Get video duration using ffprobe (lightweight, no dependencies)
        print("Getting video duration for audio extraction...")
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', self.video_path],
                capture_output=True,
                text=True,
                check=True
            )
            self.video_duration = float(result.stdout.strip())
            print(f"Video duration: {self.video_duration:.2f} seconds")
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not get video duration: {e}. Transcription may be limited.")
            self.video_duration = None
        
    def generate_tracking_data(self) -> None:
        """Generate tracking JSON if it doesn't exist"""
        if os.path.exists(self.tracking_json_path):
            return
        
        print(f"Tracking file not found: {self.tracking_json_path}")
        print(f"Generating tracking data from {self.video_path}...")
        
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Load YOLO model
        print("Loading YOLO model...")
        model = YOLO("yolo11s.pt")
        
        # Determine tracker config
        if self.tracker == "bytetrack":
            tracker_config = "bytetrack.yaml"
            tracker_name = "ByteTrack"
        elif self.tracker == "deepsort":
            tracker_config = "botsort.yaml"  # Bot-SORT is enhanced DeepSORT
            tracker_name = "Bot-SORT"
        else:
            tracker_config = "bytetrack.yaml"
            tracker_name = "ByteTrack"
        
        print(f"Running YOLO with {tracker_name} tracking (processing every {self.yolo_stride}th frame)...")
        start_time = time.time()
        
        # Run tracking with optimizations - process every Nth frame
        results = model.track(
            source=self.video_path,
            tracker=tracker_config,
            show=False,
            device=0,  # GPU 0, use 'cpu' for CPU mode
            persist=True,
            stream=True,  # Process frames as generator to avoid memory accumulation
            imgsz=640,  # Fixed input size for faster processing
            verbose=False,  # Reduce console output overhead
            vid_stride=self.yolo_stride  # Process every Nth frame
        )
        
        time_taken = time.time() - start_time
        
        # Collect results in JSON format
        output_data = {
            "video": self.video_path,
            "tracker": tracker_name,
            "time_taken_seconds": round(time_taken, 2),
            "frames": []
        }
        
        # Process results - map sequential result index to actual frame numbers
        result_index = 0
        for result in results:
            result_index += 1
            # Calculate actual frame number: frames 1, 11, 21, 31, ... (every Nth frame)
            frame_number = (result_index - 1) * self.yolo_stride + 1
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
            
            if result_index % 25 == 0:  # Print every 25 processed frames
                print(f"Processed {result_index} frames (video frames up to {frame_number})...")
        
        # Save JSON
        with open(self.tracking_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Tracking data saved to {self.tracking_json_path}")
        print(f"Total processing time: {time_taken:.2f} seconds")
        print(f"Average FPS: {frame_number / time_taken:.2f}")
    
    def load_tracking_data(self) -> Dict:
        """Load YOLO tracking JSON, generating it if it doesn't exist"""
        # Generate tracking data if it doesn't exist
        self.generate_tracking_data()
        
        print(f"Loading tracking data from {self.tracking_json_path}...")
        with open(self.tracking_json_path, 'r') as f:
            return json.load(f)
    
    def get_frame_image(self, frame_number: int) -> Image.Image:
        """Extract frame from video (thread-safe)"""
        with self.cap_lock:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.video_path)
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
    
    def process_blip(self, frame_number: int, frame_image: Image.Image) -> Dict:
        """Process single frame with BLIP"""
        start = time.time()
        inputs = self.blip_processor(frame_image, return_tensors="pt").to(self.blip_model.device)
        
        with torch.inference_mode():
            output = self.blip_model.generate(**inputs, max_new_tokens=50)
        
        caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
        processing_time = time.time() - start
        
        return {
            "description": caption,
            "blip_metadata": {
                "model": BLIP_MODEL,
                "processing_time": round(processing_time, 2)
            }
        }
    
    def process_llava(self, blip_descriptions: List[str], detection_summary: str, images: List[Image.Image] = None) -> Dict:
        """Process with LLaVA via Ollama using enhanced prompts from prompt.py
        
        Args:
            blip_descriptions: List of BLIP text descriptions
            detection_summary: Summary of object detections
            images: Optional list of PIL Images to send to LLaVA (for vision analysis)
        """
        # Build image descriptions list dynamically
        descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(blip_descriptions)])
        
        # Use enhanced prompt from prompt.py (includes technical camera/video details)
        prompt = get_llava_prompt(
            descriptions_text=descriptions_text,
            detection_summary=detection_summary,
            include_technical=True,  # Use enhanced prompts with technical details
            short=False  # Use full prompt for better quality
        )
        
        # Prepare request payload
        payload = {
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False
        }
        
        # Add images if provided and use_images is enabled
        if images and self.use_images:
            # Encode images to base64
            encoded_images = []
            for img in images:
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                encoded_images.append(img_base64)
            payload['images'] = encoded_images
        
        start = time.time()
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )
        processing_time = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            return {
                "unified_description": result.get('response', ''),
                "llava_metadata": {
                    "model": OLLAMA_MODEL,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "processing_time": round(processing_time, 2)
                }
            }
        else:
            return {
                "unified_description": f"Error: {response.status_code}",
                "llava_metadata": {
                    "model": OLLAMA_MODEL,
                    "processing_time": round(processing_time, 2)
                }
            }
    
    def transcribe_audio_segment(self, start_sec: float, end_sec: float) -> Dict:
        """Transcribe audio for a specific second of video using ffmpeg"""
        if self.whisper_model is None:
            return {
                "transcription": "",
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": "en",
                    "processing_time": 0,
                    "error": "Whisper model not initialized"
                }
            }
        
        start_time = time.time()
        temp_audio_path = None
        try:
            # Use provided start_sec and end_sec (already calculated with overlap)
            
            # Create temporary file for audio segment
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_audio_path = tmp_file.name
            
            # Use ffmpeg to extract audio segment
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', self.video_path,
                '-ss', str(start_sec),
                '-t', str(end_sec - start_sec),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # Whisper works well with 16kHz
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                temp_audio_path
            ]
            
            # Run ffmpeg (capture stderr to check for errors)
            ffmpeg_result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Validate audio file exists and has content
            if not os.path.exists(temp_audio_path):
                raise Exception("Audio file was not created by FFmpeg")
            
            file_size = os.path.getsize(temp_audio_path)
            # For 5 seconds of 16kHz mono audio, expect at least ~160KB (16kHz * 2 bytes * 5 seconds)
            # But allow smaller files in case of silence/compression
            if file_size < 500:  # Less than 500 bytes is definitely invalid for 5-second segment
                # Check stderr for clues
                stderr_msg = ffmpeg_result.stderr.decode('utf-8', errors='ignore') if ffmpeg_result.stderr else ""
                if "No audio stream" in stderr_msg or "Stream map" in stderr_msg:
                    raise Exception(f"Video has no audio track or audio extraction failed")
                raise Exception(f"Audio file too small ({file_size} bytes), likely empty or invalid")
            
            # Transcribe with Whisper (with better error handling)
            transcription_text = ""
            whisper_result = None
            try:
                whisper_result = self.whisper_model.transcribe(temp_audio_path, language="en", task="transcribe")
                transcription_text = whisper_result["text"].strip()
            except Exception as whisper_error:
                # If Whisper fails, check if it's due to empty audio
                if "reshape" in str(whisper_error).lower() or "0 elements" in str(whisper_error):
                    transcription_text = ""  # Empty audio, no transcription
                    # Continue without error - just empty transcription
                else:
                    raise  # Re-raise other Whisper errors
            
            # Clean up temporary file
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except:
                pass
            
            processing_time = time.time() - start_time
            
            return {
                "transcription": transcription_text,
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": whisper_result.get("language", "en") if whisper_result else "en",
                    "processing_time": round(processing_time, 2)
                }
            }
        except subprocess.CalledProcessError as e:
            processing_time = time.time() - start_time
            # Clean up on error
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except:
                pass
            return {
                "transcription": "",
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": "en",
                    "processing_time": round(processing_time, 2),
                    "error": f"FFmpeg error: {str(e)}"
                }
            }
        except FileNotFoundError:
            processing_time = time.time() - start_time
            return {
                "transcription": "",
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": "en",
                    "processing_time": round(processing_time, 2),
                    "error": "FFmpeg not found. Please install FFmpeg."
                }
            }
        except Exception as e:
            processing_time = time.time() - start_time
            # Clean up on error
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except:
                pass
            return {
                "transcription": "",
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": "en",
                    "processing_time": round(processing_time, 2),
                    "error": str(e)
                }
            }
    
    def group_detections(self, frames_data: List[Dict], start_frame: int, end_frame: int) -> List[Dict]:
        """Group 5 frames of detections together, deduplicate by track_id"""
        groups = []
        
        for group_idx in range(GROUPS_PER_SECOND):
            group_start = start_frame + (group_idx * FRAMES_PER_GROUP)
            group_end = min(group_start + FRAMES_PER_GROUP - 1, end_frame)
            frame_range = list(range(group_start, group_end + 1))
            
            # Collect all detections from this group
            all_detections = {}
            for frame_num in frame_range:
                if frame_num <= len(frames_data):
                    frame_data = frames_data[frame_num - 1]
                    for det in frame_data.get('detections', []):
                        track_id = det.get('track_id')
                        if track_id is not None:
                            # Deduplicate by track_id, keep highest confidence
                            if track_id not in all_detections or det['confidence'] > all_detections[track_id]['confidence']:
                                all_detections[track_id] = det.copy()
            
            # Convert to list
            group_detections = list(all_detections.values())
            unique_tracks = list(all_detections.keys())
            
            groups.append({
                "group_index": group_idx,
                "frame_range": [group_start, group_end],
                "representative_frame": group_start if group_idx in [0, 2, 5] else None,
                "detections": group_detections,
                "unique_tracks": unique_tracks,
                "total_detections": len(group_detections)
            })
        
        return groups
    
    def process_second(self, second_idx: int, frames_data: List[Dict], video_fps: float) -> Dict:
        """Process one second of video"""
        start_frame = (second_idx * FPS) + 1
        end_frame = min(start_frame + FPS - 1, len(frames_data))
        
        # Group detections
        detection_groups = self.group_detections(frames_data, start_frame, end_frame)
        
        # Get BLIP frames based on blip_split configuration
        # 1: middle frame only
        # 2: first and last frames
        # 3: first, middle, and last frames
        blip_frame_numbers = []
        middle_frame = start_frame + (end_frame - start_frame) // 2
        
        if self.blip_split == 1:
            # Middle frame only
            blip_frame_numbers = [middle_frame]
        elif self.blip_split == 2:
            # First and last frames
            blip_frame_numbers = [start_frame, end_frame]
        elif self.blip_split == 3:
            # First, middle, and last frames
            blip_frame_numbers = [start_frame, middle_frame, end_frame]
        else:
            # Default to middle if invalid value
            blip_frame_numbers = [middle_frame]
        
        # Map frames to group indices (for compatibility with existing code)
        blip_group_mapping = {}
        for frame_num in blip_frame_numbers:
            # Find which group this frame belongs to
            group_idx = -1
            for i, group in enumerate(detection_groups):
                frame_range = group["frame_range"]
                if frame_range[0] <= frame_num <= frame_range[1]:
                    group_idx = i
                    break
            # If not found in any group, use -1 for last frame or assign to closest group
            if group_idx == -1:
                if frame_num == end_frame:
                    group_idx = len(detection_groups) - 1 if detection_groups else -1
                else:
                    # Assign to closest group
                    group_idx = min(len(detection_groups) - 1, max(0, (frame_num - start_frame) // FRAMES_PER_GROUP))
            blip_group_mapping[frame_num] = group_idx
        
        # Process BLIP in parallel
        blip_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for frame_num in blip_frame_numbers:
                frame_image = self.get_frame_image(frame_num)
                if frame_image:
                    group_idx = blip_group_mapping[frame_num]
                    future = executor.submit(self.process_blip, frame_num, frame_image)
                    futures[future] = (frame_num, group_idx)
            
            for future in as_completed(futures):
                frame_num, group_idx = futures[future]
                try:
                    result = future.result()
                    blip_results.append((group_idx, frame_num, result))
                except Exception as e:
                    print(f"Error processing BLIP for frame {frame_num}: {e}")
                    blip_results.append((group_idx, frame_num, {
                        "description": f"Error: {str(e)}",
                        "blip_metadata": {"model": BLIP_MODEL, "processing_time": 0}
                    }))
        
        # Sort by group index and create blip_descriptions
        blip_results.sort(key=lambda x: x[0])
        blip_descriptions = []
        for group_idx, frame_num, result in blip_results:
            # Handle last frame (group_idx might be -1 or the last group)
            if group_idx == -1 or (group_idx >= 0 and group_idx < len(detection_groups)):
                if group_idx >= 0 and group_idx < len(detection_groups):
                    frame_range = detection_groups[group_idx]["frame_range"]
                else:
                    # Last frame - use its own frame range
                    frame_range = [end_frame, end_frame]
                
                blip_descriptions.append({
                    "group_index": group_idx if group_idx >= 0 else len(detection_groups),
                    "frame_number": frame_num,
                    "frame_range": frame_range,
                    "description": result["description"],
                    "blip_metadata": result["blip_metadata"]
                })
        
        # Create detection summary
        all_tracks = set()
        for group in detection_groups:
            all_tracks.update(group["unique_tracks"])
        detection_summary = f"Total unique tracks: {len(all_tracks)}, Groups: {len(detection_groups)}"
        
        # Process LLaVA if enabled, otherwise use BLIP descriptions
        if self.use_llava:
            blip_texts = [desc["description"] for desc in blip_descriptions]
            if not blip_texts:
                # If no BLIP descriptions, use detection summary only
                blip_texts = ["No image descriptions available"]
            
            # Collect images if use_images is enabled (use representative frames)
            images_to_send = None
            if self.use_images:
                images_to_send = []
                for desc in blip_descriptions:
                    frame_num = desc.get("frame_number")
                    if frame_num:
                        frame_image = self.get_frame_image(frame_num)
                        if frame_image:
                            images_to_send.append(frame_image)
                # If no images collected, set to None
                if not images_to_send:
                    images_to_send = None
            
            llava_result = self.process_llava(blip_texts, detection_summary, images=images_to_send)
            unified_description = llava_result["unified_description"]
            llava_metadata = llava_result["llava_metadata"]
            
            # Add note about image processing if used
            if self.use_images and images_to_send:
                llava_metadata["images_sent"] = len(images_to_send)
                llava_metadata["note"] = "LLaVA processed with both text and images"
            elif self.use_llava:
                llava_metadata["note"] = "LLaVA processed with text only"
        else:
            # Use BLIP descriptions combined as unified description
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
        
        # Build second data (transcription will be added separately)
        second_data = {
            "second": second_idx,
            "time_range": [round(second_idx, 3), round(second_idx + 0.999, 3)],
            "frame_range": [start_frame, end_frame],
            "unified_description": unified_description,
            "llava_metadata": llava_metadata,
            "blip_descriptions": blip_descriptions,
            "detection_groups": detection_groups
        }
        
        return second_data
    
    def generate(self, output_path: str):
        """Main generation function"""
        print("Starting segment tree generation...")
        start_time = time.time()
        
        # Load tracking data
        tracking_data = self.load_tracking_data()
        frames_data = tracking_data["frames"]
        video_path = tracking_data.get("video", self.video_path)
        
        # Load models
        self.load_models()
        
        # Calculate number of seconds
        total_frames = len(frames_data)
        num_seconds = (total_frames + FPS - 1) // FPS
        
        print(f"Processing {num_seconds} seconds ({total_frames} frames)...")
        
        # Process seconds in parallel
        seconds_data = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.process_second, i, frames_data, FPS): i 
                      for i in range(num_seconds)}
            
            for future in as_completed(futures):
                second_idx = futures[future]
                try:
                    result = future.result()
                    seconds_data.append(result)
                    print(f"Completed second {second_idx + 1}/{num_seconds}")
                except Exception as e:
                    print(f"Error processing second {second_idx}: {e}")
        
        # Sort by second index
        seconds_data.sort(key=lambda x: x["second"])
        
        # Generate transcriptions with 1-second overlap
        print("\nGenerating audio transcriptions with 1-second overlap...")
        transcriptions = []
        if self.video_duration and self.whisper_model:
            step = AUDIO_SEGMENT_DURATION - AUDIO_OVERLAP  # 4 seconds step (5 sec duration - 1 sec overlap)
            transcription_start = 0
            
            while transcription_start < self.video_duration:
                transcription_end = min(transcription_start + AUDIO_SEGMENT_DURATION, self.video_duration)
                
                transcription_result = self.transcribe_audio_segment(transcription_start, transcription_end)
                
                transcriptions.append({
                    "id": len(transcriptions),
                    "time_range": [round(transcription_start, 2), round(transcription_end, 2)],
                    "transcription": transcription_result["transcription"],
                    "metadata": transcription_result["transcription_metadata"]
                })
                
                print(f"Transcribed audio segment {len(transcriptions)}: {transcription_start:.1f}s - {transcription_end:.1f}s")
                
                # Move to next segment (with 1-second overlap)
                transcription_start += step
        
        # Map each second to its transcription
        # Find the transcription that best covers each second (prefer the one that starts closest)
        for second_data in seconds_data:
            second_idx = second_data["second"]
            second_time = second_idx + 0.5  # Middle of the second
            
            # Find transcription that covers this second
            best_transcription_id = None
            best_distance = float('inf')
            
            for trans in transcriptions:
                time_range = trans["time_range"]
                if time_range[0] <= second_time <= time_range[1]:
                    # This transcription covers the second
                    distance = abs(time_range[0] - second_time)
                    if distance < best_distance:
                        best_distance = distance
                        best_transcription_id = trans["id"]
            
            # If no exact match, find the closest one
            if best_transcription_id is None:
                for trans in transcriptions:
                    time_range = trans["time_range"]
                    # Check if second is near this transcription
                    if abs(time_range[0] - second_time) < 3:  # Within 3 seconds
                        distance = abs(time_range[0] - second_time)
                        if distance < best_distance:
                            best_distance = distance
                            best_transcription_id = trans["id"]
            
            second_data["transcription_id"] = best_transcription_id
        
        # Build output structure
        output_data = {
            "video": video_path,
            "fps": FPS,
            "tracker": self.tracker,
            "seconds": seconds_data,
            "transcriptions": transcriptions
        }
        
        # Save output
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\nSegment tree generated in {total_time:.2f}s")
        print(f"Output saved to {output_path}")
        
        # Clean up resources
        if self.cap:
            self.cap.release()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate segment tree from YOLO tracking data")
    parser.add_argument("--video", default="camp.mp4", help="Video file path")
    parser.add_argument("--tracking", default="camp_yolo_bytetrack.json", help="Tracking JSON file")
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "deepsort"], help="Tracker name")
    parser.add_argument("--output", default="camp_segment_tree.json", help="Output JSON file")
    parser.add_argument("--no-llava", action="store_true", default=False,
                       help="Disable LLaVA processing (default: LLaVA enabled, uses enhanced prompts with technical details)")
    parser.add_argument("--use-images", action="store_true", default=False,
                       help="Send actual frame images to LLaVA for vision analysis (default: False, text-only)")
    parser.add_argument("--yolo-stride", type=int, default=10,
                       help="Process every Nth frame with YOLO (default: 10)")
    parser.add_argument("--blip-split", type=int, default=1, choices=[1, 2, 3],
                       help="Number of frames per second for BLIP: 1=middle, 2=first+last, 3=first+middle+last (default: 1)")
    
    args = parser.parse_args()
    
    # Default is True (LLaVA enabled), unless --no-llava flag is used
    use_llava = not args.no_llava
    use_images = args.use_images
    
    generator = SegmentTreeGenerator(args.video, args.tracking, args.tracker, use_llava=use_llava, use_images=use_images, yolo_stride=args.yolo_stride, blip_split=args.blip_split)
    
    if use_llava:
        if use_images:
            print("LLaVA processing: ENABLED (text + images, using enhanced prompts with technical details)")
        else:
            print("LLaVA processing: ENABLED (text only, using enhanced prompts with technical details)")
    else:
        print("LLaVA processing: DISABLED (using BLIP descriptions only)")
    
    generator.generate(args.output)


if __name__ == "__main__":
    main()

