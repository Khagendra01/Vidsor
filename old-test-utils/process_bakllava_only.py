"""
Parallel processing with vision models (qwen3-vl:4b, bakllava, etc.) via Ollama for camp_5min.mp4.
Processes images individually in parallel (Ollama doesn't support true batch processing like BLIP).
Uses max_workers=16 for optimal performance on systems with available GPU resources.

CONFIGURATION:
- Model can be changed via OLLAMA_MODEL_OVERRIDE (default: "qwen3-vl:4b")
- Set to None to use config default, or specify any Ollama vision model name

IMPROVEMENTS FOR BETTER RESULTS:
1. Optimized prompt that explicitly states it's a SINGLE frame to prevent hallucinations about multiple images
2. Added API parameters (temperature=0.2, top_p=0.9) for more deterministic, less hallucinatory outputs
3. Structured prompt that guides the model to focus on actual visual content
4. Clear instructions to avoid mentioning batches or multiple images
"""

import json
import io
import sys
import time
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor.utils.video_utils import VideoUtils
from extractor.config import FPS, OLLAMA_URL, OLLAMA_MODEL

# Model configuration - can be overridden
# Default uses config, but you can set a specific model here
OLLAMA_MODEL_OVERRIDE = "qwen2.5:7b"  # Set to None to use config default, or specify model name

# Optimized prompt for vision models to prevent hallucinations about multiple images
# This prompt explicitly states it's a single frame and guides the model to focus on actual content
VISION_MODEL_PROMPT = """You are analyzing a single video frame. Describe what you see in this ONE image.

Focus on:
- Objects, people, and actions visible in the scene
- The setting and environment
- Spatial relationships and positions
- Any notable visual details

Important: This is ONE single frame from a video. Do not mention multiple images or batches. Describe only what is visible in this single frame."""


def process_with_vision_model(image, ollama_url: str = OLLAMA_URL, ollama_model: str = None) -> Tuple[str, float]:
    """
    Process a SINGLE image with a vision model via Ollama (e.g., qwen3-vl:4b, bakllava, etc.).
    
    NOTE: This does NOT do batch processing. Each image is sent individually.
    Ollama vision models process one image per API call.
    
    Args:
        image: PIL Image (single image, not a batch)
        ollama_url: Ollama API URL (default: http://localhost:11434/api/generate)
        ollama_model: Ollama model name (default: uses OLLAMA_MODEL_OVERRIDE or config)
        
    Returns:
        Tuple of (caption, processing_time_in_seconds)
    """
    # Use override if set, otherwise use config default
    if ollama_model is None:
        ollama_model = OLLAMA_MODEL_OVERRIDE if OLLAMA_MODEL_OVERRIDE else OLLAMA_MODEL
    
    # Encode SINGLE image to base64
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Use optimized prompt that explicitly states it's a single frame
    # This helps prevent hallucinations about multiple images (e.g., "32 images", "16 images")
    
    # Prepare request - NOTE: 'images' is a list but contains only ONE image
    # This is NOT batch processing - each API call processes exactly one image
    payload = {
        'model': ollama_model,  # Vision model from Ollama (e.g., qwen3-vl:4b, bakllava)
        'prompt': VISION_MODEL_PROMPT,
        'images': [img_base64],  # Single image in list (Ollama API format)
        'stream': False,
        # API parameters for better, more deterministic results
        # These settings reduce hallucinations and make outputs more consistent
        'options': {
            'temperature': 0.2,  # Lower temperature = more deterministic, less creative/hallucinatory
            'top_p': 0.9,  # Nucleus sampling - focus on most likely tokens
            'num_predict': 200,  # Max tokens in response (reasonable for descriptions)
        }
    }
    
    # Track processing time
    start = time.time()
    try:
        response = requests.post(ollama_url, json=payload, timeout=60)
        processing_time = time.time() - start  # Time taken for this single image
        
        if response.status_code == 200:
            result = response.json()
            caption = result.get('response', '').strip()
            return caption, processing_time
        else:
            return f"Error: {response.status_code}", processing_time
    except Exception as e:
        processing_time = time.time() - start
        return f"Error: {str(e)}", processing_time


def process_frames_parallel(
    frame_numbers: List[int],
    frame_images: List,
    ollama_url: str,
    ollama_model: str,
    max_workers: int = 2
) -> List[Dict]:
    """
    Process frames with a vision model in parallel (individual API calls per image).
    
    IMPORTANT: This does NOT do batch processing. Each image is processed individually
    via separate API calls to Ollama. The parallelization is achieved by sending multiple
    individual requests concurrently using ThreadPoolExecutor.
    
    Note: Ollama vision models do NOT support true batch processing like BLIP.
    When multiple images are sent in one request, they're treated as context for a
    single response, not as separate items to process individually. Therefore, we
    process each image individually but in parallel using workers.
    
    Args:
        frame_numbers: List of frame numbers
        frame_images: List of PIL Images (same order as frame_numbers)
        ollama_url: Ollama API URL (default: http://localhost:11434/api/generate)
        ollama_model: Ollama model name (e.g., "qwen3-vl:4b", "bakllava")
        max_workers: Max parallel workers for concurrent individual requests
        
    Returns:
        List of result dictionaries with frame_number, caption, and processing_time
    """
    results = []
    
    print(f"\nProcessing {len(frame_images)} frames with {ollama_model} in parallel (max_workers: {max_workers})...")
    
    def process_frame(frame_num, image):
        caption, proc_time = process_with_vision_model(image, ollama_url, ollama_model)
        return {
            "frame_number": frame_num,
            "caption": caption,
            "processing_time": proc_time
        }
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_frame, frame_num, image): frame_num
            for frame_num, image in zip(frame_numbers, frame_images)
        }
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"  Processed {completed}/{len(frame_images)} frames...")
    
    # Sort by frame number
    results.sort(key=lambda x: x["frame_number"])
    
    print(f"Vision model processing complete. Processed {len(results)} frames.")
    
    # Format results
    formatted_results = []
    for r in results:
        formatted_results.append({
            "frame_number": r["frame_number"],
            "second": round((r["frame_number"] - 1) / FPS, 3),
            "vision_model": {
                "caption": r["caption"],
                "processing_time": round(r["processing_time"], 3)
            }
        })
    
    return formatted_results


def main():
    """Main processing function."""
    # Configuration - paths relative to script location
    script_dir = Path(__file__).parent
    video_path = str(script_dir / "camp_5min.mp4")
    output_path = str(script_dir / "bakllava_only_results.json")
    
    # Test configuration (same as previous comparison)
    test_interval = 30  # Test every Nth frame (30 frames = 1 second at 30fps)
    max_frames_to_test = 100  # Maximum number of frames to test (or None for all)
    max_workers = 16  # Maximum workers for I/O-bound tasks (GPU is free, system has 16 threads)
    
    # Determine which model to use
    model_to_use = OLLAMA_MODEL_OVERRIDE if OLLAMA_MODEL_OVERRIDE else OLLAMA_MODEL
    
    print("="*60)
    print("VISION MODEL PARALLEL PROCESSING")
    print("="*60)
    print(f"\nModel: {model_to_use} via Ollama")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Ollama Model: {model_to_use}")
    print(f"\nVideo: {video_path}")
    print(f"Test interval: Every {test_interval} frames")
    print(f"Max frames to test: {max_frames_to_test}")
    print(f"Max workers: {max_workers} (concurrent individual requests)")
    print("\nIMPORTANT: This does NOT do batch processing.")
    print("Each image is processed individually via separate API calls.")
    print("Parallelization is achieved through concurrent requests using ThreadPoolExecutor.")
    print("Processing time is tracked per individual image request.")
    
    # Load video
    video_utils = VideoUtils(video_path)
    video_duration = video_utils.get_duration()
    
    if video_duration:
        num_frames = int(video_duration * FPS)
        print(f"Video duration: {video_duration:.2f}s ({num_frames} frames)")
    else:
        # Estimate from video file
        import cv2
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"Estimated frames: {num_frames}")
    
    # Determine which frames to test
    all_frames = list(range(1, min(num_frames + 1, (max_frames_to_test or num_frames) * test_interval + 1), test_interval))
    if max_frames_to_test:
        all_frames = all_frames[:max_frames_to_test]
    
    print(f"Testing {len(all_frames)} frames: {all_frames[:10]}{'...' if len(all_frames) > 10 else ''}")
    
    # Extract all frames
    print(f"\nExtracting {len(all_frames)} frames from video...")
    frame_images = []
    valid_frame_numbers = []
    
    for frame_num in all_frames:
        frame_image = video_utils.get_frame(frame_num)
        if frame_image:
            frame_images.append(frame_image)
            valid_frame_numbers.append(frame_num)
        else:
            print(f"  Warning: Could not extract frame {frame_num}")
    
    print(f"Extracted {len(frame_images)} frames successfully")
    
    # Process frames
    overall_start = time.time()
    
    results = process_frames_parallel(
        frame_numbers=valid_frame_numbers,
        frame_images=frame_images,
        ollama_url=OLLAMA_URL,
        ollama_model=model_to_use,
        max_workers=max_workers
    )
    
    total_time = time.time() - overall_start
    
    # Calculate statistics
    vision_model_times = [r["vision_model"]["processing_time"] for r in results]
    
    stats = {
        "total_frames": len(results),
        "total_processing_time": round(total_time, 2),
        "vision_model": {
            "avg_time_per_frame": round(sum(vision_model_times) / len(vision_model_times) if vision_model_times else 0, 3),
            "total_time": round(sum(vision_model_times), 2),
            "min_time": round(min(vision_model_times) if vision_model_times else 0, 3),
            "max_time": round(max(vision_model_times) if vision_model_times else 0, 3)
        }
    }
    
    # Prepare output
    output_data = {
        "video_path": video_path,
        "model_info": {
            "model": model_to_use,
            "source": "Ollama",
            "ollama_url": OLLAMA_URL,
            "ollama_model": model_to_use,
            "processing_method": "individual_requests_parallel",  # NOT batch processing
            "note": "Each image is processed individually via separate API calls. Parallelization achieved through concurrent requests."
        },
        "test_config": {
            "test_interval": test_interval,
            "max_frames_to_test": max_frames_to_test,
            "max_workers": max_workers,
            "frames_tested": valid_frame_numbers,
            "num_frames_tested": len(valid_frame_numbers)
        },
        "statistics": stats,
        "results": results
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    print(f"Total processing time: {total_time:.2f}s")
    
    # Print summary
    print("\nPerformance Statistics:")
    print(f"  {model_to_use}:")
    print(f"    Avg time per frame: {stats['vision_model']['avg_time_per_frame']:.3f}s")
    print(f"    Total time: {stats['vision_model']['total_time']:.2f}s")
    print(f"    Min/Max: {stats['vision_model']['min_time']:.3f}s / {stats['vision_model']['max_time']:.3f}s")
    print(f"    Frames processed: {stats['total_frames']}")
    
    # Cleanup
    video_utils.release()


if __name__ == "__main__":
    main()

