"""
Standalone script to process frames with Florence-2-base model only.
Matches the structure of process_blip_only.py for comparison.
"""

import json
import sys
import os
import time
import re
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def process_image(image_path: str, model, processor) -> tuple:
    """Process a single image with Florence-2."""
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    image = Image.open(image_path).convert("RGB")
    
    prompt = "<CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Move inputs to device and convert pixel_values to model dtype
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
    
    start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,  # Use greedy decoding to avoid past_key_values issues
            use_cache=False,  # Disable cache to avoid past_key_values issues
        )
    proc_time = time.time() - start
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    caption_result = processor.post_process_generation(generated_text, task="<CAPTION>")
    
    # Extract caption string - post_process_generation may return dict or string
    if isinstance(caption_result, dict):
        # Try different possible keys
        caption = caption_result.get("<CAPTION>") or caption_result.get("caption") or str(caption_result)
    elif isinstance(caption_result, str):
        caption = caption_result
    elif caption_result is None:
        caption = ""
    else:
        # Fallback: try to get string representation
        caption = str(caption_result)
    
    # Ensure caption is a string
    if not isinstance(caption, str):
        caption = str(caption) if caption is not None else ""
    
    # Clean up special tokens (pad tokens, etc.)
    caption = caption.replace("<pad>", "").strip()
    # Remove any remaining special tokens that might be present
    caption = re.sub(r'<[^>]+>', '', caption).strip()
    # Remove trailing newlines
    caption = caption.rstrip('\n\r')
    
    return caption, proc_time


def process_batch(image_paths: List[str], model, processor) -> List[tuple]:
    """Process multiple images in a batch with Florence-2."""
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Load all images
    images = [Image.open(path).convert("RGB") for path in image_paths]
    prompt = "<CAPTION>"
    
    # Process batch
    inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt", padding=True)
    
    # Move inputs to device and convert pixel_values to model dtype
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
    
    start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,  # Use greedy decoding to avoid past_key_values issues
            use_cache=False,  # Disable cache to avoid past_key_values issues
        )
    total_time = time.time() - start
    avg_time = total_time / len(image_paths)
    
    # Decode all captions
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    caption_results = [processor.post_process_generation(text, task="<CAPTION>") for text in generated_texts]
    
    # Extract caption strings - post_process_generation may return dict or string
    captions = []
    for caption_result in caption_results:
        if isinstance(caption_result, dict):
            # Try different possible keys
            caption = caption_result.get("<CAPTION>") or caption_result.get("caption") or str(caption_result)
        elif isinstance(caption_result, str):
            caption = caption_result
        elif caption_result is None:
            caption = ""
        else:
            caption = str(caption_result)
        
        # Ensure caption is a string
        if not isinstance(caption, str):
            caption = str(caption) if caption is not None else ""
        
        # Clean up special tokens (pad tokens, etc.)
        caption = caption.replace("<pad>", "").strip()
        # Remove any remaining special tokens that might be present
        caption = re.sub(r'<[^>]+>', '', caption).strip()
        # Remove trailing newlines
        caption = caption.rstrip('\n\r')
        
        captions.append(caption)
    
    return [(caption, avg_time) for caption in captions]


def main():
    """Main function."""
    import argparse
    import cv2
    import tempfile
    import shutil
    
    parser = argparse.ArgumentParser(description="Process frames with Florence-2")
    parser.add_argument("--video", type=str, default="camp_5min.mp4", help="Video path")
    parser.add_argument("--output", type=str, default="florence_results.json", help="Output JSON path")
    parser.add_argument("--interval", type=int, default=10, help="Frame interval (every Nth frame)")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to process (default: 300, matches BLIP)")
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size for processing (default: 48, matches extractor config)")
    args = parser.parse_args()
    
    video_path = Path(__file__).parent / args.video
    output_path = Path(__file__).parent / args.output
    
    print("="*60)
    print("FLORENCE-2 PROCESSING")
    print("="*60)
    print(f"\nVideo: {video_path}")
    print(f"Output: {output_path}")
    print(f"Frame interval: Every {args.interval} frames")
    print(f"Max frames: {args.max_frames}")
    print(f"Batch size: {args.batch_size}")
    
    # Load Florence-2-base model
    model_name = "microsoft/Florence-2-base"
    print("\nLoading Florence-2-base model...")
    start = time.time()
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Try loading without quantization first (quantization may cause generation issues)
    try:
        # Try without quantization first - Florence-2 may have issues with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"  # Avoid SDPA compatibility issues
        )
        print("   Loaded without quantization")
    except Exception as e:
        print(f"⚠️  Failed to load with device_map: {e}")
        print("   Trying basic loading...")
        try:
            # Fallback: basic loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.to(torch.device("cuda")).to(torch.float16)
            else:
                model = model.to(torch.float16)
            print("   Loaded with basic configuration")
        except Exception as e2:
            print(f"⚠️  Failed to load model: {e2}")
            raise
    
    load_time = time.time() - start
    print(f"Florence-2-base model loaded in {load_time:.2f}s")
    
    # Check which device is being used
    device = next(model.parameters()).device
    print(f"\n🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"   CUDA available: Yes")
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        if device.type == 'cuda':
            print(f"   Using GPU: {torch.cuda.get_device_name(device)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        else:
            print(f"   ⚠️  Model loaded on CPU")
    else:
        print(f"   CUDA available: No - Running on CPU")
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    print(f"\nVideo info:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    
    # Determine frames to process
    if args.max_frames:
        # Limit to max_frames if specified
        frame_numbers = list(range(1, min(total_frames + 1, args.max_frames * args.interval + 1), args.interval))
        frame_numbers = frame_numbers[:args.max_frames]
    else:
        # Process all frames at the specified interval
        frame_numbers = list(range(1, total_frames + 1, args.interval))
    
    print(f"\nProcessing {len(frame_numbers)} frames in batches of {args.batch_size}...")
    
    # Extract all frames first and save to temp files
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    valid_frame_numbers = []
    
    cap = cv2.VideoCapture(str(video_path))
    print("Extracting frames...")
    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Could not read frame {frame_num}")
            continue
        
        # Convert to PIL Image and save
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        temp_path = os.path.join(temp_dir, f"frame_{frame_num}.png")
        image.save(temp_path)
        frame_paths.append(temp_path)
        valid_frame_numbers.append(frame_num)
    cap.release()
    print(f"Extracted {len(frame_paths)} frames")
    
    # Process frames in batches
    results = []
    overall_start = time.time()
    
    for batch_idx in range(0, len(frame_paths), args.batch_size):
        batch_paths = frame_paths[batch_idx:batch_idx + args.batch_size]
        batch_frame_nums = valid_frame_numbers[batch_idx:batch_idx + args.batch_size]
        
        print(f"\nProcessing batch {batch_idx // args.batch_size + 1}/{(len(frame_paths) + args.batch_size - 1) // args.batch_size} ({len(batch_paths)} frames)...")
        
        try:
            batch_results = process_batch(batch_paths, model, processor)
            
            for frame_num, (caption, proc_time) in zip(batch_frame_nums, batch_results):
                results.append({
                    "frame_number": frame_num,
                    "second": round((frame_num - 1) / fps, 3),
                    "caption": caption,
                    "processing_time": round(proc_time, 3)
                })
                caption_str = str(caption) if caption else ""
                print(f"  Frame {frame_num}: {proc_time:.3f}s - {caption_str[:50]}...")
        except Exception as e:
            print(f"  ⚠️  Batch processing failed: {e}")
            print(f"  Falling back to individual processing...")
            # Fallback to individual processing
            for frame_num, path in zip(batch_frame_nums, batch_paths):
                try:
                    caption, proc_time = process_image(path, model, processor)
                    results.append({
                        "frame_number": frame_num,
                        "second": round((frame_num - 1) / fps, 3),
                        "caption": caption,
                        "processing_time": round(proc_time, 3)
                    })
                    caption_str = str(caption) if caption else ""
                    print(f"  Frame {frame_num}: {proc_time:.3f}s - {caption_str[:50]}...")
                except Exception as e2:
                    print(f"  ❌ Failed to process frame {frame_num}: {e2}")
    
    # Cleanup temp files
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    total_time = time.time() - overall_start
    
    # Calculate statistics
    times = [r["processing_time"] for r in results]
    stats = {
        "total_frames": len(results),
        "total_processing_time": round(total_time, 2),
        "avg_time_per_frame": round(sum(times) / len(times) if times else 0, 3),
        "min_time": round(min(times) if times else 0, 3),
        "max_time": round(max(times) if times else 0, 3),
        "fps": round(len(results) / total_time, 2) if total_time > 0 else 0
    }
    
    # Save results
    output_data = {
        "video_path": str(video_path),
        "model": model_name,
        "test_config": {
            "frame_interval": args.interval,
            "max_frames": args.max_frames,
            "batch_size": args.batch_size,
            "frames_processed": valid_frame_numbers,
            "num_frames_processed": len(valid_frame_numbers)
        },
        "statistics": stats,
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"\nStatistics:")
    print(f"  Frames processed: {stats['total_frames']}")
    print(f"  Avg time per frame: {stats['avg_time_per_frame']:.3f}s")
    print(f"  Processing speed: {stats['fps']} frames/sec")
    print(f"  Min/Max time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")


if __name__ == "__main__":
    main()

