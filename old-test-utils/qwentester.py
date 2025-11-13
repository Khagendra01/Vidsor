"""
Process camp_5min.mp4 with Qwen2-VL-7B-Instruct model.
"""

import json
import sys
import os
import time
import cv2
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Default to local model path
default_model_path = r"C:\models\Qwen2VL7B"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)


def process_image(image_path: str, processor, model) -> tuple:
    """Process a single image with Qwen."""
    img = Image.open(image_path).convert("RGB")
    
    # Qwen2-VL expects a list of conversations
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    
    start = time.time()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Use processor to prepare inputs with images
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=200)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    proc_time = time.time() - start
    
    return caption, proc_time


def process_batch(image_paths: List[str], processor, model) -> List[tuple]:
    """Process multiple images in a batch with Qwen."""
    # Load all images
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    start = time.time()
    
    # Process all images in the batch
    captions = []
    for img in images:
        # Qwen2-VL expects a list of conversations
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Use processor to prepare inputs with images
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        output = model.generate(**inputs, max_new_tokens=200)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append(caption)
    
    total_time = time.time() - start
    avg_time = total_time / len(image_paths)
    
    return [(caption, avg_time) for caption in captions]


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process frames with Qwen2-VL")
    parser.add_argument("--video", type=str, default="camp_5min.mp4", help="Video path")
    parser.add_argument("--output", type=str, default="qwen_results.json", help="Output JSON path")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval (every Nth frame)")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to process")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing (default: 8)")
    parser.add_argument("--model-path", type=str, default=default_model_path, help="Path to local model directory (default: C:\\models\\Qwen2VL7B)")
    args = parser.parse_args()
    
    video_path = Path(__file__).parent / args.video
    output_path = Path(__file__).parent / args.output
    
    print("="*60)
    print("QWEN2-VL PROCESSING")
    print("="*60)
    print(f"\nVideo: {video_path}")
    print(f"Output: {output_path}")
    print(f"Frame interval: Every {args.interval} frames")
    print(f"Max frames: {args.max_frames}")
    print(f"Batch size: {args.batch_size}")
    
    # Load Qwen model from local path
    model_path = Path(args.model_path)
    print(f"\nLoading Qwen2-VL model from: {model_path}")
    start = time.time()
    
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_path),
        device_map="auto",
        quantization_config=quant_config,
        local_files_only=True,
        trust_remote_code=True
    )
    
    load_time = time.time() - start
    print(f"Qwen2-VL model loaded in {load_time:.2f}s")
    
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
    frame_numbers = list(range(1, min(total_frames + 1, args.max_frames * args.interval + 1), args.interval))
    if args.max_frames:
        frame_numbers = frame_numbers[:args.max_frames]
    
    print(f"\nProcessing {len(frame_numbers)} frames...")
    
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
            batch_results = process_batch(batch_paths, processor, model)
            
            for frame_num, (caption, proc_time) in zip(batch_frame_nums, batch_results):
                results.append({
                    "frame_number": frame_num,
                    "second": round((frame_num - 1) / fps, 3),
                    "caption": caption,
                    "processing_time": round(proc_time, 3)
                })
                print(f"  Frame {frame_num}: {proc_time:.3f}s - {caption[:50]}...")
        except Exception as e:
            print(f"  ⚠️  Batch processing failed: {e}")
            print(f"  Error type: {type(e).__name__}")
            traceback.print_exc()
            print(f"  Falling back to individual processing...")
            # Fallback to individual processing
            for frame_num, path in zip(batch_frame_nums, batch_paths):
                try:
                    caption, proc_time = process_image(path, processor, model)
                    results.append({
                        "frame_number": frame_num,
                        "second": round((frame_num - 1) / fps, 3),
                        "caption": caption,
                        "processing_time": round(proc_time, 3)
                    })
                    print(f"  Frame {frame_num}: {proc_time:.3f}s - {caption[:50]}...")
                except Exception as e2:
                    print(f"  ❌ Failed to process frame {frame_num}: {e2}")
                    print(f"  Error type: {type(e2).__name__}")
                    traceback.print_exc()
    
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
        "model": str(model_path),
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
