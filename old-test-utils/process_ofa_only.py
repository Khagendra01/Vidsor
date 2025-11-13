"""
Standalone script to process frames with OFA large model only.
No BLIP, no conflicts - just OFA.
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from torchvision import transforms
import torch

# Disable ETag check for OFA transformers
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Try to disable ETag requirement
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

# Add OFA transformers to path BEFORE any other imports
parent_dir = Path(__file__).parent.parent
ofa_transformers_path = parent_dir / "OFA" / "transformers" / "src"
if ofa_transformers_path.exists():
    sys.path.insert(0, str(ofa_transformers_path))

from transformers import OFATokenizer, OFAModel

# OFA image preprocessing setup
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def process_image(image_path: str, model, tokenizer) -> tuple:
    """Process a single image with OFA."""
    device = next(model.parameters()).device
    # Get model dtype (should be float16 if loaded with dtype=torch.float16)
    model_dtype = next(model.parameters()).dtype
    
    image = Image.open(image_path).convert("RGB")
    patch_img = patch_resize_transform(image).unsqueeze(0).to(device).to(model_dtype)
    txt = " what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids.to(device)
    
    start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            inputs,
            patch_images=patch_img,
            num_beams=5,
            no_repeat_ngram_size=3,
            max_new_tokens=50
        )
    proc_time = time.time() - start
    
    caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption, proc_time


def process_batch(image_paths: List[str], model, tokenizer) -> List[tuple]:
    """Process multiple images in a batch with OFA."""
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Load and preprocess all images
    patch_imgs = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        patch_img = patch_resize_transform(image).to(model_dtype)
        patch_imgs.append(patch_img)
    
    # Stack into batch tensor
    patch_imgs_batch = torch.stack(patch_imgs).to(device)
    
    # Prepare text inputs (same prompt for all)
    txt = " what does the image describe?"
    inputs = tokenizer([txt] * len(image_paths), return_tensors="pt", padding=True).input_ids.to(device)
    
    start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            inputs,
            patch_images=patch_imgs_batch,
            num_beams=5,
            no_repeat_ngram_size=3,
            max_new_tokens=50
        )
    total_time = time.time() - start
    avg_time = total_time / len(image_paths)
    
    captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]
    
    return [(caption, avg_time) for caption in captions]


def main():
    """Main function."""
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser(description="Process frames with OFA")
    parser.add_argument("--video", type=str, default="camp_5min.mp4", help="Video path")
    parser.add_argument("--output", type=str, default="ofa_results.json", help="Output JSON path")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval (every Nth frame)")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to process")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing (default: 16, BLIP uses 48 but OFA may need smaller batches)")
    args = parser.parse_args()
    
    video_path = Path(__file__).parent / args.video
    output_path = Path(__file__).parent / args.output
    
    print("="*60)
    print("OFA LARGE PROCESSING")
    print("="*60)
    print(f"\nVideo: {video_path}")
    print(f"Output: {output_path}")
    print(f"Frame interval: Every {args.interval} frames")
    print(f"Max frames: {args.max_frames}")
    print(f"Batch size: {args.batch_size}")
    
    # Load OFA model
    model_name = "OFA-Sys/OFA-large-caption"
    print("\nLoading OFA model...")
    start = time.time()
    
    # Download files using huggingface_hub first to avoid ETag issues
    downloaded_path = None
    try:
        from huggingface_hub import snapshot_download
        print("📥 Downloading model files using huggingface_hub...")
        downloaded_path = snapshot_download(repo_id=model_name, local_files_only=False)
        print(f"✅ Files downloaded successfully to: {downloaded_path}")
        files_downloaded = True
    except ImportError:
        print("⚠️  huggingface_hub not available")
        print("   Installing huggingface_hub is recommended: pip install huggingface_hub")
        files_downloaded = False
    except Exception as e:
        print(f"⚠️  Could not pre-download files: {e}")
        files_downloaded = False
    
    # Load tokenizer - use downloaded path if available
    if files_downloaded and downloaded_path:
        print("📂 Loading tokenizer from downloaded path...")
        try:
            tokenizer = OFATokenizer.from_pretrained(
                downloaded_path,  # Use the actual downloaded path
                trust_remote_code=True
            )
        except Exception as e:
            print(f"⚠️  Failed to load from downloaded path: {e}")
            print("   Trying to load from model name (should use cache)...")
            try:
                tokenizer = OFATokenizer.from_pretrained(
                    model_name,
                    local_files_only=False,  # Allow it to use cache
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"❌ Failed to load tokenizer: {e2}")
                sys.exit(1)
    else:
        # Without huggingface_hub, OFA transformers will fail with ETag error
        print("\n❌ ERROR: huggingface_hub is required to download OFA models")
        print("   The OFA transformers fork requires ETag validation which needs huggingface_hub.")
        print("\n   Please install it:")
        print("   pip install huggingface_hub")
        print("\n   Then run this script again.")
        sys.exit(1)
    
    # Load model - use downloaded path if available
    # Force GPU usage if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        model_kwargs = {
            "dtype": torch.float16,
            "torch_dtype": torch.float16,
            "device_map": "cuda:0",  # Force GPU
            "use_safetensors": True,
            "use_cache": False
        }
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU")
        model_kwargs = {
            "dtype": torch.float16,
            "torch_dtype": torch.float16,
            "device_map": "cpu",
            "use_safetensors": True,
            "use_cache": False
        }
    
    if files_downloaded and downloaded_path:
        print("📂 Loading model from downloaded path...")
        model = OFAModel.from_pretrained(downloaded_path, **model_kwargs)
    else:
        model_kwargs["local_files_only"] = False
        model = OFAModel.from_pretrained(model_name, **model_kwargs)
    
    # Ensure model is on the correct device
    if torch.cuda.is_available() and next(model.parameters()).device.type != 'cuda':
        print("⚠️  Model not on GPU, moving to GPU...")
        model = model.to(device)
    
    load_time = time.time() - start
    print(f"OFA model loaded in {load_time:.2f}s")
    
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
            print(f"   ⚠️  Model loaded on CPU (device_map='auto' may have offloaded to CPU)")
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
    
    print(f"\nProcessing {len(frame_numbers)} frames in batches of {args.batch_size}...")
    
    # Extract all frames first and save to temp files
    import tempfile
    import os
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
            batch_results = process_batch(batch_paths, model, tokenizer)
            
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
            print(f"  Falling back to individual processing...")
            # Fallback to individual processing
            for frame_num, path in zip(batch_frame_nums, batch_paths):
                try:
                    caption, proc_time = process_image(path, model, tokenizer)
                    results.append({
                        "frame_number": frame_num,
                        "second": round((frame_num - 1) / fps, 3),
                        "caption": caption,
                        "processing_time": round(proc_time, 3)
                    })
                    print(f"  Frame {frame_num}: {proc_time:.3f}s - {caption[:50]}...")
                except Exception as e2:
                    print(f"  ❌ Failed to process frame {frame_num}: {e2}")
    
    # Cleanup temp files
    import shutil
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

