"""
Standalone script to process frames with OpenFlamingo-3B model only.
Uses 8-bit quantization for memory efficiency.
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class OpenFlamingoLoader:
    """Manages OpenFlamingo model loading and inference with 8-bit quantization."""
    
    def __init__(self, model_name: str = "openflamingo/OpenFlamingo-3B-vitl-mpt1b"):
        """
        Initialize OpenFlamingo loader.
        
        Args:
            model_name: Model name from HuggingFace (default: OpenFlamingo-3B-vitl-mpt1b)
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load(self):
        """Load OpenFlamingo model with 8-bit quantization on the language model only."""
        if self._loaded:
            return
        
        print(f"Loading OpenFlamingo model...")
        start = time.time()
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from open_flamingo import create_model_and_transforms
        except ImportError as e:
            raise ImportError(
                "Required libraries not found. Install with:\n"
                "  pip install open-flamingo transformers"
            ) from e
        
        # Check if 8-bit quantization is available
        use_8bit = False
        quant_config = None
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            # Check torch version compatibility
            torch_version = torch.__version__
            print(f"  Detected torch version: {torch_version}")
            
            # bitsandbytes 0.48+ requires torch >= 2.3, but open-flamingo requires torch 2.0.1
            # Try to use 8-bit anyway, it might work with older bitsandbytes
            try:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                use_8bit = True
                print("  Attempting 8-bit quantization...")
            except Exception as e:
                print(f"  Warning: 8-bit quantization not available: {e}")
                print("  Falling back to FP16 (full precision)")
                use_8bit = False
        except ImportError:
            print("  Warning: bitsandbytes not available or incompatible")
            print("  Falling back to FP16 (full precision)")
            use_8bit = False
        
        # Step 2: Load the language model
        lm_path = "anas-awadalla/mpt-1b-redpajama-200b"
        print(f"  Loading language model ({'8-bit' if use_8bit else 'FP16'})...")
        
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16
        }
        if use_8bit and quant_config:
            load_kwargs["quantization_config"] = quant_config
        
        try:
            lm = AutoModelForCausalLM.from_pretrained(lm_path, **load_kwargs)
        except Exception as e:
            if use_8bit:
                print(f"  Error loading with 8-bit quantization: {e}")
                print("  Retrying with FP16...")
                load_kwargs.pop("quantization_config", None)
                lm = AutoModelForCausalLM.from_pretrained(lm_path, **load_kwargs)
                use_8bit = False
            else:
                raise
        
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path)
        
        # Step 3: Load the Flamingo architecture (this part cannot be quantized, but it's small)
        print("  Creating Flamingo architecture...")
        self.model, self.image_processor, _ = create_model_and_transforms(
            clip_vision_encoder_path="openai/clip-vit-large-patch14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=None,  # We load LM manually
            tokenizer_path=None,     # We load tokenizer manually
            cross_attn_every_n_layers=1,
        )
        
        # Step 4: Load the Flamingo checkpoint
        print("  Loading Flamingo checkpoint...")
        from huggingface_hub import hf_hub_download
        # Use the base model name for checkpoint (3B version)
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
            "checkpoint.pt"
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
        
        # Step 5: Attach the quantized LM to the Flamingo model
        print("  Attaching quantized language model...")
        self.model.lang_encoder = lm
        self.model.tokenizer = self.tokenizer
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("  Warning: CUDA not available, using CPU (will be slow)")
        self.model = self.model.to(device)
        self.model.eval()
        
        elapsed = time.time() - start
        print(f"OpenFlamingo model loaded in {elapsed:.2f}s")
        print(f"  Language model: {'8-bit quantized' if use_8bit else 'FP16'}")
        print(f"  Vision encoder: FP16")
        print(f"  Device: {device}")
        self._loaded = True
    
    def caption(self, image: Image.Image, prompt: str = "A photo of", max_new_tokens: int = 50) -> str:
        """
        Generate caption for an image.
        
        Args:
            image: PIL Image
            prompt: Text prompt prefix
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Caption string
        """
        if not self._loaded:
            self.load()
        
        # Process image using open-flamingo's image processor
        vision_x = [self.image_processor(image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)  # Add batch and time dimensions
        
        # Tokenize text - OpenFlamingo uses special format
        lang_x = self.tokenizer(
            [f"<image>{prompt}"],
            return_tensors="pt",
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        vision_x = vision_x.to(device)
        lang_x = {k: v.to(device) for k, v in lang_x.items()}
        
        # Generate caption
        with torch.inference_mode():
            generated_text = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
            )
        
        # Decode the generated text (skip the prompt tokens)
        prompt_length = lang_x["input_ids"].shape[1]
        generated_ids = generated_text[0][prompt_length:]
        caption = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return caption.strip()
    
    def caption_batch(self, images: List[Image.Image], prompt: str = "A photo of", max_new_tokens: int = 50) -> List[str]:
        """
        Generate captions for multiple images in a batch.
        
        Args:
            images: List of PIL Images
            prompt: Text prompt prefix
            max_new_tokens: Maximum tokens to generate per image
            
        Returns:
            List of caption strings (same order as input images)
        """
        if not self._loaded:
            self.load()
        
        if not images:
            return []
        
        captions = []
        # Process images individually (OpenFlamingo batch processing may need special handling)
        for image in images:
            caption = self.caption(image, prompt=prompt, max_new_tokens=max_new_tokens)
            captions.append(caption)
        
        return captions


def process_image(image_path: str, openflamingo_loader: OpenFlamingoLoader) -> tuple:
    """Process a single image with OpenFlamingo."""
    image = Image.open(image_path).convert("RGB")
    
    start = time.time()
    caption = openflamingo_loader.caption(image, max_new_tokens=50)
    proc_time = time.time() - start
    
    return caption, proc_time


def process_batch(image_paths: List[str], openflamingo_loader: OpenFlamingoLoader) -> List[tuple]:
    """Process multiple images in a batch with OpenFlamingo."""
    # Load all images
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    start = time.time()
    captions = openflamingo_loader.caption_batch(images, max_new_tokens=50)
    total_time = time.time() - start
    avg_time = total_time / len(image_paths)
    
    return [(caption, avg_time) for caption in captions]


def main():
    """Main function."""
    import argparse
    import cv2
    import tempfile
    import shutil
    
    parser = argparse.ArgumentParser(description="Process frames with OpenFlamingo-3B")
    parser.add_argument("--video", type=str, default="camp_5min.mp4", help="Video path")
    parser.add_argument("--output", type=str, default="openflamingo_results.json", help="Output JSON path")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval (every Nth frame)")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to process")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing (default: 1, OpenFlamingo may need sequential processing)")
    args = parser.parse_args()
    
    video_path = Path(__file__).parent / args.video
    output_path = Path(__file__).parent / args.output
    
    print("="*60)
    print("OPENFLAMINGO-3B PROCESSING")
    print("="*60)
    print(f"\nVideo: {video_path}")
    print(f"Output: {output_path}")
    print(f"Frame interval: Every {args.interval} frames")
    print(f"Max frames: {args.max_frames}")
    print(f"Batch size: {args.batch_size}")
    
    # Load OpenFlamingo-3B model (will attempt 8-bit quantization if available)
    model_name = "openflamingo/OpenFlamingo-3B-vitl-mpt1b"
    print("\nLoading OpenFlamingo-3B model...")
    print(f"Model: {model_name}")
    start = time.time()
    
    openflamingo_loader = OpenFlamingoLoader(model_name=model_name)
    openflamingo_loader.load()
    
    load_time = time.time() - start
    print(f"OpenFlamingo-3B model loaded in {load_time:.2f}s")
    
    # Check which device is being used
    try:
        device = next(openflamingo_loader.model.parameters()).device
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
    except Exception as e:
        print(f"\n🔧 Device: Could not detect (model structure may vary)")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    
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
            batch_results = process_batch(batch_paths, openflamingo_loader)
            
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
                    caption, proc_time = process_image(path, openflamingo_loader)
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
        "quantization": "8-bit",
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

