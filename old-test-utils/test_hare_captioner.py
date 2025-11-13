"""
Test script for hareCaptioner-Video model.
Tests video-aware captioning with temporal understanding.
"""

import os
import sys
import time
import argparse

# Add parent directory to path to import extractor modules
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Import directly from the module file to avoid triggering package __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hare_captioner_loader",
    os.path.join(parent_dir, "extractor", "models", "hare_captioner_loader.py")
)
hare_captioner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hare_captioner_module)
HareCaptionerLoader = hare_captioner_module.HareCaptionerLoader


def main():
    """Test hareCaptioner-Video on a video file"""
    
    parser = argparse.ArgumentParser(description="Test hareCaptioner-Video model")
    parser.add_argument(
        "--video",
        type=str,
        default="kbc.mp4",
        help="Video file path (relative to old-test-utils or absolute path)"
    )
    parser.add_argument(
        "--keyframes",
        type=int,
        default=6,
        help="Number of keyframes to extract (4-8 recommended, default: 6)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Resolve video path
    if os.path.isabs(args.video):
        video_path = args.video
    else:
        video_path = os.path.join(os.path.dirname(__file__), args.video)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print(f"\nAvailable video files in old-test-utils:")
        test_dir = os.path.dirname(__file__)
        for f in os.listdir(test_dir):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"  - {f}")
        return
    
    print("=" * 60)
    print("hareCaptioner-Video Test")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Model: Lin-Chen/ShareCaptioner-Video")
    print(f"Strategy: Fast Captioning (image-grid format)")
    print(f"Keyframes: {args.keyframes}")
    print(f"Max tokens: {args.max_tokens}")
    print()
    
    # Initialize loader with 4-bit quantization for 8GB GPUs
    print("Initializing model loader...")
    print("Using 4-bit quantization (optimized for 8GB GPUs)")
    loader = HareCaptionerLoader(use_4bit=True, low_memory=True)
    
    try:
        # Generate caption
        start_time = time.time()
        print("Starting video captioning...")
        print("(This may take a while - model needs to load and process video)")
        print("-" * 60)
        
        caption = loader.caption_video(
            video_path=video_path,
            num_keyframes=args.keyframes,
            max_new_tokens=args.max_tokens
        )
        
        elapsed = time.time() - start_time
        
        print("-" * 60)
        print("Caption Generated:")
        print("-" * 60)
        print(caption)
        print("-" * 60)
        print(f"\nProcessing time: {elapsed:.2f}s")
        print("=" * 60)
        
        # Save result
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(os.path.dirname(__file__), f"hare_captioner_{video_name}_result.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Video: {video_path}\n")
            f.write(f"Model: hareCaptioner-Video (Lin-Chen/ShareCaptioner-Video)\n")
            f.write(f"Keyframes: {args.keyframes}\n")
            f.write(f"Max tokens: {args.max_tokens}\n")
            f.write(f"Processing time: {elapsed:.2f}s\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"Caption:\n")
            f.write(f"{'='*60}\n")
            f.write(f"{caption}\n")
        
        print(f"\nResult saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("Troubleshooting:")
        print("=" * 60)
        print("1. Install dependencies:")
        print("   pip install transformers accelerate bitsandbytes torch>=2.6.0")
        print("2. Check GPU memory (model requires ~14GB+ for FP16)")
        print("3. Verify internet connection for model download")
        print("4. Check the model's GitHub repo for specific usage instructions")
        print("5. Model is based on InternLM-Xcomposer2-4KHD (7B parameters)")


if __name__ == "__main__":
    main()

