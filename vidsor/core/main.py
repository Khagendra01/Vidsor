"""
Main entry point for Vidsor.
"""

import os
import argparse
from .vidsor import Vidsor


def main():
    """Main entry point for Vidsor."""
    parser = argparse.ArgumentParser(description="Vidsor - Video Editor")
    parser.add_argument("video", nargs="?", help="Path to video file (optional - can load via UI)")
    parser.add_argument("--segment-tree", help="Path to segment tree JSON (optional)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't open UI (requires video)")
    
    args = parser.parse_args()
    
    # If analyze-only mode, video is required
    if args.analyze_only:
        if not args.video:
            print("Error: Video file required for --analyze-only mode")
            return
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        
        editor = Vidsor(args.video, args.segment_tree)
        chunks = editor.analyze_video()
        print(f"\nGenerated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. {chunk.chunk_type}: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s "
                  f"(speed: {chunk.speed}x)")
        editor.close()
    else:
        # Start editor (empty or with video if provided)
        editor = Vidsor(args.video, args.segment_tree)
        editor.run()
        editor.close()


if __name__ == "__main__":
    main()

