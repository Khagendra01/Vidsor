"""
Main entry point for the extractor package.
"""

import argparse
from extractor.pipeline import SegmentTreePipeline
from extractor.config import ExtractorConfig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate segment tree from video")
    
    # Required arguments
    parser.add_argument("--video", default="camp.mp4", help="Video file path")
    parser.add_argument("--tracking", default="camp_yolo_bytetrack.json", help="Tracking JSON file")
    parser.add_argument("--output", default="camp_segment_tree.json", help="Output JSON file")
    
    # Tracker options
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "deepsort"],
                       help="Tracker name")
    parser.add_argument("--yolo-stride", type=int, default=10,
                       help="Process every Nth frame with YOLO (default: 10)")
    
    # LLaVA options
    parser.add_argument("--no-llava", action="store_true", default=False,
                       help="Disable LLaVA processing (default: enabled)")
    parser.add_argument("--use-images", action="store_true", default=False,
                       help="Send actual frame images to LLaVA (default: False)")
    
    # BLIP options
    parser.add_argument("--blip-split", type=int, default=1, choices=[1, 2, 3],
                       help="Number of frames per second for BLIP: 1=middle, 2=first+last, 3=first+middle+last (default: 1)")
    parser.add_argument("--blip-batch-size", type=int, default=48,
                       help="Batch size for BLIP processing (default: 48, adjust based on GPU memory)")
    
    # Hierarchical tree options
    parser.add_argument("--no-hierarchical", action="store_true", default=False,
                       help="Skip hierarchical tree generation (default: enabled)")
    parser.add_argument("--leaf-duration", type=float, default=5.0,
                       help="Duration of leaf nodes in seconds (default: 5.0)")
    parser.add_argument("--branching-factor", type=int, default=2,
                       help="Number of children per parent node (default: 2)")
    
    # Embeddings options
    parser.add_argument("--no-embeddings", action="store_true", default=False,
                       help="Skip embeddings generation (default: enabled)")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5",
                       help="Sentence transformer model for embeddings (default: BAAI/bge-large-en-v1.5)")
    
    # Parallel processing
    parser.add_argument("--max-workers", type=int, default=3,
                       help="Maximum number of parallel workers (default: 3)")
    
    # Model overrides
    parser.add_argument("--blip-model", help="Override BLIP model name")
    parser.add_argument("--whisper-model", help="Override Whisper model name")
    parser.add_argument("--yolo-model", help="Override YOLO model path")
    parser.add_argument("--ollama-url", help="Override Ollama URL")
    parser.add_argument("--ollama-model", help="Override Ollama model name")
    
    args = parser.parse_args()
    
    # Create config
    config = ExtractorConfig(
        video_path=args.video,
        tracking_json_path=args.tracking,
        output_path=args.output,
        tracker=args.tracker,
        yolo_stride=args.yolo_stride,
        blip_split=args.blip_split,
        blip_batch_size=args.blip_batch_size,
        use_llava=not args.no_llava,
        use_images=args.use_images,
        max_workers=args.max_workers,
        generate_hierarchical=not args.no_hierarchical,
        leaf_duration=args.leaf_duration,
        branching_factor=args.branching_factor,
        generate_embeddings=not args.no_embeddings,
        embedding_model=args.embedding_model,
        blip_model=args.blip_model,
        whisper_model=args.whisper_model,
        yolo_model=args.yolo_model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    )
    
    # Run pipeline
    pipeline = SegmentTreePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

