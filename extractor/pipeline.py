"""
Main pipeline orchestrator for video segment tree extraction.
"""

import time
from extractor.config import ExtractorConfig
from extractor.models.bakllava_loader import BakLLaVALoader
from extractor.models.whisper_loader import WhisperLoader
from extractor.models.yolo_loader import YOLOLoader
from extractor.utils.video_utils import VideoUtils
from extractor.generators.segment_tree_generator import SegmentTreeGenerator
from extractor.generators.hierarchical_generator import HierarchicalGenerator
from extractor.generators.embeddings_generator import EmbeddingsGenerator


class SegmentTreePipeline:
    """Main pipeline for generating complete segment trees."""
    
    def __init__(self, config: ExtractorConfig):
        """
        Initialize pipeline.
        
        Args:
            config: ExtractorConfig instance
        """
        self.config = config
        
        # Initialize model loaders
        self.bakllava_loader = BakLLaVALoader(
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model
        )
        self.whisper_loader = WhisperLoader(config.whisper_model)
        self.yolo_loader = YOLOLoader(config.yolo_model)
        
        # Initialize video utilities
        self.video_utils = VideoUtils(config.video_path)
        
        # Initialize generators
        self.segment_tree_generator = SegmentTreeGenerator(
            config.video_path,
            config.tracking_json_path,
            config,
            self.bakllava_loader,
            self.whisper_loader,
            self.yolo_loader,
            self.video_utils
        )
        
        self.hierarchical_generator = HierarchicalGenerator(
            config.leaf_duration,
            config.branching_factor
        ) if config.generate_hierarchical else None
        
        self.embeddings_generator = EmbeddingsGenerator(config.embedding_model) if config.generate_embeddings else None
    
    def run(self) -> str:
        """
        Run the complete pipeline.
        
        Returns:
            Path to output JSON file
        """
        print("=" * 60)
        print("VIDEO SEGMENT TREE EXTRACTION PIPELINE")
        print("=" * 60)
        print(f"\nVideo: {self.config.video_path}")
        print(f"Output: {self.config.output_path}")
        print(f"Tracker: {self.config.tracker}")
        print(f"BakLLaVA: ENABLED (via Ollama)")
        print(f"Max Workers: {self.config.max_workers}")
        print()
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Generate segment tree
            output_path = self.segment_tree_generator.generate()
            
            # Step 2: Generate hierarchical tree (if enabled)
            if self.hierarchical_generator:
                try:
                    self.hierarchical_generator.generate(output_path)
                except Exception as e:
                    print(f"Warning: Failed to generate hierarchical tree: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Step 3: Generate embeddings (if enabled)
            if self.embeddings_generator:
                try:
                    self.embeddings_generator.generate(output_path)
                except Exception as e:
                    print(f"Warning: Failed to generate embeddings: {e}")
                    import traceback
                    traceback.print_exc()
            
            total_time = time.time() - pipeline_start
            
            print("\n" + "=" * 60)
            print("ALL PROCESSING COMPLETE")
            print("=" * 60)
            print(f"\nTotal pipeline time: {total_time:.2f}s")
            print(f"Final output saved to: {output_path}")
            
            return output_path
        
        except Exception as e:
            import traceback
            print("\n" + "=" * 60)
            print("PIPELINE ERROR")
            print("=" * 60)
            print(f"\nError: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup
            self.video_utils.release()

