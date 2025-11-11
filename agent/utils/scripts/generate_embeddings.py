"""Generate embeddings for segment tree JSON file.

This script precomputes embeddings for all transcriptions and unified descriptions
in the segment tree, saving them to a cache file for faster semantic search.
"""

from agent.utils.segment_tree_utils import SegmentTreeQuery
import sys

def generate_embeddings(json_path: str = "camp_segment_tree.json", embedding_model: str = "BAAI/bge-large-en-v1.5"):
    """
    Generate and cache embeddings for the segment tree.
    
    Args:
        json_path: Path to segment tree JSON file
        embedding_model: Name of the sentence transformer model to use
    """
    print("=" * 60)
    print("GENERATING EMBEDDINGS")
    print("=" * 60)
    print(f"\nLoading segment tree from: {json_path}")
    
    # Load segment tree
    query = SegmentTreeQuery(json_path, embedding_model=embedding_model)
    
    # Get video info
    video_info = query.get_video_info()
    print(f"\nVideo: {video_info.get('video', 'N/A')}")
    print(f"Duration: {video_info.get('duration_seconds', 0)} seconds")
    print(f"FPS: {video_info.get('fps', 0)}")
    
    # Trigger embedding computation by running a semantic search
    # This will compute embeddings for all transcriptions and unified descriptions
    print("\n" + "=" * 60)
    print("COMPUTING EMBEDDINGS")
    print("=" * 60)
    print("\nThis will compute embeddings for:")
    print("  - All audio transcriptions")
    print("  - All unified descriptions (LLaVA/BLIP)")
    
    # Get model info
    model = query._get_embedding_model()
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"\nUsing model: {embedding_model} ({embedding_dim} dimensions)")
    print("\nComputing...")
    
    # Run a dummy semantic search to trigger embedding computation
    # The embeddings will be computed and cached automatically
    results = query.semantic_search(
        "test",  # Dummy query
        top_k=1,
        threshold=0.0,  # Low threshold to ensure we get at least one result
        search_transcriptions=True,
        search_unified=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("EMBEDDINGS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nEmbeddings cached to: {query._cache_path}")
    print("\nYou can now use semantic_search() for fast queries!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for segment tree")
    parser.add_argument(
        "--json",
        default="camp_segment_tree.json",
        help="Path to segment tree JSON file (default: camp_segment_tree.json)"
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-large-en-v1.5",
        help="Sentence transformer model name (default: BAAI/bge-large-en-v1.5)"
    )
    
    args = parser.parse_args()
    
    try:
        generate_embeddings(args.json, embedding_model=args.embedding_model)
    except FileNotFoundError as e:
        print(f"\nERROR: File not found: {e}")
        print(f"\nMake sure you've run generate_segment_tree_llava.py first!")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

