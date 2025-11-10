"""
Embeddings generation module.
"""

import sys
import os

# Add parent directory to path to import generate_embeddings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from generate_embeddings import generate_embeddings


class EmbeddingsGenerator:
    """Generates embeddings for segment tree."""
    
    def generate(self, json_path: str) -> None:
        """
        Generate embeddings for segment tree.
        
        Args:
            json_path: Path to segment tree JSON file
        """
        print("\n" + "=" * 60)
        print("GENERATING EMBEDDINGS")
        print("=" * 60)
        
        generate_embeddings(json_path)

