"""
Embeddings generation module.
"""

from agent.generate_embeddings import generate_embeddings


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

