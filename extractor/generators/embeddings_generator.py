"""
Embeddings generation module.
"""

from agent.utils.scripts.generate_embeddings import generate_embeddings


class EmbeddingsGenerator:
    """Generates embeddings for segment tree."""
    
    def __init__(self, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize embeddings generator.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        self.embedding_model = embedding_model
    
    def generate(self, json_path: str) -> None:
        """
        Generate embeddings for segment tree.
        
        Args:
            json_path: Path to segment tree JSON file
        """
        print("\n" + "=" * 60)
        print("GENERATING EMBEDDINGS")
        print("=" * 60)
        
        generate_embeddings(json_path, embedding_model=self.embedding_model)

