"""
Hierarchical tree generation module.
"""

import sys
import os

# Add parent directory to path to import generate_hierarchical_tree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from generate_hierarchical_tree import generate_hierarchical_tree


class HierarchicalGenerator:
    """Generates hierarchical tree from segment tree."""
    
    def __init__(self, leaf_duration: float = 5.0, branching_factor: int = 2):
        """
        Initialize hierarchical generator.
        
        Args:
            leaf_duration: Duration of leaf nodes in seconds
            branching_factor: Number of children per parent node
        """
        self.leaf_duration = leaf_duration
        self.branching_factor = branching_factor
    
    def generate(self, json_path: str) -> None:
        """
        Generate hierarchical tree and add to segment tree JSON.
        
        Args:
            json_path: Path to segment tree JSON file
        """
        print("\n" + "=" * 60)
        print("GENERATING HIERARCHICAL TREE")
        print("=" * 60)
        
        generate_hierarchical_tree(
            json_path,
            output_json_path=None,  # Overwrite the same file
            leaf_duration=self.leaf_duration,
            branching_factor=self.branching_factor
        )

