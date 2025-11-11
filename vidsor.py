"""
Backward compatibility wrapper for vidsor package.
This file allows existing code to import from vidsor.py directly.
"""

# Import from the vidsor package
from vidsor import Vidsor, Chunk, EditState

# Re-export for backward compatibility
__all__ = ['Vidsor', 'Chunk', 'EditState']
