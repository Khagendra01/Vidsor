"""
Video Segment Tree Extractor Package

A modular package for extracting and processing video segment trees with
support for object tracking, visual descriptions, audio transcription,
hierarchical tree generation, and semantic embeddings.
"""

from extractor.pipeline import SegmentTreePipeline
from extractor.config import ExtractorConfig

__all__ = ['SegmentTreePipeline', 'ExtractorConfig']
__version__ = '1.0.0'

