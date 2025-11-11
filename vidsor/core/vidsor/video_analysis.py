"""
Video analysis functionality.
"""
from typing import List
from ...models import Chunk


def analyze_with_segment_tree(self,
                              silence_threshold: float,
                              fast_forward_speed: float,
                              highlight_min_score: float) -> List[Chunk]:
    """Analyze video using segment tree data."""
    chunks = []
    duration = self.video_clip.duration
    
    # Get transcriptions to detect silence
    transcriptions = self.segment_tree.transcriptions
    transcription_ranges = []
    for tr in transcriptions:
        tr_range = tr.get("time_range", [])
        if tr_range and len(tr_range) >= 2:
            text = tr.get("transcription", "").strip()
            if text:  # Only non-empty transcriptions
                transcription_ranges.append((tr_range[0], tr_range[1]))
    
    # Sort transcription ranges
    transcription_ranges.sort(key=lambda x: x[0])
    
    # Find silent gaps
    current_time = 0.0
    for tr_start, tr_end in transcription_ranges:
        # Check for silence before this transcription
        if tr_start - current_time >= silence_threshold:
            # Silent section - mark for fast-forward
            chunks.append(Chunk(
                start_time=current_time,
                end_time=tr_start,
                chunk_type="fast_forward",
                speed=fast_forward_speed,
                description="Silent section"
            ))
        
        # Normal section with audio
        chunks.append(Chunk(
            start_time=max(current_time, tr_start),
            end_time=tr_end,
            chunk_type="normal",
            speed=1.0,
            description="Audio section"
        ))
        current_time = tr_end
    
    # Check for silence at the end
    if duration - current_time >= silence_threshold:
        chunks.append(Chunk(
            start_time=current_time,
            end_time=duration,
            chunk_type="fast_forward",
            speed=fast_forward_speed,
            description="Silent section"
        ))
    
    # Detect highlights using hierarchical tree or semantic search
    if self.segment_tree.hierarchical_tree:
        highlights = self.segment_tree.hierarchical_score_leaves_for_highlights(
            max_results=20
        )
        for highlight in highlights:
            tr = highlight.get("time_range", [])
            if tr and len(tr) >= 2:
                score = highlight.get("score", 0)
                if score >= highlight_min_score * 10:  # Adjust scale
                    # Mark as highlight
                    for chunk in chunks:
                        if (chunk.start_time <= tr[0] < chunk.end_time or
                            chunk.start_time < tr[1] <= chunk.end_time):
                            chunk.chunk_type = "highlight"
                            chunk.score = score
                            chunk.description = "Highlight moment"
    
    # Merge overlapping chunks and sort
    chunks = self._merge_chunks(chunks)
    chunks.sort(key=lambda x: x.start_time)
    
    return chunks


def analyze_simple(self, duration: float) -> List[Chunk]:
    """Simple fallback analysis without segment tree."""
    # Create a single chunk for the entire video
    return [Chunk(
        start_time=0.0,
        end_time=duration,
        chunk_type="normal",
        speed=1.0,
        description="Full video"
    )]


def merge_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
    """Merge overlapping chunks."""
    if not chunks:
        return []
    
    # Sort by start time
    sorted_chunks = sorted(chunks, key=lambda x: x.start_time)
    merged = [sorted_chunks[0]]
    
    for chunk in sorted_chunks[1:]:
        last = merged[-1]
        if chunk.start_time <= last.end_time:
            # Overlapping - merge
            last.end_time = max(last.end_time, chunk.end_time)
            # Keep the more interesting type
            if chunk.chunk_type == "highlight":
                last.chunk_type = "highlight"
            elif chunk.chunk_type == "fast_forward" and last.chunk_type == "normal":
                last.chunk_type = "fast_forward"
                last.speed = chunk.speed
        else:
            merged.append(chunk)
    
    return merged

