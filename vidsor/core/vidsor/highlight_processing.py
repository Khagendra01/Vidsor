"""
Highlight processing functionality.
"""
import os
import json
from typing import Dict, List, Tuple
from ...models import Chunk


def process_highlights(self, result: Dict, clips: List[str], time_ranges: List[Tuple[float, float]], 
                       search_results: List[Dict]):
    """
    Process highlights from agent result: extract metadata, sort by timing, and add to timeline.
    Only adds to timeline if timeline.json is empty (preserves existing data).
    
    Args:
        result: Full agent result dictionary
        clips: List of extracted clip file paths
        time_ranges: List of (start, end) time tuples
        search_results: List of search result dictionaries with metadata
    """
    print(f"[VIDSOR] _process_highlights called with {len(clips) if clips else 0} clips and {len(time_ranges) if time_ranges else 0} time ranges")
    
    # Check if timeline.json exists and has data
    timeline_path = os.path.join(self.current_project_path, "timeline.json") if self.current_project_path else None
    timeline_has_data = False
    if timeline_path and os.path.exists(timeline_path):
        try:
            with open(timeline_path, 'r') as f:
                timeline_data = json.load(f)
                chunks_data = timeline_data.get("chunks", [])
                if chunks_data:
                    timeline_has_data = True
                    print(f"[VIDSOR] Timeline.json already has {len(chunks_data)} chunks, preserving existing data")
        except Exception as e:
            print(f"[VIDSOR] Error checking timeline.json: {e}")
    
    # Only process highlights if timeline is empty
    if timeline_has_data:
        print("[VIDSOR] Timeline.json has existing data, skipping AI-generated highlights")
        return
    
    if not self.segment_tree:
        print("[VIDSOR] No segment tree available for metadata extraction")
        # Still process highlights even without segment tree for metadata
        if not time_ranges and not clips:
            return
    else:
        print(f"[VIDSOR] Segment tree available, extracting metadata")
    
    highlight_chunks = []
    
    # Create a mapping from time ranges to clips
    clip_map = {}
    if clips and time_ranges:
        for clip_path, (start, end) in zip(clips, time_ranges):
            clip_map[(start, end)] = clip_path
    
    # Process time ranges (use clips if available, otherwise time_ranges)
    ranges_to_process = []
    if clips and time_ranges and len(clips) == len(time_ranges):
        # Use clips with their corresponding time ranges (perfect match)
        for clip_path, (start, end) in zip(clips, time_ranges):
            ranges_to_process.append((start, end, clip_path))
    elif clips:
        # We have clips but maybe no time_ranges or mismatch - extract timing from filenames
        import re
        for clip_path in clips:
            # Try to extract timing from filename like "clip_1_22s_to_28s_..."
            match = re.search(r'(\d+)s_to_(\d+)s', os.path.basename(clip_path))
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                ranges_to_process.append((start, end, clip_path))
            else:
                print(f"[VIDSOR] Warning: Could not extract timing from clip filename: {clip_path}")
    elif time_ranges:
        # Only time ranges, no clips extracted yet
        for start, end in time_ranges:
            ranges_to_process.append((start, end, None))
    else:
        print("[VIDSOR] No time ranges or clips to process")
        return
    
    # Sort by start time
    ranges_to_process.sort(key=lambda x: x[0])
    
    # Extract metadata for each range
    for start, end, clip_path in ranges_to_process:
        # Get metadata from segment tree
        unified_desc = None
        audio_desc = None
        
        # Try to get description from search_results first
        if search_results:
            for search_result in search_results:
                result_time_range = search_result.get("time_range", [])
                if len(result_time_range) >= 2:
                    result_start, result_end = result_time_range[0], result_time_range[1]
                    # Check if this search result overlaps with our time range
                    if not (result_end < start or result_start > end):
                        # Get text from search result
                        text = search_result.get("text", "")
                        result_type = search_result.get("type", "")
                        
                        # Also check for transcription text in different formats
                        if not text:
                            text = search_result.get("transcription", "")
                        
                        if result_type == "unified" and not unified_desc and text:
                            unified_desc = text
                        elif (result_type == "transcription" or result_type == "audio") and not audio_desc and text:
                            audio_desc = text
                        # Handle generic search results
                        elif not result_type and text:
                            # If we don't have unified_desc yet, use this as visual
                            if not unified_desc:
                                unified_desc = text
        
        # If not found in search_results, query segment tree directly
        # Sample multiple points in the time range to get better descriptions
        if not unified_desc or not audio_desc:
            sample_times = [
                start,
                start + (end - start) * 0.25,
                (start + end) / 2,
                start + (end - start) * 0.75,
                end
            ]
            
            for sample_time in sample_times:
                second_data = self.segment_tree.get_second_by_time(sample_time)
                
                if second_data:
                    if not unified_desc:
                        desc = second_data.get("unified_description", "")
                        if desc and desc != "0":
                            unified_desc = desc
                            break  # Found one, move on
                    
                    if not audio_desc:
                        # Try to get audio transcription
                        transcription_id = second_data.get("transcription_id")
                        if transcription_id and transcription_id in self.segment_tree._transcription_map:
                            transcription = self.segment_tree._transcription_map[transcription_id]
                            audio = transcription.get("transcription", "")
                            if audio:
                                audio_desc = audio
                                break  # Found one, move on
                
                # If we found both, no need to continue
                if unified_desc and audio_desc:
                    break
        
        # Build description from available sources
        description_parts = []
        if unified_desc:
            description_parts.append(f"Visual: {unified_desc}")
        if audio_desc:
            description_parts.append(f"Audio: {audio_desc}")
        
        description = " | ".join(description_parts) if description_parts else "Highlight moment"
        
        # Calculate sequential timeline position (clips appear one after another in edited timeline)
        # start_time/end_time = position in edited timeline (sequential)
        # original_start_time/original_end_time = position in source video
        clip_duration = end - start
        if highlight_chunks:
            # Start after the last chunk
            timeline_start = highlight_chunks[-1].end_time
        else:
            # First chunk starts at 0
            timeline_start = 0.0
        timeline_end = timeline_start + clip_duration
        
        # Create chunk
        chunk = Chunk(
            start_time=timeline_start,  # Sequential position in edited timeline
            end_time=timeline_end,     # Sequential position in edited timeline
            chunk_type="highlight",
            speed=1.0,
            description=description,
            score=1.0,  # Highlights have high score
            original_start_time=start,  # Original position in source video
            original_end_time=end,      # Original position in source video
            unified_description=unified_desc,
            audio_description=audio_desc,
            clip_path=clip_path
        )
        
        highlight_chunks.append(chunk)
    
    # Add highlight chunks to edit state
    # Replace existing chunks if they're also highlights, otherwise append
    existing_highlights = [c for c in self.edit_state.chunks if c.chunk_type == "highlight"]
    if existing_highlights:
        # Replace existing highlights
        other_chunks = [c for c in self.edit_state.chunks if c.chunk_type != "highlight"]
        self.edit_state.chunks = other_chunks + highlight_chunks
    else:
        # Append to existing chunks
        self.edit_state.chunks.extend(highlight_chunks)
    
    # Sort all chunks by start_time (highlights are already sorted by original timing)
    self.edit_state.chunks.sort(key=lambda x: x.start_time)
    
    # Save timeline to timeline.json
    self._save_timeline()
    
    # Update timeline display
    print(f"[VIDSOR] Drawing timeline with {len(self.edit_state.chunks)} total chunks")
    self._draw_timeline()
    
    # Update UI state
    self._update_ui_state()
    
    # Force UI update
    if self.root:
        self.root.update_idletasks()
    
    print(f"[VIDSOR] Processed {len(highlight_chunks)} highlight clips and added to timeline")
    print(f"[VIDSOR] Total chunks in edit_state: {len(self.edit_state.chunks)}")

