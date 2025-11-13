"""Utilities to gather contextual information about timeline clips."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from agent.timeline_manager import TimelineManager
from agent.utils.segment_tree_utils import SegmentTreeQuery


def _format_object_summary(objects: Dict[str, int], max_objects: int = 5) -> List[str]:
    """Return sorted object summaries like 'person (4)'."""
    if not objects:
        return []
    sorted_objects = sorted(objects.items(), key=lambda item: item[1], reverse=True)
    return [f"{name} ({count})" for name, count in sorted_objects[:max_objects]]


def _format_descriptions(descriptions: List[Dict], max_items: int) -> List[str]:
    """Extract text from description dictionaries."""
    formatted: List[str] = []
    for desc in descriptions[:max_items]:
        text = desc.get("text") or desc.get("description") or ""
        if text:
            formatted.append(text.strip())
    return formatted


def gather_clip_contexts(
    segment_tree: Optional[SegmentTreeQuery],
    timeline_manager: Optional[TimelineManager],
    indices: List[int],
    max_visual_descriptions: int = 3,
    max_audio_descriptions: int = 2,
    max_objects: int = 5,
) -> List[Dict]:
    """
    Collect contextual information for clips referenced by timeline indices.

    Args:
        segment_tree: Loaded SegmentTreeQuery instance.
        timeline_manager: Timeline manager with current chunks.
        indices: Timeline indices to summarize.
        max_visual_descriptions: Number of visual descriptions to include per clip.
        max_audio_descriptions: Number of audio snippets to include per clip.
        max_objects: Number of object classes to include per clip.

    Returns:
        List of dictionaries containing clip context info.
    """
    if not segment_tree or not timeline_manager or not indices:
        return []

    contexts: List[Dict] = []

    for index in indices:
        chunk = timeline_manager.get_chunk(index)
        if not chunk:
            continue

        original_start = chunk.get("original_start_time")
        original_end = chunk.get("original_end_time")
        if original_start is None or original_end is None:
            continue

        duration = max(0.0, float(original_end) - float(original_start))

        # Gather object summary within clip range
        scene_summary = segment_tree.get_scene_summary(
            time_start=original_start,
            time_end=original_end,
        )
        top_objects = _format_object_summary(scene_summary.get("objects", {}), max_objects=max_objects)

        # Map float range to second indices for narrative retrieval
        start_second = max(0, int(math.floor(original_start)))
        # Use ceil-1 to ensure inclusive coverage for short clips
        end_second = max(start_second, int(math.ceil(original_end) - 1))

        combined_narrative = segment_tree.get_combined_narrative(
            start_second=start_second,
            end_second=end_second,
            include_timestamps=False,
            include_audio=True,
            separator=" ",
        )

        visual_descriptions = _format_descriptions(
            combined_narrative.get("visual_descriptions", []),
            max_items=max_visual_descriptions,
        )
        audio_descriptions = _format_descriptions(
            combined_narrative.get("audio_transcriptions", []),
            max_items=max_audio_descriptions,
        )

        narrative_text = combined_narrative.get("narrative", "")
        if narrative_text and len(narrative_text) > 360:
            narrative_text = narrative_text[:357].rstrip() + "..."

        contexts.append(
            {
                "timeline_index": index,
                "time_range": [original_start, original_end],
                "duration": duration,
                "timeline_description": chunk.get("description", ""),
                "unified_description": chunk.get("unified_description", ""),
                "audio_description": chunk.get("audio_description", ""),
                "top_objects": top_objects,
                "visual_descriptions": visual_descriptions,
                "audio_descriptions": audio_descriptions,
                "narrative": narrative_text,
            }
        )

    return contexts


