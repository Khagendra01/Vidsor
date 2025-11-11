"""Recalculate timeline start/end times for all chunks."""

from agent.timeline_manager import TimelineManager


def recalculate_timeline_times(timeline_manager: TimelineManager) -> None:
    """
    Recalculate timeline start/end times for all chunks.
    Ensures timeline times are sequential and correct.
    
    Args:
        timeline_manager: Timeline manager instance
    """
    current_time = 0.0
    for chunk in timeline_manager.chunks:
        duration = chunk["end_time"] - chunk["start_time"]
        chunk["start_time"] = current_time
        chunk["end_time"] = current_time + duration
        current_time = chunk["end_time"]

