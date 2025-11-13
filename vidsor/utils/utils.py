"""
Utility functions for Vidsor video editor.
"""


def format_time(seconds: float) -> str:
    """Format time in MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_time_detailed(seconds: float) -> str:
    """Format time in MM:SS.mmm format for precise display."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def get_chunk_color(chunk_type: str, is_hovered: bool = False, is_selected: bool = False) -> tuple:
    """
    Get professional color scheme for chunks.
    Returns (fill_color, outline_color, gradient_color) tuple.
    """
    if is_selected:
        # Selected chunks always show in green
        return ("#4CAF50", "#2E7D32", "#66BB6A")  # Green gradient for selected chunks
    elif is_hovered:
        if chunk_type == "highlight":
            return ("#FFE55C", "#FFD700", "#FFF080")  # Lighter gold
        elif chunk_type == "fast_forward":
            return ("#6BA3E8", "#4A90E2", "#8BB5F0")  # Lighter blue
        else:
            return ("#6FD88F", "#50C878", "#8FE5A8")  # Lighter green
    else:
        if chunk_type == "highlight":
            return ("#FFA500", "#FF8C00", "#FFB84D")  # Orange gradient
        elif chunk_type == "fast_forward":
            return ("#5B9BD5", "#3D6FA5", "#7AB3E0")  # Blue gradient
        else:
            return ("#4ECDC4", "#2E9B94", "#6EDDD5")  # Teal gradient

