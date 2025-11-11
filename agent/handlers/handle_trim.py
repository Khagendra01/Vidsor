"""Handler for TRIM operation."""

from typing import Dict, List, Tuple, Any
from agent.timeline_manager import TimelineManager
from agent.state import OrchestratorState


def handle_trim(
    state: OrchestratorState,
    timeline_manager: TimelineManager,
    params: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Handle TRIM operation.
    Supports:
    - delta trims from start/end (trim_from with trim_seconds)
    - set exact duration (trim_target_length)
    - remove an internal range (remove_range with start/end offsets)
    
    Parameters (backward-compatible):
    - trim_index: int (required)
    - trim_seconds: float (optional; used with trim_from start/end)
    - trim_from: "start" | "end" (optional; default end for legacy)
    - trim_target_length: float (optional; set clip duration exactly)
    - remove_range: { "start_offset": float, "end_offset": float } (optional; within-clip offsets)
    """
    if verbose:
        print("\n[OPERATION] TRIM")

    idx = params.get("trim_index")
    if idx is None:
        return {"success": False, "error": "TRIM requires trim_index"}

    if idx < 0 or idx >= len(timeline_manager.chunks):
        return {"success": False, "error": f"Invalid trim_index: {idx}"}

    chunk = timeline_manager.chunks[idx]
    # Use original times as source-of-truth for content region
    orig_start = float(chunk.get("original_start_time", chunk.get("start_time", 0.0)))
    orig_end = float(chunk.get("original_end_time", chunk.get("end_time", 0.0)))
    tl_start = float(chunk.get("start_time", 0.0))
    tl_end = float(chunk.get("end_time", tl_start))

    if orig_end <= orig_start:
        return {"success": False, "error": "Chunk has invalid original time range"}

    # Determine trim mode
    trim_target_length = params.get("trim_target_length")
    remove_range = params.get("remove_range")
    trim_seconds = params.get("trim_seconds")
    trim_from = params.get("trim_from")

    new_chunks: List[Dict] = []

    def recalc_timeline_from(start_index: int, start_time: float) -> None:
        """Recalculate timeline start/end for all chunks from start_index onward."""
        current_time = start_time
        for i in range(start_index, len(timeline_manager.chunks)):
            c = timeline_manager.chunks[i]
            dur = max(0.0, float(c["original_end_time"]) - float(c["original_start_time"]))
            c["start_time"] = current_time
            c["end_time"] = current_time + dur
            current_time = c["end_time"]

    # Case 1: remove internal range → possibly split into two chunks
    if remove_range and isinstance(remove_range, dict):
        # Support either explicit offsets or centered length
        if "center_length" in remove_range and remove_range.get("center_length") is not None:
            clen = float(remove_range.get("center_length"))
            if clen <= 0 or clen >= (orig_end - orig_start):
                return {"success": False, "error": "Invalid center_length for remove_range"}
            clip_dur = (orig_end - orig_start)
            center_time = orig_start + clip_dur / 2.0
            half = clen / 2.0
            abs_remove_start = max(orig_start, center_time - half)
            abs_remove_end = min(orig_end, center_time + half)
            rs = abs_remove_start - orig_start
            re = abs_remove_end - orig_start
        else:
            rs = float(remove_range.get("start_offset", 0.0))
            re = float(remove_range.get("end_offset", 0.0))
        if rs < 0 or re < 0 or re <= rs:
            return {"success": False, "error": "Invalid remove_range offsets"}
        if rs >= (orig_end - orig_start) or re > (orig_end - orig_start):
            return {"success": False, "error": "remove_range exceeds clip duration"}

        abs_remove_start = orig_start + rs
        abs_remove_end = orig_start + re
        # If removal eats entire clip, just delete the chunk
        if abs_remove_start <= orig_start and abs_remove_end >= orig_end:
            timeline_manager.chunks.pop(idx)
            # Recalc following
            recalc_timeline_from(idx, tl_start)
            if verbose:
                print(f"  Removed entire chunk at index {idx}")
            return {"success": True, "chunks_modified": 0, "chunks_removed": 1}

        # Build kept segments
        kept_segments: List[Tuple[float, float]] = []
        if abs_remove_start > orig_start:
            kept_segments.append((orig_start, abs_remove_start))
        if abs_remove_end < orig_end:
            kept_segments.append((abs_remove_end, orig_end))

        # Replace current chunk with kept segments (1 or 2)
        # First, remove the original
        timeline_manager.chunks.pop(idx)

        insert_pos = idx
        current_time = tl_start
        for seg_start, seg_end in kept_segments:
            duration = seg_end - seg_start
            if duration <= 0:
                continue
            new_chunk = {
                **chunk,
                "start_time": current_time,
                "end_time": current_time + duration,
                "original_start_time": seg_start,
                "original_end_time": seg_end,
            }
            # Avoid carrying references that might cause duplication issues
            new_chunk = dict(new_chunk)
            timeline_manager.chunks.insert(insert_pos, new_chunk)
            insert_pos += 1
            current_time = new_chunk["end_time"]
            new_chunks.append(new_chunk)

        # Recalc following chunks from insert_pos
        recalc_timeline_from(insert_pos, current_time)

        if verbose:
            print(f"  ✓ Removed internal range {rs:.2f}s–{re:.2f}s (offsets) from chunk {idx}")
            print(f"  Resulted in {len(new_chunks)} chunk(s)")

        return {
            "success": True,
            "chunks_added": new_chunks,
            "chunks_modified": len(new_chunks),
            "removed_range": [rs, re]
        }

    # Case 2: set exact duration
    if trim_target_length is not None:
        target = float(trim_target_length)
        if target <= 0:
            return {"success": False, "error": "trim_target_length must be > 0"}
        new_orig_end = orig_start + target
        if new_orig_end > orig_end:
            # Extending is not supported by trim; cap to original end
            new_orig_end = orig_end
            target = new_orig_end - orig_start
            if target <= 0:
                return {"success": False, "error": "Target exceeds content; nothing to keep"}
        chunk["original_end_time"] = new_orig_end
        chunk["start_time"] = tl_start
        chunk["end_time"] = tl_start + target

        # Recalc following chunks
        recalc_timeline_from(idx + 1, chunk["end_time"])

        if verbose:
            print(f"  ✓ Set clip {idx} duration to {target:.2f}s")

        return {"success": True, "chunks_modified": 1}

    # Case 3: legacy delta trim from start/end using trim_seconds + trim_from
    if trim_seconds is not None:
        delta = float(trim_seconds)
        if delta <= 0:
            return {"success": False, "error": "trim_seconds must be > 0"}

        if trim_from == "start":
            new_orig_start = orig_start + delta
            if new_orig_start >= orig_end:
                return {"success": False, "error": "Trim exceeds or equals clip duration"}
            chunk["original_start_time"] = new_orig_start
            # Update timeline times to match new duration
            new_duration = orig_end - new_orig_start
            chunk["start_time"] = tl_start
            chunk["end_time"] = tl_start + new_duration

            recalc_timeline_from(idx + 1, chunk["end_time"])
            if verbose:
                print(f"  ✓ Trimmed {delta:.2f}s from start of clip {idx}")
            return {"success": True, "chunks_modified": 1}

        # default and explicit "end"
        new_orig_end = orig_end - delta
        if new_orig_end <= orig_start:
            return {"success": False, "error": "Trim exceeds or equals clip duration"}
        chunk["original_end_time"] = new_orig_end
        new_duration = new_orig_end - orig_start
        chunk["start_time"] = tl_start
        chunk["end_time"] = tl_start + new_duration

        recalc_timeline_from(idx + 1, chunk["end_time"])
        if verbose:
            print(f"  ✓ Trimmed {delta:.2f}s from end of clip {idx}")
        return {"success": True, "chunks_modified": 1}

    return {"success": False, "error": "No valid TRIM parameters provided"}

