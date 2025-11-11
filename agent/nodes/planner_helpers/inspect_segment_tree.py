"""Inspect segment tree content and create video narrative if needed."""

import time
from typing import Optional
from agent.utils.logging_utils import get_log_helper
from agent.nodes.planner_helpers.create_video_narrative import create_video_narrative


def inspect_segment_tree(
    segment_tree,
    query: str,
    llm,
    logger=None,
    verbose: bool = False
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Inspect segment tree content and create video narrative if needed.
    
    Args:
        segment_tree: SegmentTreeQuery instance
        query: User query string
        logger: Optional logger instance
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (content_inspection, video_narrative) - both can be None
    """
    log = get_log_helper(logger, verbose)
    
    # Decide if we need to inspect segment tree content
    query_lower = query.lower()
    abstract_indicators = [
        "highlight", "highlights", "best", "important", "key", "significant",
        "interesting", "exciting", "memorable", "notable", "noteworthy",
        "moments", "scenes", "parts", "events", "action"
    ]
    needs_inspection = any(indicator in query_lower for indicator in abstract_indicators)
    
    # Also inspect if query is very vague or general
    vague_queries = ["what", "show me", "find", "get", "give me"]
    if any(vq in query_lower for vq in vague_queries) and len(query.split()) <= 5:
        needs_inspection = True
    
    content_inspection = None
    if needs_inspection and segment_tree:
        log.info("\n[INSPECTION] Inspecting segment tree to understand video content...")
        try:
            start_time = time.time()
            content_inspection = segment_tree.inspect_content(max_keywords=100, max_sample_descriptions=20)
            elapsed = time.time() - start_time
            log.info(f"  Found {content_inspection['keyword_count']} unique keywords")
            log.info(f"  Found {content_inspection['object_class_count']} object classes")
            log.info(f"  Sample keywords: {', '.join(content_inspection['all_keywords'][:15])}...")
            
            # Log sample descriptions (visual + audio mix)
            sample_descriptions = content_inspection.get('sample_descriptions', [])
            if sample_descriptions:
                visual_count = sum(1 for d in sample_descriptions if d.get('type') == 'visual')
                audio_count = sum(1 for d in sample_descriptions if d.get('type') == 'audio')
                log.info(f"  Sample descriptions: {len(sample_descriptions)} total ({visual_count} visual, {audio_count} audio)")
                log.info(f"  Sample descriptions (full, sorted chronologically):")
                for i, desc in enumerate(sample_descriptions, 1):
                    desc_type = desc.get('type', 'visual')
                    # Get time info - prefer time_range start, fallback to second
                    if desc.get('time_range'):
                        time_info = f"{desc['time_range'][0]:.1f}s"
                    else:
                        time_info = f"{desc.get('second', 0):.1f}s"
                    desc_text = desc.get('description', '').strip()
                    # Show full description (no truncation)
                    log.info(f"    {i}. [{time_info}] [{desc_type.upper()}] {desc_text}")
            
            log.info(f"  Inspection completed in {elapsed:.2f}s")
        except Exception as e:
            log.info(f"  [WARNING] Inspection failed: {e}")
            content_inspection = None
    
    # Create video narrative understanding
    video_narrative = None
    if content_inspection and needs_inspection:
        log.info("\n[NARRATION] Creating video narrative understanding...")
        video_narrative = create_video_narrative(content_inspection, query, llm, verbose=verbose, logger=logger)
        if not video_narrative:
            log.info("  [WARNING] Narrative creation failed, proceeding with raw inspection data")
    
    return content_inspection, video_narrative

