"""Create video narrative understanding from content inspection."""

import time
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from agent.utils.logging_utils import get_log_helper
from agent.utils.llm_utils import parse_json_response
from agent.utils.search.query_builder import format_content_inspection_for_narrative
from agent.prompts.planner_prompts import VIDEO_NARRATION_PROMPT


def create_video_narrative(content_inspection: dict, query: str, llm, verbose: bool = False, logger=None) -> Optional[dict]:
    """
    Create a coherent narrative understanding of the video content.
    This helps generate better search queries by understanding context, not just keywords.
    
    Args:
        content_inspection: Dictionary from segment_tree.inspect_content()
        query: User query string
        llm: Language model instance
        verbose: Whether to print verbose output
        logger: Optional logger instance
        
    Returns:
        Dictionary with video narrative understanding or None if failed
    """
    if not content_inspection:
        return None
    
    # Use shared logging helper
    log_info = get_log_helper(logger, verbose)
    
    # Build context for narration using shared formatter
    context = format_content_inspection_for_narrative(content_inspection, query)
    
    try:
        start_time = time.time()
        response = llm.invoke([
            SystemMessage(content=VIDEO_NARRATION_PROMPT),
            HumanMessage(content=context)
        ])
        elapsed = time.time() - start_time
        
        response_text = response.content.strip()
        narrative = parse_json_response(response_text, fallback=None)
        if narrative is None:
            return None
        
        log_info(f"  Video theme: {narrative.get('video_theme', 'N/A')}")
        log_info(f"  Narrative summary: {narrative.get('narrative_summary', 'N/A')[:100]}...")
        
        # Log narrative structure if available
        narrative_structure = narrative.get('narrative_structure', {})
        if narrative_structure:
            log_info(f"  Narrative structure:")
            for part_name in ['intro', 'body', 'ending']:
                part = narrative_structure.get(part_name, {})
                if part:
                    keywords = part.get('keywords', [])
                    log_info(f"    {part_name.upper()}: {', '.join(keywords[:5]) if keywords else 'N/A'}")
        
        highlight_criteria = narrative.get('highlight_criteria', {})
        if isinstance(highlight_criteria, dict):
            highlight_keywords = highlight_criteria.get('keywords', [])
            log_info(f"  Highlight keywords: {', '.join(highlight_keywords[:5]) if highlight_keywords else 'N/A'}")
        else:
            log_info(f"  Highlight criteria: {highlight_criteria}")
        
        log_info(f"  Narrative creation completed in {elapsed:.2f}s")
        
        return narrative
    except Exception as e:
        log_info(f"  [WARNING] Narrative creation failed: {e}")
        return None

