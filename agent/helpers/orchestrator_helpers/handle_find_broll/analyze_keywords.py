"""Analyze main action keywords from selected chunks."""

import math
import re
from typing import Dict, List, Set


def analyze_main_action_keywords(
    selected_chunks: List[Dict],
    segment_tree,
    verbose: bool = False
) -> List[str]:
    """
    Extract keywords from selected chunks to identify main action.
    
    Args:
        selected_chunks: List of selected timeline chunks
        segment_tree: Segment tree for narrative descriptions
        verbose: Whether to print verbose output
        
    Returns:
        Sorted list of main action keywords
    """
    keyword_set: Set[str] = set()

    if segment_tree:
        try:
            for chunk in selected_chunks:
                orig_start = chunk.get("original_start_time", chunk.get("start_time"))
                orig_end = chunk.get("original_end_time", chunk.get("end_time"))
                if orig_start is None or orig_end is None:
                    continue

                start_second = max(0, int(math.floor(orig_start)))
                end_second = max(start_second, int(math.ceil(orig_end)))
                narrative = segment_tree.get_narrative_description(start_second, end_second)

                for desc_entry in narrative.get("descriptions", []):
                    text = desc_entry.get("text", "")
                    if not text:
                        continue
                    words = re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())
                    for word in words:
                        if len(word) > 4:
                            keyword_set.add(word)
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Failed to extract keywords from segment tree: {e}")

    if not keyword_set:
        for chunk in selected_chunks:
            desc = chunk.get("unified_description", chunk.get("description", ""))
            words = re.findall(r"[a-zA-Z][a-zA-Z'-]*", desc.lower())
            keyword_set.update(word for word in words if len(word) > 4)

    main_keywords = sorted(keyword_set)
    if verbose and main_keywords:
        print(f"  Main action keywords (top 5): {', '.join(main_keywords[:5])}")
    
    return main_keywords

