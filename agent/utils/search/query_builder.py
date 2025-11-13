"""Query message building utilities for search queries."""

from typing import Optional, Dict, Any, List


def format_content_inspection(content_inspection: dict) -> Dict[str, Any]:
    """
    Format content inspection data for prompts.
    
    Args:
        content_inspection: Dictionary from segment_tree.inspect_content()
        
    Returns:
        Dictionary with formatted content inspection data:
        - sample_descriptions_text: Formatted list of sample descriptions
        - summary: Content summary
        - keywords: Comma-separated keywords string
        - object_classes: Comma-separated object classes string
    """
    # Format sample descriptions
    sample_descriptions_text = []
    for i, desc in enumerate(content_inspection.get('sample_descriptions', [])[:10], 1):
        desc_type = desc.get('type', 'visual')
        time_info = f"[{desc.get('second', desc.get('time_range', [0])[0]):.1f}s]"
        full_desc = desc['description']
        sample_descriptions_text.append(f"{i}. {time_info} [{desc_type.upper()}] {full_desc}")
    
    return {
        "sample_descriptions_text": sample_descriptions_text,
        "sample_descriptions": "\n".join(sample_descriptions_text),
        "summary": content_inspection.get('summary', ''),
        "keywords": ', '.join(content_inspection.get('all_keywords', [])[:30]),
        "object_classes": ', '.join(sorted(content_inspection.get('object_classes', {}).keys())[:20])
    }


def format_video_narrative(video_narrative: dict) -> Dict[str, Any]:
    """
    Format video narrative data for prompts.
    
    Args:
        video_narrative: Dictionary with video narrative understanding
        
    Returns:
        Dictionary with formatted narrative data:
        - narrative_structure: Formatted narrative structure text
        - highlight_criteria: Formatted highlight criteria text
        - narrative_theme: Video theme
        - narrative_summary: Narrative summary
        - key_objects: Comma-separated key objects string
    """
    narrative_structure = video_narrative.get('narrative_structure', {})
    narrative_parts = []
    
    for part_name in ['intro', 'body', 'ending']:
        part = narrative_structure.get(part_name, {})
        if part:
            keywords = part.get('keywords', [])
            description = part.get('description', 'N/A')
            narrative_parts.append(f"- {part_name.upper()}: {description}")
            narrative_parts.append(f"  Keywords: {', '.join(keywords) if keywords else 'N/A'}")
    
    highlight_criteria = video_narrative.get('highlight_criteria', {})
    highlight_text = ""
    if highlight_criteria:
        highlight_keywords = highlight_criteria.get('keywords', [])
        highlight_desc = highlight_criteria.get('description', 'N/A')
        highlight_text = f"\nHighlight Criteria: {highlight_desc}\nHighlight Keywords: {', '.join(highlight_keywords) if highlight_keywords else 'N/A'}\n"
    
    return {
        "narrative_structure": "\n".join(narrative_parts),
        "highlight_criteria": highlight_text,
        "narrative_theme": video_narrative.get('video_theme', 'N/A'),
        "narrative_summary": video_narrative.get('narrative_summary', 'N/A'),
        "key_objects": ', '.join(video_narrative.get('key_objects', []))
    }


def format_content_inspection_for_narrative(content_inspection: dict, query: str) -> str:
    """
    Format content inspection data for video narrative creation prompt.
    
    Args:
        content_inspection: Dictionary from segment_tree.inspect_content()
        query: User query string
        
    Returns:
        Formatted context string for narrative creation
    """
    all_keywords = content_inspection['all_keywords'][:100]  # Get more keywords
    context = f"""Video Content Inspection:

AVAILABLE KEYWORDS (use ONLY these keywords - do not invent new ones):
{', '.join(all_keywords)}

Object classes detected: {', '.join(sorted(content_inspection['object_classes'].keys())[:20])}

Sample descriptions from video:
"""
    for i, desc in enumerate(content_inspection['sample_descriptions'][:10], 1):
        desc_type = desc.get('type', 'visual')
        time_info = f"[{desc.get('second', desc.get('time_range', [0])[0]):.1f}s]"
        context += f"{i}. {time_info} [{desc_type.upper()}] {desc['description'][:150]}\n"
    
    context += f"\nUser Query: {query}\n"
    context += "\nCRITICAL: Create a narrative structure (intro, body, ending) using ONLY keywords from the list above. "
    context += "Each narrative part should have keywords that actually exist in the hierarchical tree. "
    context += "What would be considered highlights using these actual keywords?"
    
    return context


def build_search_query_message(
    query: str,
    content_inspection: Optional[dict] = None,
    video_narrative: Optional[dict] = None,
    clip_contexts: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Build search query message using template pattern.
    Replaces multiple string concatenations with structured template.
    
    Args:
        query: User query string
        content_inspection: Optional content inspection data
        video_narrative: Optional video narrative data
        
    Returns:
        Formatted message string
    """
    # Base template
    template_parts = ["User query: {query}\n"]
    
    def _truncate(text: str, limit: int = 180) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    if content_inspection:
        # Format content inspection data
        formatted_inspection = format_content_inspection(content_inspection)
        
        # Content inspection section
        template_parts.append("""I've inspected the video content. Here's what's available:

{summary}

Sample keywords from video: {keywords}
Object classes: {object_classes}

CRITICAL: Sample descriptions from the video (analyze their style and vocabulary):
{sample_descriptions}

IMPORTANT: These are ACTUAL descriptions that will be searched. Analyze:
1. Vocabulary style: What words/phrases are used? (concrete nouns, action verbs, technical terms)
2. Sentence structure: How are descriptions phrased?
3. Common patterns: What patterns repeat? (e.g., "person doing X", "object visible", "camera perspective")
4. Content focus: What aspects are described? (objects, actions, locations, camera details)

Generate semantic queries that use SIMILAR vocabulary and structure to these descriptions.
For example, if descriptions say "person holding fish", generate queries like "person holding fish" not "amazing fishing moment".
""")
        
        # Add narrative section if available
        if video_narrative:
            formatted_narrative = format_video_narrative(video_narrative)
            
            template_parts.append("""
Video Narrative Understanding:
- Theme: {narrative_theme}
- Summary: {narrative_summary}

Narrative Structure (use keywords from each part):
{narrative_structure}
{highlight_criteria}
Key Objects: {key_objects}

CRITICAL INSTRUCTIONS FOR SEMANTIC QUERIES:
1. Analyze the sample descriptions above - they show HOW descriptions are written
2. Generate semantic queries using the SAME vocabulary and style as those descriptions
3. Use concrete, factual language: "person doing X", "object visible", "location Y"
4. Avoid abstract concepts: NO "amazing", "exciting", "memorable", "adventure" - use concrete actions/objects
5. Match description patterns: If descriptions say "person holding fish", query should say "person holding fish"
6. Include objects, actions, and locations that appear in the sample descriptions
7. For hierarchical keywords: Use actual keywords from the narrative structure
8. The goal: Generate queries that would match descriptions through semantic similarity (cosine similarity)
""")
        else:
            template_parts.append("""
Based on the sample descriptions above, generate semantic queries that match their style:
- Use concrete, factual language from the descriptions
- Match vocabulary: If descriptions say "person", "fish", "backpack", use those exact terms
- Match structure: If descriptions say "person holding X", use "person holding X" in queries
- Avoid abstract concepts: NO "amazing", "exciting", "memorable" - use concrete actions/objects
- Example: If descriptions say "person holding fish, backpack visible", query should be "person holding fish, backpack visible" (NOT "amazing fishing moment")
""")
    
    if clip_contexts:
        formatted_clips = []
        for i, ctx in enumerate(clip_contexts[:5], 1):
            duration = ctx.get("duration")
            time_range = ctx.get("time_range", [])
            duration_text = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "N/A"
            time_text = (
                f"{time_range[0]:.1f}s - {time_range[1]:.1f}s"
                if len(time_range) == 2
                else "N/A"
            )
            timeline_desc = ctx.get("timeline_description") or ctx.get("unified_description") or ""
            visual_desc = ctx.get("visual_descriptions") or []
            audio_desc = ctx.get("audio_descriptions") or []
            object_desc = ctx.get("top_objects") or []

            clip_lines = [
                f"{i}. Timeline index {ctx.get('timeline_index')} | Duration: {duration_text} | Source time: {time_text}",
            ]
            if timeline_desc:
                clip_lines.append(f"   Timeline label: {_truncate(timeline_desc, 160)}")
            if object_desc:
                clip_lines.append(f"   Dominant objects: {', '.join(object_desc)}")
            if visual_desc:
                clip_lines.append(f"   Visual cues: {', '.join(_truncate(v, 140) for v in visual_desc)}")
            if audio_desc:
                clip_lines.append(f"   Audio cues: {', '.join(_truncate(a, 140) for a in audio_desc)}")
            if ctx.get("narrative"):
                clip_lines.append(f"   Narrative recap: {_truncate(ctx['narrative'], 200)}")

            formatted_clips.append("\n".join(clip_lines))

        template_parts.append(
            """\nExisting timeline clip context (these clips will be replaced with shorter alternatives):\n"""
            + "\n".join(formatted_clips)
            + """
\nReplacement requirements:
- Find segments that capture the SAME core actions/objects/audio themes as the referenced clips
- Prefer clips that are SHORTER than the originals while staying faithful to the described content
- If multiple strong candidates exist, prioritize higher density of matching objects/actions
- Avoid generic filler; maintain the narrative intent implied by the clip context
"""
        )

    template_parts.append("Generate search queries and keywords for ALL search types. Return JSON only.")
    
    # Combine and format template
    full_template = "\n".join(template_parts)
    
    # Format with data
    if content_inspection:
        format_dict = {
            "query": query,
            "summary": formatted_inspection["summary"],
            "keywords": formatted_inspection["keywords"],
            "object_classes": formatted_inspection["object_classes"],
            "sample_descriptions": formatted_inspection["sample_descriptions"]
        }
        
        if video_narrative:
            formatted_narrative = format_video_narrative(video_narrative)
            format_dict.update({
                "narrative_theme": formatted_narrative["narrative_theme"],
                "narrative_summary": formatted_narrative["narrative_summary"],
                "narrative_structure": formatted_narrative["narrative_structure"],
                "highlight_criteria": formatted_narrative["highlight_criteria"],
                "key_objects": formatted_narrative["key_objects"]
            })
        
        format_dict["clip_contexts"] = clip_contexts
        return full_template.format(**format_dict)
    else:
        # Simple case without inspection
        format_dict = {"query": query, "clip_contexts": clip_contexts}
        return full_template.format(**format_dict)

