"""
Enhanced prompts for video analysis that include technical camera/video details.
These prompts ask models to describe both content AND technical aspects of video footage.
"""

# Base prompt for LLaVA unified descriptions with technical details
LLAVA_UNIFIED_PROMPT = """Based on these image descriptions and object detections, provide a unified detailed description of this 1-second scene. Include:

1. Content Description - What is happening in the scene:
   - Objects, people, actions, and location
   - Visual details (colors, positions, movements)
   - Scene context and setting

2. Camera Perspective and Technical Details:
   - Camera perspective: Is this first-person (POV), third-person, overhead, or other?
   - Camera positioning: How is the camera positioned? (handheld, mounted, body-mounted, backpack-mounted, tripod, etc.)
   - Field of view: Is it wide-angle showing many objects simultaneously, or narrow focus on specific subjects?
   - Camera movement: Is the camera stable, shaky, panning, or following motion?
   - Lens characteristics: Any signs of wide-angle distortion, fisheye effect, or telephoto compression?

3. Video Production Style and Characteristics:
   - Video style: Action camera (GoPro-style), documentary, narrative, test footage, etc.
   - Lighting: Natural outdoor lighting, indoor lighting, low light, etc.
   - Frame composition: What objects/people are visible and their spatial relationships
   - Any technical artifacts: Motion blur, lens flare, compression artifacts, etc.

Image Descriptions:
{descriptions_text}

Object Detections Summary:
{detection_summary}

Provide a comprehensive description covering both content and technical aspects of the video."""


# Enhanced prompt for BLIP (if using prompt-based BLIP variant)
BLIP_ENHANCED_PROMPT = """Describe this image in detail, including:
- What objects, people, and actions are visible
- The camera perspective and angle
- The field of view (wide or narrow)
- Any technical characteristics of the video frame"""


# Alternative shorter prompt for faster processing
LLAVA_UNIFIED_PROMPT_SHORT = """Based on these image descriptions and object detections, provide a unified detailed description of this 1-second scene covering:

1. Content: What is happening (objects, people, actions, location)
2. Camera: Perspective (first-person/third-person), positioning (handheld/mounted), field of view (wide/narrow)
3. Style: Video production style and technical characteristics

Image Descriptions:
{descriptions_text}

Object Detections Summary:
{detection_summary}

Provide a comprehensive description."""


# Prompt specifically for camera technical analysis
CAMERA_TECHNICAL_PROMPT = """Analyze the technical camera and video characteristics of this scene:

1. Camera Perspective:
   - First-person (POV), third-person, overhead, or other?
   - Evidence of camera mounting (body-mounted, backpack-mounted, handheld, etc.)

2. Lens and Field of View:
   - Wide-angle (showing many objects), normal, or telephoto?
   - Any lens distortion visible?
   - How much of the scene is visible?

3. Camera Movement and Stability:
   - Stable, shaky, or smooth movement?
   - Handheld vs mounted characteristics
   - Any camera motion patterns?

4. Video Production Style:
   - Action camera style, documentary, narrative, test footage?
   - Professional vs consumer camera characteristics

Image Descriptions:
{descriptions_text}

Object Detections Summary:
{detection_summary}

Focus specifically on technical camera and video production aspects."""


# Prompt for content-only (original style, for comparison)
LLAVA_CONTENT_ONLY_PROMPT = """Based on these image descriptions and object detections, provide a unified detailed description of this 1-second scene:

Image Descriptions:
{descriptions_text}

Object Detections Summary:
{detection_summary}

Provide a comprehensive description covering the entire second."""


def get_llava_prompt(descriptions_text: str, detection_summary: str, include_technical: bool = True, short: bool = False) -> str:
    """
    Get LLaVA prompt with or without technical details.
    
    Args:
        descriptions_text: Formatted BLIP descriptions
        detection_summary: Object detection summary
        include_technical: Whether to include technical camera/video details
        short: Whether to use shorter prompt version
    
    Returns:
        Formatted prompt string
    """
    if include_technical:
        if short:
            template = LLAVA_UNIFIED_PROMPT_SHORT
        else:
            template = LLAVA_UNIFIED_PROMPT
    else:
        template = LLAVA_CONTENT_ONLY_PROMPT
    
    return template.format(
        descriptions_text=descriptions_text,
        detection_summary=detection_summary
    )


def get_camera_technical_prompt(descriptions_text: str, detection_summary: str) -> str:
    """
    Get prompt focused specifically on camera technical analysis.
    
    Args:
        descriptions_text: Formatted BLIP descriptions
        detection_summary: Object detection summary
    
    Returns:
        Formatted prompt string
    """
    return CAMERA_TECHNICAL_PROMPT.format(
        descriptions_text=descriptions_text,
        detection_summary=detection_summary
    )


# Example usage:
if __name__ == "__main__":
    # Example usage
    example_descriptions = """1. a man holding a camera in the woods
2. a man in a forest holding a camera
3. a man in the woods with a camera"""
    
    example_detections = "Person detected with high confidence. Backpack detected."
    
    print("=" * 80)
    print("EXAMPLE: Enhanced Prompt with Technical Details")
    print("=" * 80)
    print(get_llava_prompt(example_descriptions, example_detections, include_technical=True))
    print()
    
    print("=" * 80)
    print("EXAMPLE: Short Version")
    print("=" * 80)
    print(get_llava_prompt(example_descriptions, example_detections, include_technical=True, short=True))
    print()
    
    print("=" * 80)
    print("EXAMPLE: Camera Technical Only")
    print("=" * 80)
    print(get_camera_technical_prompt(example_descriptions, example_detections))
    print()
    
    print("=" * 80)
    print("EXAMPLE: Content Only (Original)")
    print("=" * 80)
    print(get_llava_prompt(example_descriptions, example_detections, include_technical=False))

