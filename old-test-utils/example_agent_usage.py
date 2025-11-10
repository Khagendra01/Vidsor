"""
Example usage of the video clip extraction agent.
"""

from video_clip_agent import run_agent

def example_fish_catching():
    """Example: Find moments where they catch fish."""
    print("=" * 60)
    print("Example 1: Finding fish catching moments")
    print("=" * 60)
    
    query = "find moments where they catch fish"
    result = run_agent(
        query=query,
        json_path="camp_segment_tree.json",
        video_path="camp.mp4",
        model_name="gpt-4o-mini"
    )
    
    print(f"\nQuery: {query}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Time ranges: {result.get('time_ranges', [])}")
    print(f"Clips saved: {len(result.get('output_clips', []))}")
    
    if result.get('output_clips'):
        print("\nSaved clips:")
        for clip in result['output_clips']:
            print(f"  - {clip}")


def example_audio_search():
    """Example: Search for specific audio mentions."""
    print("\n" + "=" * 60)
    print("Example 2: Searching audio transcriptions")
    print("=" * 60)
    
    query = "find when someone says Alaska"
    result = run_agent(
        query=query,
        json_path="camp_segment_tree.json",
        video_path="camp.mp4",
        model_name="gpt-4o-mini"
    )
    
    print(f"\nQuery: {query}")
    if result.get('needs_clarification'):
        print(f"Clarification needed: {result.get('clarification_question')}")
    else:
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Time ranges: {result.get('time_ranges', [])}")
        print(f"Clips saved: {len(result.get('output_clips', []))}")


def example_object_search():
    """Example: Find specific objects."""
    print("\n" + "=" * 60)
    print("Example 3: Finding objects")
    print("=" * 60)
    
    query = "show me all boats in the video"
    result = run_agent(
        query=query,
        json_path="camp_segment_tree.json",
        video_path="camp.mp4",
        model_name="gpt-4o-mini"
    )
    
    print(f"\nQuery: {query}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Time ranges: {result.get('time_ranges', [])}")
    print(f"Clips saved: {len(result.get('output_clips', []))}")


if __name__ == "__main__":
    # Run examples
    try:
        example_fish_catching()
        # example_audio_search()
        # example_object_search()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("2. Installed required packages: pip install -r requirements_agent.txt")
        print("3. The video file (camp.mp4) and JSON file (camp_segment_tree.json) exist")

