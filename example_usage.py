"""
Example usage of segment tree utilities for AI agents.
Demonstrates how to query video information from segment tree JSON.
"""

from segment_tree_utils import load_segment_tree
from generate_indexed_segment_tree import IndexedSegmentTreeQuery


def example_basic_queries():
    """Example: Basic queries using the utility functions."""
    print("=" * 60)
    print("Example 1: Basic Queries")
    print("=" * 60)
    
    # Load the segment tree
    query = load_segment_tree("camp_segment_tree.json")
    
    # Get video info
    info = query.get_video_info()
    print(f"\nVideo: {info['video']}")
    print(f"Duration: {info['duration_seconds']} seconds")
    print(f"FPS: {info['fps']}")
    
    # Find all boats
    boats = query.find_objects_by_class("boat")
    print(f"\nFound {len(boats)} boat detections")
    if boats:
        print(f"First boat at second {boats[0]['second']}, time {boats[0]['time_range']}")
    
    # Find all persons
    persons = query.find_objects_by_class("person")
    print(f"\nFound {len(persons)} person detections")
    
    # Get all unique classes
    all_classes = query.get_all_classes()
    print(f"\nAll object classes: {list(all_classes.keys())}")
    print(f"Class counts: {all_classes}")


def example_track_queries():
    """Example: Tracking specific objects across time."""
    print("\n" + "=" * 60)
    print("Example 2: Track Queries")
    print("=" * 60)
    
    query = load_segment_tree("camp_segment_tree.json")
    
    # Get all tracks
    all_tracks = query.get_all_tracks()
    print(f"\nTotal unique tracks: {len(all_tracks)}")
    
    # Follow a specific track
    if all_tracks:
        track_id = list(all_tracks)[0]
        timeline = query.find_track_timeline(track_id)
        print(f"\nTrack {track_id} appears in {len(timeline)} detection groups")
        if timeline:
            print(f"  First appearance: second {timeline[0]['second']}")
            print(f"  Last appearance: second {timeline[-1]['second']}")


def example_description_search():
    """Example: Searching descriptions for keywords."""
    print("\n" + "=" * 60)
    print("Example 3: Description Search")
    print("=" * 60)
    
    query = load_segment_tree("camp_segment_tree.json")
    
    # Search for "mountain"
    results = query.search_descriptions("mountain")
    print(f"\nFound 'mountain' in {len(results)} seconds")
    for result in results[:3]:  # Show first 3
        print(f"  Second {result['second']}: {result['matches'][0]['description'][:60]}...")
    
    # Search for "fishing"
    results = query.search_descriptions("fishing")
    print(f"\nFound 'fishing' in {len(results)} seconds")
    
    # Search for "boat"
    results = query.search_descriptions("boat")
    print(f"Found 'boat' in {len(results)} seconds")


def example_time_range_queries():
    """Example: Querying specific time ranges."""
    print("\n" + "=" * 60)
    print("Example 4: Time Range Queries")
    print("=" * 60)
    
    query = load_segment_tree("camp_segment_tree.json")
    
    # Get scene summary for first 10 seconds
    summary = query.get_scene_summary(time_start=0, time_end=10)
    print(f"\nScene summary (0-10 seconds):")
    print(f"  Seconds covered: {summary['seconds_count']}")
    print(f"  Objects detected: {list(summary['objects'].keys())}")
    print(f"  Total detections: {summary['total_detections']}")
    
    # Get all objects in a specific time range
    objects = query.find_objects_in_time_range(5, 15)
    print(f"\nObjects in time range 5-15 seconds:")
    print(f"  Unique classes: {list(objects['objects'].keys())}")
    print(f"  Unique tracks: {len(objects['unique_tracks'])}")
    print(f"  Total detections: {objects['total_detections']}")


def example_indexed_queries():
    """Example: Using the indexed version for faster queries."""
    print("\n" + "=" * 60)
    print("Example 5: Indexed Queries (Fast)")
    print("=" * 60)
    
    try:
        # Try to load indexed version
        indexed_query = IndexedSegmentTreeQuery("camp_segment_tree_indexed.json")
        
        # Get statistics
        stats = indexed_query.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total seconds: {stats['total_seconds']}")
        print(f"  Unique classes: {stats['unique_classes']}")
        print(f"  Unique tracks: {stats['unique_tracks']}")
        
        # Fast class lookup
        boats = indexed_query.find_objects_by_class_fast("boat")
        print(f"\nFast lookup: Found {len(boats)} boat occurrences")
        
        # Fast track lookup
        if boats:
            track_id = boats[0]['track_id']
            timeline = indexed_query.find_track_timeline_fast(track_id)
            print(f"Fast lookup: Track {track_id} has {len(timeline)} occurrences")
        
        # Fast keyword search
        results = indexed_query.search_descriptions_fast("mountain")
        print(f"Fast lookup: Found 'mountain' in {len(results)} seconds")
        
        # Most common classes
        top_classes = indexed_query.get_most_common_classes(5)
        print(f"\nTop 5 most common classes:")
        for class_name, count in top_classes:
            print(f"  {class_name}: {count}")
            
    except FileNotFoundError:
        print("\nIndexed file not found. Run generate_indexed_segment_tree.py first:")
        print("  python generate_indexed_segment_tree.py --input camp_segment_tree.json --output camp_segment_tree_indexed.json")


def example_ai_agent_use_case():
    """Example: How an AI agent would use these utilities."""
    print("\n" + "=" * 60)
    print("Example 6: AI Agent Use Case")
    print("=" * 60)
    
    query = load_segment_tree("camp_segment_tree.json")
    
    # Simulate an AI agent query: "What objects appear in the video?"
    print("\nAI Agent Query: 'What objects appear in the video?'")
    all_classes = query.get_all_classes()
    print(f"Answer: The video contains {len(all_classes)} types of objects:")
    for class_name, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {class_name}: {count} detections")
    
    # Simulate: "When does fishing happen?"
    print("\nAI Agent Query: 'When does fishing happen?'")
    fishing_results = query.search_descriptions("fishing")
    if fishing_results:
        for result in fishing_results:
            print(f"  Fishing scene at second {result['second']}, time {result['time_range']}")
    else:
        print("  No explicit 'fishing' mentions found")
    
    # Simulate: "Show me all boats"
    print("\nAI Agent Query: 'Show me all boats'")
    boats = query.find_objects_by_class("boat")
    print(f"  Found {len(boats)} boat detections:")
    for i, boat in enumerate(boats[:5], 1):  # Show first 5
        print(f"    {i}. Second {boat['second']}, track {boat['detection']['track_id']}, "
              f"confidence {boat['detection']['confidence']:.2f}")
    
    # Simulate: "What happens between 10-20 seconds?"
    print("\nAI Agent Query: 'What happens between 10-20 seconds?'")
    summary = query.get_scene_summary(time_start=10, time_end=20)
    print(f"  Time range: {summary['time_range']}")
    print(f"  Objects: {list(summary['objects'].keys())}")
    if summary['descriptions']:
        print(f"  Descriptions: {len(summary['descriptions'])} found")
        for desc in summary['descriptions'][:2]:
            print(f"    - [{desc['type']}] {desc['text'][:60]}...")


def example_narrative_description():
    """Example: Get concatenated narrative description for a time range."""
    print("\n" + "=" * 60)
    print("Example 7: Narrative Description (Concatenated)")
    print("=" * 60)
    
    query = load_segment_tree("camp_segment_tree.json")
    
    # Get narrative for seconds 0-5
    print("\nNarrative for seconds 0-5:")
    narrative = query.get_narrative_description(0, 5)
    print(f"  Time range: {narrative['time_range']}")
    print(f"  Descriptions found: {narrative['description_count']}")
    print(f"\n  Narrative:")
    print(f"  {narrative['narrative'][:300]}...")
    
    # Get narrative with timestamps
    print("\nNarrative for seconds 3-7 (with timestamps):")
    narrative = query.get_narrative_description(3, 7, include_timestamps=True)
    print(f"  {narrative['narrative'][:400]}...")
    
    # Get narrative for a longer range
    print("\nNarrative for seconds 10-20:")
    narrative = query.get_narrative_description(10, 20)
    print(f"  Seconds covered: {narrative['seconds_covered']}")
    print(f"  Description count: {narrative['description_count']}")
    print(f"\n  Full narrative:")
    print(f"  {narrative['narrative'][:500]}...")
    
    # Show individual descriptions
    print(f"\n  Individual descriptions ({len(narrative['descriptions'])}):")
    for i, desc in enumerate(narrative['descriptions'][:5], 1):
        print(f"    {i}. [{desc['type']}] Second {desc['second']}: {desc['text'][:70]}...")


if __name__ == "__main__":
    # Run all examples
    example_basic_queries()
    example_track_queries()
    example_description_search()
    example_time_range_queries()
    example_indexed_queries()
    example_ai_agent_use_case()
    example_narrative_description()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

