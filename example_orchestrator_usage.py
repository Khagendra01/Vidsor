"""Example usage of the orchestrator agent."""

from agent.orchestrator_runner import run_orchestrator

# Example 1: Find highlights
print("=" * 60)
print("EXAMPLE 1: Find Highlights")
print("=" * 60)

result = run_orchestrator(
    query="find the highlights of the video",
    timeline_path="projects/asdf/timeline.json",
    json_path="projects/asdf/segment_tree.json",
    video_path="projects/asdf/video/camp_5min.mp4",
    model_name="gpt-4o-mini",
    verbose=True
)

print(f"\nSuccess: {result.get('success')}")
print(f"Operation: {result.get('operation')}")

# Example 2: Cut timeline chunks
print("\n" + "=" * 60)
print("EXAMPLE 2: Cut Timeline Chunks")
print("=" * 60)

result = run_orchestrator(
    query="cut timeline index 0 and 1",
    timeline_path="projects/asdf/timeline.json",
    json_path="projects/asdf/segment_tree.json",
    video_path="projects/asdf/video/camp_5min.mp4",
    model_name="gpt-4o-mini",
    verbose=True
)

print(f"\nSuccess: {result.get('success')}")
print(f"Operation: {result.get('operation')}")

# Example 3: Replace chunks
print("\n" + "=" * 60)
print("EXAMPLE 3: Replace Timeline Chunks")
print("=" * 60)

result = run_orchestrator(
    query="replace timeline index 0-2 with cooking clips",
    timeline_path="projects/asdf/timeline.json",
    json_path="projects/asdf/segment_tree.json",
    video_path="projects/asdf/video/camp_5min.mp4",
    model_name="gpt-4o-mini",
    verbose=True
)

print(f"\nSuccess: {result.get('success')}")
print(f"Operation: {result.get('operation')}")

print("\n" + "=" * 60)
print("Examples complete!")
print("=" * 60)

