import opentimelineio as otio
import os

# Path to the video file
video_path = "kbc.mp4"

if not os.path.exists(video_path):
    print(f"Error: {video_path} not found!")
    exit(1)

print(f"Processing {video_path}...")

# Create a timeline
timeline = otio.schema.Timeline(name="kbc_timeline")

# Create a track
track = otio.schema.Track(name="video_track")

# Create a clip from the video file
# Set a default time range (24 fps, 100 frames as placeholder)
# In production, you'd get actual duration from video metadata
default_duration = otio.opentime.RationalTime(100, 24)
media_ref = otio.schema.ExternalReference(
    target_url=video_path,
    available_range=otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, 24),
        duration=default_duration
    )
)

clip = otio.schema.Clip(
    name="kbc_clip",
    media_reference=media_ref,
    source_range=otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, 24),
        duration=default_duration
    )
)

# Add clip to track
track.append(clip)

# Add track to timeline
timeline.tracks.append(track)

# Print timeline information
print(f"\nTimeline created: {timeline.name}")
print(f"Number of tracks: {len(timeline.tracks)}")
print(f"Duration: {timeline.duration()}")

# Save as OTIO file
output_file = "kbc.otio"
otio.adapters.write_to_file(timeline, output_file)
print(f"\nTimeline saved to {output_file}")

# Display detailed information
print("\n=== Timeline Details ===")
for i, track in enumerate(timeline.tracks):
    print(f"\nTrack {i+1}: {track.name}")
    for j, item in enumerate(track):
        print(f"  Item {j+1}: {item.name}")
        if hasattr(item, 'media_reference') and item.media_reference:
            print(f"    Media: {item.media_reference.target_url}")

print("\nProcessing complete!")

