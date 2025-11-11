"""Check what movements/actions are in the extracted clips."""
import json

# Load JSON directly to avoid heavy imports
with open("camp_segment_tree.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Time ranges from the extracted clips
clips = [
    (33, 42),
    (43, 52),
    (58, 67),
    (118, 127),
    (153, 162),
    (198, 207),
    (323, 332),
    (348, 357),
    (423, 432),
    (463, 472),
]

seconds_data = data.get("seconds", [])
transcriptions = data.get("transcriptions", [])

print("\n" + "=" * 80)
print("MOVEMENTS/ACTIONS FOUND IN EXTRACTED CLIPS")
print("=" * 80)

for i, (start, end) in enumerate(clips, 1):
    print(f"\n{'='*80}")
    print(f"CLIP {i}: {start}s - {end}s")
    print(f"{'='*80}")
    
    # Get visual descriptions for this time range
    print("\n[VISUAL DESCRIPTIONS]")
    descriptions_found = False
    for second_data in seconds_data:
        time_range = second_data.get("time_range", [])
        if time_range and time_range[0] >= start and time_range[0] < end:
            unified_desc = second_data.get("unified_description", "")
            if unified_desc and unified_desc.lower() != "0":
                print(f"  {time_range[0]:.1f}s: {unified_desc[:200]}")
                descriptions_found = True
    
    if not descriptions_found:
        print("  (No visual descriptions found)")
    
    # Get transcriptions for this time range
    print("\n[AUDIO TRANSCRIPTIONS]")
    found_transcriptions = []
    for tr in transcriptions:
        tr_range = tr.get("time_range", [])
        if tr_range and len(tr_range) >= 2:
            tr_start, tr_end = tr_range[0], tr_range[1]
            if tr_end >= start and tr_start <= end:
                found_transcriptions.append(tr)
    
    if found_transcriptions:
        for tr in found_transcriptions:
            text = tr.get("transcription", "").strip()
            if text:
                tr_range = tr.get("time_range", [])
                print(f"  {tr_range[0]:.1f}s-{tr_range[1]:.1f}s: {text}")
    else:
        print("  (No transcriptions found)")
    
    # Get detected objects
    print("\n[DETECTED OBJECTS]")
    objects_found = {}
    for second_data in seconds_data:
        time_range = second_data.get("time_range", [])
        if time_range and time_range[0] >= start and time_range[0] < end:
            for group in second_data.get("detection_groups", []):
                for detection in group.get("detections", []):
                    class_name = detection.get("class_name", "")
                    if class_name:
                        objects_found[class_name] = objects_found.get(class_name, 0) + 1
    
    if objects_found:
        for obj, count in sorted(objects_found.items(), key=lambda x: x[1], reverse=True):
            print(f"  {obj}: {count} detection(s)")
    else:
        print("  (No objects detected)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total clips analyzed: {len(clips)}")

