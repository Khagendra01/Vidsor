"""
Generate a comprehensive JSON file with all BLIP vs OFA frame-by-frame comparisons.
Organized for easy reading and analysis.
"""

import json
from pathlib import Path
from typing import Dict, List


def generate_comparison_json(blip_path: str = "blip_results.json", 
                            ofa_path: str = "ofa_results.json",
                            output_path: str = "frame_by_frame_comparison.json"):
    """Generate comprehensive frame-by-frame comparison JSON."""
    script_dir = Path(__file__).parent
    blip_file = script_dir / blip_path
    ofa_file = script_dir / ofa_path
    output_file = script_dir / output_path
    
    print("Loading results...")
    # Load results
    with open(blip_file, 'r') as f:
        blip_data = json.load(f)
    with open(ofa_file, 'r') as f:
        ofa_data = json.load(f)
    
    blip_stats = blip_data["statistics"]
    ofa_stats = ofa_data["statistics"]
    blip_dict = {r["frame_number"]: r for r in blip_data["results"]}
    ofa_dict = {r["frame_number"]: r for r in ofa_data["results"]}
    
    # Get common frames
    common_frames = sorted(set(blip_dict.keys()) & set(ofa_dict.keys()))
    
    print(f"Processing {len(common_frames)} frames...")
    
    # Calculate similarity for each frame
    frame_comparisons = []
    for frame_num in common_frames:
        blip_result = blip_dict[frame_num]
        ofa_result = ofa_dict[frame_num]
        
        blip_caption = blip_result["caption"]
        ofa_caption = ofa_result["caption"]
        
        # Calculate similarity
        blip_words = set(blip_caption.lower().split())
        ofa_words = set(ofa_caption.lower().split())
        common_words = blip_words & ofa_words
        total_unique_words = len(blip_words | ofa_words)
        similarity = (len(common_words) / total_unique_words * 100) if total_unique_words > 0 else 0
        
        # Determine which is more detailed
        blip_word_count = len(blip_caption.split())
        ofa_word_count = len(ofa_caption.split())
        blip_char_count = len(blip_caption)
        ofa_char_count = len(ofa_caption)
        
        frame_comparisons.append({
            "frame_number": frame_num,
            "second": blip_result["second"],
            "blip": {
                "caption": blip_caption,
                "processing_time": blip_result["processing_time"],
                "word_count": blip_word_count,
                "character_count": blip_char_count
            },
            "ofa": {
                "caption": ofa_caption,
                "processing_time": ofa_result["processing_time"],
                "word_count": ofa_word_count,
                "character_count": ofa_char_count
            },
            "comparison": {
                "similarity_percent": round(similarity, 1),
                "common_words": sorted(list(common_words)),
                "common_word_count": len(common_words),
                "speed_difference": round(ofa_result["processing_time"] - blip_result["processing_time"], 3),
                "speedup_ratio": round(ofa_result["processing_time"] / blip_result["processing_time"], 2) if blip_result["processing_time"] > 0 else 0,
                "blip_faster": blip_result["processing_time"] < ofa_result["processing_time"],
                "word_count_difference": ofa_word_count - blip_word_count,
                "character_count_difference": ofa_char_count - blip_char_count
            }
        })
    
    # Create comprehensive output
    output_data = {
        "metadata": {
            "video_path": str(blip_data["video_path"]),
            "blip_model": blip_data["model"],
            "ofa_model": ofa_data["model"],
            "total_frames_compared": len(common_frames),
            "test_config": {
                "frame_interval": blip_data["test_config"]["frame_interval"],
                "max_frames": blip_data["test_config"]["max_frames"],
                "batch_size": blip_data["test_config"]["batch_size"]
            }
        },
        "summary_statistics": {
            "speed": {
                "blip": {
                    "total_time": blip_stats["total_processing_time"],
                    "avg_time_per_frame": blip_stats["avg_time_per_frame"],
                    "fps": blip_stats["fps"],
                    "min_time": blip_stats["min_time"],
                    "max_time": blip_stats["max_time"]
                },
                "ofa": {
                    "total_time": ofa_stats["total_processing_time"],
                    "avg_time_per_frame": ofa_stats["avg_time_per_frame"],
                    "fps": ofa_stats["fps"],
                    "min_time": ofa_stats["min_time"],
                    "max_time": ofa_stats["max_time"]
                },
                "comparison": {
                    "blip_faster_by": round(ofa_stats["avg_time_per_frame"] / blip_stats["avg_time_per_frame"], 1) if blip_stats["avg_time_per_frame"] > 0 else 0,
                    "time_difference_per_frame": round(ofa_stats["avg_time_per_frame"] - blip_stats["avg_time_per_frame"], 3),
                    "total_time_difference": round(ofa_stats["total_processing_time"] - blip_stats["total_processing_time"], 2)
                }
            },
            "quality": {
                "blip": {
                    "avg_caption_length": round(sum(f["blip"]["character_count"] for f in frame_comparisons) / len(frame_comparisons), 1),
                    "avg_words_per_caption": round(sum(f["blip"]["word_count"] for f in frame_comparisons) / len(frame_comparisons), 1)
                },
                "ofa": {
                    "avg_caption_length": round(sum(f["ofa"]["character_count"] for f in frame_comparisons) / len(frame_comparisons), 1),
                    "avg_words_per_caption": round(sum(f["ofa"]["word_count"] for f in frame_comparisons) / len(frame_comparisons), 1)
                },
                "avg_similarity": round(sum(f["comparison"]["similarity_percent"] for f in frame_comparisons) / len(frame_comparisons), 1)
            }
        },
        "frame_comparisons": frame_comparisons
    }
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Comparison JSON saved to: {output_file}")
    print(f"   Total frames: {len(common_frames)}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"BLIP is {output_data['summary_statistics']['speed']['comparison']['blip_faster_by']:.1f}x faster")
    print(f"Average similarity: {output_data['summary_statistics']['quality']['avg_similarity']:.1f}%")
    print(f"BLIP avg words: {output_data['summary_statistics']['quality']['blip']['avg_words_per_caption']:.1f}")
    print(f"OFA avg words: {output_data['summary_statistics']['quality']['ofa']['avg_words_per_caption']:.1f}")
    print("="*80)
    
    return output_file


if __name__ == "__main__":
    import sys
    
    blip_file = sys.argv[1] if len(sys.argv) > 1 else "blip_results.json"
    ofa_file = sys.argv[2] if len(sys.argv) > 2 else "ofa_results.json"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "frame_by_frame_comparison.json"
    
    generate_comparison_json(blip_file, ofa_file, output_file)

