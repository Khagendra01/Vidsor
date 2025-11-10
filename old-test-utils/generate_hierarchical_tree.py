"""
Generate hierarchical keyword tree from segment tree JSON.
Creates a tree structure with 5-second leaf nodes containing filtered keywords,
and parent nodes that concatenate children's keywords up to the root.
"""

import json
import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
import math


# Comprehensive stop words list
STOP_WORDS = {
    # Articles
    'the', 'a', 'an',
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    # Common verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'can', 'shall', 'ought',
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
    'through', 'during', 'including', 'against', 'among', 'throughout', 'despite',
    'towards', 'upon', 'concerning', 'to', 'of', 'in', 'for', 'on', 'at', 'by',
    # Conjunctions
    'and', 'or', 'but', 'if', 'then', 'so', 'than', 'as', 'because', 'since', 'while',
    'although', 'though', 'unless', 'until', 'when', 'where', 'whether',
    # Adverbs
    'very', 'just', 'only', 'also', 'even', 'still', 'already', 'yet', 'more', 'most',
    'here', 'there', 'where', 'when', 'how', 'why', 'what', 'which', 'who', 'whom',
    # Other common words
    'one', 'two', 'three', 'first', 'second', 'third', 'some', 'any', 'all', 'each',
    'every', 'both', 'few', 'many', 'other', 'such', 'no', 'nor', 'not', 'too',
    'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should',
    'now', 'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew',
    'think', 'thought', 'take', 'took', 'make', 'made', 'give', 'gave', 'say', 'said',
    'tell', 'told', 'ask', 'asked', 'try', 'tried', 'use', 'used', 'find', 'found',
    'want', 'wanted', 'need', 'needed', 'look', 'looked', 'work', 'worked', 'call',
    'called', 'seem', 'seemed', 'feel', 'felt', 'become', 'became', 'leave', 'left',
    'put', 'let', 'help', 'helped', 'show', 'showed', 'move', 'moved', 'live', 'lived',
    'believe', 'believed', 'bring', 'brought', 'happen', 'happened', 'write', 'wrote',
    'sit', 'sat', 'stand', 'stood', 'lose', 'lost', 'pay', 'paid', 'meet', 'met',
    'include', 'included', 'continue', 'continued', 'set', 'set', 'learn', 'learned',
    'change', 'changed', 'lead', 'led', 'understand', 'understood', 'watch', 'watched',
    'follow', 'followed', 'stop', 'stopped', 'create', 'created', 'speak', 'spoke',
    'read', 'read', 'allow', 'allowed', 'add', 'added', 'spend', 'spent', 'grow', 'grew',
    'open', 'opened', 'walk', 'walked', 'win', 'won', 'offer', 'offered', 'remember',
    'remembered', 'love', 'loved', 'consider', 'considered', 'appear', 'appeared',
    'buy', 'bought', 'wait', 'waited', 'serve', 'served', 'die', 'died', 'send', 'sent',
    'build', 'built', 'stay', 'stayed', 'fall', 'fell', 'cut', 'cut', 'reach', 'reached',
    'kill', 'killed', 'raise', 'raised', 'pass', 'passed', 'sell', 'sold', 'decide',
    'decided', 'return', 'returned', 'explain', 'explained', 'develop', 'developed',
    'carry', 'carried', 'break', 'broke', 'receive', 'received', 'agree', 'agreed',
    'support', 'supported', 'hit', 'hit', 'produce', 'produced', 'eat', 'ate',
    'cover', 'covered', 'catch', 'caught', 'draw', 'drew', 'choose', 'chose'
}


def extract_keywords(text: str, min_length: int = 2) -> Set[str]:
    """
    Extract keywords from text, removing stop words.
    
    Args:
        text: Text to extract keywords from
        min_length: Minimum keyword length
        
    Returns:
        Set of lowercase keywords
    """
    if not text or text.lower() == "0":
        return set()
    
    # Split on whitespace and punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stop words and short words
    keywords = {w for w in words if len(w) >= min_length and w not in STOP_WORDS}
    
    return keywords


def get_transcriptions_for_time_range(transcriptions: List[Dict], 
                                      time_start: float, 
                                      time_end: float) -> List[Dict]:
    """Get all transcriptions that overlap with a time range."""
    results = []
    for trans in transcriptions:
        tr_range = trans.get("time_range", [])
        if not tr_range or len(tr_range) < 2:
            continue
        tr_start, tr_end = tr_range[0], tr_range[1]
        # Check if transcription overlaps with the requested range
        if tr_end >= time_start and tr_start <= time_end:
            results.append(trans)
    return sorted(results, key=lambda x: x.get("time_range", [0])[0])


def create_leaf_node(node_id: str, 
                    time_start: float, 
                    time_end: float,
                    seconds_data: List[Dict],
                    transcriptions: List[Dict]) -> Dict[str, Any]:
    """
    Create a 5-second leaf node by combining visual and audio data.
    
    Args:
        node_id: Unique node identifier
        time_start: Start time in seconds
        time_end: End time in seconds
        seconds_data: All second-level data
        transcriptions: All transcription data
        
    Returns:
        Leaf node dictionary
    """
    # Get seconds in this range
    relevant_seconds = []
    for sec in seconds_data:
        sec_time_range = sec.get("time_range", [])
        if sec_time_range and sec_time_range[0] >= time_start and sec_time_range[0] < time_end:
            relevant_seconds.append(sec)
    
    # Get transcriptions in this range
    relevant_transcriptions = get_transcriptions_for_time_range(
        transcriptions, time_start, time_end
    )
    
    # Combine visual descriptions
    visual_texts = []
    source_second_indices = []
    for sec in relevant_seconds:
        unified_desc = sec.get("unified_description", "")
        if unified_desc and unified_desc.lower() != "0":
            visual_texts.append(unified_desc)
        source_second_indices.append(sec.get("second", 0))
    
    # Combine audio transcriptions
    audio_texts = []
    source_transcription_ids = []
    for trans in relevant_transcriptions:
        trans_text = trans.get("transcription", "").strip()
        if trans_text:
            audio_texts.append(trans_text)
        source_transcription_ids.append(trans.get("id"))
    
    # Combine all text
    visual_text = " ".join(visual_texts)
    audio_text = " ".join(audio_texts)
    combined_text = f"{visual_text} {audio_text}".strip()
    
    # Extract keywords
    visual_keywords = set()
    for text in visual_texts:
        visual_keywords.update(extract_keywords(text))
    
    audio_keywords = set()
    for text in audio_texts:
        audio_keywords.update(extract_keywords(text))
    
    all_keywords = visual_keywords.union(audio_keywords)
    
    return {
        "node_id": node_id,
        "time_range": [time_start, time_end],
        "duration": time_end - time_start,
        "keywords": sorted(list(all_keywords)),
        "keyword_count": len(all_keywords),
        "children": None,
        "level": None,  # Will be set later
        "parent": None,  # Will be set later
        "source_seconds": sorted(source_second_indices),
        "source_transcriptions": sorted(source_transcription_ids),
        "visual_text": visual_text,
        "audio_text": audio_text,
        "combined_text": combined_text
    }


def create_parent_node(node_id: str,
                       children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a parent node by merging children.
    
    Args:
        node_id: Unique node identifier
        children: List of child nodes
        
    Returns:
        Parent node dictionary
    """
    if not children:
        return None
    
    # Merge time ranges
    time_ranges = [child["time_range"] for child in children]
    time_start = min(tr[0] for tr in time_ranges)
    time_end = max(tr[1] for tr in time_ranges)
    
    # Merge keywords (deduplicate)
    all_keywords = set()
    for child in children:
        all_keywords.update(child.get("keywords", []))
    
    # Get child node IDs
    child_ids = [child["node_id"] for child in children]
    
    return {
        "node_id": node_id,
        "time_range": [time_start, time_end],
        "duration": time_end - time_start,
        "keywords": sorted(list(all_keywords)),
        "keyword_count": len(all_keywords),
        "children": child_ids,
        "level": None,  # Will be set later
        "parent": None  # Will be set later
    }


def build_hierarchical_tree(seconds_data: List[Dict],
                           transcriptions: List[Dict],
                           leaf_duration: float = 5.0,
                           branching_factor: int = 2) -> Dict[str, Any]:
    """
    Build hierarchical tree structure.
    
    Args:
        seconds_data: All second-level data
        transcriptions: All transcription data
        leaf_duration: Duration of leaf nodes in seconds (default: 5)
        branching_factor: Number of children per parent (default: 2)
        
    Returns:
        Complete hierarchical tree structure
    """
    print(f"Building hierarchical tree with {leaf_duration}s leaf nodes...")
    
    # Calculate total duration
    if not seconds_data:
        return None
    
    total_duration = seconds_data[-1].get("time_range", [0, 0])[1]
    num_leaves = math.ceil(total_duration / leaf_duration)
    
    print(f"  Total duration: {total_duration:.1f}s")
    print(f"  Number of leaf nodes: {num_leaves}")
    
    # Create leaf nodes
    print("  Creating leaf nodes...")
    leaf_nodes = {}
    for i in range(num_leaves):
        time_start = i * leaf_duration
        time_end = min((i + 1) * leaf_duration, total_duration)
        node_id = f"leaf_{i}"
        
        leaf_node = create_leaf_node(
            node_id, time_start, time_end, seconds_data, transcriptions
        )
        leaf_nodes[node_id] = leaf_node
        
        if (i + 1) % 50 == 0:
            print(f"    Created {i + 1}/{num_leaves} leaf nodes...")
    
    print(f"  Created {len(leaf_nodes)} leaf nodes")
    
    # Build parent levels bottom-up
    print("  Building parent levels...")
    all_nodes = leaf_nodes.copy()
    current_level_nodes = list(leaf_nodes.values())
    level = 1
    node_counter = 0
    
    while len(current_level_nodes) > 1:
        parent_nodes = {}
        num_parents = math.ceil(len(current_level_nodes) / branching_factor)
        
        for i in range(num_parents):
            start_idx = i * branching_factor
            end_idx = min(start_idx + branching_factor, len(current_level_nodes))
            children = current_level_nodes[start_idx:end_idx]
            
            node_id = f"node_{level}_{i}"
            parent_node = create_parent_node(node_id, children)
            
            if parent_node:
                # Set parent reference in children
                for child in children:
                    child["parent"] = node_id
                    child["level"] = level - 1
                
                parent_nodes[node_id] = parent_node
                all_nodes[node_id] = parent_node
        
        print(f"    Level {level}: {len(parent_nodes)} nodes")
        current_level_nodes = list(parent_nodes.values())
        level += 1
    
    # Set root
    if current_level_nodes:
        root = current_level_nodes[0]
        root["level"] = level - 1
        root["parent"] = None
    
    # Set levels for all nodes
    for node in all_nodes.values():
        if node.get("level") is None:
            node["level"] = level - 1
    
    # Create indexes
    print("  Creating indexes...")
    by_time_index = {}
    by_keyword_index = defaultdict(list)
    
    for node_id, node in all_nodes.items():
        time_range = node["time_range"]
        time_key = f"{time_range[0]:.1f}-{time_range[1]:.1f}"
        by_time_index[time_key] = node_id
        
        # Index by keywords
        for keyword in node.get("keywords", []):
            by_keyword_index[keyword].append(node_id)
    
    # Calculate tree metadata
    total_levels = max(node.get("level", 0) for node in all_nodes.values()) + 1
    
    tree_metadata = {
        "leaf_duration": leaf_duration,
        "branching_factor": branching_factor,
        "total_levels": total_levels,
        "total_nodes": len(all_nodes),
        "total_keywords": len(by_keyword_index),
        "stop_words_removed": sorted(list(STOP_WORDS))
    }
    
    return {
        "root": root,
        "nodes": all_nodes,
        "indexes": {
            "by_time": by_time_index,
            "by_keyword": dict(by_keyword_index)
        },
        "tree_metadata": tree_metadata
    }


def generate_hierarchical_tree(input_json_path: str,
                              output_json_path: Optional[str] = None,
                              leaf_duration: float = 5.0,
                              branching_factor: int = 2):
    """
    Generate hierarchical tree and add it to the segment tree JSON.
    
    Args:
        input_json_path: Path to input segment tree JSON
        output_json_path: Path to output JSON (if None, overwrites input)
        leaf_duration: Duration of leaf nodes in seconds
        branching_factor: Number of children per parent node
    """
    print(f"Loading segment tree from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    seconds_data = data.get("seconds", [])
    transcriptions = data.get("transcriptions", [])
    
    print(f"Loaded {len(seconds_data)} seconds and {len(transcriptions)} transcriptions")
    
    # Build hierarchical tree
    hierarchical_tree = build_hierarchical_tree(
        seconds_data, transcriptions, leaf_duration, branching_factor
    )
    
    if not hierarchical_tree:
        print("Error: Failed to build hierarchical tree")
        return
    
    # Add hierarchical tree to data
    data["hierarchical_tree"] = hierarchical_tree
    
    # Save output
    output_path = output_json_path or input_json_path
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nHierarchical tree generated successfully!")
    print(f"  Total nodes: {hierarchical_tree['tree_metadata']['total_nodes']}")
    print(f"  Total levels: {hierarchical_tree['tree_metadata']['total_levels']}")
    print(f"  Total unique keywords: {hierarchical_tree['tree_metadata']['total_keywords']}")
    print(f"  Leaf duration: {leaf_duration}s")
    print(f"  Branching factor: {branching_factor}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate hierarchical keyword tree from segment tree JSON"
    )
    parser.add_argument(
        "input_json",
        help="Path to input segment tree JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output JSON file (default: overwrites input)"
    )
    parser.add_argument(
        "--leaf-duration",
        type=float,
        default=5.0,
        help="Duration of leaf nodes in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--branching-factor",
        type=int,
        default=2,
        help="Number of children per parent node (default: 2)"
    )
    
    args = parser.parse_args()
    
    generate_hierarchical_tree(
        args.input_json,
        args.output,
        args.leaf_duration,
        args.branching_factor
    )

