"""
Print a complete path from root to a leaf node with all values.
"""

import json
import sys


def load_tree(json_path: str):
    """Load the hierarchical tree from JSON."""
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "hierarchical_tree" not in data:
        print("Error: hierarchical_tree not found in JSON file")
        sys.exit(1)
    
    return data["hierarchical_tree"]


def get_path_to_leaf(tree, leaf_id: str):
    """Get the complete path from root to a leaf node."""
    nodes = tree.get("nodes", {})
    root = tree.get("root", {})
    
    if leaf_id not in nodes:
        print(f"Error: Leaf node '{leaf_id}' not found")
        return None
    
    # Build path from leaf to root
    path = []
    current = nodes[leaf_id]
    
    while current:
        path.append(current)
        parent_id = current.get("parent")
        if parent_id and parent_id in nodes:
            current = nodes[parent_id]
        else:
            # Check if current is root
            if current.get("node_id") == root.get("node_id"):
                break
            current = None
    
    # Add root if not already in path
    if path and path[-1].get("node_id") != root.get("node_id"):
        path.append(root)
    
    # Reverse to get root -> leaf order
    path.reverse()
    
    return path


def print_path_details(path):
    """Print detailed information for each node in the path."""
    print("\n" + "=" * 80)
    print("PATH FROM ROOT TO LEAF")
    print("=" * 80)
    print(f"Total nodes in path: {len(path)}\n")
    
    for i, node in enumerate(path):
        node_id = node.get("node_id", "unknown")
        time_range = node.get("time_range", [])
        duration = node.get("duration", 0)
        level = node.get("level", -1)
        keyword_count = node.get("keyword_count", 0)
        keywords = node.get("keywords", [])
        children = node.get("children", [])
        parent = node.get("parent")
        
        # Determine node type
        if i == 0:
            node_type = "ROOT"
        elif children is None:
            node_type = "LEAF"
        else:
            node_type = f"LEVEL {level}"
        
        print("-" * 80)
        print(f"Node {i+1}/{len(path)}: {node_id} ({node_type})")
        print("-" * 80)
        print(f"  Time Range:     {time_range[0]:.1f}s - {time_range[1]:.1f}s")
        print(f"  Duration:       {duration:.1f}s")
        print(f"  Level:           {level}")
        print(f"  Keyword Count:   {keyword_count}")
        print(f"  Parent:          {parent if parent else 'None (ROOT)'}")
        print(f"  Children:        {len(children) if children else 0} nodes")
        if children:
            print(f"  Child IDs:       {', '.join(children[:5])}{'...' if len(children) > 5 else ''}")
        
        # Print keywords (first 30)
        print(f"\n  Keywords ({len(keywords)} total):")
        if keywords:
            keywords_display = keywords[:30]
            print(f"    {', '.join(keywords_display)}")
            if len(keywords) > 30:
                print(f"    ... and {len(keywords) - 30} more")
        else:
            print("    (none)")
        
        # Print visual and audio text for leaf nodes
        if node_type == "LEAF":
            visual_text = node.get("visual_text", "")
            audio_text = node.get("audio_text", "")
            combined_text = node.get("combined_text", "")
            
            if visual_text:
                print(f"\n  Visual Text:")
                print(f"    {visual_text[:200]}{'...' if len(visual_text) > 200 else ''}")
            
            if audio_text:
                print(f"\n  Audio Text:")
                print(f"    {audio_text[:200]}{'...' if len(audio_text) > 200 else ''}")
            
            if combined_text:
                print(f"\n  Combined Text:")
                print(f"    {combined_text[:200]}{'...' if len(combined_text) > 200 else ''}")
            
            source_seconds = node.get("source_seconds", [])
            source_transcriptions = node.get("source_transcriptions", [])
            if source_seconds:
                print(f"\n  Source Seconds:  {source_seconds}")
            if source_transcriptions:
                print(f"  Source Transcriptions: {source_transcriptions}")
        
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Print a path from root to a leaf node with all values"
    )
    parser.add_argument(
        "json_file",
        help="Path to segment tree JSON file"
    )
    parser.add_argument(
        "--leaf",
        default="leaf_0",
        help="Leaf node ID to trace path to (default: leaf_0)"
    )
    
    args = parser.parse_args()
    
    # Load tree
    tree = load_tree(args.json_file)
    
    # Get path
    path = get_path_to_leaf(tree, args.leaf)
    
    if not path:
        print(f"Error: Could not find path to {args.leaf}")
        sys.exit(1)
    
    # Print path details
    print_path_details(path)
    
    print("=" * 80)
    print("Path visualization:")
    print("=" * 80)
    for i, node in enumerate(path):
        node_id = node.get("node_id", "unknown")
        time_range = node.get("time_range", [])
        duration = node.get("duration", 0)
        level = node.get("level", -1)
        
        arrow = " → " if i < len(path) - 1 else ""
        print(f"{node_id} ({time_range[0]:.1f}s-{time_range[1]:.1f}s, {duration:.1f}s, level {level}){arrow}", end="")
    print("\n")


if __name__ == "__main__":
    main()

