"""
Visualize and explore the hierarchical tree structure.
Shows tree statistics, structure, and allows keyword searches.
"""

import json
import sys
from typing import Dict, List, Any, Optional
from collections import defaultdict


def load_tree(json_path: str) -> Dict[str, Any]:
    """Load the hierarchical tree from JSON."""
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "hierarchical_tree" not in data:
        print("Error: hierarchical_tree not found in JSON file")
        sys.exit(1)
    
    return data["hierarchical_tree"]


def print_statistics(tree: Dict[str, Any]):
    """Print tree statistics."""
    metadata = tree.get("tree_metadata", {})
    root = tree.get("root", {})
    
    print("\n" + "=" * 70)
    print("HIERARCHICAL TREE STATISTICS")
    print("=" * 70)
    print(f"Total nodes: {metadata.get('total_nodes', 0)}")
    print(f"Total levels: {metadata.get('total_levels', 0)}")
    print(f"Leaf duration: {metadata.get('leaf_duration', 0)}s")
    print(f"Branching factor: {metadata.get('branching_factor', 0)}")
    print(f"Total unique keywords: {metadata.get('total_keywords', 0)}")
    print(f"Root time range: {root.get('time_range', [])}")
    print(f"Root duration: {root.get('duration', 0):.1f}s")
    print(f"Root keywords: {root.get('keyword_count', 0)}")
    print(f"Stop words removed: {len(metadata.get('stop_words_removed', []))}")


def print_tree_structure(tree: Dict[str, Any], max_levels: int = 4):
    """Print tree structure showing levels and node counts."""
    nodes = tree.get("nodes", {})
    
    # Count nodes by level
    level_counts = defaultdict(int)
    for node in nodes.values():
        level = node.get("level", -1)
        level_counts[level] += 1
    
    print("\n" + "=" * 70)
    print("TREE STRUCTURE BY LEVEL")
    print("=" * 70)
    
    sorted_levels = sorted(level_counts.keys(), reverse=True)
    for level in sorted_levels[:max_levels]:
        count = level_counts[level]
        if level == sorted_levels[0]:
            level_name = "ROOT"
        elif level == 0:
            level_name = "LEAVES"
        else:
            level_name = f"LEVEL {level}"
        
        # Calculate approximate duration at this level
        sample_node = next((n for n in nodes.values() if n.get("level") == level), None)
        if sample_node:
            duration = sample_node.get("duration", 0)
            print(f"{level_name:15} | Nodes: {count:4} | ~Duration: {duration:7.1f}s per node")
        else:
            print(f"{level_name:15} | Nodes: {count:4}")


def print_sample_leaf_nodes(tree: Dict[str, Any], num_samples: int = 5):
    """Print sample leaf nodes."""
    nodes = tree.get("nodes", {})
    leaves = [n for n in nodes.values() if n.get("level") == 0]
    
    print("\n" + "=" * 70)
    print(f"SAMPLE LEAF NODES (showing {min(num_samples, len(leaves))} of {len(leaves)})")
    print("=" * 70)
    
    for i, leaf in enumerate(leaves[:num_samples]):
        time_range = leaf.get("time_range", [])
        keywords = leaf.get("keywords", [])
        visual_text = leaf.get("visual_text", "")[:100]
        audio_text = leaf.get("audio_text", "")[:100]
        
        print(f"\nLeaf {i+1}: {leaf.get('node_id')}")
        print(f"  Time: {time_range[0]:.1f}s - {time_range[1]:.1f}s ({leaf.get('duration', 0):.1f}s)")
        print(f"  Keywords ({len(keywords)}): {', '.join(keywords[:15])}{'...' if len(keywords) > 15 else ''}")
        if visual_text:
            print(f"  Visual: {visual_text}...")
        if audio_text:
            print(f"  Audio: {audio_text}...")


def print_sample_parent_nodes(tree: Dict[str, Any], num_samples: int = 3):
    """Print sample parent nodes at different levels."""
    nodes = tree.get("nodes", {})
    
    # Get nodes at different levels (excluding leaves and root)
    parent_nodes = [n for n in nodes.values() 
                   if n.get("level") is not None 
                   and n.get("level") > 0 
                   and n.get("children") is not None]
    
    # Group by level
    by_level = defaultdict(list)
    for node in parent_nodes:
        level = node.get("level")
        by_level[level].append(node)
    
    print("\n" + "=" * 70)
    print("SAMPLE PARENT NODES BY LEVEL")
    print("=" * 70)
    
    sorted_levels = sorted(by_level.keys(), reverse=True)
    for level in sorted_levels[:num_samples]:
        level_nodes = by_level[level]
        if not level_nodes:
            continue
        
        node = level_nodes[0]  # Show first node at this level
        time_range = node.get("time_range", [])
        keywords = node.get("keywords", [])
        children = node.get("children", [])
        
        print(f"\nLevel {level} Node: {node.get('node_id')}")
        print(f"  Time: {time_range[0]:.1f}s - {time_range[1]:.1f}s ({node.get('duration', 0):.1f}s)")
        print(f"  Children: {len(children)} nodes")
        print(f"  Keywords ({len(keywords)}): {', '.join(keywords[:20])}{'...' if len(keywords) > 20 else ''}")


def search_keyword(tree: Dict[str, Any], keyword: str, max_results: int = 10):
    """Search for a keyword and show which nodes contain it."""
    indexes = tree.get("indexes", {})
    by_keyword = indexes.get("by_keyword", {})
    
    keyword_lower = keyword.lower()
    matching_nodes = by_keyword.get(keyword_lower, [])
    
    print("\n" + "=" * 70)
    print(f"KEYWORD SEARCH: '{keyword}'")
    print("=" * 70)
    print(f"Found in {len(matching_nodes)} node(s)")
    
    if not matching_nodes:
        print("No matches found.")
        return
    
    nodes = tree.get("nodes", {})
    print(f"\nShowing first {min(max_results, len(matching_nodes))} matches:\n")
    
    for i, node_id in enumerate(matching_nodes[:max_results]):
        node = nodes.get(node_id)
        if not node:
            continue
        
        time_range = node.get("time_range", [])
        level = node.get("level", -1)
        keyword_count = node.get("keyword_count", 0)
        
        level_name = "ROOT" if level == max(n.get("level", -1) for n in nodes.values()) else \
                    "LEAF" if level == 0 else f"LEVEL {level}"
        
        print(f"{i+1}. {node_id} ({level_name})")
        print(f"   Time: {time_range[0]:.1f}s - {time_range[1]:.1f}s | Keywords: {keyword_count}")


def show_keyword_distribution(tree: Dict[str, Any], top_n: int = 20):
    """Show most common keywords."""
    indexes = tree.get("indexes", {})
    by_keyword = indexes.get("by_keyword", {})
    
    # Sort by number of nodes containing each keyword
    keyword_counts = [(kw, len(nodes)) for kw, nodes in by_keyword.items()]
    keyword_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 70)
    print(f"TOP {top_n} MOST COMMON KEYWORDS")
    print("=" * 70)
    
    for i, (keyword, count) in enumerate(keyword_counts[:top_n], 1):
        print(f"{i:2}. {keyword:20} appears in {count:3} node(s)")


def traverse_path_to_leaf(tree: Dict[str, Any], leaf_id: str):
    """Show the path from root to a specific leaf."""
    nodes = tree.get("nodes", {})
    root = tree.get("root", {})
    
    if leaf_id not in nodes:
        print(f"Error: Leaf node '{leaf_id}' not found")
        return
    
    # Build path from leaf to root
    path = []
    current = nodes[leaf_id]
    
    while current:
        path.append(current)
        parent_id = current.get("parent")
        if parent_id and parent_id in nodes:
            current = nodes[parent_id]
        else:
            break
    
    # Add root if not already in path
    if path and path[-1].get("node_id") != root.get("node_id"):
        path.append(root)
    
    print("\n" + "=" * 70)
    print(f"PATH FROM ROOT TO {leaf_id}")
    print("=" * 70)
    
    for i, node in enumerate(reversed(path)):
        time_range = node.get("time_range", [])
        level = node.get("level", -1)
        keyword_count = node.get("keyword_count", 0)
        children = node.get("children", [])
        
        indent = "  " * i
        level_name = "ROOT" if i == 0 else "LEAF" if level == 0 else f"LEVEL {level}"
        
        print(f"{indent}{level_name}: {node.get('node_id')}")
        print(f"{indent}  Time: {time_range[0]:.1f}s - {time_range[1]:.1f}s ({node.get('duration', 0):.1f}s)")
        print(f"{indent}  Keywords: {keyword_count}")
        if children:
            print(f"{indent}  Children: {len(children)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize hierarchical tree structure"
    )
    parser.add_argument(
        "json_file",
        help="Path to segment tree JSON file with hierarchical_tree"
    )
    parser.add_argument(
        "--search",
        "-s",
        help="Search for a keyword"
    )
    parser.add_argument(
        "--top-keywords",
        type=int,
        default=20,
        help="Show top N keywords (default: 20)"
    )
    parser.add_argument(
        "--leaf-path",
        help="Show path from root to a specific leaf node (e.g., leaf_0)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample nodes to show (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Load tree
    tree = load_tree(args.json_file)
    
    # Print statistics
    print_statistics(tree)
    
    # Print structure
    print_tree_structure(tree)
    
    # Show samples
    print_sample_leaf_nodes(tree, args.samples)
    print_sample_parent_nodes(tree, 3)
    
    # Show keyword distribution
    show_keyword_distribution(tree, args.top_keywords)
    
    # Keyword search
    if args.search:
        search_keyword(tree, args.search)
    
    # Show path to leaf
    if args.leaf_path:
        traverse_path_to_leaf(tree, args.leaf_path)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print("\nUsage examples:")
    print(f"  Search keyword: python {sys.argv[0]} {args.json_file} --search fishing")
    print(f"  Show leaf path: python {sys.argv[0]} {args.json_file} --leaf-path leaf_0")


if __name__ == "__main__":
    main()

