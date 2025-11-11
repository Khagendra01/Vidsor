"""
Visualize hierarchical tree as an image or vector graphic.
Supports PNG (image) and SVG (vector) output formats.
"""

import json
import sys
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False


def load_tree(json_path: str) -> Dict[str, Any]:
    """Load the hierarchical tree from JSON."""
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "hierarchical_tree" not in data:
        print("Error: hierarchical_tree not found in JSON file")
        sys.exit(1)
    
    return data["hierarchical_tree"]


def create_tree_layout(nodes: Dict[str, Any], root_id: str) -> Dict[str, Tuple[float, float]]:
    """
    Create a hierarchical layout for the tree.
    Returns a dictionary mapping node_id to (x, y) coordinates.
    """
    # Group nodes by level
    by_level = defaultdict(list)
    for node_id, node in nodes.items():
        level = node.get("level", -1)
        by_level[level].append((node_id, node))
    
    # Sort levels
    sorted_levels = sorted(by_level.keys(), reverse=True)
    max_level = max(sorted_levels) if sorted_levels else 0
    
    layout = {}
    
    # Calculate positions
    for level in sorted_levels:
        level_nodes = by_level[level]
        num_nodes = len(level_nodes)
        
        # Y position based on level (root at top)
        y = max_level - level
        
        # X positions evenly spaced
        if num_nodes == 1:
            x = 0.0
        else:
            spacing = 1.0 / (num_nodes - 1) if num_nodes > 1 else 1.0
            for i, (node_id, _) in enumerate(level_nodes):
                x = i * spacing - 0.5  # Center around 0
                layout[node_id] = (x, y)
    
    return layout


def visualize_with_matplotlib(tree: Dict[str, Any], 
                              output_path: str,
                              max_levels: Optional[int] = None,
                              show_keywords: bool = False,
                              figsize: Tuple[int, int] = (16, 12)):
    """Visualize tree using matplotlib."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for image output")
        print("Install with: pip install matplotlib")
        return False
    
    nodes = tree.get("nodes", {})
    root = tree.get("root", {})
    root_id = root.get("node_id")
    
    if not root_id:
        print("Error: Root node not found")
        return False
    
    # Filter nodes by max_levels if specified
    if max_levels is not None:
        max_level = max(n.get("level", -1) for n in nodes.values())
        min_level = max(0, max_level - max_levels + 1)
        nodes = {nid: n for nid, n in nodes.items() 
                if n.get("level", -1) >= min_level}
        nodes[root_id] = root
    
    # Create layout
    layout = create_tree_layout(nodes, root_id)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Color scheme by level
    max_level = max(n.get("level", -1) for n in nodes.values())
    colors = plt.cm.viridis([l / max_level for l in range(max_level + 1)])
    
    # Draw nodes
    node_patches = {}
    for node_id, (x, y) in layout.items():
        node = nodes.get(node_id)
        if not node:
            continue
        
        level = node.get("level", 0)
        color = colors[level] if level < len(colors) else colors[-1]
        
        # Node size based on keyword count
        keyword_count = node.get("keyword_count", 0)
        size = 0.1 + (keyword_count / 100) * 0.3  # Scale between 0.1 and 0.4
        
        # Draw node
        if node.get("children") is None:  # Leaf node
            shape = mpatches.Circle((x, y), size, color=color, ec='black', lw=1.5, zorder=3)
        else:  # Parent node
            shape = mpatches.Rectangle((x - size, y - size), size * 2, size * 2,
                                      color=color, ec='black', lw=1.5, zorder=3)
        
        ax.add_patch(shape)
        node_patches[node_id] = shape
        
        # Add label
        label = node_id.replace("node_", "n").replace("leaf_", "l")
        if len(label) > 8:
            label = label[:8]
        ax.text(x, y - size - 0.15, label, ha='center', va='top', 
               fontsize=8, fontweight='bold')
        
        # Add keyword count
        ax.text(x, y + size + 0.1, str(keyword_count), ha='center', va='bottom',
               fontsize=7, style='italic')
    
    # Draw edges (parent-child relationships)
    for node_id, node in nodes.items():
        if node_id not in layout:
            continue
        
        children = node.get("children")
        if not children:
            continue
        
        x1, y1 = layout[node_id]
        
        for child_id in children:
            if child_id not in layout:
                continue
            x2, y2 = layout[child_id]
            
            # Draw arrow
            arrow = FancyArrowPatch((x1, y1 - 0.1), (x2, y2 + 0.1),
                                   arrowstyle='->', mutation_scale=15,
                                   color='gray', lw=1, alpha=0.6, zorder=1)
            ax.add_patch(arrow)
    
    # Add title
    metadata = tree.get("tree_metadata", {})
    title = f"Hierarchical Tree Structure\n"
    title += f"Nodes: {metadata.get('total_nodes', 0)} | "
    title += f"Levels: {metadata.get('total_levels', 0)} | "
    title += f"Keywords: {metadata.get('total_keywords', 0)}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[0], label='Root'),
        mpatches.Patch(facecolor=colors[len(colors)//2], label='Intermediate'),
        mpatches.Patch(facecolor=colors[-1], label='Leaves'),
        mpatches.Circle((0, 0), 0.1, color='black', label='Leaf Node'),
        mpatches.Rectangle((0, 0), 0.2, 0.2, color='black', label='Parent Node')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved successfully!")
    
    return True


def visualize_with_graphviz(tree: Dict[str, Any],
                            output_path: str,
                            max_levels: Optional[int] = None,
                            format: str = 'png'):
    """Visualize tree using Graphviz (better for large trees)."""
    if not HAS_GRAPHVIZ:
        print("Error: graphviz is required for vector output")
        print("Install with: pip install graphviz")
        print("Also install Graphviz software: https://graphviz.org/download/")
        return False
    
    nodes = tree.get("nodes", {})
    root = tree.get("root", {})
    root_id = root.get("node_id")
    
    if not root_id:
        print("Error: Root node not found")
        return False
    
    # Filter nodes by max_levels if specified
    if max_levels is not None:
        max_level = max(n.get("level", -1) for n in nodes.values())
        min_level = max(0, max_level - max_levels + 1)
        nodes = {nid: n for nid, n in nodes.items() 
                if n.get("level", -1) >= min_level}
        nodes[root_id] = root
    
    # Create graph
    dot = Digraph(comment='Hierarchical Tree')
    dot.attr(rankdir='TB')  # Top to bottom
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Add nodes
    max_level = max(n.get("level", -1) for n in nodes.values())
    for node_id, node in nodes.items():
        level = node.get("level", 0)
        keyword_count = node.get("keyword_count", 0)
        time_range = node.get("time_range", [])
        duration = node.get("duration", 0)
        
        # Color by level (darker = higher level)
        hue = 0.6 - (level / max_level) * 0.4  # Blue to green gradient
        color = f"{hue} 0.8 0.9"
        
        # Node label
        if node.get("children") is None:  # Leaf
            label = f"{node_id}\\n{time_range[0]:.0f}-{time_range[1]:.0f}s\\n{keyword_count} keywords"
            shape = 'ellipse'
        else:  # Parent
            children_count = len(node.get("children", []))
            label = f"{node_id}\\n{time_range[0]:.0f}-{time_range[1]:.0f}s\\n{keyword_count} kw | {children_count} children"
            shape = 'box'
        
        dot.node(node_id, label, fillcolor=color, shape=shape)
    
    # Add edges
    for node_id, node in nodes.items():
        children = node.get("children")
        if not children:
            continue
        
        for child_id in children:
            if child_id in nodes:
                dot.edge(node_id, child_id)
    
    # Render
    print(f"Rendering visualization to {output_path}...")
    try:
        dot.render(output_path, format=format, cleanup=True)
        print(f"Saved successfully as {output_path}.{format}")
        return True
    except Exception as e:
        print(f"Error rendering: {e}")
        print("Make sure Graphviz is installed: https://graphviz.org/download/")
        return False


def create_simple_svg(tree: Dict[str, Any],
                     output_path: str,
                     max_levels: Optional[int] = None):
    """Create a simple SVG visualization (no external dependencies)."""
    nodes = tree.get("nodes", {})
    root = tree.get("root", {})
    root_id = root.get("node_id")
    
    if not root_id:
        print("Error: Root node not found")
        return False
    
    # Filter nodes by max_levels if specified
    if max_levels is not None:
        max_level = max(n.get("level", -1) for n in nodes.values())
        min_level = max(0, max_level - max_levels + 1)
        nodes = {nid: n for nid, n in nodes.items() 
                if n.get("level", -1) >= min_level}
        nodes[root_id] = root
    
    # Create layout
    layout = create_tree_layout(nodes, root_id)
    
    # Calculate dimensions
    if not layout:
        print("Error: Could not create layout")
        return False
    
    coords = list(layout.values())
    min_x, max_x = min(x for x, y in coords), max(x for x, y in coords)
    min_y, max_y = min(y for x, y in coords), max(y for x, y in coords)
    
    width = 1200
    height = 800
    margin = 50
    
    # Scale and translate coordinates
    x_range = max_x - min_x if max_x != min_x else 1
    y_range = max_y - min_y if max_y != min_y else 1
    
    def transform(x, y):
        sx = ((x - min_x) / x_range) * (width - 2 * margin) + margin
        sy = ((max_y - y) / y_range) * (height - 2 * margin) + margin
        return sx, sy
    
    # Generate SVG
    svg_lines = []
    svg_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_lines.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
    svg_lines.append('<defs>')
    svg_lines.append('  <style>')
    svg_lines.append('    .node-text { font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }')
    svg_lines.append('    .node-label { font-weight: bold; font-size: 9px; }')
    svg_lines.append('    .edge { stroke: #666; stroke-width: 1.5; fill: none; }')
    svg_lines.append('  </style>')
    svg_lines.append('</defs>')
    
    # Draw edges first (so they appear behind nodes)
    for node_id, node in nodes.items():
        if node_id not in layout:
            continue
        
        children = node.get("children")
        if not children:
            continue
        
        x1, y1 = transform(*layout[node_id])
        
        for child_id in children:
            if child_id not in layout:
                continue
            x2, y2 = transform(*layout[child_id])
            
            svg_lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="edge"/>')
    
    # Draw nodes
    max_level = max(n.get("level", -1) for n in nodes.values())
    for node_id, (x, y) in layout.items():
        node = nodes.get(node_id)
        if not node:
            continue
        
        sx, sy = transform(x, y)
        level = node.get("level", 0)
        keyword_count = node.get("keyword_count", 0)
        time_range = node.get("time_range", [])
        
        # Color by level
        hue = int(240 - (level / max_level) * 120)  # Blue to green
        color = f"hsl({hue}, 70%, 80%)"
        
        # Node size
        size = 20 + (keyword_count / 10)
        
        # Draw node
        if node.get("children") is None:  # Leaf
            svg_lines.append(f'  <circle cx="{sx}" cy="{sy}" r="{size}" fill="{color}" stroke="black" stroke-width="2"/>')
        else:  # Parent
            svg_lines.append(f'  <rect x="{sx - size}" y="{sy - size}" width="{size * 2}" height="{size * 2}" fill="{color}" stroke="black" stroke-width="2" rx="3"/>')
        
        # Add label
        label = node_id.replace("node_", "n").replace("leaf_", "l")
        if len(label) > 10:
            label = label[:10]
        svg_lines.append(f'  <text x="{sx}" y="{sy - size - 5}" class="node-text node-label">{label}</text>')
        svg_lines.append(f'  <text x="{sx}" y="{sy + size + 12}" class="node-text">{keyword_count}kw</text>')
    
    # Add title
    metadata = tree.get("tree_metadata", {})
    title = f"Hierarchical Tree - {metadata.get('total_nodes', 0)} nodes, {metadata.get('total_levels', 0)} levels"
    svg_lines.append(f'  <text x="{width // 2}" y="25" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">{title}</text>')
    
    svg_lines.append('</svg>')
    
    # Write file
    print(f"Saving SVG to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_lines))
    
    print(f"Saved successfully!")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize hierarchical tree as image or vector graphic"
    )
    parser.add_argument(
        "json_file",
        help="Path to segment tree JSON file with hierarchical_tree"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="tree_visualization",
        help="Output file path (without extension, default: tree_visualization)"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "svg", "pdf", "jpg"],
        default="png",
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        help="Maximum number of levels to show (default: all)"
    )
    parser.add_argument(
        "--method",
        choices=["matplotlib", "graphviz", "svg"],
        default="matplotlib",
        help="Visualization method (default: matplotlib)"
    )
    
    args = parser.parse_args()
    
    # Load tree
    tree = load_tree(args.json_file)
    
    # Determine output path
    if args.format == "svg" and args.method == "svg":
        output_path = args.output if args.output.endswith(".svg") else f"{args.output}.svg"
    else:
        output_path = args.output if "." in args.output else f"{args.output}.{args.format}"
    
    # Visualize
    success = False
    if args.method == "matplotlib":
        if args.format == "svg":
            print("Note: matplotlib SVG support is limited. Use --method svg for better SVG output.")
        success = visualize_with_matplotlib(tree, output_path, args.max_levels)
    elif args.method == "graphviz":
        success = visualize_with_graphviz(tree, output_path, args.max_levels, args.format)
    elif args.method == "svg":
        if args.format != "svg":
            print("Warning: SVG method only outputs SVG format. Ignoring --format.")
        success = create_simple_svg(tree, output_path, args.max_levels)
    
    if success:
        print(f"\nVisualization saved to: {output_path}")
    else:
        print("\nVisualization failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

