"""Quick test script for semantic search functionality."""

from segment_tree_utils import SegmentTreeQuery

# Load the segment tree
print("Loading segment tree...")
query = SegmentTreeQuery("camp_segment_tree.json")

# Test semantic search
print("\n" + "="*60)
print("Testing Semantic Search")
print("="*60)

# Example query
test_query = "man pointing at camera"
print(f"\nQuery: '{test_query}'")
print("\nSearching...")

results = query.semantic_search(test_query, top_k=5, threshold=0.3)

print(f"\nFound {len(results)} results:\n")
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.3f} | Type: {result['type']} | Time: {result['time_range']}")
    print(f"   Text: {result['text'][:100]}...")
    print()

