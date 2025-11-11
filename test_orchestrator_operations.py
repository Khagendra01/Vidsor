"""Test script for Orchestrator Operation Handlers - Step 3 Testing"""

import os
import json
import shutil
from agent.timeline_manager import TimelineManager
from agent.orchestrator_handlers import (
    create_timeline_chunk,
    handle_cut,
    handle_find_highlights,
    handle_replace,
    handle_insert,
    handle_find_broll
)
from agent.orchestrator_state import OrchestratorState

# Import LLM classes for planner
try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def get_llm():
    """Get LLM instance for testing."""
    if HAS_OPENAI:
        try:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except:
            if HAS_ANTHROPIC:
                return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
            else:
                return None
    elif HAS_ANTHROPIC:
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    else:
        return None


def create_mock_planner_agent(time_ranges_list):
    """
    Create a mock planner agent that returns predefined time ranges.
    
    Args:
        time_ranges_list: List of time ranges to return
        
    Returns:
        Mock planner agent function
    """
    def mock_planner(state):
        return {
            "time_ranges": time_ranges_list,
            "search_results": [
                {
                    "time_range": tr,
                    "description": f"Mock description for {tr[0]:.1f}s-{tr[1]:.1f}s",
                    "unified_description": f"Mock unified description for {tr[0]:.1f}s-{tr[1]:.1f}s",
                    "audio_description": f"Mock audio for {tr[0]:.1f}s-{tr[1]:.1f}s"
                }
                for tr in time_ranges_list
            ],
            "confidence": 0.8,
            "needs_clarification": False
        }
    return mock_planner


def setup_test_timeline(test_path: str) -> TimelineManager:
    """Create a test timeline with sample chunks."""
    # Copy original timeline if it exists
    original_path = "projects/asdf/timeline.json"
    if os.path.exists(original_path):
        shutil.copy(original_path, test_path)
        manager = TimelineManager(test_path, verbose=False)
        manager.load()
    else:
        # Create new timeline
        manager = TimelineManager(test_path, verbose=False)
        manager.load()  # Creates empty timeline
        
        # Add some test chunks
        test_chunks = [
            create_timeline_chunk(22.0, 28.0, 0.0, description="Walking through forest"),
            create_timeline_chunk(65.0, 70.0, 6.0, description="Preparing equipment"),
            create_timeline_chunk(126.0, 131.0, 11.0, description="Setting up camp"),
        ]
        manager.chunks = test_chunks
        manager.save()
    
    return manager


def test_create_timeline_chunk():
    """Test timeline chunk creation"""
    print("\n" + "=" * 60)
    print("TEST 1: Create Timeline Chunk")
    print("=" * 60)
    
    chunk = create_timeline_chunk(
        original_start=22.0,
        original_end=28.0,
        timeline_start=0.0,
        chunk_type="highlight",
        description="Test chunk",
        unified_description="Test unified",
        audio_description="Test audio",
        score=0.9
    )
    
    assert chunk["original_start_time"] == 22.0
    assert chunk["original_end_time"] == 28.0
    assert chunk["start_time"] == 0.0
    assert chunk["end_time"] == 6.0  # 0.0 + (28.0 - 22.0)
    assert chunk["chunk_type"] == "highlight"
    assert chunk["description"] == "Test chunk"
    assert chunk["score"] == 0.9
    
    print("✓ Timeline chunk created correctly")
    return True


def test_handle_cut():
    """Test CUT operation"""
    print("\n" + "=" * 60)
    print("TEST 2: CUT Operation")
    print("=" * 60)
    
    test_path = "test_timeline_cut.json"
    manager = setup_test_timeline(test_path)
    
    original_count = len(manager.chunks)
    print(f"  Original chunks: {original_count}")
    
    # Create mock state
    state = {
        "user_query": "cut timeline index 0",
        "video_path": "test.mp4",
        "json_path": "test.json",
        "segment_tree": None,
        "verbose": False,
    }
    
    # Cut first chunk
    result = handle_cut(
        state,
        manager,
        {"timeline_indices": [0]},
        verbose=True
    )
    
    assert result["success"] == True
    assert len(result["chunks_removed"]) == 1
    assert len(manager.chunks) == original_count - 1
    
    # Verify timeline positions recalculated
    if manager.chunks:
        assert manager.chunks[0]["start_time"] == 0.0, "Timeline should start at 0.0"
        print(f"  ✓ Timeline starts at {manager.chunks[0]['start_time']:.1f}s")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("✓ CUT operation successful")
    return True


def test_handle_cut_multiple():
    """Test CUT operation with multiple indices"""
    print("\n" + "=" * 60)
    print("TEST 3: CUT Multiple Chunks")
    print("=" * 60)
    
    test_path = "test_timeline_cut_multiple.json"
    manager = setup_test_timeline(test_path)
    
    original_count = len(manager.chunks)
    print(f"  Original chunks: {original_count}")
    
    state = {
        "user_query": "cut timeline index 0 and 1",
        "video_path": "test.mp4",
        "json_path": "test.json",
        "segment_tree": None,
        "verbose": False,
    }
    
    # Cut first two chunks
    result = handle_cut(
        state,
        manager,
        {"timeline_indices": [0, 1]},
        verbose=True
    )
    
    assert result["success"] == True
    assert len(result["chunks_removed"]) == 2
    assert len(manager.chunks) == original_count - 2
    
    # Verify timeline continuity
    if len(manager.chunks) > 0:
        assert manager.chunks[0]["start_time"] == 0.0
        print(f"  ✓ Timeline continuity maintained")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("✓ CUT multiple chunks successful")
    return True


def test_handle_replace():
    """Test REPLACE operation"""
    print("\n" + "=" * 60)
    print("TEST 4: REPLACE Operation")
    print("=" * 60)
    
    test_path = "test_timeline_replace.json"
    manager = setup_test_timeline(test_path)
    
    original_count = len(manager.chunks)
    print(f"  Original chunks: {original_count}")
    
    # Create mock planner that returns replacement content
    replacement_ranges = [(200.0, 210.0), (220.0, 230.0)]
    mock_planner = create_mock_planner_agent(replacement_ranges)
    
    state = {
        "user_query": "replace timeline index 0 with cooking clips",
        "video_path": "test.mp4",
        "json_path": "test.json",
        "segment_tree": None,
        "verbose": False,
    }
    
    # Replace first chunk
    result = handle_replace(
        state,
        manager,
        {
            "timeline_indices": [0],
            "search_query": "cooking clips"
        },
        mock_planner,
        verbose=True
    )
    
    assert result["success"] == True
    assert len(result["chunks_removed"]) == 1
    assert len(result["chunks_added"]) == 1
    assert len(manager.chunks) == original_count  # Same count, replaced
    
    # Verify new chunk has correct source times
    new_chunk = result["chunks_added"][0]
    assert new_chunk["original_start_time"] == 200.0
    assert new_chunk["original_end_time"] == 210.0
    
    # Verify timeline continuity
    if manager.chunks:
        assert manager.chunks[0]["start_time"] == 0.0
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("✓ REPLACE operation successful")
    return True


def test_handle_insert():
    """Test INSERT operation"""
    print("\n" + "=" * 60)
    print("TEST 5: INSERT Operation")
    print("=" * 60)
    
    test_path = "test_timeline_insert.json"
    manager = setup_test_timeline(test_path)
    
    original_count = len(manager.chunks)
    print(f"  Original chunks: {original_count}")
    
    # Create mock planner
    insert_ranges = [(100.0, 105.0)]
    mock_planner = create_mock_planner_agent(insert_ranges)
    
    state = {
        "user_query": "add clip between timeline index 1 and 2",
        "video_path": "test.mp4",
        "json_path": "test.json",
        "segment_tree": None,
        "verbose": False,
    }
    
    # Insert between chunks 1 and 2
    result = handle_insert(
        state,
        manager,
        {
            "insert_between_indices": [1, 2],
            "search_query": "transition clip"
        },
        mock_planner,
        verbose=True
    )
    
    assert result["success"] == True
    assert len(result["chunks_inserted"]) == 1
    assert len(manager.chunks) == original_count + 1
    
    # Verify insert position
    assert result["insert_position"] == 2
    
    # Verify timeline continuity
    if len(manager.chunks) > 2:
        # Chunk 1 end should equal chunk 2 start
        chunk1_end = manager.chunks[1]["end_time"]
        chunk2_start = manager.chunks[2]["start_time"]
        assert abs(chunk1_end - chunk2_start) < 0.01, "Timeline should be continuous"
        print(f"  ✓ Timeline continuity: chunk 1 ends at {chunk1_end:.1f}s, chunk 2 starts at {chunk2_start:.1f}s")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("✓ INSERT operation successful")
    return True


def test_handle_find_broll():
    """Test FIND_BROLL operation"""
    print("\n" + "=" * 60)
    print("TEST 6: FIND_BROLL Operation")
    print("=" * 60)
    
    test_path = "test_timeline_broll.json"
    manager = setup_test_timeline(test_path)
    
    original_count = len(manager.chunks)
    print(f"  Original chunks: {original_count}")
    
    # Create mock planner for B-roll
    broll_ranges = [(50.0, 55.0), (80.0, 85.0)]
    mock_planner = create_mock_planner_agent(broll_ranges)
    
    state = {
        "user_query": "find B-roll for timeline 0 to 1",
        "video_path": "test.mp4",
        "json_path": "test.json",
        "segment_tree": None,
        "verbose": False,
    }
    
    # Find B-roll for first two chunks
    result = handle_find_broll(
        state,
        manager,
        {"timeline_indices": [0, 1]},
        mock_planner,
        verbose=True
    )
    
    assert result["success"] == True
    assert len(result["chunks_created"]) == 2
    assert len(manager.chunks) == original_count + 2
    
    # Verify B-roll chunks are marked correctly
    for chunk in result["chunks_created"]:
        assert chunk["chunk_type"] == "broll"
    
    # Verify insert position (after last selected chunk)
    assert result["insert_position"] == 2  # After index 1
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("✓ FIND_BROLL operation successful")
    return True


def test_timeline_continuity_after_operations():
    """Test that timeline maintains continuity after multiple operations"""
    print("\n" + "=" * 60)
    print("TEST 7: Timeline Continuity")
    print("=" * 60)
    
    test_path = "test_timeline_continuity.json"
    manager = setup_test_timeline(test_path)
    
    state = {
        "user_query": "test",
        "video_path": "test.mp4",
        "json_path": "test.json",
        "segment_tree": None,
        "verbose": False,
    }
    
    # Perform multiple operations
    print("  Performing CUT...")
    handle_cut(state, manager, {"timeline_indices": [0]}, verbose=False)
    
    print("  Performing INSERT...")
    insert_ranges = [(150.0, 155.0)]
    mock_planner = create_mock_planner_agent(insert_ranges)
    handle_insert(
        state,
        manager,
        {"insert_after_index": 0, "search_query": "test"},
        mock_planner,
        verbose=False
    )
    
    # Verify timeline continuity
    print("  Verifying timeline continuity...")
    for i in range(len(manager.chunks) - 1):
        current_end = manager.chunks[i]["end_time"]
        next_start = manager.chunks[i + 1]["start_time"]
        assert abs(current_end - next_start) < 0.01, f"Gap between chunk {i} and {i+1}"
    
    print(f"  ✓ Timeline has {len(manager.chunks)} chunks, all continuous")
    
    # Verify timeline starts at 0
    if manager.chunks:
        assert manager.chunks[0]["start_time"] == 0.0
        print(f"  ✓ Timeline starts at 0.0s")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("✓ Timeline continuity maintained")
    return True


def test_handle_cut_invalid_indices():
    """Test CUT with invalid indices"""
    print("\n" + "=" * 60)
    print("TEST 8: CUT with Invalid Indices")
    print("=" * 60)
    
    test_path = "test_timeline_invalid.json"
    manager = setup_test_timeline(test_path)
    
    state = {
        "user_query": "cut timeline index 999",
        "video_path": "test.mp4",
        "json_path": "test.json",
        "segment_tree": None,
        "verbose": False,
    }
    
    # Try to cut invalid index
    result = handle_cut(
        state,
        manager,
        {"timeline_indices": [999]},
        verbose=True
    )
    
    assert result["success"] == False
    assert "error" in result
    print(f"  ✓ Correctly rejected invalid index: {result['error']}")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    return True


def test_handle_replace_no_query():
    """Test REPLACE without search query"""
    print("\n" + "=" * 60)
    print("TEST 9: REPLACE without Query")
    print("=" * 60)
    
    test_path = "test_timeline_replace_no_query.json"
    manager = setup_test_timeline(test_path)
    
    state = {
        "user_query": "replace timeline index 0",
        "video_path": "test.mp4",
        "json_path": "test.json",
        "segment_tree": None,
        "verbose": False,
    }
    
    mock_planner = create_mock_planner_agent([])
    
    # Try to replace without query
    result = handle_replace(
        state,
        manager,
        {
            "timeline_indices": [0],
            "search_query": None  # No query
        },
        mock_planner,
        verbose=True
    )
    
    assert result["success"] == False
    assert "error" in result
    print(f"  ✓ Correctly rejected replace without query: {result['error']}")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ORCHESTRATOR OPERATION HANDLERS TESTS - STEP 3")
    print("=" * 60)
    
    tests = [
        ("Create Timeline Chunk", test_create_timeline_chunk),
        ("CUT Operation", test_handle_cut),
        ("CUT Multiple Chunks", test_handle_cut_multiple),
        ("REPLACE Operation", test_handle_replace),
        ("INSERT Operation", test_handle_insert),
        ("FIND_BROLL Operation", test_handle_find_broll),
        ("Timeline Continuity", test_timeline_continuity_after_operations),
        ("CUT Invalid Indices", test_handle_cut_invalid_indices),
        ("REPLACE without Query", test_handle_replace_no_query),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Step 3 is complete.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")


if __name__ == "__main__":
    main()

