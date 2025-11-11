"""Test script for TimelineManager - Step 1 Testing"""

import os
import json
from agent.timeline_manager import TimelineManager


def test_load_existing_timeline():
    """Test loading existing timeline.json"""
    print("\n" + "=" * 60)
    print("TEST 1: Load Existing Timeline")
    print("=" * 60)
    
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=True)
    
    try:
        timeline_data = manager.load()
        print(f"✓ Successfully loaded timeline")
        print(f"  Version: {timeline_data.get('version')}")
        print(f"  Chunks: {len(manager.chunks)}")
        print(f"  Total duration: {manager.calculate_timeline_duration():.2f}s")
        return True
    except Exception as e:
        print(f"✗ Failed to load timeline: {e}")
        return False


def test_save_timeline():
    """Test saving timeline.json"""
    print("\n" + "=" * 60)
    print("TEST 2: Save Timeline")
    print("=" * 60)
    
    # Load existing timeline
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=True)
    manager.load()
    
    original_chunk_count = len(manager.chunks)
    
    # Save to a test file
    test_path = "test_timeline_output.json"
    manager.timeline_path = test_path
    
    try:
        success = manager.save()
        if success:
            print(f"✓ Successfully saved timeline to {test_path}")
            
            # Verify file exists and is valid JSON
            if os.path.exists(test_path):
                with open(test_path, 'r') as f:
                    saved_data = json.load(f)
                print(f"  Saved chunks: {len(saved_data.get('chunks', []))}")
                print(f"  Matches original: {len(saved_data.get('chunks', [])) == original_chunk_count}")
                
                # Cleanup
                os.remove(test_path)
                print(f"  ✓ Cleaned up test file")
                return True
            else:
                print(f"✗ File was not created")
                return False
        else:
            print(f"✗ Save operation returned False")
            return False
    except Exception as e:
        print(f"✗ Failed to save timeline: {e}")
        if os.path.exists(test_path):
            os.remove(test_path)
        return False


def test_get_chunk():
    """Test getting chunk by index"""
    print("\n" + "=" * 60)
    print("TEST 3: Get Chunk by Index")
    print("=" * 60)
    
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=True)
    manager.load()
    
    # Test valid index
    chunk = manager.get_chunk(0)
    if chunk:
        print(f"✓ Got chunk at index 0")
        print(f"  Description: {chunk.get('description', 'N/A')[:50]}...")
        print(f"  Timeline: {chunk.get('start_time')}s - {chunk.get('end_time')}s")
        print(f"  Source: {chunk.get('original_start_time')}s - {chunk.get('original_end_time')}s")
    else:
        print(f"✗ Failed to get chunk at index 0")
        return False
    
    # Test invalid index
    chunk = manager.get_chunk(999)
    if chunk is None:
        print(f"✓ Correctly returned None for invalid index")
    else:
        print(f"✗ Should return None for invalid index")
        return False
    
    return True


def test_validate_indices():
    """Test index validation"""
    print("\n" + "=" * 60)
    print("TEST 4: Validate Indices")
    print("=" * 60)
    
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=True)
    manager.load()
    
    # Test valid indices
    is_valid, error = manager.validate_indices([0, 1, 2])
    if is_valid:
        print(f"✓ Valid indices [0, 1, 2] passed validation")
    else:
        print(f"✗ Valid indices failed: {error}")
        return False
    
    # Test invalid indices (out of range)
    is_valid, error = manager.validate_indices([0, 999])
    if not is_valid:
        print(f"✓ Invalid indices correctly rejected: {error}")
    else:
        print(f"✗ Should reject invalid indices")
        return False
    
    # Test negative index
    is_valid, error = manager.validate_indices([-1])
    if not is_valid:
        print(f"✓ Negative index correctly rejected: {error}")
    else:
        print(f"✗ Should reject negative index")
        return False
    
    return True


def test_calculate_duration():
    """Test timeline duration calculation"""
    print("\n" + "=" * 60)
    print("TEST 5: Calculate Timeline Duration")
    print("=" * 60)
    
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=True)
    manager.load()
    
    duration = manager.calculate_timeline_duration()
    print(f"✓ Calculated timeline duration: {duration:.2f}s")
    
    # Verify it matches the last chunk's end_time
    if manager.chunks:
        last_chunk = manager.chunks[-1]
        expected_duration = last_chunk.get("end_time", 0)
        if abs(duration - expected_duration) < 0.01:
            print(f"  ✓ Matches last chunk end_time: {expected_duration:.2f}s")
            return True
        else:
            print(f"  ✗ Mismatch: expected {expected_duration:.2f}s, got {duration:.2f}s")
            return False
    
    return True


def test_get_timeline_range():
    """Test getting source video time range for timeline indices"""
    print("\n" + "=" * 60)
    print("TEST 6: Get Timeline Range")
    print("=" * 60)
    
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=True)
    manager.load()
    
    # Get range for first 3 chunks
    time_range = manager.get_timeline_range([0, 1, 2])
    if time_range:
        start, end = time_range
        print(f"✓ Got time range for indices [0, 1, 2]")
        print(f"  Source video range: {start:.2f}s - {end:.2f}s")
        print(f"  Duration: {end - start:.2f}s")
        
        # Verify it's correct
        chunks = manager.get_chunks([0, 1, 2])
        expected_start = min(c.get("original_start_time") for c in chunks)
        expected_end = max(c.get("original_end_time") for c in chunks)
        
        if abs(start - expected_start) < 0.01 and abs(end - expected_end) < 0.01:
            print(f"  ✓ Range matches expected values")
            return True
        else:
            print(f"  ✗ Range mismatch")
            return False
    else:
        print(f"✗ Failed to get time range")
        return False


def test_validate_chunk():
    """Test chunk validation"""
    print("\n" + "=" * 60)
    print("TEST 7: Validate Chunk")
    print("=" * 60)
    
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=True)
    manager.load()
    
    # Test valid chunk
    valid_chunk = manager.chunks[0] if manager.chunks else None
    if valid_chunk:
        is_valid, error = manager.validate_chunk(valid_chunk)
        if is_valid:
            print(f"✓ Valid chunk passed validation")
        else:
            print(f"✗ Valid chunk failed: {error}")
            return False
    
    # Test invalid chunk (missing field)
    invalid_chunk = {"start_time": 0.0}  # Missing end_time
    is_valid, error = manager.validate_chunk(invalid_chunk)
    if not is_valid:
        print(f"✓ Invalid chunk correctly rejected: {error}")
    else:
        print(f"✗ Should reject invalid chunk")
        return False
    
    # Test invalid chunk (negative times)
    invalid_chunk2 = {
        "start_time": -1.0,
        "end_time": 5.0,
        "original_start_time": 0.0,
        "original_end_time": 5.0
    }
    is_valid, error = manager.validate_chunk(invalid_chunk2)
    if not is_valid:
        print(f"✓ Negative times correctly rejected: {error}")
    else:
        print(f"✗ Should reject negative times")
        return False
    
    return True


def test_validate_timeline():
    """Test timeline validation"""
    print("\n" + "=" * 60)
    print("TEST 8: Validate Timeline")
    print("=" * 60)
    
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=True)
    manager.load()
    
    is_valid, errors = manager.validate_timeline()
    if is_valid:
        print(f"✓ Timeline is valid")
        return True
    else:
        print(f"✗ Timeline validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TIMELINE MANAGER TESTS - STEP 1")
    print("=" * 60)
    
    tests = [
        ("Load Existing Timeline", test_load_existing_timeline),
        ("Save Timeline", test_save_timeline),
        ("Get Chunk by Index", test_get_chunk),
        ("Validate Indices", test_validate_indices),
        ("Calculate Duration", test_calculate_duration),
        ("Get Timeline Range", test_get_timeline_range),
        ("Validate Chunk", test_validate_chunk),
        ("Validate Timeline", test_validate_timeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
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
        print("\n🎉 All tests passed! Step 1 is complete.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")


if __name__ == "__main__":
    main()

