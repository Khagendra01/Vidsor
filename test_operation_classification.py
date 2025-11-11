"""Test script for Operation Classification - Step 2 Testing"""

import os
import sys
from agent.timeline_manager import TimelineManager
from agent.orchestrator_operations import classify_operation, validate_operation_params, _extract_timeline_indices, _extract_search_query

# Import LLM classes
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
                print("WARNING: No LLM available, using heuristic only")
                return None
    elif HAS_ANTHROPIC:
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    else:
        print("WARNING: No LLM available, using heuristic only")
        return None


def test_extract_timeline_indices():
    """Test timeline index extraction"""
    print("\n" + "=" * 60)
    print("TEST 1: Extract Timeline Indices")
    print("=" * 60)
    
    chunk_count = 9
    
    test_cases = [
        ("cut timeline index 0", [0]),
        ("remove index 1 and 2", [1, 2]),
        ("replace timeline 0-2", [0, 1, 2]),
        ("the first two clips", [0, 1]),
        ("last 3 clips", [6, 7, 8]),
        ("timeline index 0 to 4", [0, 1, 2, 3, 4]),
    ]
    
    all_passed = True
    for query, expected in test_cases:
        result = _extract_timeline_indices(query, chunk_count)
        if set(result) == set(expected):
            print(f"✓ '{query}' → {result}")
        else:
            print(f"✗ '{query}' → {result} (expected {expected})")
            all_passed = False
    
    return all_passed


def test_extract_search_query():
    """Test search query extraction"""
    print("\n" + "=" * 60)
    print("TEST 2: Extract Search Query")
    print("=" * 60)
    
    test_cases = [
        ("replace timeline index 0 with cooking clips", "cooking clips"),
        ("find highlights", "highlights"),
        ("add a clip of people fishing", "people fishing"),
        ("show me the best moments", "the best moments"),
    ]
    
    all_passed = True
    for query, expected in test_cases:
        result = _extract_search_query(query)
        if result and expected.lower() in result.lower():
            print(f"✓ '{query}' → '{result}'")
        else:
            print(f"✗ '{query}' → '{result}' (expected contains '{expected}')")
            all_passed = False
    
    return all_passed


def test_classify_operation():
    """Test operation classification"""
    print("\n" + "=" * 60)
    print("TEST 3: Classify Operation (with LLM)")
    print("=" * 60)
    
    llm = get_llm()
    if not llm:
        print("⚠ Skipping LLM tests (no API key)")
        return True
    
    timeline_path = "projects/asdf/timeline.json"
    manager = TimelineManager(timeline_path, verbose=False)
    manager.load()
    chunk_count = manager.get_chunk_count()
    duration = manager.calculate_timeline_duration()
    
    test_cases = [
        ("find the highlights of the video", "FIND_HIGHLIGHTS"),
        ("cut timeline index 0 and 1", "CUT"),
        ("replace timeline index 0-2 with cooking clips", "REPLACE"),
        ("add a clip between timeline index 1 and 2", "INSERT"),
        ("find B-roll for timeline 0 to 2", "FIND_BROLL"),
    ]
    
    all_passed = True
    for query, expected_op in test_cases:
        try:
            result = classify_operation(query, chunk_count, duration, llm, verbose=True)
            operation = result.get("operation")
            
            if operation == expected_op:
                print(f"✓ '{query}' → {operation}")
            else:
                print(f"✗ '{query}' → {operation} (expected {expected_op})")
                all_passed = False
        except Exception as e:
            print(f"✗ '{query}' → Error: {e}")
            all_passed = False
    
    return all_passed


def test_validate_operation_params():
    """Test parameter validation"""
    print("\n" + "=" * 60)
    print("TEST 4: Validate Operation Parameters")
    print("=" * 60)
    
    chunk_count = 9
    
    # Test valid CUT
    is_valid, error = validate_operation_params("CUT", {"timeline_indices": [0, 1]}, chunk_count)
    if is_valid:
        print(f"✓ Valid CUT parameters passed")
    else:
        print(f"✗ Valid CUT parameters failed: {error}")
        return False
    
    # Test invalid CUT (out of range)
    is_valid, error = validate_operation_params("CUT", {"timeline_indices": [999]}, chunk_count)
    if not is_valid:
        print(f"✓ Invalid CUT parameters correctly rejected: {error}")
    else:
        print(f"✗ Should reject invalid CUT parameters")
        return False
    
    # Test valid REPLACE
    is_valid, error = validate_operation_params("REPLACE", {"timeline_indices": [0, 1, 2]}, chunk_count)
    if is_valid:
        print(f"✓ Valid REPLACE parameters passed")
    else:
        print(f"✗ Valid REPLACE parameters failed: {error}")
        return False
    
    # Test invalid INSERT (no position)
    is_valid, error = validate_operation_params("INSERT", {}, chunk_count)
    if not is_valid:
        print(f"✓ Invalid INSERT parameters correctly rejected: {error}")
    else:
        print(f"✗ Should reject invalid INSERT parameters")
        return False
    
    # Test valid INSERT (with position)
    is_valid, error = validate_operation_params("INSERT", {"insert_between_indices": [1, 2]}, chunk_count)
    if is_valid:
        print(f"✓ Valid INSERT parameters passed")
    else:
        print(f"✗ Valid INSERT parameters failed: {error}")
        return False
    
    # Test valid TRIM
    is_valid, error = validate_operation_params("TRIM", {"trim_index": 0, "trim_seconds": 2.0}, chunk_count)
    if is_valid:
        print(f"✓ Valid TRIM parameters passed")
    else:
        print(f"✗ Valid TRIM parameters failed: {error}")
        return False
    
    # Test invalid TRIM (no index)
    is_valid, error = validate_operation_params("TRIM", {"trim_seconds": 2.0}, chunk_count)
    if not is_valid:
        print(f"✓ Invalid TRIM parameters correctly rejected: {error}")
    else:
        print(f"✗ Should reject invalid TRIM parameters")
        return False
    
    return True


def test_heuristic_classification():
    """Test heuristic classification fallback"""
    print("\n" + "=" * 60)
    print("TEST 5: Heuristic Classification (Fallback)")
    print("=" * 60)
    
    from agent.orchestrator_operations import _classify_operation_heuristic
    
    chunk_count = 9
    
    test_cases = [
        ("find highlights", "FIND_HIGHLIGHTS"),
        ("cut timeline index 0", "CUT"),
        ("replace index 1 with cooking", "REPLACE"),
        ("add clip between 1 and 2", "INSERT"),
        ("find B-roll for timeline 0-2", "FIND_BROLL"),
    ]
    
    all_passed = True
    for query, expected_op in test_cases:
        result = _classify_operation_heuristic(query, chunk_count, verbose=False)
        operation = result.get("operation")
        
        if operation == expected_op:
            print(f"✓ '{query}' → {operation}")
        else:
            print(f"✗ '{query}' → {operation} (expected {expected_op})")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("OPERATION CLASSIFICATION TESTS - STEP 2")
    print("=" * 60)
    
    tests = [
        ("Extract Timeline Indices", test_extract_timeline_indices),
        ("Extract Search Query", test_extract_search_query),
        ("Classify Operation (LLM)", test_classify_operation),
        ("Validate Operation Parameters", test_validate_operation_params),
        ("Heuristic Classification", test_heuristic_classification),
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
        print("\n🎉 All tests passed! Step 2 is complete.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")


if __name__ == "__main__":
    main()

