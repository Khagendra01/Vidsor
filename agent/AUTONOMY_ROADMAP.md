# Agent Autonomy & Capability Roadmap

## Overview

This document outlines a comprehensive roadmap to make the video editing agent more capable and autonomous. The improvements are organized by priority and impact, focusing on both immediate wins and long-term enhancements.

**Current Status:** The agent has a solid foundation (8.2/10 rating) with good architecture, but needs improvements in autonomy, error handling, and advanced capabilities.

---

## ✅ Already Implemented

The following improvements have already been applied (see `CRITICAL_FIXES_APPLIED.md`):

- ✅ **Planner Agent Caching** - Planner agent created once and reused
- ✅ **Parallel Search Execution** - All search types run concurrently (2-3x faster)
- ✅ **Retry Logic for LLM Calls** - Automatic retry with exponential backoff
- ✅ **Basic Self-Validation** - Search results are validated before use

---

## 🔴 HIGH PRIORITY - Core Autonomy Features

### 1. Operation Batching ⭐ **HIGHEST IMPACT**

**Current Limitation:** Only one operation per query.

**Goal:** Support multiple operations in a single query.

**Example Queries:**
```
"cut index 0 and trim index 1 by 2 seconds"
"find highlights then add B-roll for timeline 0-2"
"replace clip 1 with cooking, then zoom in on the man"
"cut the first clip, find highlights, and add B-roll"
```

**Implementation Approach:**
```python
def parse_batch_operations(query: str, llm) -> List[Dict]:
    """
    Parse query into multiple operations.
    Returns list of operation dictionaries.
    """
    # Use LLM to identify operation boundaries
    # Split: "cut X and trim Y" → [{"operation": "CUT", ...}, {"operation": "TRIM", ...}]
    pass

def execute_batch_operations(state, operations: List[Dict]):
    """Execute operations in sequence with context preservation."""
    results = []
    for op in operations:
        result = execute_operation(state, op)
        # Update state for next operation
        state = update_state_with_result(state, result)
        results.append(result)
    return results
```

**Benefits:**
- More natural user interaction
- Better context preservation between operations
- Faster multi-step workflows
- Reduced user intervention

**Files to Modify:**
- `agent/orchestrator_operations.py` - Add batch parsing
- `agent/nodes/orchestrator.py` - Add batch execution logic
- `agent/prompts/orchestrator_prompts.py` - Update prompts for batch operations

**Estimated Impact:** 🔥 **Very High** - Enables complex workflows

---

### 2. Transaction Support (Undo/Redo) ⭐ **CRITICAL FOR SAFETY** ✅ **IMPLEMENTED**

**Current Limitation:** No rollback mechanism if operations fail.

**Goal:** Support transaction-based operations with rollback capability.

**Status:** ✅ **COMPLETED** - Transaction support has been implemented and integrated into the orchestrator.

**Implementation:**
```python
class TimelineTransaction:
    """Manages timeline state for atomic operations."""
    
    def __init__(self, timeline_manager: TimelineManager):
        self.timeline_manager = timeline_manager
        self.backup = copy.deepcopy(timeline_manager.chunks)
        self.operations = []
        self.committed = False
    
    def add_operation(self, operation: Dict):
        """Track operation for rollback."""
        self.operations.append({
            "type": operation.get("operation"),
            "params": operation.get("parameters"),
            "timestamp": datetime.now()
        })
    
    def commit(self):
        """Save changes permanently."""
        self.timeline_manager.save()
        self.committed = True
    
    def rollback(self):
        """Restore previous state."""
        if not self.committed:
            self.timeline_manager.chunks = self.backup
            self.timeline_manager.save()
            return True
        return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Operation failed, rollback
            self.rollback()
        return False
```

**Usage:**
```python
with TimelineTransaction(timeline_manager) as tx:
    tx.add_operation(operation_result)
    result = execute_operation(state, operation_params)
    if result["success"]:
        tx.commit()
    else:
        # Automatic rollback on failure
        pass
```

**Benefits:**
- Safe operation execution
- Automatic error recovery
- Foundation for undo/redo
- Better user confidence

**Files Created:**
- ✅ `agent/utils/transaction.py` - Transaction manager with `TimelineTransaction` class

**Files Modified:**
- ✅ `agent/nodes/orchestrator.py` - All operations now wrapped in transactions
- ✅ `agent/utils/__init__.py` - Added transaction exports

**Implementation Details:**
- All timeline operations are now wrapped in `TimelineTransaction` context manager
- Automatic rollback on operation failure or exception
- Timeline validation before commit
- Transaction info included in operation results
- Supports both context manager and manual rollback

**Usage Example:**
```python
# Automatically used in orchestrator:
with TimelineTransaction(timeline_manager, verbose=verbose) as tx:
    tx.add_operation(operation_result)
    result = execute_operation(...)
    if result["success"]:
        tx.commit()  # Validates and saves
    # Auto-rollback on exception or if not committed
```

**Estimated Impact:** 🔥 **Very High** - Critical for reliability ✅ **ACHIEVED**

---

### 3. Result Caching ⭐ **PERFORMANCE BOOST**

**Current Limitation:** Re-searches identical queries every time.

**Goal:** Cache search results and LLM responses for repeated queries.

**Implementation:**
```python
from functools import lru_cache
import hashlib
import json
from typing import Optional

class ResultCache:
    """Cache for search results and LLM responses."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _cache_key(self, query: str, context: Dict) -> str:
        """Generate cache key from query and context."""
        # Include segment tree hash if available
        tree_hash = context.get("segment_tree_hash", "")
        key_data = f"{query}:{tree_hash}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, context: Dict) -> Optional[Dict]:
        """Get cached result if available."""
        key = self._cache_key(query, context)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, query: str, context: Dict, result: Dict):
        """Cache a result."""
        key = self._cache_key(query, context)
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = result
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }
```

**Usage:**
```python
# In planner agent
cache = ResultCache(max_size=100)
cached_result = cache.get(query, {"segment_tree_hash": tree_hash})
if cached_result:
    return cached_result

# Execute search
result = execute_search(...)
cache.set(query, {"segment_tree_hash": tree_hash}, result)
return result
```

**Benefits:**
- Faster repeated queries (instant for cached results)
- Reduced API costs
- Better performance for common queries
- Improved user experience

**Files to Create:**
- `agent/utils/result_cache.py` - Caching utilities

**Files to Modify:**
- `agent/nodes/planner.py` - Add caching to search execution
- `agent/orchestrator_operations.py` - Cache operation classifications

**Estimated Impact:** 🔥 **High** - Significant performance improvement

---

### 4. Self-Correction Loop ⭐ **AUTONOMY BOOST** ✅ **IMPLEMENTED**

**Current Limitation:** Basic validation exists, but no automatic refinement.

**Goal:** Automatically refine poor results without user intervention.

**Status:** ✅ **COMPLETED** - Self-correction loop has been implemented and integrated into the orchestrator.

**Implementation:**
```python
def self_correct_loop(
    state: OrchestratorState,
    operation: str,
    params: Dict,
    max_iterations: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Execute operation with automatic self-correction.
    
    Attempts to improve results through iterative refinement.
    """
    best_result = None
    best_confidence = 0.0
    
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n[SELF-CORRECTION] Iteration {iteration}/{max_iterations}")
        
        # Execute operation
        result = execute_operation(state, operation, params)
        
        # Validate result
        validation = validate_operation_result(result, state)
        confidence = validation.get("confidence", 0.0)
        
        if verbose:
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Valid: {validation.get('is_valid', False)}")
        
        # Track best result
        if confidence > best_confidence:
            best_result = result
            best_confidence = confidence
        
        # If result is good enough, return
        if validation.get("is_valid", False) and confidence >= 0.7:
            if verbose:
                print(f"  ✓ Result is acceptable, returning")
            return {
                **result,
                "iterations": iteration,
                "self_corrected": iteration > 1
            }
        
        # If not last iteration, refine
        if iteration < max_iterations:
            refinement = suggest_refinement(validation, result, state)
            if refinement:
                if verbose:
                    print(f"  → Refining: {refinement.get('action', 'N/A')}")
                params = apply_refinement(params, refinement)
            else:
                # No refinement possible, return best result
                break
    
    # Return best result found
    if verbose:
        print(f"\n[SELF-CORRECTION] Returning best result (confidence: {best_confidence:.2f})")
    
    return {
        **best_result,
        "iterations": max_iterations,
        "self_corrected": True,
        "final_confidence": best_confidence
    }

def suggest_refinement(validation: Dict, result: Dict, state: Dict) -> Optional[Dict]:
    """Suggest how to refine the operation."""
    issues = validation.get("issues", [])
    if not issues:
        return None
    
    # Analyze issues and suggest fixes
    suggestions = validation.get("suggestions", [])
    if suggestions:
        # Convert suggestions to parameter adjustments
        return {
            "action": "adjust_parameters",
            "adjustments": parse_suggestions(suggestions)
        }
    
    return None
```

**Benefits:**
- Higher success rate
- Less user intervention needed
- Better quality results
- More autonomous behavior

**Files Created:**
- ✅ `agent/utils/self_correction.py` - Self-correction utilities with validation and refinement

**Files Modified:**
- ✅ `agent/nodes/orchestrator.py` - Integrated self-correction for search-based operations
- ✅ `agent/utils/__init__.py` - Added self-correction exports
- ✅ `agent/orchestrator_runner.py` - Added model_name and enable_self_correction to state

**Implementation Details:**
- Automatic validation of operation results using LLM
- Iterative refinement with up to 3 attempts
- Operation-specific validation (FIND_HIGHLIGHTS, REPLACE, INSERT, FIND_BROLL)
- Parameter adjustment based on validation feedback
- Tracks best result across iterations
- Configurable via `enable_self_correction` state flag

**Usage:**
```python
# Automatically enabled for: FIND_HIGHLIGHTS, REPLACE, INSERT, FIND_BROLL
# Can be disabled by setting enable_self_correction=False in state
result = self_correct_loop(
    state=state,
    timeline_manager=timeline_manager,
    operation="FIND_HIGHLIGHTS",
    params=operation_params,
    operation_handler=handler,
    max_iterations=3,
    confidence_threshold=0.7,
    verbose=verbose
)
```

**Estimated Impact:** 🔥 **High** - Significant autonomy improvement ✅ **ACHIEVED**

**Files to Create:**
- `agent/utils/self_correction.py` - Self-correction utilities

**Files to Modify:**
- `agent/nodes/orchestrator.py` - Add self-correction wrapper
- `agent/utils/processing/refinement.py` - Enhance validation

**Estimated Impact:** 🔥 **High** - Significant autonomy improvement

---

### 5. Multi-Step Planning ⭐ **COMPLEX WORKFLOWS**

**Current Limitation:** Single-step execution only.

**Goal:** Break complex queries into multiple steps and execute them autonomously.

**Implementation:**
```python
def create_multi_step_plan(query: str, llm, context: Dict) -> List[Dict]:
    """
    Break complex query into executable steps.
    
    Example:
    "Create a highlight reel with B-roll" →
    [
        {"step": 1, "operation": "FIND_HIGHLIGHTS", "query": "find highlights"},
        {"step": 2, "operation": "FIND_BROLL", "query": "find B-roll for highlights"},
        {"step": 3, "operation": "REORDER", "query": "organize clips"}
    ]
    """
    system_prompt = """You are a video editing workflow planner. Break complex queries into steps.
    
    Return JSON:
    {
        "steps": [
            {
                "step_number": 1,
                "operation": "FIND_HIGHLIGHTS",
                "query": "find highlights",
                "depends_on": [],
                "reasoning": "First find the highlights"
            },
            {
                "step_number": 2,
                "operation": "FIND_BROLL",
                "query": "find B-roll for timeline 0-2",
                "depends_on": [1],
                "reasoning": "Then add B-roll for the highlights"
            }
        ]
    }"""
    
    user_message = f"Query: {query}\n\nBreak this into executable steps."
    
    result = invoke_llm_with_json(llm, system_prompt, user_message)
    return result.get("steps", [])

def execute_multi_step_plan(state: OrchestratorState, steps: List[Dict]) -> Dict:
    """Execute multi-step plan with dependency resolution."""
    results = []
    completed_steps = set()
    
    # Sort steps by dependencies
    sorted_steps = resolve_dependencies(steps)
    
    for step in sorted_steps:
        step_num = step["step_number"]
        dependencies = step.get("depends_on", [])
        
        # Check dependencies are complete
        if not all(dep in completed_steps for dep in dependencies):
            continue
        
        # Execute step
        result = execute_operation_step(state, step)
        results.append({
            "step": step_num,
            "result": result
        })
        completed_steps.add(step_num)
        
        # Update state for next step
        state = update_state_for_next_step(state, result)
    
    return {
        "success": all(r["result"].get("success") for r in results),
        "steps_completed": len(results),
        "results": results
    }
```

**Benefits:**
- Handles complex workflows autonomously
- Better planning and organization
- More natural user interaction
- Reduces need for multiple queries

**Files to Create:**
- `agent/utils/multi_step_planner.py` - Multi-step planning utilities

**Files to Modify:**
- `agent/nodes/orchestrator.py` - Add multi-step execution
- `agent/prompts/orchestrator_prompts.py` - Add planning prompts

**Estimated Impact:** 🔥 **High** - Enables complex autonomous workflows

---

## 🟡 MEDIUM PRIORITY - Intelligence Features

### 6. Context Memory

**Goal:** Remember previous operations and user preferences.

**Implementation:**
```python
class ContextMemory:
    """Maintains context across operations."""
    
    def __init__(self):
        self.operation_history = []
        self.user_preferences = {}
        self.common_patterns = {}
        self.feedback_history = []
    
    def remember_operation(self, operation: Dict, result: Dict):
        """Store operation in history."""
        self.operation_history.append({
            "operation": operation,
            "result": result,
            "timestamp": datetime.now()
        })
    
    def get_relevant_context(self, query: str) -> Dict:
        """Get relevant context for a query."""
        # Find similar past operations
        # Extract patterns
        # Return context
        pass
    
    def learn_from_feedback(self, feedback: Dict):
        """Learn from user feedback."""
        self.feedback_history.append(feedback)
        # Update preferences
        pass
```

**Benefits:**
- Personalized behavior
- Learning from history
- Better suggestions
- Improved context awareness

---

### 7. Proactive Suggestions

**Goal:** Suggest next actions based on current state.

**Implementation:**
```python
def suggest_next_actions(timeline_state: Dict, query_history: List[str]) -> List[str]:
    """Suggest what user might want to do next."""
    suggestions = []
    
    # Analyze timeline state
    if len(timeline_state.get("chunks", [])) == 0:
        suggestions.append("You have an empty timeline. Would you like to find highlights?")
    
    # Analyze recent operations
    if "FIND_HIGHLIGHTS" in query_history[-1]:
        suggestions.append("You've added highlights. Would you like to add B-roll?")
    
    # Analyze clip durations
    for chunk in timeline_state.get("chunks", []):
        duration = chunk.get("end_time", 0) - chunk.get("start_time", 0)
        if duration > 10:
            suggestions.append(f"Clip {chunk.get('index')} is {duration:.1f}s long. Consider trimming?")
    
    return suggestions
```

**Benefits:**
- More helpful agent
- Better user experience
- Guides users through workflows
- Reduces learning curve

---

### 8. Error Classification and Recovery

**Goal:** Distinguish retryable vs permanent errors and handle accordingly.

**Implementation:**
```python
class ErrorClassifier:
    """Classify errors and determine recovery strategy."""
    
    RETRYABLE_ERRORS = (
        TimeoutError,
        ConnectionError,
        RateLimitError,
        TemporaryAPIError
    )
    
    PERMANENT_ERRORS = (
        ValueError,
        FileNotFoundError,
        InvalidOperationError
    )
    
    def classify(self, error: Exception) -> Dict:
        """Classify error and suggest recovery."""
        error_type = type(error)
        
        if isinstance(error, self.RETRYABLE_ERRORS):
            return {
                "type": "retryable",
                "strategy": "retry_with_backoff",
                "max_attempts": 3
            }
        elif isinstance(error, self.PERMANENT_ERRORS):
            return {
                "type": "permanent",
                "strategy": "fail_with_message",
                "user_message": self._generate_error_message(error)
            }
        else:
            return {
                "type": "unknown",
                "strategy": "log_and_fail"
            }
```

**Benefits:**
- Smarter error handling
- Automatic recovery from transient errors
- Better user feedback
- Reduced false failures

---

### 9. Dry-Run Mode

**Goal:** Preview operations before applying them.

**Implementation:**
```python
def handle_operation(
    state: OrchestratorState,
    operation: str,
    params: Dict,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Execute operation with optional dry-run mode."""
    
    if dry_run:
        # Preview changes without applying
        preview = preview_operation(state, operation, params)
        return {
            "success": True,
            "dry_run": True,
            "preview": preview,
            "changes": preview.get("changes", [])
        }
    else:
        # Execute normally
        return execute_operation(state, operation, params)
```

**Benefits:**
- Safer experimentation
- Better user confidence
- Preview before committing
- Reduced mistakes

---

## 🟢 LOWER PRIORITY - Advanced Features

### 10. Async/Parallel Execution
- Use `asyncio` for I/O-bound operations
- Parallel execution of independent operations
- Better resource utilization

### 11. Metrics and Observability
- Track operation success rates
- Performance metrics
- User behavior analytics
- Integration with monitoring tools

### 12. State Snapshots
- Save timeline states at key points
- Quick recovery from errors
- Version history

### 13. Operation History
- Complete audit trail
- Replay operations
- Undo/redo with history

### 14. Video Quality Analysis
- Analyze clip quality
- Suggest improvements
- Detect issues automatically

---

## Implementation Priority

### Phase 1: Foundation (Autonomy) - **START HERE**
1. ✅ Operation Batching
2. ✅ Transaction Support
3. ✅ Self-Correction Loop

**Timeline:** 2-3 weeks
**Impact:** 🔥🔥🔥 Very High

### Phase 2: Performance (Capability)
4. ✅ Result Caching
5. ✅ Multi-Step Planning
6. ✅ Error Classification

**Timeline:** 2-3 weeks
**Impact:** 🔥🔥 High

### Phase 3: Intelligence (Advanced)
7. Context Memory
8. Proactive Suggestions
9. Dry-Run Mode

**Timeline:** 3-4 weeks
**Impact:** 🔥 Medium

---

## Quick Wins (Can Implement Immediately)

### 1. Result Caching
- **Effort:** Low (1-2 days)
- **Impact:** High
- **Complexity:** Simple LRU cache

### 2. Operation Batching
- **Effort:** Medium (3-5 days)
- **Impact:** Very High
- **Complexity:** Moderate (parsing + execution)

### 3. Transaction Support
- **Effort:** Low (2-3 days)
- **Impact:** Very High
- **Complexity:** Simple backup/restore

---

## Success Metrics

### Autonomy Metrics
- **Self-Correction Rate:** % of operations that self-correct successfully
- **User Intervention Rate:** % of operations requiring user input
- **Multi-Step Success Rate:** % of multi-step plans completed successfully

### Performance Metrics
- **Cache Hit Rate:** % of queries served from cache
- **Average Operation Time:** Time to complete operations
- **Error Recovery Rate:** % of errors automatically recovered

### Quality Metrics
- **Operation Success Rate:** % of operations that succeed
- **User Satisfaction:** Feedback scores
- **Timeline Quality:** Metrics on final timeline quality

---

## Notes

- All improvements should maintain backward compatibility
- Test each feature thoroughly before moving to next
- Document all new features in code and README
- Consider user feedback when prioritizing features

---

## References

- `ARCHITECTURE_ANALYSIS.md` - Detailed architecture analysis
- `CRITICAL_FIXES_APPLIED.md` - Already implemented improvements
- `agent/nodes/orchestrator.py` - Main orchestrator implementation
- `agent/orchestrator_operations.py` - Operation classification

---

**Last Updated:** 2024
**Status:** Planning Phase
**Next Steps:** Begin Phase 1 implementation

