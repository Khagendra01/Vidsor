# Enhanced Agent Autonomy & Intelligence Plan

## Overview

This document outlines a comprehensive plan to transform the video editing agent into a truly autonomous, intelligent system that can learn, adapt, and improve itself. The plan is organized into 5 phases, each building on the previous to create increasingly sophisticated agentic behavior.

**Current Status:** Agent has basic self-correction but it's limited to parameter tweaking and only works for 4/8 operations. This plan will make it universally intelligent and autonomous.

---

## 🎯 Core Problem Analysis

### Current Limitations

1. **Limited Self-Correction**
   - Only works for 4 operations: FIND_HIGHLIGHTS, REPLACE, INSERT, FIND_BROLL
   - Missing: CUT, REORDER, TRIM, APPLY_EFFECT
   - Only tweaks parameters (threshold, max_results), doesn't fix root causes

2. **Inefficient Iteration**
   - Runs all 3 iterations even when first result is good
   - No early stopping for high-quality results
   - Wastes time and API calls

3. **No Learning**
   - Doesn't remember what worked before
   - Doesn't adapt to video types or user preferences
   - No cross-session knowledge

4. **No Proactivity**
   - Doesn't anticipate needs
   - Doesn't prevent errors before they happen
   - Only reacts, never acts proactively

5. **No Meta-Learning**
   - Doesn't improve its own learning process
   - Doesn't optimize its strategies
   - No self-improvement loop

---

## 📋 Implementation Plan: 5 Phases

### Phase 1: Foundation - Universal Self-Correction ⭐ **START HERE**

**Goal:** Make self-correction work for ALL operations with intelligent early stopping and behavior-level fixes.

#### 1.1 Extend Self-Correction to All Operations

**Current:** Only 4 operations have self-correction
**Target:** All 8 operations (FIND_HIGHLIGHTS, CUT, REPLACE, INSERT, FIND_BROLL, REORDER, TRIM, APPLY_EFFECT)

**Implementation:**
- Add validation functions for CUT, REORDER, TRIM, APPLY_EFFECT
- Create operation-specific validators
- Enable self-correction for all operations in orchestrator

**Files to Modify:**
- `agent/utils/self_correction.py` - Add validation functions
- `agent/nodes/orchestrator.py` - Enable for all operations

**Estimated Time:** 1-2 days

---

#### 1.2 Intelligent Early Stopping ⚡ **CRITICAL FOR PERFORMANCE**

**Problem:** Currently runs all 3 iterations even when first result is excellent (confidence 0.70, but `is_valid=False`)

**Solution:** Multi-tier early stopping logic

**Implementation:**
```python
def should_stop_early(validation: Dict, iteration: int, max_iterations: int) -> bool:
    """
    Determine if we should stop early based on quality metrics.
    """
    confidence = validation.get("confidence", 0.0)
    is_valid = validation.get("is_valid", False)
    issues = validation.get("issues", [])
    
    # Tier 1: Perfect result - stop immediately
    if is_valid and confidence >= 0.9:
        return True, "Perfect result achieved"
    
    # Tier 2: Good result with minor issues - stop if confidence is high
    if confidence >= 0.85 and len(issues) <= 1:
        return True, "High confidence with minor issues"
    
    # Tier 3: Acceptable result - stop if no critical issues
    if is_valid and confidence >= 0.7:
        return True, "Acceptable result"
    
    # Tier 4: No improvement potential - stop if confidence plateaued
    if iteration >= 2 and confidence < 0.5:
        return True, "Low confidence, unlikely to improve"
    
    # Tier 5: No refinement possible - stop if LLM says no refinement
    if not validation.get("needs_refinement", True):
        return True, "No refinement needed"
    
    return False, "Continue iteration"
```

**Additional Improvements:**
- **Confidence Plateau Detection:** If confidence doesn't improve between iterations, stop early
- **Issue Severity Analysis:** Minor issues (missing descriptions) vs critical issues (no results)
- **Adaptive Thresholds:** Lower threshold for simple operations, higher for complex ones
- **Time Budget:** Stop if taking too long (configurable timeout)

**Files to Modify:**
- `agent/utils/self_correction.py` - Add `should_stop_early()` function
- Update `self_correct_loop()` to use early stopping

**Estimated Time:** 1 day

**Expected Impact:** 
- **50-70% reduction** in self-correction time for good results
- **Faster response** for users
- **Lower API costs**

---

#### 1.3 Behavior-Level Refinement (Not Just Parameters)

**Problem:** Current refinement only adjusts parameters (threshold, max_results), doesn't fix root causes

**Solution:** Enable behavior-level actions that actually change how operations work

**Implementation:**
```python
# New refinement action types
REFINEMENT_ACTIONS = {
    # Description extraction
    "add_descriptions_from_segment_tree": {
        "handler": extract_descriptions_from_segment_tree,
        "applies_to": ["FIND_HIGHLIGHTS", "FIND_BROLL"],
        "triggers": ["no descriptions", "lack context"]
    },
    
    # Diversity filtering
    "enable_diversity_filtering": {
        "handler": apply_diversity_filter,
        "applies_to": ["FIND_HIGHLIGHTS"],
        "triggers": ["lack diversity", "too similar"]
    },
    
    # Alternative search strategies
    "switch_to_semantic_only": {
        "handler": use_semantic_search_only,
        "applies_to": ["FIND_HIGHLIGHTS", "FIND_BROLL"],
        "triggers": ["too many false positives"]
    },
    
    # Context expansion
    "expand_time_range_context": {
        "handler": include_neighboring_segments,
        "applies_to": ["FIND_HIGHLIGHTS", "FIND_BROLL"],
        "triggers": ["missing context", "incomplete results"]
    },
    
    # Validation improvements
    "retry_with_stricter_validation": {
        "handler": apply_stricter_validation,
        "applies_to": ["FIND_BROLL"],
        "triggers": ["low quality matches"]
    }
}

def apply_behavior_refinement(
    operation: str,
    validation: Dict,
    result: Dict,
    state: OrchestratorState
) -> Dict:
    """
    Apply behavior-level refinements based on validation issues.
    """
    issues = validation.get("issues", [])
    suggestions = validation.get("suggestions", [])
    
    # Map issues to behavior actions
    actions_to_apply = []
    for issue in issues:
        for action_name, action_config in REFINEMENT_ACTIONS.items():
            if operation in action_config["applies_to"]:
                if any(trigger in issue.lower() for trigger in action_config["triggers"]):
                    actions_to_apply.append(action_name)
    
    # Execute actions
    for action_name in actions_to_apply:
        action_config = REFINEMENT_ACTIONS[action_name]
        result = action_config["handler"](result, state)
    
    return result
```

**Files to Create:**
- `agent/utils/behavior_refinements.py` - Behavior-level refinement handlers

**Files to Modify:**
- `agent/utils/self_correction.py` - Integrate behavior refinements
- `agent/handlers/handle_find_highlights.py` - Support behavior actions

**Estimated Time:** 2-3 days

**Expected Impact:**
- **Fixes root causes**, not just symptoms
- **Actually solves** validation issues
- **Better results** with fewer iterations

---

#### 1.4 Enhanced Description Extraction

**Problem:** Chunks lack meaningful descriptions (only timestamps)

**Solution:** Multi-source description extraction with intelligent fallbacks

**Implementation:**
```python
def extract_chunk_descriptions(
    time_range: Tuple[float, float],
    search_results: List[Dict],
    segment_tree: Dict,
    state: OrchestratorState
) -> Tuple[str, str, str]:
    """
    Extract descriptions from multiple sources with intelligent fallback.
    
    Priority:
    1. Search results (if high quality match)
    2. Segment tree (visual + audio descriptions)
    3. Audio transcripts (if available)
    4. Generated description from context
    """
    start_time, end_time = time_range
    
    # Try search results first
    description = match_search_result_to_time_range(search_results, start_time)
    if description and len(description) > 20:  # Meaningful description
        return description
    
    # Try segment tree
    segment_descriptions = query_segment_tree_for_range(
        segment_tree, start_time, end_time
    )
    if segment_descriptions:
        visual_desc = segment_descriptions.get("visual", "")
        audio_desc = segment_descriptions.get("audio", "")
        combined = combine_descriptions(visual_desc, audio_desc)
        if combined:
            return combined
    
    # Try audio transcripts
    audio_transcript = extract_audio_transcript(state, start_time, end_time)
    if audio_transcript:
        return f"Audio: {audio_transcript[:100]}"
    
    # Generate from context (last resort)
    return generate_description_from_context(time_range, state)
```

**Files to Modify:**
- `agent/helpers/orchestrator_helpers/match_search_result_to_time_range.py` - Enhance matching
- `agent/helpers/orchestrator_helpers/create_chunks_from_time_ranges.py` - Add segment tree lookup

**Files to Create:**
- `agent/utils/description_extraction.py` - Multi-source description extraction

**Estimated Time:** 2 days

---

#### 1.5 Diversity Filtering

**Problem:** Results are too similar (e.g., seconds 33-37 all clustered)

**Solution:** Temporal diversity enforcement with cluster-aware selection

**Implementation:**
```python
def apply_diversity_filter(
    scored_seconds: List[Tuple[int, float]],
    min_distance: float = 10.0,  # Minimum seconds between highlights
    max_results: int = 50
) -> List[Tuple[int, float]]:
    """
    Filter results to ensure temporal diversity.
    
    Strategy:
    1. Group nearby seconds into clusters
    2. Select best moment from each cluster
    3. Ensure minimum distance between selected moments
    4. Prioritize high-scoring moments that are well-spaced
    """
    # Sort by score (descending)
    sorted_seconds = sorted(scored_seconds, key=lambda x: x[1], reverse=True)
    
    # Cluster nearby seconds
    clusters = cluster_temporally(sorted_seconds, cluster_radius=5.0)
    
    # Select from clusters with diversity constraint
    selected = []
    for cluster in clusters:
        # Get best moment from cluster
        best_moment = max(cluster, key=lambda x: x[1])
        
        # Check if far enough from already selected
        if not selected or all(abs(best_moment[0] - s[0]) >= min_distance 
                              for s in selected):
            selected.append(best_moment)
        
        if len(selected) >= max_results:
            break
    
    return selected
```

**Files to Create:**
- `agent/utils/diversity_filter.py` - Diversity filtering utilities

**Files to Modify:**
- `agent/nodes/planner.py` - Integrate diversity filtering

**Estimated Time:** 1-2 days

---

### Phase 1 Summary

**Total Estimated Time:** 7-10 days

**Key Deliverables:**
- ✅ Self-correction for all 8 operations
- ✅ Intelligent early stopping (50-70% time reduction)
- ✅ Behavior-level refinements (fixes root causes)
- ✅ Enhanced description extraction
- ✅ Diversity filtering

**Expected Impact:**
- **Faster operations** (early stopping)
- **Better results** (behavior fixes)
- **More autonomous** (works for all operations)

---

## Phase 2: Adaptive Intelligence (Context-Aware Behavior)

**Goal:** Agent adapts behavior based on context, history, and patterns.

### 2.1 Context-Aware Parameter Selection

**Implementation:**
- Learn optimal thresholds per video type (vlogs vs tutorials vs documentaries)
- Adapt search strategies based on video content characteristics
- Adjust diversity requirements based on video length

**Files to Create:**
- `agent/utils/context_learning.py` - Context-aware parameter selection
- `agent/utils/video_type_classifier.py` - Classify video types

**Estimated Time:** 3-4 days

---

### 2.2 Pattern Recognition and Learning

**Implementation:**
- Remember what worked before for similar queries
- Learn user preferences (e.g., prefers longer highlights)
- Detect common failure patterns and prevent them

**Files to Create:**
- `agent/utils/pattern_learning.py` - Pattern recognition
- `agent/utils/preference_learning.py` - User preference tracking

**Estimated Time:** 3-4 days

---

### 2.3 Multi-Strategy Execution

**Implementation:**
- Try multiple approaches in parallel, pick best
- Fallback chains: if strategy A fails, try B, then C
- Ensemble methods: combine results from multiple strategies

**Files to Create:**
- `agent/utils/multi_strategy_executor.py` - Parallel strategy execution

**Estimated Time:** 2-3 days

---

### Phase 2 Summary

**Total Estimated Time:** 8-11 days

**Key Deliverables:**
- Context-aware behavior adaptation
- Pattern recognition and learning
- Multi-strategy execution

---

## Phase 3: Proactive Autonomy (Anticipatory Behavior)

**Goal:** Agent anticipates needs and prevents errors before they happen.

### 3.1 Pre-Operation Validation

**Implementation:**
- Check if operation will likely fail before executing
- Suggest alternatives if current approach is risky
- Validate timeline state before operations

**Files to Create:**
- `agent/utils/pre_operation_validator.py` - Pre-execution validation

**Estimated Time:** 2-3 days

---

### 3.2 Proactive Suggestions

**Implementation:**
- "Your timeline is empty, want to find highlights?"
- "This clip is 15s, consider trimming?"
- "You added highlights, want B-roll?"

**Files to Create:**
- `agent/utils/proactive_suggestions.py` - Suggestion engine

**Estimated Time:** 2-3 days

---

### 3.3 Error Prevention

**Implementation:**
- Detect invalid timeline indices before CUT/REPLACE
- Warn about operations that might break narrative flow
- Suggest safer alternatives

**Files to Modify:**
- `agent/nodes/orchestrator.py` - Add pre-operation checks

**Estimated Time:** 1-2 days

---

### Phase 3 Summary

**Total Estimated Time:** 5-8 days

**Key Deliverables:**
- Pre-operation validation
- Proactive suggestions
- Error prevention

---

## Phase 4: Meta-Learning (Learning to Learn)

**Goal:** Agent improves its own learning and decision-making.

### 4.1 Self-Improvement Loop

**Implementation:**
- Track which refinements actually work
- Learn which validation criteria are most predictive
- Optimize self-correction strategies over time

**Files to Create:**
- `agent/utils/meta_learning.py` - Self-improvement tracking

**Estimated Time:** 3-4 days

---

### 4.2 Strategy Selection Learning

**Implementation:**
- Learn which search strategies work best for which queries
- Remember successful parameter combinations
- Build a "strategy library" of proven approaches

**Files to Create:**
- `agent/utils/strategy_library.py` - Strategy knowledge base

**Estimated Time:** 3-4 days

---

### 4.3 Cross-Session Learning

**Implementation:**
- Save successful patterns to disk
- Load learned strategies on startup
- Share knowledge across video projects

**Files to Create:**
- `agent/utils/knowledge_persistence.py` - Save/load learned patterns

**Estimated Time:** 2-3 days

---

### Phase 4 Summary

**Total Estimated Time:** 8-11 days

**Key Deliverables:**
- Self-improvement loop
- Strategy selection learning
- Cross-session learning

---

## Phase 5: Full Autonomy (Complete Workflow Handling)

**Goal:** Agent can handle complex, multi-step workflows without guidance.

### 5.1 Goal Decomposition

**Implementation:**
- Break high-level goals into sub-goals
- Execute sub-goals autonomously
- Handle dependencies and ordering

**Files to Enhance:**
- `agent/utils/multi_step_planner.py` - Enhanced goal decomposition

**Estimated Time:** 3-4 days

---

### 5.2 Autonomous Quality Control

**Implementation:**
- Automatically improve results without being asked
- Refine timeline for better narrative flow
- Optimize clip durations and transitions

**Files to Create:**
- `agent/utils/autonomous_quality_control.py` - Auto-improvement engine

**Estimated Time:** 4-5 days

---

### 5.3 Self-Directed Exploration

**Implementation:**
- Try alternative approaches when confidence is low
- Explore parameter space intelligently
- Learn from exploration results

**Files to Create:**
- `agent/utils/exploration_engine.py` - Intelligent exploration

**Estimated Time:** 3-4 days

---

### Phase 5 Summary

**Total Estimated Time:** 10-13 days

**Key Deliverables:**
- Goal decomposition
- Autonomous quality control
- Self-directed exploration

---

## 📊 Overall Timeline

| Phase | Duration | Priority | Impact |
|-------|----------|----------|--------|
| **Phase 1: Foundation** | 7-10 days | 🔴 **HIGHEST** | 🔥 **CRITICAL** |
| Phase 2: Adaptive Intelligence | 8-11 days | 🟡 Medium | 🔥 High |
| Phase 3: Proactive Autonomy | 5-8 days | 🟡 Medium | 🔥 High |
| Phase 4: Meta-Learning | 8-11 days | 🟢 Low | 🔥 Medium |
| Phase 5: Full Autonomy | 10-13 days | 🟢 Low | 🔥 Medium |
| **Total** | **38-53 days** | | |

---

## 🎯 Immediate Action Items (Phase 1, Week 1)

### Day 1-2: Early Stopping
- [ ] Implement `should_stop_early()` function
- [ ] Add confidence plateau detection
- [ ] Add issue severity analysis
- [ ] Test with various scenarios

### Day 3-4: Behavior Refinements
- [ ] Create `behavior_refinements.py`
- [ ] Implement description extraction action
- [ ] Implement diversity filtering action
- [ ] Integrate into self-correction loop

### Day 5-6: Description Extraction
- [ ] Enhance `match_search_result_to_time_range.py`
- [ ] Add segment tree lookup
- [ ] Add audio transcript fallback
- [ ] Test description quality

### Day 7: Diversity Filtering
- [ ] Create `diversity_filter.py`
- [ ] Implement clustering algorithm
- [ ] Integrate into planner
- [ ] Test diversity improvement

### Day 8-10: Universal Self-Correction
- [ ] Add validation for CUT, REORDER, TRIM, APPLY_EFFECT
- [ ] Enable self-correction for all operations
- [ ] Test all operations
- [ ] Documentation

---

## 📈 Expected Outcomes

### After Phase 1
- ✅ **50-70% faster** self-correction (early stopping)
- ✅ **Better results** (behavior fixes, descriptions, diversity)
- ✅ **All operations** self-correcting
- ✅ **Root cause fixes** instead of parameter tweaks

### After Phase 2
- ✅ **Adaptive behavior** to video types
- ✅ **Learning from history**
- ✅ **Multiple strategies** tried intelligently

### After Phase 3
- ✅ **Error prevention** before they happen
- ✅ **Proactive suggestions**
- ✅ **Better user experience**

### After Phase 4
- ✅ **Self-improving** capabilities
- ✅ **Strategy optimization**
- ✅ **Cross-session learning**

### After Phase 5
- ✅ **Complex workflows** handled autonomously
- ✅ **Auto quality improvements**
- ✅ **Intelligent exploration**

---

## 🔧 Configuration Options

Add to `orchestrator_runner.py`:

```python
state = {
    # Self-correction settings
    "enable_self_correction": True,
    "max_self_correction_iterations": 3,
    "self_correction_confidence_threshold": 0.7,
    "enable_early_stopping": True,  # NEW
    "early_stopping_aggressiveness": "moderate",  # conservative, moderate, aggressive
    
    # Behavior refinement settings
    "enable_behavior_refinements": True,  # NEW
    "auto_extract_descriptions": True,  # NEW
    "auto_apply_diversity": True,  # NEW
    
    # Learning settings
    "enable_context_learning": False,  # Phase 2
    "enable_pattern_learning": False,  # Phase 2
    "enable_meta_learning": False,  # Phase 4
}
```

---

## 🧪 Testing Strategy

### Phase 1 Testing
1. **Early Stopping Tests:**
   - Test with perfect results (should stop at iteration 1)
   - Test with good results (should stop at iteration 2)
   - Test with poor results (should run all 3)

2. **Behavior Refinement Tests:**
   - Test description extraction fixes "no descriptions" issue
   - Test diversity filtering fixes "lack diversity" issue

3. **Description Extraction Tests:**
   - Test with search results available
   - Test with only segment tree
   - Test with only audio transcripts
   - Test with no sources (fallback)

4. **Diversity Filtering Tests:**
   - Test with clustered results
   - Test with well-spaced results
   - Test with various video lengths

---

## 📝 Notes

- **Start with Phase 1** - It fixes immediate issues and enables future phases
- **Early stopping is critical** - Will significantly improve user experience
- **Behavior refinements are key** - They fix root causes, not symptoms
- **Test thoroughly** - Each phase builds on previous, so quality matters

---

## 🚀 Next Steps

1. **Review and approve** this plan
2. **Start Phase 1, Day 1-2** (Early Stopping)
3. **Iterate and test** as we go
4. **Document learnings** for future phases

---

**Last Updated:** 2025-01-13
**Status:** Ready for Implementation
**Priority:** Phase 1 - Foundation (CRITICAL)

