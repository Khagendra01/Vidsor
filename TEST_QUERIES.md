# Orchestrator Agent - Test Queries

This document provides a comprehensive list of test queries to verify the orchestrator agent implementation.

## 🎯 Test Strategy

Test queries are organized by operation type and complexity. Start with basic operations and progress to more complex scenarios.

---

## ✅ FIND_HIGHLIGHTS Tests

### Basic Highlights
```bash
# Test 1: Simple highlights query
"find the highlights of the video"

# Test 2: Alternative phrasing
"show me the best moments"

# Test 3: More specific
"find highlights from the video"

# Test 4: With context
"find the highlights and add them to the timeline"
```

### Expected Behavior:
- Calls planner agent with "highlights" query
- Planner searches video and returns time ranges
- Orchestrator creates timeline chunks from time ranges
- Timeline.json is updated with new chunks
- Timeline positions are calculated automatically

---

## ✂️ CUT Tests

### Single Chunk Removal
```bash
# Test 5: Cut single chunk by index
"cut timeline index 0"

# Test 6: Cut with alternative phrasing
"remove timeline index 0"

# Test 7: Cut with "delete"
"delete timeline index 0"
```

### Multiple Chunk Removal
```bash
# Test 8: Cut multiple chunks (explicit indices)
"cut timeline index 0 and 1"

# Test 9: Cut range
"cut timeline index 0-2"

# Test 10: Cut first chunks
"cut the first two clips"

# Test 11: Cut last chunks
"cut the last 3 clips"

# Test 12: Cut with "remove"
"remove timeline index 0, 1, and 2"
```

### Expected Behavior:
- Chunks are removed from timeline
- Timeline positions are recalculated (no gaps)
- Timeline starts at 0.0s
- Timeline.json is updated

---

## 🔄 REPLACE Tests

### Basic Replacement
```bash
# Test 13: Replace single chunk
"replace timeline index 0 with cooking clips"

# Test 14: Replace multiple chunks
"replace timeline index 0-2 with people fishing"

# Test 15: Replace with specific content
"replace the first clip with sunset scenes"
```

### Replacement with Different Content Types
```bash
# Test 16: Replace with activity
"replace timeline index 1 with camping activities"

# Test 17: Replace with object search
"replace timeline index 0 with clips showing boats"

# Test 18: Replace with semantic query
"replace timeline index 2 with moments of laughter"
```

### Expected Behavior:
- Old chunks are removed
- Planner searches for replacement content
- New chunks are created from planner results
- Timeline positions are recalculated
- Timeline.json is updated

---

## ➕ INSERT Tests

### Insert Between Chunks
```bash
# Test 19: Insert between two indices
"add a clip between timeline index 1 and 2"

# Test 20: Insert with search query
"add a transition clip between timeline index 0 and 1"

# Test 21: Insert nature scene
"insert a nature scene between timeline index 2 and 3"
```

### Insert Before/After
```bash
# Test 22: Insert before index
"add a clip before timeline index 1"

# Test 23: Insert after index
"add a clip after timeline index 0"

# Test 24: Insert with specific content
"insert a cooking scene after timeline index 2"
```

### Expected Behavior:
- Planner searches for content to insert
- New chunks are created
- Chunks are inserted at correct position
- Subsequent chunks are shifted
- Timeline positions are recalculated
- Timeline.json is updated

---

## 🎬 FIND_BROLL Tests

### Basic B-Roll
```bash
# Test 25: Find B-roll for single chunk
"find B-roll for timeline index 0"

# Test 26: Find B-roll for range
"find B-roll for timeline 0 to 2"

# Test 27: Find B-roll for first chunks
"find B-roll for the first three clips"
```

### B-Roll with Context
```bash
# Test 28: Find B-roll with specific request
"find B-roll for timeline index 1-3, show nature"

# Test 29: Find B-roll for selected segments
"find B-roll for timeline index 0 and 1"
```

### Expected Behavior:
- Selected chunks are analyzed for main action
- B-roll query is built (excludes main action keywords)
- Planner searches for complementary content
- B-roll chunks are created and inserted
- Timeline positions are recalculated
- Timeline.json is updated

---

## 🔀 Complex Workflows

### Multi-Step Editing
```bash
# Test 30: Create highlights, then edit
"find highlights"
"cut timeline index 0"
"replace timeline index 1 with cooking clips"

# Test 31: Edit existing timeline
"cut the first two clips"
"add a transition between timeline index 0 and 1"
"find B-roll for timeline index 0-2"

# Test 32: Replace and insert
"replace timeline index 0 with sunset scenes"
"insert a nature clip between timeline index 1 and 2"
```

### Expected Behavior:
- Each operation executes correctly
- Timeline maintains continuity across operations
- Timeline.json is updated after each operation
- No index errors or gaps

---

## ⚠️ Error Handling Tests

### Invalid Operations
```bash
# Test 33: Invalid index (too high)
"cut timeline index 999"

# Test 34: Invalid index (negative)
"cut timeline index -1"

# Test 35: Replace without query
"replace timeline index 0"

# Test 36: Insert without position
"add a clip"

# Test 37: Empty query
""
```

### Expected Behavior:
- Operations are rejected with clear error messages
- Timeline is not modified
- Error is returned in operation_result

---

## 📊 Testing Checklist

### Basic Operations
- [ ] FIND_HIGHLIGHTS works
- [ ] CUT single chunk works
- [ ] CUT multiple chunks works
- [ ] REPLACE works
- [ ] INSERT works
- [ ] FIND_BROLL works

### Timeline Integrity
- [ ] Timeline starts at 0.0s after operations
- [ ] No gaps between chunks
- [ ] Timeline positions recalculated correctly
- [ ] Timeline.json saved correctly

### Edge Cases
- [ ] Invalid indices rejected
- [ ] Missing queries handled
- [ ] Empty timeline handled
- [ ] Multiple operations in sequence

### Integration
- [ ] Planner agent called correctly
- [ ] Time ranges converted to chunks
- [ ] Descriptions extracted from search results
- [ ] Timeline manager works correctly

---

## 🧪 Interactive Testing

### Step-by-Step Test Session

1. **Start Interactive Mode:**
```bash
python -m agent.orchestrator_runner \
    --timeline projects/asdf/timeline.json \
    --json projects/asdf/segment_tree.json \
    --video projects/asdf/video/camp_5min.mp4
```

2. **Check Initial Timeline:**
```
Query: show
```

3. **Find Highlights:**
```
Query: find the highlights of the video
```

4. **Check Timeline:**
```
Query: show
```

5. **Cut First Clip:**
```
Query: cut timeline index 0
```

6. **Check Timeline:**
```
Query: show
```

7. **Replace Second Clip:**
```
Query: replace timeline index 0 with cooking clips
```

8. **Check Timeline:**
```
Query: show
```

9. **Add B-Roll:**
```
Query: find B-roll for timeline 0 to 2
```

10. **Final Check:**
```
Query: show
```

11. **Exit:**
```
Query: quit
```

---

## 📝 Test Results Template

For each test, record:

```
Test #: [Number]
Query: [Query string]
Expected: [What should happen]
Actual: [What actually happened]
Status: [PASS/FAIL]
Notes: [Any observations]
```

---

## 🎯 Quick Test Suite

Run these 5 queries to quickly verify core functionality:

1. `"find the highlights of the video"` - Tests FIND_HIGHLIGHTS
2. `"cut timeline index 0"` - Tests CUT
3. `"replace timeline index 0 with cooking clips"` - Tests REPLACE
4. `"add a clip between timeline index 1 and 2"` - Tests INSERT
5. `"find B-roll for timeline 0 to 2"` - Tests FIND_BROLL

If all 5 work, core functionality is verified! ✅

---

## 🔍 Debugging Tips

### If Operations Fail:

1. **Check Timeline:**
   - Use `show` command in interactive mode
   - Verify timeline.json file directly

2. **Check Planner:**
   - Verify segment_tree.json exists
   - Check if planner returns time_ranges

3. **Check Indices:**
   - Timeline indices start at 0
   - Use `show` to see current indices

4. **Check Logs:**
   - Run with `--verbose` flag
   - Check error messages in output

### Common Issues:

- **"No timeline indices provided"** → Query doesn't include indices
- **"Invalid timeline index"** → Index doesn't exist (use `show` to check)
- **"No search query provided"** → REPLACE needs "with [query]"
- **"No highlights found"** → Planner couldn't find content (try different query)

---

**Happy Testing! 🚀**

