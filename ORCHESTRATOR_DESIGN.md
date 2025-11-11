# Orchestrator/Director Agent Design Plan

## 🎬 Vision
The Orchestrator Agent acts as a **movie director** that manages the entire video editing workflow. It coordinates between user intent, timeline state, and the planner/executor agents to create cohesive video narratives.

---

## 📋 Core Responsibilities

### 1. **Timeline State Management**
- Maintains `timeline.json` as the source of truth for the current edit
- Tracks:
  - **Timeline indices** (0, 1, 2...) - position in the edited sequence
  - **Source video times** (original_start_time, original_end_time) - where clips come from
  - **Clip metadata** (descriptions, scores, chunk_type)
  - **Narrative flow** (continuity, pacing, transitions)

### 2. **Operation Types**

#### **A. Content Discovery Operations**
- `FIND_HIGHLIGHTS` - "find the highlights of the video"
  - Calls planner agent with query
  - Receives time ranges from planner
  - Creates new timeline chunks from results
  - Replaces or appends to timeline

#### **B. Timeline Editing Operations**
- `CUT` - Remove clip(s) at timeline indices
  - Example: "cut timeline index 0 and 1"
  - Removes chunks, adjusts subsequent start_times
  
- `REPLACE` - Replace clip(s) at timeline indices
  - Example: "replace timeline index 0-2 with new clips"
  - Option A: Use existing planner query to find replacement
  - Option B: User provides new query for replacement
  
- `INSERT` - Add clip(s) between existing clips
  - Example: "add a clip between timeline index 1 and 2"
  - Finds appropriate clip using planner
  - Inserts at specified position
  - Adjusts all subsequent start_times

- `DELETE` - Remove specific clip(s)
  - Similar to CUT but more explicit

#### **C. B-Roll Operations**
- `FIND_BROLL` - Find B-roll between two timeline segments
  - User selects: "timeline index 0 to 2, find B-roll"
  - Extracts source time range from selected chunks
  - Queries planner: "find clips between [start_time] and [end_time] that are different from main action"
  - Inserts B-roll clips between or alongside main clips

#### **D. Refinement Operations**
- `REORDER` - Change clip sequence
  - Example: "move timeline index 3 before index 1"
  
- `TRIM` - Adjust clip boundaries
  - Example: "trim timeline index 0 by 2 seconds from start"
  
- `SPEED_ADJUST` - Change playback speed
  - Example: "make timeline index 2 play at 1.5x speed"

---

## 🧠 Agentic Decision-Making

### **Operation Classification**
The orchestrator must understand user intent:

```
User Query → Intent Analysis → Operation Type → Execution Plan
```

**Intent Patterns:**
- "find highlights" → `FIND_HIGHLIGHTS`
- "cut/replace timeline index X" → `CUT` / `REPLACE`
- "add clip between X and Y" → `INSERT`
- "find B-roll for timeline X-Y" → `FIND_BROLL`
- "move/reorder" → `REORDER`
- "trim/adjust" → `TRIM`

### **Context Awareness**
- **Timeline State**: Knows current timeline structure
- **Previous Operations**: Remembers recent edits
- **Narrative Continuity**: Ensures logical flow between clips
- **Source Video Mapping**: Maps timeline indices to source video times

---

## 🔄 Workflow Architecture

### **High-Level Flow**

```
User Query
    ↓
[Orchestrator Agent]
    ├─→ Intent Analysis (LLM)
    ├─→ Operation Classification
    ├─→ Timeline State Load
    │
    ├─→ [If FIND_HIGHLIGHTS]
    │   └─→ Call Planner Agent
    │       └─→ Receive time_ranges
    │       └─→ Create timeline chunks
    │       └─→ Update timeline.json
    │
    ├─→ [If CUT/REPLACE/INSERT]
    │   ├─→ Parse timeline indices
    │   ├─→ Extract source time ranges
    │   ├─→ [If REPLACE needs new content]
    │   │   └─→ Call Planner Agent with query
    │   ├─→ Modify timeline chunks
    │   └─→ Update timeline.json
    │
    ├─→ [If FIND_BROLL]
    │   ├─→ Extract time range from selected chunks
    │   ├─→ Query Planner: "find B-roll between X-Y"
    │   ├─→ Receive alternative clips
    │   └─→ Insert as B-roll chunks
    │
    └─→ [Executor Agent] (if clips need extraction)
        └─→ Extract new clips
        └─→ Update clip_paths in timeline
```

---

## 📊 State Management

### **Orchestrator State Extension**

```python
class OrchestratorState(AgentState):
    # Existing AgentState fields...
    
    # Timeline management
    timeline_path: str  # Path to timeline.json
    timeline_chunks: List[Dict]  # Current timeline chunks
    timeline_version: str  # Version for tracking changes
    
    # Operation context
    current_operation: Optional[str]  # "FIND_HIGHLIGHTS", "CUT", etc.
    operation_params: Optional[Dict]  # Operation-specific parameters
    
    # B-roll context
    selected_timeline_indices: Optional[List[int]]  # For B-roll operations
    broll_time_range: Optional[Tuple[float, float]]  # Source time range for B-roll
    
    # Narrative context
    editing_history: List[Dict]  # History of operations
    narrative_notes: Optional[str]  # Director's notes about narrative flow
```

---

## 🎯 Best Practices: Director-Like Behavior

### **1. Narrative Continuity**
- **Check transitions**: When inserting/replacing, ensure clips flow naturally
- **Pacing awareness**: Consider clip durations and overall timeline length
- **Visual continuity**: Avoid jarring cuts (unless intentional)
- **Audio continuity**: Consider audio transitions between clips

### **2. Intelligent Clip Selection**
- **Context-aware search**: When replacing, understand what came before/after
- **Complementary content**: B-roll should complement, not duplicate main action
- **Quality scoring**: Prefer higher-scored clips when multiple options exist

### **3. User Intent Interpretation**
- **Ambiguity resolution**: "cut the first two" → timeline indices 0,1
- **Relative references**: "after the cooking clip" → find clip with cooking, then operate
- **Implicit operations**: "this is too long" → suggest trim operation

### **4. Timeline Integrity**
- **Index validation**: Ensure timeline indices exist before operations
- **Time range validation**: Ensure source times are within video duration
- **Gap handling**: When cutting, decide whether to close gaps or leave them
- **Overlap prevention**: Ensure clips don't overlap in timeline

---

## 🔍 Operation Examples

### **Example 1: Find Highlights**
```
User: "find the highlights of the video"
Orchestrator:
  1. Classify: FIND_HIGHLIGHTS
  2. Call Planner: query="highlights"
  3. Receive: time_ranges = [(22, 28), (65, 70), ...]
  4. Create timeline chunks:
     - chunk 0: start_time=0, end_time=6, original_start_time=22, ...
     - chunk 1: start_time=6, end_time=11, original_start_time=65, ...
  5. Save timeline.json
  6. Call Executor to extract clips
```

### **Example 2: Replace Timeline Segments**
```
User: "replace timeline index 0 and 1 with clips of people cooking"
Orchestrator:
  1. Classify: REPLACE
  2. Parse: timeline_indices = [0, 1]
  3. Load timeline: Get chunks at indices 0, 1
  4. Extract source times: original_start_time=22, original_end_time=70
  5. Call Planner: query="people cooking"
  6. Receive: time_ranges = [(120, 130), (145, 155)]
  7. Replace chunks:
     - chunk 0: Update original_start_time=120, original_end_time=130, ...
     - chunk 1: Update original_start_time=145, original_end_time=155, ...
  8. Recalculate timeline start_times (maintain sequence)
  9. Save timeline.json
  10. Call Executor to extract new clips
```

### **Example 3: Find B-Roll**
```
User: "for timeline index 0 to 2, find B-roll"
Orchestrator:
  1. Classify: FIND_BROLL
  2. Parse: selected_indices = [0, 1, 2]
  3. Load timeline: Get chunks at indices 0, 1, 2
  4. Extract time range:
     - start = min(original_start_time of chunks)
     - end = max(original_end_time of chunks)
     - Result: (22, 131)
  5. Analyze main action: "walking, equipment, camping"
  6. Call Planner: query="find B-roll between 22s and 131s, show nature, scenery, wide shots, different from main action"
  7. Receive: time_ranges = [(30, 35), (50, 55), ...]  # Different moments
  8. Create B-roll chunks:
     - Mark as chunk_type="broll"
     - Insert between or alongside main chunks
  9. Save timeline.json
  10. Call Executor to extract B-roll clips
```

### **Example 4: Insert Between Clips**
```
User: "add a transition clip between timeline index 1 and 2"
Orchestrator:
  1. Classify: INSERT
  2. Parse: insert_position = between index 1 and 2
  3. Analyze context:
     - Before: chunk 1 (equipment preparation)
     - After: chunk 2 (camping setup)
  4. Call Planner: query="transition, nature, establishing shot, between equipment and camping"
  5. Receive: time_ranges = [(80, 85)]
  6. Create new chunk:
     - start_time = chunk[1].end_time (11.0)
     - end_time = 11.0 + duration
     - original_start_time = 80, original_end_time = 85
  7. Shift subsequent chunks: chunk[2].start_time += duration
  8. Save timeline.json
  9. Call Executor to extract clip
```

---

## 🎨 System Prompt Design

### **Orchestrator System Prompt (Draft)**

```
You are a professional video editor and movie director orchestrating a video editing workflow.

Your role:
1. **Timeline Management**: Maintain a coherent timeline.json that represents the edited video sequence
2. **Operation Execution**: Execute editing operations (cut, replace, insert, B-roll) with narrative awareness
3. **Context Awareness**: Understand the current timeline state and how operations affect narrative flow
4. **Quality Control**: Ensure clips flow naturally, maintain pacing, and create compelling narratives

Timeline Structure:
- Each chunk has: start_time/end_time (timeline position), original_start_time/original_end_time (source video)
- Timeline indices start at 0
- Operations must maintain timeline continuity

Operation Types:
- FIND_HIGHLIGHTS: Find and add highlights to timeline
- CUT: Remove chunks at specified timeline indices
- REPLACE: Replace chunks with new content (may require planner query)
- INSERT: Add clips between existing chunks
- FIND_BROLL: Find complementary B-roll for selected timeline segments
- REORDER: Change clip sequence
- TRIM: Adjust clip boundaries

Best Practices:
- Maintain narrative continuity between clips
- Consider pacing and rhythm
- Ensure smooth transitions
- Validate timeline indices before operations
- Preserve timeline integrity (no overlaps, valid times)
- When replacing, understand context (what came before/after)
- B-roll should complement, not duplicate main action
```

---

## 🔗 Integration Points

### **With Planner Agent**
- Orchestrator calls planner for content discovery
- Passes queries: "highlights", "people cooking", "B-roll between X-Y"
- Receives time_ranges
- Converts time_ranges to timeline chunks

### **With Executor Agent**
- Orchestrator calls executor after timeline updates
- Passes new/updated chunks that need clip extraction
- Executor extracts clips and updates clip_paths

### **With Tracking Data**
- For B-roll operations, may query tracking.json
- Find moments with different object compositions
- Identify wide shots vs close-ups
- Track object movements for continuity

---

## 📝 Timeline.json Operations

### **Chunk Creation**
```python
def create_timeline_chunk(
    original_start: float,
    original_end: float,
    timeline_start: float,
    chunk_type: str = "highlight",
    description: str = "",
    score: float = 1.0
) -> Dict:
    return {
        "start_time": timeline_start,
        "end_time": timeline_start + (original_end - original_start),
        "chunk_type": chunk_type,
        "speed": 1.0,
        "description": description,
        "score": score,
        "original_start_time": original_start,
        "original_end_time": original_end,
        "clip_path": None  # Will be set by executor
    }
```

### **Timeline Updates**
- After CUT: Remove chunks, shift subsequent start_times
- After REPLACE: Update chunk metadata, maintain timeline positions
- After INSERT: Add chunks, shift subsequent start_times
- After REORDER: Recalculate all start_times

---

## 🚀 Implementation Phases

### **Phase 1: Basic Operations**
- Timeline loading/saving
- FIND_HIGHLIGHTS
- CUT operation
- Basic REPLACE

### **Phase 2: Advanced Operations**
- INSERT with context awareness
- FIND_BROLL
- REORDER
- TRIM

### **Phase 3: Director Intelligence**
- Narrative continuity checking
- Automatic transition suggestions
- Pacing analysis
- Quality scoring integration

---

## ❓ Open Questions

1. **Gap Handling**: When cutting clips, should we:
   - Close gaps automatically?
   - Leave gaps for user to fill?
   - Suggest filler content?

2. **B-Roll Placement**: Should B-roll:
   - Replace main clips?
   - Insert alongside (picture-in-picture)?
   - Create separate timeline track?

3. **Timeline Versioning**: Should we:
   - Keep edit history?
   - Support undo/redo?
   - Version timeline.json?

4. **Multi-Track Support**: Should timeline support:
   - Multiple video tracks?
   - Audio tracks?
   - Overlay tracks?

---

## 📌 Next Steps

1. **Define OrchestratorState** - Extend AgentState
2. **Create Orchestrator Agent** - Main decision-making node
3. **Implement Timeline Manager** - Load/save/update timeline.json
4. **Implement Operation Handlers** - CUT, REPLACE, INSERT, etc.
5. **Integrate with Planner** - Call planner for content discovery
6. **Integrate with Executor** - Call executor for clip extraction
7. **Add Director Intelligence** - Narrative continuity, pacing, quality

