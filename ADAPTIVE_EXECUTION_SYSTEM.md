# Adaptive Execution System - Complete Guide

**Date:** October 14, 2025
**Status:** ✅ FULLY ADAPTIVE - Tasks change based on meeting results

---

## 🎯 Problem Solved

**Your Concern:**
> "Tech Analyst MUST use ≥2 tools (run_script for GPU check, CLIP install) i am not sure if it would be CLIP install everytime. and the execution should do all the pirority tasks listed in the meeting in order"

**Solution:**
✅ Tasks are now **dynamically generated** from previous meeting's action items
✅ Execution follows **priority order** (HIGH → MEDIUM → LOW)
✅ No hardcoded tasks - everything adapts to what was decided

---

## 🔄 How It Works

### Meeting Flow:

```
Meeting N (Planning):
├─ Agents discuss strategy
├─ Make decisions
├─ Assign tasks with priorities
└─ Save to actions_YYYYMMDD_HHMMSS.json
     │
     │ [Automatic generation]
     ↓
Meeting N+1 Topic Auto-Generated:
├─ Read previous meeting files
├─ Extract tasks from actions.json
├─ Group by agent and priority
├─ Create NEXT_MEETING_TOPIC.md
     │
     │ [Meeting N+1 starts]
     ↓
Meeting N+1 (Execution):
├─ Executors read previous meeting
├─ See THEIR specific tasks
├─ Execute in priority order (HIGH → MEDIUM → LOW)
├─ Use appropriate tools
├─ Save results
└─ Create NEW actions.json
     │
     │ [Automatic generation]
     ↓
Meeting N+2 Topic Auto-Generated:
└─ Repeat with NEW tasks...
```

---

## 📊 Example: How Tasks Adapt

### Scenario 1: After Strategic Planning Meeting

**Meeting N decisions:**
- Decided: Focus on V1 success story (Option C)
- Research Director assigned: Document V1's MMR methodology (HIGH priority)
- Tech Analyst assigned: Extract V1 metrics (HIGH priority)

**Meeting N+1 auto-generated topic:**
```markdown
### Research Director - 1 task assigned

#### HIGH Priority Tasks (1):
**Task 1:** Document V1's MMR methodology
- Type: analysis
- Priority: HIGH
- Suggested tools: read_file(), list_files()
- Action required: Read V1 code, create documentation

### Tech Analyst - 1 task assigned

#### HIGH Priority Tasks (1):
**Task 1:** Extract V1 metrics from comprehensive_evaluation.json
- Type: analysis
- Priority: HIGH
- Suggested tools: read_file(), run_script()
- Action required: Load JSON, extract metrics, analyze
```

**Meeting N+1 execution:**
- Research Director uses `read_file('research/v1_model.py')`
- Research Director creates `V1_MMR_DOCUMENTATION.md`
- Tech Analyst uses `read_file('comprehensive_evaluation.json')`
- Tech Analyst creates `v1_metrics_analysis.py`

### Scenario 2: After Go/No-Go Decision

**Meeting N decisions:**
- Decided: Proceed with Plan A (attention collapse framework)
- Research Director assigned: Test diagnostic tool on CLIP (HIGH priority)
- Tech Analyst assigned: Install CLIP model (HIGH priority), prepare dataset (MEDIUM)

**Meeting N+1 auto-generated topic:**
```markdown
### Research Director - 1 task assigned

#### HIGH Priority Tasks (1):
**Task 1:** Test diagnostic tool on CLIP model
- Type: experiment
- Priority: HIGH
- Suggested tools: run_script()
- Action required: Run diagnostic, analyze results

### Tech Analyst - 2 tasks assigned

#### HIGH Priority Tasks (1):
**Task 1:** Install CLIP model and dependencies
- Type: setup
- Priority: HIGH
- Suggested tools: run_script()
- Action required: Install transformers, load CLIP

#### MEDIUM Priority Tasks (1):
**Task 2:** Prepare COCO dataset (n=100 samples)
- Type: setup
- Priority: MEDIUM
- Suggested tools: run_script()
- Action required: Download, preprocess dataset
```

**Meeting N+1 execution:**
- Tech Analyst executes HIGH priority first: `run_script('pip install transformers')`
- Research Director waits for CLIP to be ready
- Tech Analyst completes CLIP setup, moves to MEDIUM: dataset prep
- Research Director then runs: `run_script('test_diagnostic_clip.py')`

### Scenario 3: After Experiment Results

**Meeting N decisions:**
- Decided: CLIP results promising, expand to ALIGN
- Research Director assigned: Implement ALIGN adapter (HIGH priority)
- Tech Analyst assigned: Run ALIGN experiments (MEDIUM priority)
- Pre-Architect assigned: Review architecture (LOW priority)

**Meeting N+1 auto-generated topic:**
```markdown
### Research Director - 1 task assigned

#### HIGH Priority Tasks (1):
**Task 1:** Implement ALIGN adapter for diagnostic tool
- Type: implementation
- Priority: HIGH
- Suggested tools: read_file(), run_script()
- Action required: Study ALIGN, create adapter, test

### Tech Analyst - 1 task assigned

#### MEDIUM Priority Tasks (1):
**Task 1:** Run experiments on ALIGN with n=100 samples
- Type: experiment
- Priority: MEDIUM
- Suggested tools: run_script()
- Action required: Test ALIGN, collect results
```

**Meeting N+1 execution:**
- Research Director executes HIGH: implements ALIGN adapter
- Tech Analyst executes MEDIUM: runs experiments once adapter ready
- Pre-Architect reviews (no tools needed, LOW priority)

---

## 🤖 Automatic Topic Generation

### Script: `generate_next_meeting_topic.py`

**Location:** `unified-team/generate_next_meeting_topic.py`

**What it does:**
1. Reads latest `actions_YYYYMMDD_HHMMSS.json`
2. Groups tasks by agent and priority
3. Determines appropriate tools based on task type
4. Generates `NEXT_MEETING_TOPIC.md`

**Usage:**
```bash
cd unified-team
python3 generate_next_meeting_topic.py
```

**Output:**
```
======================================================================
🤖 GENERATING NEXT MEETING TOPIC
======================================================================

📂 Loaded: actions_20251014_033652.json
   Total actions: 21

📊 Task Distribution:
   research_director: 5 total (H:0, M:5, L:0)
   tech_analyst: 0 total (H:0, M:0, L:0)
   pre_architect: 5 total (H:0, M:5, L:0)

✅ Generated: unified-team/NEXT_MEETING_TOPIC.md

💡 Use this topic for your next meeting:
   coordinator.run_meeting(open('unified-team/NEXT_MEETING_TOPIC.md').read())
======================================================================
```

### Automation Options

**Option 1: Post-Meeting Hook**
Add to coordinator after each meeting:
```python
# After meeting completes
import subprocess
subprocess.run(['python3', 'unified-team/generate_next_meeting_topic.py'])

# Next meeting uses generated topic
next_topic = open('unified-team/NEXT_MEETING_TOPIC.md').read()
coordinator.run_meeting(next_topic)
```

**Option 2: Manual Between Meetings**
Run script manually after reviewing meeting results:
```bash
# Review latest meeting
cat unified-team/reports/transcript_*.md | tail -100

# Generate next topic
python3 unified-team/generate_next_meeting_topic.py

# Review generated topic
cat unified-team/NEXT_MEETING_TOPIC.md

# Run meeting with generated topic (if satisfied)
# In Python/Colab:
# coordinator.run_meeting(open('unified-team/NEXT_MEETING_TOPIC.md').read())
```

**Option 3: Integrated Into Autonomous System**
Modify `unified_autonomous_system.ipynb` to auto-generate:
```python
# In meeting loop, after each meeting:
for meeting_num in range(max_meetings):
    if meeting_num == 0:
        # First meeting uses strategic planning topic
        topic = STRATEGIC_PLANNING_TOPIC
    else:
        # Subsequent meetings use auto-generated topics
        subprocess.run(['python3', 'unified-team/generate_next_meeting_topic.py'])
        topic = open('unified-team/NEXT_MEETING_TOPIC.md').read()

    result = coordinator.run_meeting(topic)
```

---

## 📋 Priority-Based Execution

### How Priority Works:

**HIGH Priority:**
- Must be completed or explicitly blocked
- Executed first before any other tasks
- Blockers reported immediately
- If blocked, noted and moved to next HIGH task

**MEDIUM Priority:**
- Attempted after all HIGH tasks
- Can be deferred if time limited
- Should aim for ≥50% completion

**LOW Priority:**
- Attempted only if time permits
- Can be postponed to next meeting
- No penalty for not completing

### Example Execution Order:

**Tasks assigned:**
1. Research Director: Task A (HIGH), Task B (MEDIUM), Task C (LOW)
2. Tech Analyst: Task D (HIGH), Task E (MEDIUM)

**Execution sequence:**
```
1. Research Director starts Task A (HIGH)
2. Tech Analyst starts Task D (HIGH)
3. Research Director completes A → starts Task B (MEDIUM)
4. Tech Analyst completes D → starts Task E (MEDIUM)
5. Research Director completes B → starts Task C (LOW) if time permits
6. Tech Analyst completes E → supports Research Director
```

**Status report:**
```markdown
## Research Director
- Task A (HIGH): ✅ COMPLETE
- Task B (MEDIUM): ✅ COMPLETE
- Task C (LOW): ⏳ DEFERRED (time limited)

## Tech Analyst
- Task D (HIGH): ✅ COMPLETE
- Task E (MEDIUM): ✅ COMPLETE
```

---

## 🛠️ Tool Recommendations

The script automatically suggests tools based on:

**Task Type:**
- `experiment` → `run_script()`
- `analysis` → `read_file()`
- `setup` → `run_script()`
- `implementation` → `read_file()`, `run_script()`
- `review` → `read_file()`

**Description Keywords:**
- "read", "review", "analyze" → `read_file()`
- "run", "test", "execute" → `run_script()`
- "find", "locate", "search" → `list_files()`

**Example:**
```markdown
**Task:** Test diagnostic tool on CLIP model
- Type: experiment
- Suggested tools: run_script()
- Action: Run test_diagnostic_clip.py, collect results
```

Executors can adapt suggestions based on actual needs.

---

## ✅ Verification

**After meeting, verify execution:**

1. **Check tools used:**
   ```bash
   grep "Tools Executed:" unified-team/reports/transcript_*.md | tail -1
   # Expected: Tools Executed: ≥4 (if both executors had tasks)
   ```

2. **Verify priority order:**
   ```bash
   # Check that HIGH priority tasks mentioned before MEDIUM/LOW
   grep -A 5 "Task.*HIGH" unified-team/reports/transcript_*.md
   ```

3. **Check task completion:**
   ```bash
   # Look for status reports in transcript
   grep -A 3 "Status:" unified-team/reports/transcript_*.md
   ```

---

## 📁 Files in Adaptive System

### Created Files:

1. **ADAPTIVE_EXECUTION_TOPIC.md**
   - Template and explanation of adaptive system
   - Execution protocol and rules
   - Examples of dynamic task adaptation

2. **generate_next_meeting_topic.py**
   - Automated topic generator
   - Reads actions.json, creates NEXT_MEETING_TOPIC.md
   - Groups tasks by agent and priority

3. **NEXT_MEETING_TOPIC.md** (auto-generated)
   - Dynamic meeting topic for next meeting
   - Contains specific tasks from previous meeting
   - Updates after each meeting

4. **ADAPTIVE_EXECUTION_SYSTEM.md** (this file)
   - Complete guide to adaptive system
   - Examples and workflows
   - Integration instructions

---

## 🎯 Key Advantages

### vs. Hardcoded Tasks:

**Hardcoded (OLD):**
```markdown
Tech Analyst MUST:
- Install CLIP
- Check GPU
- Download dataset
```
❌ Same tasks every meeting
❌ Not responsive to decisions
❌ Wastes time if already done

**Adaptive (NEW):**
```markdown
Tech Analyst tasks from previous meeting:
- HIGH: [whatever was decided]
- MEDIUM: [whatever was assigned]
```
✅ Tasks change based on decisions
✅ Responsive to meeting results
✅ No duplicate work

### Benefits:

1. **Responsive:** Tasks adapt to decisions made
2. **Priority-driven:** Important work happens first
3. **No waste:** Don't repeat completed work
4. **Flexible:** Handles changing requirements
5. **Traceable:** Clear link from decision → task → execution

---

## 🚀 Getting Started

### Step 1: Run Your Current Meeting
```python
# On Colab - your autonomous system is already running
# Wait for first meeting to complete
```

### Step 2: Generate Next Topic
```bash
# After first meeting completes
cd /content/drive/MyDrive/cv_multimodal/project/computer-vision-clean
python3 unified-team/generate_next_meeting_topic.py
```

### Step 3: Review Generated Topic
```bash
cat unified-team/NEXT_MEETING_TOPIC.md
```

### Step 4: Use for Next Meeting
```python
# In Colab
topic = open('unified-team/NEXT_MEETING_TOPIC.md').read()
result = coordinator.run_meeting(topic)
```

### Step 5: Automate (Optional)
Integrate into autonomous system so it auto-generates after each meeting.

---

## 📊 Success Metrics

**After 5 meetings, you should see:**

1. **Task diversity:** Different tasks each meeting (not repeated)
2. **Priority adherence:** HIGH tasks completed before MEDIUM/LOW
3. **Tool usage:** ≥2 tools per executor per meeting
4. **Progress:** Clear advancement toward milestones
5. **Adaptation:** Tasks respond to blockers and results

**Example progression:**
- Meeting 1: Strategic planning → assigns tasks
- Meeting 2: Execute assigned tasks → discovers blocker
- Meeting 3: Resolve blocker (NEW task) → continue original work
- Meeting 4: Complete original work → move to next milestone
- Meeting 5: New milestone tasks → fresh priorities

---

## 🎓 Summary

**Your Requirements:**
✅ Tasks change based on meeting results
✅ Execution follows priority order (HIGH → MEDIUM → LOW)
✅ All assigned tasks attempted (not just first one)
✅ Tools used appropriately for each task type
✅ Results saved and documented

**System Features:**
✅ Automatic topic generation from action items
✅ Priority-based task organization
✅ Tool recommendations based on task type
✅ Status tracking and reporting
✅ No hardcoded tasks - fully dynamic

**Status:** 🟢 READY FOR ADAPTIVE EXECUTION

---

**Next Action:** Wait for first meeting to complete, then run `generate_next_meeting_topic.py` to see adaptive system in action.
