# Meeting Execution Verification Checklist

**Purpose:** Verify that agents are EXECUTING tasks (not just planning) in meetings

**Created:** October 14, 2025 12:40 PM
**For:** Post-meeting verification after unified system meetings

---

## Quick Status Check

### ✅ EXECUTION MEETING Criteria (Must meet ALL):
- [ ] **Tools Executed:** Count > 0 (visible in meeting logs)
- [ ] **Concrete Artifacts:** Files created/modified during meeting
- [ ] **Research Director used tools:** read_file(), run_script(), or list_files()
- [ ] **Tech Analyst used tools:** run_script(), read_file(), or list_files()
- [ ] **Results saved:** Experiment logs, scripts, or data files created

### ⚠️ PLANNING MEETING Criteria (If only these are true):
- [ ] **Tools Executed:** Count = 0
- [ ] **Only discussion:** No file modifications or script executions
- [ ] **No concrete artifacts:** No new files created
- [ ] **Next steps discussed:** But not implemented

---

## Detailed Verification Steps

### Step 1: Check Meeting Transcript

**File:** `unified-team/reports/transcript_YYYYMMDD_HHMMSS.md`

**Look for:**
```markdown
## Tools Executed

1. **read_file** (research_director)
   - File: research/attention_analysis.py
   - Result: Successfully read 350 lines
   - Key findings: [...]

2. **run_script** (tech_analyst)
   - Script: setup_clip.py
   - Result: CLIP model loaded successfully
   - Output: [...]

3. **list_files** (research_director)
   - Directory: research/
   - Found: 15 files
   - Relevant: attention_analysis.py, v2_model.py
```

**Verification:**
- ✅ **EXECUTION:** Tools section exists and has ≥2 entries
- ⚠️ **PLANNING:** Tools section shows "Tools Executed: 0"

---

### Step 2: Check File System for Artifacts

**Commands:**
```bash
# Check if new files created during meeting
ls -lt unified-team/experiments/ | head -10
ls -lt unified-team/scripts/ | head -10
ls -lt research/ | head -10

# Look for recently modified files (within last 2 hours)
find unified-team/ -type f -mmin -120 -ls

# Check for experiment results
ls -lt unified-team/experiments/results/
```

**Expected Artifacts (for EXECUTION meeting):**
- **Tool development:** `universal_attention_analyzer.py`, `clip_adapter.py`
- **Environment setup:** `setup_clip.py`, `requirements.txt`
- **Experiment logs:** `week1_day1_results.txt`, `clip_test_output.log`
- **Data files:** `coco_samples_50.json`, `attention_weights.pkl`

**Verification:**
- ✅ **EXECUTION:** ≥1 new file created or ≥2 files modified
- ⚠️ **PLANNING:** No new files, only transcript and summary created

---

### Step 3: Review Agent Tool Usage

**Check Research Director (Primary Executor):**

**Expected tool calls:**
1. `list_files('research/')` → Find existing diagnostic code
2. `read_file('research/attention_analysis.py')` → Understand current implementation
3. `run_script('test_universal_analyzer.py')` → Test generalized approach

**Verification:**
```bash
# Check transcript for Research Director tool usage
grep -A 5 "research_director" unified-team/reports/transcript_*.md | grep -E "(read_file|run_script|list_files)"
```

**Expected output:**
```
research_director: read_file('research/attention_analysis.py')
research_director: list_files('research/')
research_director: run_script('test_adapter.py')
```

- ✅ **EXECUTION:** Research Director used ≥1 tool
- ⚠️ **PLANNING:** Research Director used 0 tools

---

**Check Tech Analyst (Secondary Executor):**

**Expected tool calls:**
1. `run_script('pip install transformers torch')` → Install dependencies
2. `run_script('setup_clip.py')` → Load CLIP model
3. `read_file('requirements.txt')` → Check existing dependencies
4. `list_files('data/')` → Find datasets

**Verification:**
```bash
# Check transcript for Tech Analyst tool usage
grep -A 5 "tech_analyst" unified-team/reports/transcript_*.md | grep -E "(read_file|run_script|list_files)"
```

**Expected output:**
```
tech_analyst: run_script('setup_clip.py')
tech_analyst: run_script('pip install transformers')
tech_analyst: list_files('data/')
```

- ✅ **EXECUTION:** Tech Analyst used ≥1 tool
- ⚠️ **PLANNING:** Tech Analyst used 0 tools

---

### Step 4: Check Trajectory Logs

**File:** `sessions/session_YYYYMMDD_HHMMSS/trajectories/trajectory_001.json`

**What to check:**
```python
import json

with open('sessions/session_20251014_162454/trajectories/trajectory_001.json') as f:
    traj = json.load(f)

print(f"Tools executed: {traj['metadata']['tools_executed']}")
print(f"Participants: {traj['metadata']['participants']}")

# Check agent actions
for msg in traj['messages']:
    if 'tool_call' in msg:
        print(f"{msg['agent']}: {msg['tool_call']['tool']}({msg['tool_call']['args']})")
```

**Expected output (EXECUTION meeting):**
```
Tools executed: 5
Participants: ['research_director', 'tech_analyst', 'pre_architect', 'data_analyst', 'critic']

research_director: read_file({'file_path': 'research/attention_analysis.py'})
research_director: list_files({'directory': 'research/'})
tech_analyst: run_script({'script_path': 'setup_clip.py'})
tech_analyst: run_script({'script_path': 'pip', 'args': 'install transformers'})
research_director: run_script({'script_path': 'test_adapter.py'})
```

**Verification:**
- ✅ **EXECUTION:** tools_executed > 0, multiple tool calls logged
- ⚠️ **PLANNING:** tools_executed = 0

---

### Step 5: Check Deployment State Changes

**File:** `unified-team/state/deployment_state.json`

**Check for SLO status changes:**
```bash
cat unified-team/state/deployment_state.json | grep -A 5 "slo_status"
```

**Expected progression:**

**After Planning Meeting:**
```json
"slo_status": {
  "compliance": false,
  "latency": false,
  "error_rate": false
}
```

**After Execution Meeting (with experiments run):**
```json
"slo_status": {
  "compliance": true,   // ← Changed after experiments
  "latency": true,      // ← Latency measured
  "error_rate": true    // ← Error rate acceptable
}
```

**Verification:**
- ✅ **EXECUTION:** At least one SLO status changed to true
- ⚠️ **PLANNING:** All SLO status remain false

---

## Summary Checklist

### Quick 60-Second Verification

Run these commands after meeting completes:

```bash
# 1. Check transcript for tool usage (should be > 0)
echo "=== TOOL USAGE ==="
grep "Tools Executed:" unified-team/reports/transcript_*.md | tail -1

# 2. Check for new files in last 2 hours
echo "=== NEW FILES ==="
find unified-team/ research/ -type f -mmin -120 -ls | wc -l

# 3. Check executor tool calls
echo "=== RESEARCH DIRECTOR TOOLS ==="
grep -c "research_director.*\(read_file\|run_script\|list_files\)" unified-team/reports/transcript_*.md | tail -1

echo "=== TECH ANALYST TOOLS ==="
grep -c "tech_analyst.*\(read_file\|run_script\|list_files\)" unified-team/reports/transcript_*.md | tail -1

# 4. Check deployment state
echo "=== SLO STATUS ==="
cat unified-team/state/deployment_state.json | grep -A 5 "slo_status"
```

**Expected output for EXECUTION meeting:**
```
=== TOOL USAGE ===
Tools Executed: 5

=== NEW FILES ===
7

=== RESEARCH DIRECTOR TOOLS ===
3

=== TECH ANALYST TOOLS ===
2

=== SLO STATUS ===
"slo_status": {
  "compliance": true,
  "latency": true,
  "error_rate": false
}
```

---

## Red Flags (Planning Only - No Execution)

### 🚩 Red Flag 1: Zero Tool Usage
```
Tools Executed: 0
```
**Problem:** Agents only discussed tasks, didn't execute anything
**Action:** Review WEEK_1_MEETING_TOPIC.md to ensure explicit tool requirements

### 🚩 Red Flag 2: No File Modifications
```bash
find unified-team/ research/ -type f -mmin -120 -ls
# Returns: 0 results (only transcript and summary)
```
**Problem:** No concrete work produced
**Action:** Verify agents have file write permissions and tool access

### 🚩 Red Flag 3: Executors Didn't Use Tools
```bash
grep "research_director.*\(read_file\|run_script\|list_files\)" transcript.md
# Returns: 0 matches

grep "tech_analyst.*\(read_file\|run_script\|list_files\)" transcript.md
# Returns: 0 matches
```
**Problem:** Executors participated but only as planners
**Action:** Update agent prompts to emphasize EXECUTION responsibility

### 🚩 Red Flag 4: Deployment State Unchanged
```json
"slo_status": {
  "compliance": false,
  "latency": false,
  "error_rate": false
}
```
**Problem:** No experiments run, no metrics collected
**Action:** Verify MLflow integration and metric collection enabled

---

## Corrective Actions

### If Meeting Was Planning Only (When Execution Expected):

**1. Review Meeting Topic Requirements**
```bash
cat unified-team/WEEK_1_MEETING_TOPIC.md | grep -A 10 "PART 3: IMMEDIATE TASKS"
```
- Check if tool usage was explicitly required
- Verify executor tasks were clearly defined
- Ensure success criteria included tool execution

**2. Update Next Meeting Topic**
Create stronger execution emphasis:
```markdown
## MANDATORY TOOL EXECUTION (CANNOT BE SKIPPED)

### Research Director (EXECUTOR):
**MUST USE TOOLS - NOT OPTIONAL:**
1. ✅ MUST call list_files('research/') to locate existing code
2. ✅ MUST call read_file() on attention_analysis.py
3. ✅ MUST call run_script() to test implementation

**If you do not use tools, the meeting is considered incomplete.**

### Tech Analyst (EXECUTOR):
**MUST USE TOOLS - NOT OPTIONAL:**
1. ✅ MUST call run_script() to install CLIP
2. ✅ MUST call run_script() to test CLIP loading
3. ✅ MUST call list_files() to find datasets

**If you do not use tools, the meeting is considered incomplete.**
```

**3. Check Agent Prompts**
Review `agents/research_director.md` and `agents/tech_analyst.md`:
- Do prompts emphasize EXECUTION over planning?
- Are tool usage requirements clear?
- Is there explicit instruction to use tools during meetings?

**4. Verify Tool Access**
```python
# Test if agents can access tools
from unified_coordinator import UnifiedCoordinator

coordinator = UnifiedCoordinator()
tools = coordinator.get_available_tools()
print("Available tools:", tools)

# Expected: ['read_file', 'run_script', 'list_files', 'write_file']
```

**5. Manual Test of Tool Execution**
```python
# Force a tool execution test
result = coordinator.research_director.use_tool(
    tool='list_files',
    args={'directory': 'research/'}
)
print("Tool result:", result)
```

---

## Success Pattern (Reference)

### Example of SUCCESSFUL Execution Meeting

**Transcript excerpt:**
```markdown
## Meeting: Week 1 Day 1 Execution
Date: 2025-10-14 14:30
Participants: 5 agents

### Research Director (19:32:15):
I'll begin by locating our existing diagnostic code.

**Tool: list_files('research/')**
Result: Found 12 files including attention_analysis.py

Now I'll review the current implementation.

**Tool: read_file('research/attention_analysis.py')**
Result: Read 350 lines. Current implementation uses PyTorch hooks to extract attention weights from our V2 model. Key insight: The hook-based approach is architecture-specific.

Let me test a generalized version.

**Tool: run_script('test_universal_analyzer.py')**
Result: Script executed successfully. Output shows the adapter pattern works for CLIP.

### Tech Analyst (19:35:42):
I'll set up the CLIP environment.

**Tool: run_script('pip install transformers torch torchvision')**
Result: Packages installed successfully. transformers==4.35.0, torch==2.1.0

**Tool: run_script('setup_clip.py')**
Result: CLIP model loaded successfully. Model size: 428MB. Ready for attention extraction.

**Tool: list_files('data/')**
Result: Found COCO captions dataset with 5000 samples. Selecting 50 for Week 1.

## Tools Executed: 6

## Artifacts Created:
- test_universal_analyzer.py (Research Director)
- setup_clip.py (Tech Analyst)
- requirements.txt (Tech Analyst)
- week1_day1_log.txt (Both)

## Results Saved:
- experiments/results/clip_baseline_50_samples.json
- experiments/logs/week1_day1_execution.log
```

**Verification:**
- ✅ Tools executed: 6 (target: ≥2)
- ✅ Both executors used tools (Research Director: 3, Tech Analyst: 3)
- ✅ Artifacts created: 4 files
- ✅ Results saved: 2 output files
- ✅ Concrete progress: CLIP environment set up, universal analyzer tested

**Deployment state after meeting:**
```json
"slo_status": {
  "compliance": true,   // ← Changed (experiments completed)
  "latency": true,      // ← Measured (tool execution time tracked)
  "error_rate": false   // ← Still false (no errors encountered)
}
```

---

## Usage Instructions

### After Each Meeting:

1. **Wait 5 minutes** for all files to be saved
2. **Run Quick 60-Second Verification** (commands above)
3. **Check results:**
   - ✅ All green → EXECUTION meeting successful
   - ⚠️ Any yellow/red → PLANNING meeting (investigate if execution was expected)
4. **If planning only when execution expected:**
   - Review corrective actions section
   - Update next meeting topic with stronger tool requirements
   - Check agent prompt configurations

### Weekly Review:

1. **Count execution vs. planning meetings:**
   ```bash
   grep "Tools Executed:" unified-team/reports/transcript_*.md | \
     awk -F': ' '{if ($2 > 0) exec++; else plan++} END {print "Execution:", exec, "Planning:", plan}'
   ```

2. **Expected ratio for Week 1:**
   - Days 1-2: 50% execution (tool development + setup)
   - Days 3-5: 80% execution (cross-architecture testing)
   - Days 6-7: 30% execution (analysis + Go/No-Go decision)

3. **Red flag:** <30% execution meetings during Days 3-5

---

## Key Insight

**The difference between PLANNING and EXECUTION:**

**Planning Meeting:**
```
Agent: "I propose we generalize attention_analysis.py using an adapter pattern..."
```

**Execution Meeting:**
```
Agent: "I propose we generalize attention_analysis.py using an adapter pattern. Let me test this now."
**Tool: read_file('research/attention_analysis.py')**
Result: [reads file]
"Based on the code I just reviewed, here's my implementation..."
**Tool: run_script('test_adapter.py')**
Result: [tests implementation]
"The test results show..."
```

**User's requirement:** "they need to set up environment and carry out experiment and do research and save the results and summaries"

This means:
- ✅ **DO:** Use tools to implement, test, save results
- ❌ **DON'T:** Only discuss what should be implemented

---

**Checklist Status:** Ready to use for post-meeting verification
**Next Action:** Run verification after first execution meeting completes
