# Quick Verification Guide - 60 Seconds

**After each meeting, run these commands to verify execution:**

---

## ✅ Quick Check (Copy-Paste Ready)

### On Colab:
```bash
# Navigate to project
cd /content/drive/MyDrive/cv_multimodal/project/computer-vision-clean

# Run automated check
python unified-team/POST_MEETING_EXECUTION_CHECK.py

# Expected: ✅ EXECUTION MEETING CONFIRMED
```

### On Local Machine:
```bash
# Navigate to Google Drive folder
cd "/Users/guyan/Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/computer-vision-clean"

# Run automated check
python unified-team/POST_MEETING_EXECUTION_CHECK.py session_20251014_162454

# Expected: ✅ EXECUTION MEETING CONFIRMED
```

---

## 📊 What You're Looking For

### ✅ GOOD (Execution Meeting):
```
✅ EXECUTION MEETING CONFIRMED

Tools executed: 5
Executors active: Research Director, Tech Analyst
New files created: 3
SLO status changed: Yes

New artifacts:
  - unified-team/experiments/week1_day1_setup.log
  - research/diagnostic_tools/universal_analyzer_plan.md
  - unified-team/scripts/test_clip_load.py
```

### ⚠️ BAD (Planning Only):
```
⚠️ PLANNING MEETING DETECTED

No tool execution detected
No new files created
No executor activity logged

⚠️ WARNING: If execution was expected, check:
  1. Agent prompts emphasize EXECUTION
  2. Meeting topic explicitly requires tools
  3. Executors have tool access permissions
```

---

## 🔍 Manual Verification (If Script Fails)

### Check 1: Tool Count (5 seconds)
```bash
grep "Tools Executed:" unified-team/reports/transcript_*.md | tail -1
```
**Expected:** `Tools Executed: 5` (or any number > 0)
**Bad:** `Tools Executed: 0`

### Check 2: Executor Activity (10 seconds)
```bash
# Check Research Director
grep "research_director" unified-team/reports/transcript_*.md | grep -E "(read_file|run_script|list_files)" | tail -3

# Check Tech Analyst
grep "tech_analyst" unified-team/reports/transcript_*.md | grep -E "(read_file|run_script|list_files)" | tail -3
```
**Expected:** Multiple lines showing tool calls
**Bad:** No output

### Check 3: New Files (5 seconds)
```bash
find unified-team/experiments research/ -type f -mmin -120 -ls | head -10
```
**Expected:** List of recently created files
**Bad:** Empty output (or only transcript/summary files)

---

## 🚨 If Planning Only (Action Required)

### Step 1: Confirm Issue
```bash
# Double-check the transcript
tail -50 unified-team/reports/transcript_*.md
```
Look for "Tools Executed: 0" - if present, execution was missing.

### Step 2: Review Meeting Topic
```bash
# Check what agents were asked to do
head -150 unified-team/WEEK_1_MEETING_TOPIC.md
```
Verify "🔴 MANDATORY TOOL EXECUTION" section is present.

### Step 3: Prepare Stronger Next Meeting
Create file: `unified-team/URGENT_EXECUTION_MEETING.md`
```markdown
# URGENT: Tool Execution Required

Research Director and Tech Analyst, the previous meeting had ZERO tool usage.

YOU MUST USE TOOLS IN THIS MEETING. THIS IS NOT OPTIONAL.

## Research Director - DO NOW:
1. Run: list_files('research/')
2. Run: read_file('research/[file from step 1]')
3. Report what you found

## Tech Analyst - DO NOW:
1. Run: run_script('nvidia-smi')
2. Run: run_script('pip install transformers')
3. Report installation status

FAILURE TO USE TOOLS = MEETING INCOMPLETE
```

### Step 4: Check Agent Tool Access
```python
# On Colab - test if tools work
from unified_coordinator import UnifiedCoordinator
import yaml

with open('unified-team/configs/team.yaml') as f:
    config = yaml.safe_load(f)

coordinator = UnifiedCoordinator(config, Path('.'))

# Test Research Director tool access
print("Testing Research Director tools...")
result = coordinator.agents['research_director'].available_tools
print(f"Available tools: {result}")

# Test Tech Analyst tool access
print("\nTesting Tech Analyst tools...")
result = coordinator.agents['tech_analyst'].available_tools
print(f"Available tools: {result}")
```

---

## 📈 Success Metrics

### After Week 1 (7 meetings):
```bash
# Count execution vs planning meetings
grep "Tools Executed:" unified-team/reports/transcript_*.md | \
  awk -F': ' '{if ($2 > 0) exec++; else plan++} END {print "Execution:", exec, "Planning:", plan}'
```

**Expected for Week 1:**
- Execution meetings: 4-5 (Days 1-5)
- Planning meetings: 2-3 (Day 1, Day 7)

**Red flag:**
- Execution meetings: 0-1 (system not working)
- Planning meetings: 6-7 (agents not using tools)

---

## 💡 Tips

### Tip 1: Check Immediately After Meeting
Don't wait hours - verify within 5-10 minutes of meeting completion.

### Tip 2: Use Monitoring Dashboard
Colab notebook cell #16 shows live tool execution count.

### Tip 3: Set Expectations
First meeting (strategic planning): Tools = 0-2 (acceptable)
Second meeting onwards: Tools = 4+ (required)

### Tip 4: Watch for Patterns
If 2+ consecutive meetings show "Tools Executed: 0", system needs intervention.

---

## 📞 Escalation Path

### Level 1: Normal (Tools > 0)
✅ System working correctly
✅ Continue monitoring
✅ No action needed

### Level 2: Warning (Tools = 0, first time)
⚠️ Review meeting topic
⚠️ Check agent prompts
⚠️ Prepare stronger next topic

### Level 3: Critical (Tools = 0, 2+ times)
🚨 Agent prompts need updating
🚨 Meeting topics need restructuring
🚨 Consider manual tool execution test
🚨 Review coordinator configuration

---

## 🎯 One-Line Verification

**Fastest check (copy-paste after each meeting):**
```bash
grep "Tools Executed:" unified-team/reports/transcript_*.md | tail -1 && echo "✅ Check complete"
```

**Expected output:**
```
Tools Executed: 5
✅ Check complete
```

---

**Keep this guide handy for quick post-meeting verification!**
