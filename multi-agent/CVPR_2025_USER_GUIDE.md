# CVPR 2025 Autonomous Research System - User Guide

**Version:** 1.0
**Date:** October 14, 2025
**Mission:** Submit CVPR 2025 paper on "Diagnosing and Preventing Attention Collapse in Multimodal Fusion"

---

## 📖 Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [How to Run a Cycle](#how-to-run-a-cycle)
4. [Understanding the Output](#understanding-the-output)
5. [Manual Checkpoint Review](#manual-checkpoint-review)
6. [Troubleshooting](#troubleshooting)
7. [File Locations](#file-locations)
8. [Week 1 Specific Guide](#week-1-specific-guide)
9. [FAQ](#faq)

---

## 🚀 Quick Start

**Goal:** Get your first autonomous research cycle running in 15 minutes.

### **Prerequisites**

- ✅ Google account with Colab access
- ✅ Google Drive with project files
- ✅ API keys in `.env` file (Anthropic, OpenAI, Google)
- ✅ A100 GPU quota (or T4/V100 as fallback)

### **5-Minute Quickstart**

**Step 1:** Open Google Colab
```
https://colab.research.google.com/
```

**Step 2:** Upload notebook from Google Drive
```
MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/cvpr_autonomous_execution_cycle.ipynb
```

**Step 3:** Select GPU
```
Runtime → Change runtime type → GPU → A100 (or T4)
```

**Step 4:** Run all cells
```
Runtime → Run all (or Ctrl+F9)
```

**Step 5:** Wait for execution to complete (~2-4 hours for HIGH priority tasks)

**Step 6:** Review results
```
Read: reports/handoff/execution_progress_update.md
```

**Done!** You've completed your first autonomous research cycle.

---

## 🎯 System Overview

### **What This System Does**

The autonomous research system manages **Planning-Executive cycles** for your CVPR 2025 paper research:

```
┌─────────────────────────────────────────────────────────┐
│                   AUTONOMOUS CYCLE                      │
└─────────────────────────────────────────────────────────┘

Planning Team (4 agents)
  ├─ Strategic Leader (Opus 4)
  ├─ Empirical Validation Lead (Sonnet 4)
  ├─ Critical Evaluator (GPT-4)
  └─ Gemini Research Advisor (Gemini Flash)
       │
       ↓ Generates
  pending_actions.json (7 tasks: 4 HIGH, 3 MEDIUM)
       │
       ↓ Executes in Colab
Executive Team (3 agents)
  ├─ Ops Commander (Sonnet 4)
  ├─ Quality & Safety Officer (Sonnet 4)
  └─ Infrastructure Monitor (Sonnet 4)
       │
       ↓ Produces
  execution_progress_update.md (Results & progress)
       │
       ↓ You Review
  Manual Checkpoint (Approve/Pause/Reject)
       │
       ↓ Next Cycle
  Planning Team reviews results → new pending_actions.json
```

### **Key Features**

✅ **Priority-Based Execution:** HIGH → MEDIUM → LOW
✅ **Real Deployment Tools:** Python, MLflow, GPU, actual experiments
✅ **Progress Tracking:** Every task logged, agent responses saved
✅ **Manual Checkpoints:** You review and approve between cycles
✅ **Crash Recovery:** Auto-sync to Google Drive
✅ **Evidence-Based:** MLflow run_ids, file paths for reproducibility

---

## 🔄 How to Run a Cycle

### **Cycle Structure**

Each cycle has 3 phases:

1. **Planning** → Generate tasks (`pending_actions.json`)
2. **Execution** → Run tasks in Colab (agents work on research)
3. **Review** → Manual checkpoint (you approve/reject)

### **Running Execution Phase (Most Common)**

This is what you'll do most often - running the Executive Team to execute tasks.

#### **Step-by-Step: Run Executive Team in Colab**

**1. Open Colab Notebook**

From Google Drive:
```
MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/cvpr_autonomous_execution_cycle.ipynb
```

Or upload from local:
```
/Users/guyan/computer_vision/computer-vision/research/colab/cvpr_autonomous_execution_cycle.ipynb
```

**2. Check Runtime**

```
Runtime → Change runtime type
  ├─ Runtime type: Python 3
  ├─ Hardware accelerator: GPU
  └─ GPU type: A100 (recommended) or T4
```

**3. Run Setup Cells (Cells 1-3)**

```python
# Cell 1: Check GPU
!nvidia-smi
# Expected: NVIDIA A100-SXM4-80GB

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Expected: "Mounted at /content/drive"

# Cell 3: Load API Keys
# Expected: "✅ ALL API KEYS LOADED SUCCESSFULLY"
```

**4. Run All Remaining Cells**

```
Runtime → Run after (from cell 4)
```

Or run all at once:
```
Runtime → Run all
```

**5. Monitor Execution**

Watch the output for:
```
📥 Reading pending_actions.json...
   ⭐ HIGH: 4 tasks
   🟠 MEDIUM: 3 tasks

🤖 Initializing Executive Team...
   ✅ Ops Commander (claude-sonnet-4)
   ✅ Quality & Safety Officer (claude-sonnet-4)
   ✅ Infrastructure Monitor (claude-sonnet-4)

🚀 Starting Task task_1: Adapt attention_analysis.py for CLIP
   Priority: HIGH
   Started: 2025-10-14T...
   ✅ ops_commander responded (3241 chars)
   ✅ quality_safety responded (1832 chars)
   📄 Output: Updated research/attention_analysis.py
   ✅ Task completed in 45.2s
   Status: completed
```

**6. Wait for Completion**

Expected duration:
- **HIGH tasks (4):** 2-4 hours total
- **MEDIUM tasks (3):** 1-2 hours total
- **Full cycle:** 3-6 hours

You can:
- Leave tab open and check periodically
- Close tab (execution continues in background)
- Check status by re-opening notebook

**7. Check Results**

When execution completes, you'll see:
```
================================================================================
✅ EXECUTIVE TEAM EXECUTION COMPLETE
================================================================================
📊 Total tasks: 7
✅ Completed: 6
❌ Failed: 1
⏱️  Duration: 245.3s

📄 Progress report: reports/handoff/execution_progress_update.md
```

### **Running Planning Phase (Less Common)**

Only run when you need to generate new tasks (after reviewing execution results).

```bash
# On your local machine
cd /Users/guyan/computer_vision/computer-vision/multi-agent
python scripts/run_planning_meeting.py
```

This will:
1. Planning Team reviews `execution_progress_update.md`
2. Assesses progress toward goals
3. Generates new `pending_actions.json`

**When to run:**
- After completing a cycle and approving results
- When you need to pivot strategy
- After Week 1 GO/NO-GO decision

---

## 📊 Understanding the Output

### **Key Output Files**

After execution, you'll have these files:

#### **1. execution_progress_update.md** (Most Important)

**Location:** `reports/handoff/execution_progress_update.md`

**What it contains:**
```markdown
# Executive Team Progress Update

**Date:** 2025-10-14 18:30:00
**Mission:** CVPR 2025 Week 1 - Cross-architecture validation

## Execution Summary
Total Tasks: 7
Completed: 6 ✅
Failed: 1 ❌
Execution Time: 245.3s

## Task Results

### Task 1: Adapt attention_analysis.py for CLIP
**Status:** ✅ COMPLETED
**Duration:** 45.2s
**Outputs:**
- Updated: research/attention_analysis.py
- Created: research/clip_adapter.py
- MLflow run_id: clip_adapter_001

**Agent Responses:**
- ops_commander: "Successfully adapted attention analysis framework..."
- quality_safety: "Code review complete, all tests pass..."
```

**How to read it:**
- ✅ Check "Execution Summary" - how many tasks completed?
- ✅ Read each task's status and outputs
- ✅ Look for MLflow run_ids (evidence of experiments)
- ✅ Note any failures or blockers

#### **2. execution_results_*.json** (Detailed Data)

**Location:** `reports/execution/results/execution_results_20251014_183000.json`

**What it contains:**
```json
{
  "total_tasks": 7,
  "completed": 6,
  "failed": 1,
  "total_duration_seconds": 245.3,
  "task_results": [
    {
      "task_id": "task_1",
      "action": "Adapt attention_analysis.py for CLIP",
      "status": "completed",
      "agent_responses": {...},
      "outputs": [...],
      "errors": []
    }
  ]
}
```

**How to use it:**
- Programmatic access to results
- Copy to analysis scripts
- Track metrics over time

#### **3. execution_dashboard_*.png** (Visual Summary)

**Location:** `reports/execution/results/execution_dashboard_20251014_183000.png`

**What it shows:**
- Pie chart: Completed vs Failed tasks
- Bar chart: Tasks by priority (HIGH/MEDIUM/LOW)
- Bar chart: Execution time per task

#### **4. pending_actions.json** (Next Cycle Input)

**Location:** `reports/handoff/pending_actions.json`

**Generated by:** Planning Team
**Used by:** Executive Team (next cycle)

**Structure:**
```json
{
  "meeting_id": "cvpr_planning_week1_cycle2",
  "decisions": [
    {
      "priority": "HIGH",
      "action": "Run ALIGN diagnostic experiment",
      "owner": "ops_commander",
      "deadline": "2025-10-16T18:00:00Z",
      "acceptance_criteria": [...],
      "evidence_paths": [...]
    }
  ]
}
```

---

## ⏸️ Manual Checkpoint Review

**This is the most important part!** After each cycle, you review results before continuing.

### **What to Check**

#### **1. Task Completion Rate**

```markdown
## Execution Summary
Total Tasks: 7
Completed: 6 ✅  ← Is this acceptable?
Failed: 1 ❌     ← Why did this fail?
```

**Decision criteria:**
- ✅ **All HIGH tasks completed:** Approve
- ⚠️ **Some HIGH tasks failed:** Review why, maybe re-run
- ❌ **Most HIGH tasks failed:** Reject, fix issues

#### **2. Output Quality**

For each completed task, check:

**✅ Files created?**
```markdown
**Outputs:**
- Updated: research/attention_analysis.py ← File exists?
- Created: research/clip_adapter.py       ← File exists?
```

**✅ Evidence provided?**
```markdown
- MLflow run_id: clip_adapter_001  ← Can verify in MLflow?
- Evidence: research/attention_analysis.py#L245  ← Line number?
```

**✅ Acceptance criteria met?**
```markdown
**Acceptance Criteria:**
- [x] CLIP model loads successfully
- [x] Attention weights extracted
- [x] Modality Contribution Score computed
- [ ] Visualization pipeline generates plots  ← Not done!
```

#### **3. Progress Toward Goals**

**Week 1 Goal:** Determine if attention collapse is widespread

Check:
```markdown
## Week 1 Progress Toward GO/NO-GO Decision

- [x] Diagnostic tools work on CLIP
- [ ] Diagnostic tools work on ALIGN
- [x] Statistical framework designed
- [x] CLIP diagnostic completed

**CLIP Results:**
- Modality Contribution Score: Text 89.3%, Vision 10.7%
- Attention Imbalance: 79.3% (approaching 80% threshold)
- Statistical Significance: p=0.002 ✅

**Recommendation:** Continue - need ALIGN for 2nd datapoint
```

**Ask yourself:**
- Are we moving toward the Week 1 GO/NO-GO decision?
- Do results support or contradict our hypothesis?
- What's needed for next cycle?

#### **4. Blockers & Risks**

Look for:
```markdown
**Blockers:**
- ❌ ALIGN model not accessible (licensing issues)
- ⚠️ GPU quota running low (3 hours remaining)
- ⚠️ CLIP results just below 80% threshold (79.3%)
```

**Decision:** Can we address these before next cycle?

### **Making the Decision**

After reviewing, you have 3 choices:

#### **✅ APPROVE - Continue to Next Cycle**

**When:**
- Most/all HIGH tasks completed
- Outputs meet acceptance criteria
- Progress toward goals is good
- No major blockers

**Action:**
```bash
# Run Planning Team to generate next cycle
cd multi-agent
python scripts/run_planning_meeting.py
```

#### **⏸️ PAUSE - Fix Issues First**

**When:**
- Some HIGH tasks failed
- Outputs incomplete or low quality
- Major blocker identified
- Need to debug or redesign

**Action:**
- Investigate failures
- Fix code/environment issues
- Re-run Executive Team with same `pending_actions.json`

#### **❌ REJECT - Re-Execute Cycle**

**When:**
- Most HIGH tasks failed
- Critical outputs missing
- Wrong direction (strategy pivot needed)

**Action:**
- Restart Colab runtime
- Fix root cause (API keys, environment, etc.)
- Re-run entire cycle

---

## 🔧 Troubleshooting

### **Common Issues & Solutions**

#### **Issue 1: API Authentication Failed**

**Error:**
```
❌ Error: "Could not resolve authentication method..."
```

**Cause:** API keys not loaded

**Fix:**
See `API_KEY_FIX.md` or `ERROR_ANALYSIS_AND_FIX.md`

**Quick fix:**
1. Check `.env` file exists: `/content/drive/MyDrive/cv_multimodal/project/.env`
2. Verify keys are not in quotes: `ANTHROPIC_API_KEY=sk-ant-xxx` (not `"sk-ant-xxx"`)
3. Re-run API key loading cell
4. Verify: `✅ ALL API KEYS LOADED SUCCESSFULLY`

#### **Issue 2: GPU Not Available**

**Error:**
```
RuntimeError: CUDA out of memory
```

**Fix:**
```
Runtime → Change runtime type → GPU → A100
```

If A100 not available:
- Use T4 (slower but works)
- Reduce batch size in experiments
- Run tasks sequentially instead of parallel

#### **Issue 3: Colab Disconnected**

**What happens:**
- Colab tab shows "Reconnecting..."
- Execution may stop

**Fix:**
- All progress saved to Google Drive (auto-sync every 5s)
- Re-open notebook
- Check `execution_progress_update.md` for last completed task
- Resume from checkpoint (if implemented) or re-run failed tasks

#### **Issue 4: Task Failed with Error**

**Check:**
```markdown
### Task 3: Design statistical framework
**Status:** ❌ FAILED
**Errors:**
- "ModuleNotFoundError: No module named 'scipy'"
```

**Fix:**
1. Install missing dependency: `!pip install scipy`
2. Re-run just that task (modify notebook to skip completed tasks)
3. Or re-run entire cycle

#### **Issue 5: No Progress Report Generated**

**Check:**
1. Did all cells run? (scroll through notebook)
2. Is Google Drive mounted? (check mount cell)
3. File path correct? (check `reports/handoff/`)

**Fix:**
- Check intermediate results: `reports/execution/results/execution_results_*.json`
- Re-run report generation cells manually

---

## 📁 File Locations

### **Google Drive Structure**

```
MyDrive/cv_multimodal/project/computer-vision-clean/
├── multi-agent/
│   ├── reports/
│   │   ├── handoff/
│   │   │   ├── pending_actions.json ← Planning → Executive
│   │   │   ├── execution_progress_update.md ← Executive → You
│   │   │   └── CVPR_WEEK1_HANDOFF_TO_EXECUTIVE_TEAM.md (guide)
│   │   ├── planning/
│   │   │   ├── transcripts/
│   │   │   ├── summaries/
│   │   │   └── actions/
│   │   └── execution/
│   │       ├── summaries/
│   │       │   └── execution_summary_20251014_183000.md
│   │       └── results/
│   │           ├── execution_results_20251014_183000.json
│   │           └── execution_dashboard_20251014_183000.png
│   ├── state/
│   │   └── cycle_state.json (cycle tracking)
│   ├── autonomous_cycle_coordinator.py
│   ├── CVPR_2025_USER_GUIDE.md ← You are here!
│   ├── AUTONOMOUS_CYCLE_SYSTEM_GUIDE.md
│   ├── API_KEY_FIX.md
│   └── ERROR_ANALYSIS_AND_FIX.md
├── research/
│   ├── colab/
│   │   └── cvpr_autonomous_execution_cycle.ipynb ← Run this!
│   ├── attention_analysis.py (diagnostic tool)
│   ├── 01_v1_production_line/
│   ├── 02_v2_research_line/
│   └── 03_cotrr_lightweight_line/
└── CVPR_2025_MISSION_BRIEFING_FOR_5_AGENT_SYSTEM.md
```

### **Local Machine Structure**

```
/Users/guyan/computer_vision/computer-vision/
├── multi-agent/
│   ├── scripts/
│   │   ├── run_planning_meeting.py ← Run Planning Team
│   │   ├── run_execution_meeting.py
│   │   └── run_cvpr_planning_meeting.py
│   ├── agents/
│   │   └── prompts/
│   │       ├── planning_team/ (4 agents)
│   │       └── executive_team/ (3 agents)
│   └── autonomous_cycle_coordinator.py
└── research/
    └── colab/
        └── cvpr_autonomous_execution_cycle.ipynb
```

### **Quick Access Shortcuts**

**On Google Drive (web):**
- Handoff files: `cv_multimodal/project/computer-vision-clean/multi-agent/reports/handoff/`
- Results: `cv_multimodal/project/computer-vision-clean/multi-agent/reports/execution/results/`
- Notebook: `cv_multimodal/project/computer-vision-clean/research/colab/`

**On Local Machine:**
- Planning Team: `cd /Users/guyan/computer_vision/computer-vision/multi-agent && python scripts/run_planning_meeting.py`
- Docs: `cd /Users/guyan/computer_vision/computer-vision/multi-agent && open .`

---

## 📅 Week 1 Specific Guide

### **Week 1 Goal (Oct 14-20)**

**Question:** Is attention collapse (0.15% visual contribution) unique to our V2 model, or widespread across multimodal architectures?

**Method:** Test CLIP, ALIGN, Flamingo with diagnostic tools

**Decision (Oct 20):**
- **GO:** ≥2 models show >80% imbalance → Field-wide paper
- **PIVOT:** Only V2 shows collapse → Focused case study
- **DIAGNOSTIC:** Tools work on all models → Framework contribution

### **Week 1 Tasks (Already Assigned)**

**Cycle 1 (Oct 14-15) - HIGH Priority:**
1. ⭐ Adapt `attention_analysis.py` for CLIP (24h)
2. ⭐ Set up CLIP/OpenCLIP environment (24h)

**Cycle 2 (Oct 16-17) - HIGH Priority:**
3. ⭐ Design statistical validation framework (48h)
4. ⭐ Run first CLIP diagnostic (72h)

**Cycle 3 (Oct 18-19) - MEDIUM Priority:**
5. 🟠 Literature review
6. 🟠 ALIGN environment setup
7. 🟠 Draft paper outline

### **Week 1 Success Criteria**

By Oct 20, you should have:
- [ ] ≥3 models tested (CLIP, ALIGN, +1)
- [ ] Statistical evidence collected (p<0.05)
- [ ] CLIP diagnostic complete with MLflow run_id
- [ ] ALIGN diagnostic attempted
- [ ] Clear GO/PIVOT/DIAGNOSTIC recommendation

### **Week 1 Workflow**

**Monday-Tuesday (Oct 14-15):**
- Run Cycle 1 in Colab
- Review: Are tools adapted? Is CLIP environment ready?
- Approve → Planning Team generates Cycle 2 tasks

**Wednesday-Thursday (Oct 16-17):**
- Run Cycle 2 in Colab
- Review: CLIP results? Statistical significance?
- Check: Is attention imbalance >80%? Is p<0.05?
- Approve → Planning Team generates Cycle 3 tasks

**Friday-Saturday (Oct 18-19):**
- Run Cycle 3 in Colab
- Review: ALIGN results? Paper outline?
- Compile Week 1 findings

**Sunday (Oct 20):**
- Planning Team meeting: Week 1 GO/NO-GO Decision
- Review all evidence
- Decide paper scope
- Generate Week 2 plan

---

## ❓ FAQ

### **General Questions**

**Q: How long does one cycle take?**
A: 3-6 hours for full cycle (4 HIGH + 3 MEDIUM tasks). HIGH tasks alone take 2-4 hours.

**Q: Can I run multiple cycles per day?**
A: Yes! With manual checkpoints, you can run 2-3 cycles per day if you review results quickly.

**Q: Do I need to keep Colab tab open?**
A: No, execution continues in background. But keep runtime active (Colab disconnects after ~90 min idle).

**Q: What happens if Colab crashes?**
A: All progress auto-synced to Google Drive. Check `execution_progress_update.md` for last completed task.

**Q: How much does each cycle cost?**
A: API costs depend on task complexity. Estimate $1-3 per cycle (Claude + GPT-4 + Gemini calls).

### **Technical Questions**

**Q: Which GPU should I use?**
A: A100 (best), V100 (good), T4 (works but slower). Priority: A100 > V100 > T4.

**Q: Can I modify pending_actions.json?**
A: Yes! Edit priorities, add/remove tasks, change deadlines. Planning Team generates it, but you can customize.

**Q: How do I skip a task?**
A: Edit `pending_actions.json`, remove that task's entry. Or set priority to LOW so it runs last.

**Q: Can I run tasks locally instead of Colab?**
A: Yes, but Colab has free GPU. For local, use `python scripts/run_execution_meeting.py`.

**Q: How do I see agent responses in detail?**
A: Check `execution_progress_update.md` for summaries, or `execution_results_*.json` for full responses.

### **Research Questions**

**Q: What if CLIP doesn't show attention collapse?**
A: Test ALIGN next. If neither shows collapse, PIVOT to focused V2 case study (still publishable).

**Q: What if we can't test ALIGN?**
A: Test Flamingo/BLIP instead. Need ≥3 models total for field-wide claim.

**Q: What counts as "attention collapse"?**
A: >80% attention imbalance (e.g., text 89%, vision 11%) with statistical significance (p<0.05).

**Q: When do we write the paper?**
A: Week 3 (Oct 28-Nov 3). Weeks 1-2 are experiments, Week 3 is writing, Week 4 is finalization.

**Q: What if we miss the Nov 6 abstract deadline?**
A: Focus on Nov 13 full paper deadline. Abstract can be written last-minute (250 words).

### **Troubleshooting Questions**

**Q: API keys not loading?**
A: See `API_KEY_FIX.md`. Check `.env` file path and format.

**Q: All tasks failed immediately?**
A: API authentication issue. See `ERROR_ANALYSIS_AND_FIX.md`.

**Q: One task failed but others passed?**
A: Check that task's error in `execution_progress_update.md`. May be dependency issue.

**Q: How do I re-run just one failed task?**
A: Edit `pending_actions.json` to only include that task, re-run Colab.

**Q: Progress report not generated?**
A: Check if all cells ran. Re-run last few cells manually.

---

## 📚 Additional Resources

### **Documentation Files**

**Quick Reference:**
- `CVPR_2025_USER_GUIDE.md` ← You are here!
- `CVPR_WEEK1_KICKOFF_COMPLETE.md` - Week 1 summary
- `AUTONOMOUS_CYCLE_DEPLOYMENT_READY.md` - Deployment guide

**Detailed Guides:**
- `AUTONOMOUS_CYCLE_SYSTEM_GUIDE.md` - Complete system documentation
- `CVPR_2025_MISSION_BRIEFING_FOR_5_AGENT_SYSTEM.md` - Research mission
- `CVPR_PAPER_PLAN_SUMMARY.md` - Paper strategy

**Technical Docs:**
- `API_KEY_FIX.md` - API authentication fix
- `ERROR_ANALYSIS_AND_FIX.md` - Error recovery
- `CVPR_AUTONOMOUS_SYSTEM_ENHANCEMENTS.md` - Future improvements

### **File Locations**

All documentation synced to:
```
Google Drive: MyDrive/cv_multimodal/project/computer-vision-clean/multi-agent/
Local: /Users/guyan/computer_vision/computer-vision/multi-agent/
```

---

## ✅ Checklist for Success

**Before Each Cycle:**
- [ ] `pending_actions.json` exists
- [ ] Google Drive mounted
- [ ] API keys loaded (all 3 show ✅)
- [ ] GPU selected (A100 preferred)
- [ ] Previous cycle approved

**During Execution:**
- [ ] Tasks execute in priority order (HIGH first)
- [ ] Agent responses logged
- [ ] Outputs being generated
- [ ] No errors in console

**After Execution:**
- [ ] Read `execution_progress_update.md`
- [ ] Check task completion (how many HIGH done?)
- [ ] Verify outputs (files, MLflow run_ids)
- [ ] Review errors/blockers
- [ ] Make decision: Approve/Pause/Reject

**Week 1 Milestones:**
- [ ] Oct 15: Cycle 1 complete (tools adapted, CLIP ready)
- [ ] Oct 17: Cycle 2 complete (CLIP diagnostic done)
- [ ] Oct 19: Cycle 3 complete (ALIGN attempted, paper outline)
- [ ] Oct 20: Week 1 Decision (GO/PIVOT/DIAGNOSTIC)

---

## 🎯 Summary

**What you do:**
1. Run Colab notebook (`cvpr_autonomous_execution_cycle.ipynb`)
2. Wait for execution (3-6 hours)
3. Review results (`execution_progress_update.md`)
4. Approve/Pause/Reject
5. Run Planning Team (if approved)
6. Repeat

**What the system does:**
- Reads tasks from `pending_actions.json`
- Executes in priority order (HIGH → MEDIUM → LOW)
- Uses real deployment tools (Python, MLflow, GPU)
- Tracks progress and generates reports
- Auto-syncs to Google Drive
- Prepares next cycle trigger

**Your role:**
- **Manual checkpoints** - Review and approve between cycles
- **Strategic decisions** - GO/PIVOT/DIAGNOSTIC choices
- **Quality control** - Verify outputs meet standards
- **Blocker resolution** - Fix issues when they arise

**Goal:**
Submit CVPR 2025 paper by Nov 13, with Week 1 GO/NO-GO decision on Oct 20.

---

**Status:** ✅ **USER GUIDE COMPLETE**
**Version:** 1.0
**Last Updated:** October 14, 2025
**Next Update:** After Week 1 (Oct 20)

Good luck with Week 1! 🚀🎓
