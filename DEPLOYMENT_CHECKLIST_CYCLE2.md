# Cycle 2 Deployment Checklist

**Date:** 2025-10-15
**Target:** Deploy task-type aware enforcement + run experimental tasks

---

## 📋 Pre-Flight Checklist

### **1. Backup Current State**
- [ ] Colab notebook still open? (If yes, File → Save a copy to Google Drive)
- [ ] Desktop backup exists? (`/Users/guyan/Desktop/"cvpr_autonomous_execution_cycle_ipynb"的副本.ipynb`)
- [ ] Current execution summary saved? (`execution_summary_20251015_073009.md`)

### **2. Review Documentation**
- [ ] Read: `ENFORCEMENT_UPDATE_QUICK_SUMMARY.md` (this is the quick version)
- [ ] Reference: `TASK_TYPE_AWARE_ENFORCEMENT_COMPLETE.md` (detailed version)
- [ ] Context: `MLFLOW_REQUIREMENT_ANALYSIS.md` (why this update was needed)

---

## 🔧 Deployment Steps

### **Step 1: Update Notebook Cell 16** ⚠️ CRITICAL

**Action:** Update Phase 5.5 Evidence Verification to be task-type aware

**Instructions:**
1. Open Colab: `cvpr_autonomous_execution_cycle.ipynb`
2. Find Cell 16 (should have header: `# PHASE 5.5: EVIDENCE VERIFICATION`)
3. **Select ALL content in Cell 16**
4. **Delete** (Ctrl+A, then Backspace)
5. Open local file: `CELL_16_TASK_TYPE_AWARE_UPDATE.md`
6. Scroll to section: **"Updated Cell 16 Code"**
7. Copy entire code block (from `# ============================================================` to `print("="*80)`)
8. Paste into Cell 16
9. **Run the cell** (Shift+Enter) → Should see function definitions, no errors

**Verification:**
```python
# In a new cell, test:
print(should_require_mlflow("Execute CLIP diagnostic"))  # True
print(should_require_mlflow("Design statistical framework"))  # False
```

- [ ] Cell 16 updated successfully
- [ ] Test verification passed

---

### **Step 2: Upload Cycle 2 Pending Actions**

**Action:** Replace Cycle 1 task list with Cycle 2 experimental tasks

**Option A: Command Line**
```bash
cp /Users/guyan/computer_vision/computer-vision/pending_actions_cycle2_experimental.json \
   "/Users/guyan/Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/computer-vision-clean/multi-agent/reports/handoff/pending_actions.json"
```

**Option B: Manual Copy**
1. Open: `pending_actions_cycle2_experimental.json`
2. Copy all content
3. Navigate to Google Drive folder: `multi-agent/reports/handoff/`
4. Open: `pending_actions.json`
5. Select all, paste, save

**Verification:**
```python
# In Colab Cell 1, after loading:
import json
with open(f"{MULTI_AGENT_ROOT}/reports/handoff/pending_actions.json") as f:
    actions = json.load(f)
    print(f"Meeting ID: {actions['meeting_id']}")  # Should show: cvpr_planning_cycle2_20251015_080000
    print(f"Tasks: {len(actions['decisions'])}")   # Should show: 3
```

- [ ] `pending_actions.json` uploaded
- [ ] Verification shows 3 tasks for Cycle 2

---

### **Step 3: Verify Agent Prompts (Word Limits Removed)**

**Check:** Ensure all word limits removed from agent prompts (completed earlier)

**Files to verify:**
- `multi-agent/agents/prompts/executive_team/01_quality_safety_officer.md`
- `multi-agent/agents/prompts/executive_team/02_ops_commander.md`
- `multi-agent/agents/prompts/executive_team/03_infrastructure_performance_monitor.md`

**Quick check:**
```bash
grep "≤.*words" multi-agent/agents/prompts/executive_team/*.md
# Should return: (no matches)
```

- [ ] Word limits confirmed removed

---

## 🚀 Execution Steps

### **Step 4: Run Cycle 2 Execution**

**Action:** Execute notebook cells to run experimental tasks

**Cell Execution Order:**
1. **Cell 1-5:** Setup, imports, configuration
   - [ ] Executed without errors

2. **Cell 6:** Load `pending_actions.json`
   - [ ] Shows 3 tasks loaded
   - [ ] Meeting ID: `cvpr_planning_cycle2_20251015_080000`

3. **Cell 7-10:** Initialize tracking, display tasks
   - [ ] Task 1: CLIP Integration (HIGH priority)
   - [ ] Task 4: CLIP Diagnostic (HIGH priority)
   - [ ] Task 5: ALIGN/CoCa (HIGH priority)

4. **Cell 11:** Task Execution Loop (3-Agent Approval)
   - [ ] Task 1 executing...
   - [ ] Task 4 executing...
   - [ ] Task 5 executing...
   - **Monitor:** Each task should show Ops Commander, Quality & Safety, Infrastructure responses
   - **Watch for:** MLflow run_id mentions in agent responses

5. **Cell 12-15:** Progress reports, summary generation
   - [ ] Execution summary generated
   - [ ] Timestamp used: `YYYYMMDD_HHMMSS`

6. **Cell 16:** Phase 5.5 Evidence Verification (UPDATED!)
   - **Expected output:**
     ```
     🔍 EVIDENCE VERIFICATION - CHECKING TASK COMPLETION CLAIMS
     📋 VERIFYING ALL COMPLETED TASKS (TASK-TYPE AWARE)

     🔍 Verifying Task 1: Re-execute Task 1: CLIP Integration...
        📊 Task Type: EXPERIMENTAL (requires MLflow tracking)
        ✅ MLflow Run: abc123... (or ❌ if missing)
        ...

     🔍 Verifying Task 4: Execute Task 4: CLIP Diagnostic...
        📊 Task Type: EXPERIMENTAL (requires MLflow tracking)
        ...

     🔍 Verifying Task 5: Complete Task 5: ALIGN/CoCa...
        📊 Task Type: EXPERIMENTAL (requires MLflow tracking)
        ...
     ```

   - [ ] All tasks show "📊 EXPERIMENTAL" type
   - [ ] Phase 5.5 verification completed

7. **Cell 17-21:** Final reports, handoff generation
   - [ ] Reports saved to Google Drive
   - [ ] Execution complete

---

## 🔍 Verification Checkpoints

### **During Execution:**

**Task 1 Checkpoint:**
- [ ] Ops Commander response mentions "MLflow"
- [ ] Response includes: `run_id: [alphanumeric]`
- [ ] Files created in `runs/clip_integration/`
- [ ] Agent claims "COMPLETED"

**Task 4 Checkpoint:**
- [ ] COCO dataset loaded (n≥100 samples)
- [ ] MCS scores computed
- [ ] Statistical tests run (p-value reported)
- [ ] MLflow run_id in response
- [ ] Files created in `runs/clip_diagnostic/`

**Task 5 Checkpoint:**
- [ ] ALIGN access attempted (or CoCa fallback used)
- [ ] Access status documented
- [ ] If successful: MCS computed, MLflow logged
- [ ] If blocked: Fallback plan executed
- [ ] Files created in `runs/align_diagnostic/`

---

### **After Cell 16 (Phase 5.5):**

**Success Case:**
```
✅ EVIDENCE VERIFICATION PASSED
   All 3 completed tasks have verified evidence
```
- [ ] ✅ All tasks passed verification
- [ ] Ready for GO/NO-GO decision

**Failure Case:**
```
❌ EVIDENCE VERIFICATION FAILED
   X tasks lack required evidence
   Failed tasks: [1, 4, 5]
```
- [ ] ❌ Review `multi-agent/error.md` for details
- [ ] Check which tasks missing MLflow run_id
- [ ] Re-run failed tasks if needed

---

## 📊 Expected Outcomes

### **Cycle 2 Success Criteria:**

**Minimum (GO/NO-GO Ready):**
- [ ] Task 1: CLIP integration complete with MLflow
- [ ] Task 4: CLIP diagnostic complete with p-value
- [ ] Task 5: 2nd external model diagnostic complete

**Ideal:**
- [ ] All 3 tasks pass Phase 5.5 verification
- [ ] CLIP shows attention collapse (p<0.05)
- [ ] 2nd model shows attention collapse (p<0.05)
- [ ] Statistical evidence strong (Cohen's d > 0.5)

**GO Decision Evidence:**
- ≥2 external models show attention collapse
- p<0.05 for MCS vs balanced baseline
- Cohen's d > 0.5 (medium effect)

**Next:** Week 1 GO/NO-GO meeting on October 20, 2025

---

## 🚨 Troubleshooting

### **Issue: Cell 16 shows error**
**Symptom:** `NameError: name 'should_require_mlflow' is not defined`
**Fix:** Function not defined before loop. Re-check Cell 16 code copied correctly.

### **Issue: Task shows ❌ but should pass**
**Symptom:** Documentation task failed for "No MLflow tracking"
**Fix:** Cell 16 not updated correctly. Verify task-type detection working.

### **Issue: Task shows ✅ but should fail**
**Symptom:** Experimental task passed without MLflow
**Fix:** Task action doesn't match experimental keywords. Update keyword list.

### **Issue: MLflow run_id not found**
**Symptom:** "❌ No MLflow run_id found in responses"
**Fix:** Agent didn't log to MLflow. Check `mlruns/` directory, verify MLflow tracking enabled.

---

## 📁 Files Reference

**Local Files Created:**
- `CELL_16_TASK_TYPE_AWARE_UPDATE.md` - Cell 16 updated code
- `pending_actions_cycle2_experimental.json` - Cycle 2 task list
- `TASK_TYPE_AWARE_ENFORCEMENT_COMPLETE.md` - Full documentation
- `ENFORCEMENT_UPDATE_QUICK_SUMMARY.md` - Quick summary
- `MLFLOW_REQUIREMENT_ANALYSIS.md` - Analysis document
- `DEPLOYMENT_CHECKLIST_CYCLE2.md` - This file

**Google Drive Files (after execution):**
- `multi-agent/reports/execution/summaries/execution_summary_[timestamp].md`
- `multi-agent/reports/handoff/execution_progress_update_[timestamp].md`
- `multi-agent/error.md` (if Phase 5.5 fails)
- `runs/clip_integration/*` (Task 1 outputs)
- `runs/clip_diagnostic/*` (Task 4 outputs)
- `runs/align_diagnostic/*` (Task 5 outputs)

---

## ✅ Final Checklist

**Before Running:**
- [ ] Notebook Cell 16 updated
- [ ] Cycle 2 pending_actions.json uploaded
- [ ] Word limits removed from agent prompts
- [ ] Backup created (Colab + Desktop)

**During Execution:**
- [ ] Monitor agent responses for MLflow mentions
- [ ] Watch for run_id in Ops Commander reports
- [ ] Check file creation in `runs/` directories

**After Execution:**
- [ ] Phase 5.5 verification passed
- [ ] All 3 tasks marked "completed"
- [ ] Execution summary saved to Google Drive
- [ ] Ready for Week 1 GO/NO-GO meeting

---

## 🎯 Success Metrics

**Cycle 2 Complete When:**
✅ 3/3 experimental tasks pass Phase 5.5
✅ CLIP diagnostic has statistical results (p-value, CI95)
✅ 2nd external model diagnostic complete
✅ Evidence ready for GO/NO-GO decision (October 20, 2025)

---

**System Status:** ✅ READY TO DEPLOY
**Next Action:** Update Cell 16 in Colab → Upload pending_actions.json → Run execution

**Good luck with Cycle 2! 🚀**
