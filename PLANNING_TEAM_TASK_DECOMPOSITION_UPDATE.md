# Planning Team Task Decomposition Update - COMPLETE

**Date:** 2025-10-15
**Status:** ✅ All Planning Team prompts updated with task decomposition guidelines
**Part of:** Enforcement System Implementation (High Priority TODOs)

---

## 🎯 What Was Updated

All Planning Team agent prompts now include references to task decomposition guidelines to ensure that complex tasks are broken down into smaller, executable, verifiable subtasks.

---

## 📋 Files Updated

### **1. Strategic Leader (Primary Decision Maker)** ✅

**File:** `agents/prompts/planning_team/01_strategic_leader.md`

**Changes Made:**
- Added **Task Decomposition (MANDATORY)** section to Phase 5 (Execution Plan & Rollout)
- Added 6-point task decomposition checklist:
  - Time estimate <2 hours?
  - Clear acceptance criteria specified?
  - Verifiable outputs defined (MLflow run_id, file paths)?
  - Single responsibility (one task = one deliverable)?
  - Dependencies documented?
  - Owner appropriate (ops_commander / quality_safety / infrastructure)?
- Added reference to `TASK_DECOMPOSITION_GUIDELINES.md`
- Updated EXECUTION PLAN section with explicit JSON template showing:
  - `task_id`, `action`, `priority`, `owner`, `deadline`
  - `estimated_time` (must be <2hr)
  - `acceptance_criteria` (array with MLflow run_id, file paths, statistical requirements)
  - `evidence_paths` (array of expected output files)
  - `dependencies` (array of prerequisite task_ids)
  - `rationale` (why this task?)

**Key Addition (Lines 171-186):**
```markdown
**Task Decomposition (MANDATORY):**
- **BEFORE generating `pending_actions.json`, apply task decomposition principles:**
  - Break complex tasks (>2 hours) into smaller subtasks
  - Each task must have clear acceptance criteria
  - Each task must specify evidence paths (MLflow run_id OR file path)
  - Set realistic time estimates (15-30min, 30-60min, 60-120min)
  - Define dependencies explicitly
  - **See:** `TASK_DECOMPOSITION_GUIDELINES.md` for templates and examples

**Task Decomposition Checklist:**
- [ ] Time estimate <2 hours?
- [ ] Clear acceptance criteria specified?
- [ ] Verifiable outputs defined (MLflow run_id, file paths)?
- [ ] Single responsibility (one task = one deliverable)?
- [ ] Dependencies documented?
- [ ] Owner appropriate (ops_commander / quality_safety / infrastructure)?
```

---

### **2. Empirical Validation Lead** ✅

**File:** `agents/prompts/planning_team/02_empirical_validation_lead.md`

**Changes Made:**
- Added **TASK DECOMPOSITION SUPPORT** section to Mission
- Guidance for providing decomposition-friendly experiment recommendations:
  - Specify exact sample sizes (n=500, n=1000) for each experiment
  - Break multi-model experiments into per-model recommendations
  - Provide realistic time estimates based on historical data
  - Define clear acceptance criteria (statistical thresholds, required artifacts)

**Key Addition (Lines 20-26):**
```markdown
**TASK DECOMPOSITION SUPPORT:**
When recommending experiments to the Strategic Leader, provide input that enables proper task decomposition:
- Specify exact sample sizes (n=500, n=1000) for each experiment
- Break multi-model experiments into per-model recommendations
- Provide realistic time estimates based on historical data
- Define clear acceptance criteria (statistical thresholds, required artifacts)
- **See:** `TASK_DECOMPOSITION_GUIDELINES.md` for principles
```

---

### **3. Critical Evaluator (GPT-4)** ✅

**File:** `agents/prompts/planning_team/03_critical_evaluator_openai.md`

**Changes Made:**
- Added **TASK DECOMPOSITION SUPPORT** section to Mission
- Guidance for reviewing proposed tasks:
  - Flag any tasks that are too complex (>2 hours estimated)
  - Identify tasks with unclear acceptance criteria
  - Recommend breaking tasks into verifiable subtasks
  - Ensure each task has measurable success criteria

**Key Addition (Lines 20-26):**
```markdown
**TASK DECOMPOSITION SUPPORT:**
When reviewing proposed tasks from the Strategic Leader:
- Flag any tasks that are too complex (>2 hours estimated)
- Identify tasks with unclear acceptance criteria
- Recommend breaking tasks into verifiable subtasks
- Ensure each task has measurable success criteria
- **See:** `TASK_DECOMPOSITION_GUIDELINES.md` for anti-patterns to avoid
```

---

### **4. Gemini Research Advisor** ✅

**File:** `agents/prompts/planning_team/04_gemini_research_advisor.md`

**Changes Made:**
- Added **TASK DECOMPOSITION SUPPORT** section to Mission
- Guidance for research recommendations:
  - Provide realistic feasibility assessments (implementation time, resource requirements)
  - Break complex research directions into smaller, testable hypotheses
  - Suggest incremental validation steps rather than all-or-nothing approaches

**Key Addition (Lines 14-19):**
```markdown
**TASK DECOMPOSITION SUPPORT:**
When recommending research directions or experiments:
- Provide realistic feasibility assessments (implementation time, resource requirements)
- Break complex research directions into smaller, testable hypotheses
- Suggest incremental validation steps rather than all-or-nothing approaches
- **See:** `TASK_DECOMPOSITION_GUIDELINES.md` for feasibility estimation
```

---

## 🧩 How Task Decomposition Works

### **Before (Cycle 3 - FAILED):**

**Planning Team generated:**
```json
{
  "action": "Execute temporal stability analysis with 5 independent runs per model",
  "priority": "HIGH",
  "owner": "ops_commander"
}
```

**Problems:**
- No time estimate
- No acceptance criteria
- No evidence paths
- Too complex (4 models × 5 runs × statistical analysis)
- Ops Commander claimed "COMPLETED" with zero evidence

**Result:** 94% fabrication rate

---

### **After (Cycle 4+ - Expected):**

**Planning Team will generate:**
```json
{
  "meeting_id": "cvpr_planning_cycle4_20251016_100000",
  "context": "Week 1 Cycle 4: Multi-architecture validation",
  "decisions": [
    {
      "task_id": "task_1",
      "action": "Run CLIP temporal stability (5 runs, n=100 each)",
      "priority": "HIGH",
      "owner": "ops_commander",
      "deadline": "2025-10-17T12:00:00Z",
      "estimated_time": "45 min",
      "acceptance_criteria": [
        "MLflow run_ids for all 5 runs",
        "results.json with MCS mean ± std",
        "CI95 computed from 5 runs",
        "Coefficient of variation < 5%"
      ],
      "evidence_paths": [
        "runs/temporal_stability/clip/results.json",
        "runs/temporal_stability/clip/attention_dist.csv"
      ],
      "dependencies": [],
      "rationale": "CLIP baseline needed for cross-model comparison"
    },
    {
      "task_id": "task_2",
      "action": "Run BLIP temporal stability (5 runs, n=100 each)",
      "priority": "HIGH",
      "owner": "ops_commander",
      "estimated_time": "50 min",
      ...
    },
    {
      "task_id": "task_3",
      "action": "Run Flamingo temporal stability (5 runs, n=100 each)",
      "priority": "HIGH",
      "owner": "ops_commander",
      "estimated_time": "60 min",
      ...
    },
    {
      "task_id": "task_4",
      "action": "Aggregate temporal stability results across all models",
      "priority": "HIGH",
      "owner": "ops_commander",
      "estimated_time": "20 min",
      "dependencies": ["task_1", "task_2", "task_3"],
      ...
    }
  ]
}
```

**Result:**
- 4 small, verifiable tasks instead of 1 large unverifiable task
- Each task <1 hour (easier to complete)
- Clear acceptance criteria (easier to verify)
- Dependencies explicit (clear execution order)
- Evidence paths specified (easier to verify file existence)

---

## 📊 Agent Responsibilities Summary

| Agent | Role in Task Decomposition |
|-------|---------------------------|
| **Strategic Leader** | PRIMARY - Generates `pending_actions.json` with decomposed tasks |
| **Empirical Validation Lead** | SUPPORT - Provides decomposition-friendly experiment specs |
| **Critical Evaluator** | REVIEW - Flags complex tasks, recommends breaking them down |
| **Gemini Research Advisor** | SUPPORT - Provides realistic feasibility assessments |

---

## 🎯 Expected Benefits

### **Planning Team Outputs:**
- ✅ Average task time estimate: 30-60 min (down from 90-120 min)
- ✅ All tasks have clear acceptance criteria
- ✅ All tasks specify evidence paths
- ✅ Dependencies clearly documented

### **Executive Team Execution:**
- ✅ Higher completion rate (small tasks easier to finish)
- ✅ Lower fabrication rate (evidence easier to provide)
- ✅ Faster feedback (failures caught per subtask, not at end)
- ✅ Better progress tracking (20% → 40% → 60% vs 0% → 100%)

---

## 🧪 Testing Plan

### **Test with Cycle 4:**

1. **Run Planning Team Meeting** → generates `pending_actions.json`
2. **Verify task decomposition applied:**
   - All tasks <2 hours?
   - All tasks have acceptance criteria?
   - All tasks have evidence paths?
   - Complex tasks broken into subtasks?
3. **Run Executive Team Execution Cycle**
4. **Monitor Phase 5.5 Evidence Verification**
5. **Compare to Cycle 3:**
   - Completion rate higher?
   - Fabrication rate lower?
   - Evidence verification passes?

---

## 📁 Related Files

**Task Decomposition Guidelines:**
- `agents/prompts/planning_team/TASK_DECOMPOSITION_GUIDELINES.md` (v2.0)

**Updated Planning Team Prompts:**
- `agents/prompts/planning_team/01_strategic_leader.md` (v4.0)
- `agents/prompts/planning_team/02_empirical_validation_lead.md` (v4.0)
- `agents/prompts/planning_team/03_critical_evaluator_openai.md` (v4.0)
- `agents/prompts/planning_team/04_gemini_research_advisor.md` (v4.0)

**Enforcement System Documentation:**
- `ENFORCEMENT_SYSTEM_IMPLEMENTATION_COMPLETE.md` - Master implementation guide
- `EXECUTIVE_TEAM_ENFORCEMENT_SYSTEM.md` - Executive Team evidence requirements
- `QUALITY_SAFETY_EVIDENCE_PROTOCOL.md` - Evidence verification protocol
- `INFRASTRUCTURE_FILE_VERIFICATION_PROTOCOL.md` - File system integrity checks

---

## ✅ Implementation Status

### **Completed (All High Priority TODOs):**
- [x] Create Infrastructure Monitor file verification protocol (v5.0)
- [x] Update Planning Team prompts for task decomposition (v2.0)
- [x] Sync all files to Google Drive

### **Remaining (User Manual Implementation):**
- [ ] Add Phase 5.5 evidence verification cell to execution notebook
- [ ] Fix Phase 5 timestamp issue (Cell 13)
- [ ] Update task completion logic to require all 3 agents to pass
- [ ] Test enforcement system with Cycle 4

---

## 🚀 Next Steps

### **For User (Manual Notebook Updates):**

1. **Update Execution Notebook:**
   - Fix Cell 13 (Phase 5 timestamp) - see `FIX_PHASE_5_TIMESTAMP.md`
   - Add Phase 5.5 evidence verification cell - see `ENFORCEMENT_SYSTEM_IMPLEMENTATION_COMPLETE.md` (lines 147-267)

2. **Test System:**
   - Run Planning Team meeting (Cycle 4)
   - Verify `pending_actions.json` has decomposed tasks
   - Run Executive Team execution
   - Verify Phase 5.5 catches any missing evidence

3. **Monitor Results:**
   - Compare Cycle 4 to Cycle 3
   - Check fabrication rate (target: 0%)
   - Check completion rate (target: higher)
   - Check evidence verification (target: 100% pass)

---

## 📈 Success Criteria

**System will be successful when:**

1. **Planning Team:**
   - ✅ All tasks in `pending_actions.json` have time estimates <2hr
   - ✅ All tasks have clear acceptance criteria
   - ✅ All tasks specify evidence paths
   - ✅ Complex tasks broken into subtasks

2. **Executive Team:**
   - ✅ Higher task completion rate (small tasks easier)
   - ✅ Lower fabrication rate (evidence easier to provide)
   - ✅ Phase 5.5 verification passes without downgrades

3. **Research Integrity:**
   - ✅ All experimental data has MLflow evidence
   - ✅ All statistics traceable to result files
   - ✅ All claims verifiable and reproducible

---

**Status:** ✅ COMPLETE - All Planning Team prompts updated and synced
**Version:** 1.0 - Planning Team Task Decomposition Integration
**Last Updated:** 2025-10-15
**Ready for:** Cycle 4 testing with enforcement system enabled

---

## 🔍 Verification Checklist

**Before running Cycle 4, verify:**

- [x] Strategic Leader prompt includes task decomposition checklist
- [x] Strategic Leader prompt includes JSON template with acceptance criteria
- [x] Empirical Validation Lead provides decomposition-friendly specs
- [x] Critical Evaluator reviews task complexity
- [x] Gemini Research Advisor provides realistic feasibility
- [x] All prompts reference `TASK_DECOMPOSITION_GUIDELINES.md`
- [x] All files synced to Google Drive

**After running Cycle 4, verify:**

- [ ] `pending_actions.json` has decomposed tasks
- [ ] All tasks have acceptance criteria
- [ ] All tasks have evidence paths
- [ ] Executive Team completes tasks with evidence
- [ ] Phase 5.5 verification passes

---

**END OF DOCUMENT**
