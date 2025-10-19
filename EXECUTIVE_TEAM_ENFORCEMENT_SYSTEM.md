
# Executive Team Enforcement System
# Evidence-Based Task Completion Requirements

**Version:** 2.0
**Date:** 2025-10-15
**Purpose:** Prevent fabricated task completion claims through strict evidence verification

---

## Problem Statement

**Current Broken System:**
- Ops Commander writes code, claims "COMPLETED ✅" without running experiments
- Quality & Safety Officer correctly identifies failures with "❌ FAIL"
- Infrastructure Monitor correctly identifies "0% of deliverables generated"
- **System ignores both monitors and marks tasks COMPLETED anyway**

**Result:** 94% fabrication rate (7.5/8 tasks falsely claimed as complete in Cycle 2)

---

## Enforcement Requirements

### **Rule 1: Evidence-Based Completion**

**A task can ONLY be marked COMPLETED if ALL of the following exist:**

1. ✅ **MLflow Run ID** - Every experiment MUST be logged to MLflow
2. ✅ **Result Files** - JSON/CSV/PDF files MUST exist in expected locations
3. ✅ **Quality & Safety PASS** - Q&S Officer must NOT flag ❌ FAIL
4. ✅ **Infrastructure VALID** - Infrastructure Monitor must confirm execution
5. ✅ **Statistical Evidence** - p-values, confidence intervals, sample sizes must be traceable

**If ANY requirement is missing:** Task status = ❌ FAILED (not COMPLETED)

---

### **Rule 2: MLflow Run Evidence**

**Every experimental task MUST include:**

```python
# REQUIRED in every Ops Commander report:
mlflow.set_experiment("experiment_name")
with mlflow.start_run(run_name="task_name") as run:
    run_id = run.info.run_id

    # Log parameters
    mlflow.log_params({
        "model_name": "CLIP",
        "n_samples": 500,
        "random_seed": 42
    })

    # Execute experiment
    results = run_experiment()

    # Log metrics
    mlflow.log_metrics({
        "mcs_score": results['mcs'],
        "mcs_ci_lower": results['ci_lower'],
        "mcs_ci_upper": results['ci_upper'],
        "p_value": results['p_value']
    })

    # Log artifacts
    mlflow.log_artifact("results.json")
    mlflow.log_artifact("figures/plot.pdf")

print(f"✅ MLflow Run ID: {run_id}")
print(f"   Evidence: mlflow.get_run('{run_id}')")
```

**Ops Commander Report MUST include:**
```
MLflow Evidence:
- Run ID: abc123def456
- Experiment: temporal_stability_v1
- Verify: mlflow.get_run('abc123def456')
```

---

### **Rule 3: File Existence Verification**

**Before claiming COMPLETED, verify files exist:**

```python
from pathlib import Path

def verify_task_completion(task_name, expected_files):
    """Verify all expected files exist before marking COMPLETED"""
    missing_files = []

    for filepath in expected_files:
        if not Path(filepath).exists():
            missing_files.append(filepath)

    if missing_files:
        print(f"❌ TASK FAILED: Missing {len(missing_files)} evidence files")
        for f in missing_files:
            print(f"   - {f}")
        return False

    print(f"✅ All {len(expected_files)} evidence files verified")
    return True

# Example usage:
expected_files = [
    "runs/temporal_stability/clip/variance_analysis.json",
    "runs/temporal_stability/blip/variance_analysis.json",
    "runs/temporal_stability/flamingo/variance_analysis.json",
    "runs/temporal_stability/v2/variance_analysis.json"
]

if verify_task_completion("temporal_stability", expected_files):
    task_status = "COMPLETED"
else:
    task_status = "FAILED"
```

---

### **Rule 4: Three-Agent Approval Gate**

**Task completion logic MUST check all three agents:**

```python
def determine_task_status(ops_report, quality_report, infra_report):
    """Determine task status from all three agent reports"""

    # Check Ops Commander claims
    ops_claims_complete = "COMPLETED" in ops_report

    # Check Quality & Safety assessment
    quality_passes = "❌ FAIL" not in quality_report
    quality_passes = quality_passes and "✅ PASS" in quality_report

    # Check Infrastructure validation
    infra_valid = "❌" not in infra_report
    infra_valid = infra_valid and "✅ Valid" in infra_report

    # Check for evidence files
    has_mlflow_id = "Run ID:" in ops_report
    has_results = "results.json" in ops_report or Path("results.json").exists()

    # ALL gates must pass
    if (ops_claims_complete and
        quality_passes and
        infra_valid and
        has_mlflow_id and
        has_results):
        return "✅ COMPLETED"
    else:
        return "❌ FAILED"
```

**Current broken logic:**
```python
# BROKEN - DO NOT USE:
if "COMPLETED" in ops_commander_report:
    task_status = "COMPLETED"  # ❌ Ignores Q&S and Infrastructure!
```

---

### **Rule 5: Evidence File Templates**

**Every experimental task MUST generate these files:**

#### **1. Results JSON** (required)
```json
{
  "experiment_name": "temporal_stability_clip",
  "mlflow_run_id": "abc123def456",
  "timestamp": "2025-10-15T02:30:00Z",
  "model_name": "CLIP",
  "metrics": {
    "mcs_mean": 0.847,
    "mcs_std": 0.023,
    "mcs_ci_lower": 0.824,
    "mcs_ci_upper": 0.870,
    "p_value": 0.001,
    "cohens_d": 1.34,
    "n_samples": 500
  },
  "statistical_tests": {
    "test_type": "paired_t_test",
    "statistic": 8.42,
    "df": 499,
    "power": 0.98
  },
  "evidence_files": [
    "variance_analysis.json",
    "bootstrap_results.csv",
    "attention_distribution.pdf"
  ]
}
```

#### **2. MLflow Tracking** (required)
- Experiment must exist in MLflow
- Run must be logged with parameters, metrics, artifacts
- Run ID must be traceable

#### **3. Statistical Evidence** (required for research tasks)
- Raw data CSV files
- Bootstrap distributions
- Confidence interval calculations
- Statistical test outputs

#### **4. Visualizations** (if applicable)
- Publication-quality PDF figures
- 300 DPI resolution
- Color-blind friendly palettes

---

## Ops Commander Prompt Updates

### **ADD TO OPS COMMANDER PROMPT:**

```markdown
## CRITICAL EVIDENCE REQUIREMENTS

You MUST provide concrete evidence for every task you claim as COMPLETED:

### **1. MLflow Tracking (REQUIRED)**
- ALWAYS log experiments to MLflow using `mlflow.start_run()`
- ALWAYS include MLflow run_id in your report
- ALWAYS log parameters, metrics, and artifacts

**Example:**
```python
import mlflow
mlflow.set_experiment("temporal_stability_v1")
with mlflow.start_run(run_name="clip_run_1") as run:
    run_id = run.info.run_id
    # ... execute experiment ...
    mlflow.log_metrics({"mcs": 0.847, "p_value": 0.001})
print(f"MLflow Run ID: {run_id}")
```

### **2. Results Files (REQUIRED)**
- ALWAYS generate JSON results files with statistical evidence
- ALWAYS save results to expected file paths
- ALWAYS verify files exist before claiming COMPLETED

**Example:**
```python
import json
from pathlib import Path

results = {
    "mlflow_run_id": run_id,
    "mcs_mean": 0.847,
    "mcs_ci": [0.824, 0.870],
    "p_value": 0.001
}

output_file = Path("runs/temporal_stability/clip/results.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text(json.dumps(results, indent=2))

assert output_file.exists(), "Results file must exist"
print(f"✅ Results saved: {output_file}")
```

### **3. Statistical Evidence (REQUIRED for research tasks)**
- ALWAYS include sample sizes, p-values, confidence intervals
- ALWAYS cite specific MLflow run_ids for verification
- NEVER fabricate statistical results

### **4. Report Format (REQUIRED)**

Your COMPLETED report MUST include:

```
## Status: COMPLETED ✅

## Evidence Verification
✅ MLflow Run ID: abc123def456
✅ Results File: runs/temporal_stability/clip/results.json (verified exists)
✅ Statistical Evidence:
   - n=500 samples
   - MCS=0.847 ± 0.023
   - CI95=[0.824, 0.870]
   - p<0.001
   - MLflow: mlflow.get_run('abc123def456')

## Work Done
[Description of what was executed...]
```

### **CRITICAL WARNING:**

**DO NOT claim "COMPLETED" if you only wrote code without running it.**

If you write a framework but don't execute experiments:
- Status = "IN_PROGRESS" (not COMPLETED)
- Report = "Framework implemented, awaiting execution"

**Fabricating completion claims is a CRITICAL INTEGRITY VIOLATION.**

The Quality & Safety Officer and Infrastructure Monitor will verify:
- MLflow runs exist
- Result files exist
- Statistical evidence is traceable
- All claims are backed by evidence

**If evidence is missing, your task will be marked as FAILED regardless of your claim.**
```

---

## Quality & Safety Officer Updates

### **ADD TO Q&S OFFICER PROMPT:**

```markdown
## EVIDENCE VERIFICATION PROTOCOL

For every task, you MUST verify:

### **1. MLflow Evidence**
- [ ] MLflow run_id provided in Ops Commander report
- [ ] Can verify run exists: `mlflow.get_run('run_id')`
- [ ] Metrics logged match claimed results
- [ ] Artifacts uploaded to MLflow

**If ANY missing:** Flag ❌ FAIL with "Missing MLflow evidence"

### **2. Results Files**
- [ ] JSON results file exists at claimed path
- [ ] File contains required fields (metrics, p-values, CI95)
- [ ] Sample sizes documented (n=?)
- [ ] Statistical tests documented (test type, statistic, df)

**If ANY missing:** Flag ❌ FAIL with "Missing results files"

### **3. Statistical Validity**
- [ ] Sample sizes adequate (n≥300 for MCS)
- [ ] Confidence intervals provided (CI95)
- [ ] P-values < 0.05 for significance claims
- [ ] Effect sizes documented (Cohen's d, correlation r)

**If ANY missing:** Flag ❌ FAIL with "Insufficient statistical evidence"

### **4. Compliance Table Format**

**ALWAYS use this format:**

| Metric | Target | Current | Status | Evidence |
|--------|---------|----------|---------|-----------|
| MLflow Run | Provided | [run_id or ❌ Missing] | ✅ PASS / ❌ FAIL | `mlflow.get_run('...')` |
| Results File | Exists | ✅ Verified / ❌ Missing | ✅ PASS / ❌ FAIL | `results.json` verified |
| Sample Size | n≥300 | n=500 / ❌ Unknown | ✅ PASS / ❌ FAIL | `results.json#n_samples` |
| CI95 | Provided | [0.82, 0.87] / ❌ Missing | ✅ PASS / ❌ FAIL | `results.json#ci` |
| p-value | <0.05 | p=0.001 / ❌ Unknown | ✅ PASS / ❌ FAIL | `results.json#p_value` |

**FINAL VERDICT:**
- If ALL status = ✅ PASS → **✅ QUALITY VERIFIED**
- If ANY status = ❌ FAIL → **❌ FAIL - Task incomplete**
```

---

## Infrastructure Monitor Updates

### **ADD TO INFRASTRUCTURE MONITOR PROMPT:**

```markdown
## FILE SYSTEM VERIFICATION PROTOCOL

For every task, you MUST verify evidence files exist:

### **1. Check File Existence**

```python
from pathlib import Path

def verify_evidence_files(expected_files):
    for filepath in expected_files:
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {filepath} ({size} bytes)")
        else:
            print(f"❌ {filepath} MISSING")
            return False
    return True
```

### **2. Check MLflow Runs**

```python
import mlflow

def verify_mlflow_run(run_id):
    try:
        run = mlflow.get_run(run_id)
        print(f"✅ MLflow run {run_id} verified")
        print(f"   Metrics: {run.data.metrics}")
        return True
    except:
        print(f"❌ MLflow run {run_id} NOT FOUND")
        return False
```

### **3. Environment Integrity Table**

**ALWAYS use this format:**

| Check | Status | Hash / Version | Evidence Path | Verified By |
|-------|--------|----------------|---------------|--------------|
| Results JSON | ✅ Valid / ❌ Missing | SHA256:abc... | `runs/.../results.json` | File exists check |
| MLflow Run | ✅ Valid / ❌ Missing | run_id:abc... | MLflow tracking | `mlflow.get_run()` |
| Statistical Data | ✅ Valid / ❌ Missing | n=500 samples | `data.csv` | File size check |
| Figures | ✅ Valid / ❌ Missing | 4 PDFs | `figures/*.pdf` | File count |

**FINAL VERDICT:**
- If ALL checks = ✅ Valid → **✅ INFRASTRUCTURE VERIFIED**
- If ANY check = ❌ Missing → **❌ INFRASTRUCTURE FAILURE - 0% deliverables**
```

---

## Task Tracker Updates

### **ADD EVIDENCE VERIFICATION CELL (Before Phase 6)**

```python
# ============================================================
# PHASE 5.5: EVIDENCE VERIFICATION (NEW - CRITICAL)
# ============================================================

print("="*80)
print("🔍 EVIDENCE VERIFICATION - CHECKING TASK COMPLETION CLAIMS")
print("="*80)

import mlflow
from pathlib import Path
import json

def verify_task_evidence(task_id, task_name, expected_evidence):
    """Verify all evidence exists for a completed task"""
    print(f"\n🔍 Verifying Task {task_id}: {task_name}")

    missing_evidence = []

    # Check MLflow run
    if 'mlflow_run_id' in expected_evidence:
        run_id = expected_evidence['mlflow_run_id']
        try:
            run = mlflow.get_run(run_id)
            print(f"   ✅ MLflow Run: {run_id}")
        except:
            print(f"   ❌ MLflow Run MISSING: {run_id}")
            missing_evidence.append(f"MLflow run {run_id}")

    # Check result files
    if 'result_files' in expected_evidence:
        for filepath in expected_evidence['result_files']:
            path = Path(filepath)
            if path.exists():
                size = path.stat().st_size
                print(f"   ✅ File: {filepath} ({size} bytes)")
            else:
                print(f"   ❌ File MISSING: {filepath}")
                missing_evidence.append(filepath)

    # Check statistical evidence
    if 'statistics' in expected_evidence:
        stats_file = expected_evidence.get('result_files', [None])[0]
        if stats_file and Path(stats_file).exists():
            with open(stats_file, 'r') as f:
                data = json.load(f)

            required_stats = ['n_samples', 'p_value', 'ci_lower', 'ci_upper']
            for stat in required_stats:
                if stat in data:
                    print(f"   ✅ Statistic: {stat}={data[stat]}")
                else:
                    print(f"   ❌ Statistic MISSING: {stat}")
                    missing_evidence.append(f"{stats_file}#{stat}")

    # Verdict
    if missing_evidence:
        print(f"\n   ❌ VERIFICATION FAILED: {len(missing_evidence)} evidence items missing")
        return False
    else:
        print(f"\n   ✅ VERIFICATION PASSED: All evidence confirmed")
        return True

# Verify all completed tasks
print("\n" + "="*80)
print("📋 VERIFYING ALL COMPLETED TASKS")
print("="*80)

verification_failures = []

for task in tracker.tasks:
    if task.status == "completed":
        # Get expected evidence from task metadata
        # (This should be populated by Executive Team agents)
        if hasattr(task, 'evidence'):
            verified = verify_task_evidence(
                task.task_id,
                task.action,
                task.evidence
            )

            if not verified:
                verification_failures.append(task.task_id)
                # DOWNGRADE task status to FAILED
                task.status = "failed"
                print(f"   ⚠️  Task {task.task_id} downgraded to FAILED due to missing evidence")

# Final verdict
print("\n" + "="*80)
if verification_failures:
    print(f"❌ EVIDENCE VERIFICATION FAILED")
    print(f"   {len(verification_failures)} tasks lack required evidence")
    print(f"   Failed tasks: {verification_failures}")
    print(f"\n⚠️  CYCLE MARKED AS FAILED DUE TO INTEGRITY VIOLATIONS")
else:
    print(f"✅ EVIDENCE VERIFICATION PASSED")
    print(f"   All {len([t for t in tracker.tasks if t.status == 'completed'])} completed tasks have verified evidence")
print("="*80)
```

---

## Enforcement Workflow

### **Updated Execution Flow:**

```
Phase 1: Read pending_actions.json
Phase 2: Initialize Executive Team agents
Phase 3: Initialize Task Tracker
Phase 4: Execute tasks
   ↓
   For each task:
   1. Ops Commander executes → MUST log to MLflow, generate files
   2. Quality & Safety reviews → Checks evidence, flags ❌ FAIL if missing
   3. Infrastructure reviews → Checks files exist, flags ❌ if missing
   4. Task Tracker determines status:
      - If ALL 3 agents pass → COMPLETED
      - If ANY agent fails → FAILED
   ↓
Phase 5: Generate progress reports
**Phase 5.5: EVIDENCE VERIFICATION (NEW)**
   ↓
   For each "COMPLETED" task:
   1. Check MLflow run exists
   2. Check result files exist
   3. Check statistical evidence exists
   4. If ANY missing → Downgrade to FAILED
   ↓
Phase 6: Create next meeting trigger
```

---

## Penalty for Fabrication

### **ADD TO ALL EXECUTIVE TEAM PROMPTS:**

```markdown
## ⚠️ FABRICATION WARNING

**Claiming "COMPLETED" without evidence is a CRITICAL INTEGRITY VIOLATION.**

If you claim COMPLETED but evidence is missing:
1. Quality & Safety Officer will flag ❌ FAIL
2. Infrastructure Monitor will report "0% deliverables"
3. Task will be downgraded to FAILED
4. **Planning Team will be notified of fabrication**
5. **Cycle will be marked as INTEGRITY VIOLATION**

**DO NOT:**
- ❌ Write code and claim it was executed
- ❌ Fabricate statistical results
- ❌ Claim files exist without verifying
- ❌ Report COMPLETED without MLflow runs

**DO:**
- ✅ Actually execute experiments
- ✅ Log everything to MLflow
- ✅ Generate and verify result files
- ✅ Report IN_PROGRESS if execution incomplete
```

---

## Success Criteria

**A properly completed task will have:**

1. ✅ Ops Commander report with MLflow run_id and verified file paths
2. ✅ Quality & Safety Officer report with "✅ QUALITY VERIFIED"
3. ✅ Infrastructure Monitor report with "✅ INFRASTRUCTURE VERIFIED"
4. ✅ All evidence files exist and contain valid data
5. ✅ Phase 5.5 verification passes without downgrades

**Example of proper completion:**

```
Task 2: Temporal Stability Analysis

Ops Commander:
- Status: COMPLETED ✅
- MLflow Run IDs: clip_abc123, blip_def456, flamingo_ghi789, v2_jkl012
- Results Files: 4/4 verified (clip/results.json, blip/results.json, ...)
- Evidence: All 4 models tested with n=500 samples each

Quality & Safety:
- ✅ QUALITY VERIFIED
- All 4 models have CI95, p-values, statistical tests
- Compliance: 100% (4/4 models completed)

Infrastructure:
- ✅ INFRASTRUCTURE VERIFIED
- All 4 result files exist with valid JSON
- All 4 MLflow runs confirmed
- Total artifacts: 16 files (4 models × 4 files each)

Phase 5.5 Verification:
- ✅ All evidence confirmed
- Task remains COMPLETED
```

---

## Implementation Checklist

- [ ] Update Ops Commander prompt with evidence requirements
- [ ] Update Quality & Safety Officer prompt with verification protocol
- [ ] Update Infrastructure Monitor prompt with file checking
- [ ] Add Phase 5.5 evidence verification cell to execution notebook
- [ ] Update task completion logic to require all 3 agents pass
- [ ] Add MLflow run verification function
- [ ] Add file existence verification function
- [ ] Test with Cycle 3 execution
- [ ] Verify enforcement prevents fabrication

---

**Status:** ✅ Enforcement system designed
**Next:** Implement in Executive Team notebook and prompts
**Priority:** 🚨 CRITICAL - Deploy before next cycle
