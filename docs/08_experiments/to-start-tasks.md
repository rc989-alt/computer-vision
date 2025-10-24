

1. Column schema (what each column means)
2. Ready-to-copy header row
3. Example filled rows (3 demo tasks)
4. Metric definitions (so you can put them in appendix of the paper)

This is written assuming you’re comparing:

* **Baseline A** = Single Agent (no planning, no approval, no verification, just “do it”)
* **Baseline B** = Self-Reflecting Agent (tries to fix itself, but no governance / no approval gate / no evidence binding)
* **Ours** = Your Full Multi-Phase System (Planning → Approval → Execution → Verification → Reflection w/ retry and evidence-bound completion)

If you have other baselines, just add columns.

---

## 1. Column schema

You will record one row per *task instance*.

**Task metadata**

* `Task_ID`
  Short ID like T01, T02, etc.
* `Task_Name`
  Human-readable name: “Run attention-collapse diagnosis on V2,” “Calibrate CLIP head,” etc.
* `Domain`
  e.g. "Vision Safety Eval", "Model Training", "Retrieval Benchmark", "Data Cleaning".
* `Acceptance_Criteria`
  Short summary of what counts as success for this task (must match what Phase 1/2 wrote into pending_actions.json).

**For each system variant (Baseline_A, Baseline_B, Ours) you’ll log:**

* `Success?` (Y/N)
  Did it actually satisfy Acceptance_Criteria?
* `Hallucinated_Success?` (Y/N)
  Did the agent/system *claim* success even though validation says it's not actually done?
  (This is super important. High = bad.)
* `Num_Attempts`
  How many execution attempts / retries were needed until it stopped.

  * Baseline_A: usually 1 (it just tries once).
  * Baseline_B: might try self-fix loops.
  * Ours: includes Phase 4.5 / 5.5 / 6.5 retries.
* `Human_Intervention_Minutes`
  How many minutes a human spent unblocking it (clarifying specs, fixing code, approving risky patches, etc.).
  This will be huge for CHI framing.
* `Verification_Pass?` (Y/N)
  Did it pass Phase 6 automatic verification (i.e. did we get valid MLflow run_id / evidence bundle)?
  For baselines that don’t have a Phase 6 verifier, log whether the output *would* have passed if we ran the same checker script.
* `Final_Run_ID`
  For Ours: MLflow run_id or equivalent artifact ID that supports the claimed result.
  For baselines: put “N/A” if no evidence artifact.

**Global outcome columns**

* `Notes`
  1–2 line summary of what went wrong or right.
* `Critical_Failure?` (Y/N)
  Did it try to do something dangerous / clearly broken (e.g. delete data, fabricate metrics, infinite loop)?
  This supports your “governance / safety” section.

---

## 2. Header row (copy this directly into the first row of your spreadsheet)

Task_ID | Task_Name | Domain | Acceptance_Criteria | Baseline_A_Success | Baseline_A_Hallucinated_Success | Baseline_A_Num_Attempts | Baseline_A_Human_Intervention_Minutes | Baseline_A_Verification_Pass | Baseline_A_Final_Run_ID | Baseline_B_Success | Baseline_B_Hallucinated_Success | Baseline_B_Num_Attempts | Baseline_B_Human_Intervention_Minutes | Baseline_B_Verification_Pass | Baseline_B_Final_Run_ID | Ours_Success | Ours_Hallucinated_Success | Ours_Num_Attempts | Ours_Human_Intervention_Minutes | Ours_Verification_Pass | Ours_Final_Run_ID | Critical_Failure | Notes

That is your canonical schema.

---

## 3. Example filled rows

### Example Row 1

Task: run a diagnostic experiment and produce quantified attention-collapse metrics with plots

T01 | Run Attention Collapse Diagnosis on V2 | Vision Safety Eval | Produce collapse score table + plots for Model V2; results must be logged to MLflow with run_id | N | Y | 1 | 12 | N | N/A | N | Y | 2 | 10 | N | N/A | Y | N | 2 | 3 | Y | mlflow://runs/84ac9... | N | Baseline_A said “done” but no plots and no run_id. Baseline_B produced partial code but failed to save artifacts. Our system retried after failed verification, generated run_id, and attached metrics.

Interpretation:

* Baseline_A lied (Hallucinated_Success=Y).
* Baseline_B also lied.
* Ours succeeded, low human time, and passed verification.

---

### Example Row 2

Task: add a new evaluation script and integrate into the pipeline without breaking existing jobs

T02 | Integrate New Retrieval@K Evaluator | Retrieval Benchmark | Add eval_retrieval.py, run top-10 nDCG, store metrics + evaluation JSON in artifacts | N | Y | 1 | 25 | N | N/A | Y | N | 3 | 18 | Y | tmp_local_only | Y | N | 3 | 7 | Y | mlflow://runs/b173e... | N | Baseline_A immediately claimed success but script import crashed. Baseline_B fixed import after self-reflection but never persisted artifacts to shared storage. Our system failed approval first (Phase 4), auto-patched path, re-ran, passed verification, produced final run_id.

Interpretation:

* This shows governance gates and retry loops doing real work.

---

### Example Row 3

Task: generate a summary report for yesterday’s experiments in human-readable markdown

T03 | Summarize Experiment Results to Markdown | Reporting / Documentation | Produce summary.md including methodology, key metrics, and failure notes for all runs in last 24h | Y | N | 1 | 5 | Y | local_dump_0923 | Y | N | 2 | 4 | Y | local_dump_0923 | Y | N | 2 | 2 | Y | mlflow://runs/summary_2025_10_23 | N | All systems can draft text, but only ours attached verifiable run links + failure causes section. Human time dropped to 2 min because Planning already defined the exact markdown sections.

Interpretation:

* This shows reduced human editing time thanks to Planning phase (pre-specified acceptance criteria / sections).

---

## 4. How you will use this table in the paper

From this sheet you derive three figures/tables:

### (A) Overall performance table (aggregate across tasks)

You compute, across all tasks:

* Success rate (%) for each system
* Hallucinated success rate (%)
* Avg human intervention minutes
* Avg attempts
* % tasks with verification pass

This becomes your “Table 1: Overall Results”.

Example shape:

| System     | True_Success_Rate (%) | Hallucinated_Success_Rate (%) | Avg_Human_Intervention_Min | Avg_Num_Attempts | Verification_Pass_Rate (%) |
| ---------- | --------------------- | ----------------------------- | -------------------------- | ---------------- | -------------------------- |
| Baseline A | 30                    | 70                            | 15.3                       | 1.1              | 10                         |
| Baseline B | 45                    | 40                            | 11.2                       | 2.2              | 25                         |
| Ours       | 85                    | 5                             | 4.1                        | 2.3              | 90                         |

This is your knockout punch.

### (B) Ablation figure

You run “Ours” with modules turned off:

* No Reflection
* No Verification
* No Memory

For each ablation, re-run a subset of tasks and log just:

* True success rate
* Hallucinated success rate
* Human intervention minutes

That becomes a bar chart (or 3-bar groups) in the paper’s Ablation section.

### (C) Case studies

Pick 1–2 rows where Baseline_A hallucinated and Ours passed verification with evidence.
Copy-paste those narratives directly into the “Case Study” subsection.

---

## 5. Metric definitions (you paste this in the Appendix / Experimental Setup section)

* **Success? (Y/N)**
  Y if and only if the task output satisfies the Acceptance_Criteria that was defined at planning time (Phase 1/2), and the output passes automated verification if applicable. If any required artifact (report, metric table, plot, run_id) is missing or corrupted, mark N.

* **Hallucinated_Success? (Y/N)**
  Y if the agent/system declared or implied completion (“Task complete”, “All metrics generated”, etc.) while the acceptance check or verifier shows the requirements were *not* actually met.
  This measures false confidence / bluffing.

* **Num_Attempts**
  Total distinct execution attempts until the system stopped trying.
  For Ours, include automatic retries triggered by approval rejection (Phase 4.5), execution failure (Phase 5.5), or missing evidence (Phase 6.5).

* **Human_Intervention_Minutes**
  Cumulative wall-clock minutes a human spent stepping in:
  clarifying requirements, editing code, manually approving risky actions, unblocking resource paths.
  Reading logs silently does not count; only active intervention does.

* **Verification_Pass? (Y/N)**
  Y if output survived the system’s verifier (Phase 6): e.g. MLflow run_id exists and is queryable, metrics/plots generated with no crash, file hashes valid, etc.
  For baselines that don’t have Phase 6, we retroactively run the same checker script on their output and log whether it would have passed.

* **Final_Run_ID**
  The artifact or MLflow run identifier that backs the claimed result.
  “N/A” if no artifact is produced or nothing is logged.

* **Critical_Failure? (Y/N)**
  Y if the system attempted something unsafe (e.g. destructive file ops without approval) or produced fabricated numeric results with no underlying artifact.
  This supports your “governance / safety” argument.

---

## 6. Tiered Implementation Plan

Based on `benchmark_tasks.md` and `system_merged_execution_v3_6.ipynb`, here's a **prioritized, phased approach** to validate the execution system.

---

## TIER 1: Foundation Tasks (Start Here - Week of 10/24)

These test core infrastructure without complex retry/reflection logic.

### **T04: Data Path Validation** ✅ EASIEST START

**What it tests:** Phase 0 bootstrap, deterministic fallback

**Implementation checklist:**
1. Create task in `pending_actions.json` with invalid `TEST_DATA_DIR` path
2. Run `ensure_phase_0()` and verify it detects missing path
3. Check that synthetic fallback data is generated
4. Confirm telemetry logs `fallback_used=True`

**Expected outputs:**
- ✅ Phase 0 completes without crash
- ✅ Synthetic data created in `TEST_DATA_DIR`
- ✅ Tracker logs show fallback flag
- ✅ No human intervention needed

**Task definition template:**
```json
{
  "task_id": "T04",
  "description": "Validate data path with invalid directory and use fallback",
  "acceptance_criteria": "Fallback synthetic data created and logged in telemetry",
  "owner": "Infrastructure Agent",
  "deadline": "2025-10-24",
  "inputs": {
    "TEST_DATA_DIR": "/invalid/path/to/data"
  }
}
```

---

### **T05: Results Report Persistence**

**What it tests:** Basic artifact generation, MLflow tagging

**Implementation checklist:**
1. Define task with metadata: `id`, `owner`, `acceptance_criteria`
2. Run through Phase 5 with simple script that generates:
   - `reports/execution_summary/{task_id}_results.json`
   - `reports/execution_summary/{task_id}_summary.md`
3. Verify MLflow tags match task metadata
4. Check Phase 6 finds and validates both files

**Expected outputs:**
- ✅ JSON + MD files exist with correct schema
- ✅ MLflow run has tags: `task_id`, `owner`, `status`
- ✅ Phase 6 verification passes
- ✅ Files contain required sections (methodology, results, metrics)

**Task definition template:**
```json
{
  "task_id": "T05",
  "description": "Generate structured results JSON and Markdown summary",
  "acceptance_criteria": "Files in reports/execution_summary/, MLflow tags present",
  "owner": "Ops Agent",
  "deadline": "2025-10-24",
  "required_artifacts": [
    "results.json",
    "summary.md"
  ]
}
```

---

### **T12: Environment Pinning**

**What it tests:** Reproducibility metadata capture

**Implementation checklist:**
1. Run simple task (e.g., "print CLIP model version")
2. Check execution telemetry captures:
   - Python version
   - Key packages (torch, transformers, mlflow)
   - Environment variables (`CUDA_VISIBLE_DEVICES`, `HF_TOKEN`)
3. Verify this appears in:
   - MLflow run tags
   - `results.json` under `environment` key

**Expected outputs:**
- ✅ Environment block logged to MLflow
- ✅ Results JSON has `environment.python_version`
- ✅ Results JSON has `environment.packages[]` array
- ✅ Sensitive vars redacted (HF_TOKEN shows only first 4 chars)

---

## TIER 2: Core Execution Capabilities (After Tier 1 passes)

### **T01: CLIP Attention Extraction** 🔥 HIGH VALUE

**What it tests:** Built-in attention patch, artifact export

**Implementation checklist:**
1. Task: "Extract CLIP cross-attention for `TEST_IMAGE_PATH` + prompt='a dog on a beach'"
2. Verify Phase 5 execution:
   - Uses attention patch from `multi-agent/scripts/V3_6_CRITICAL_FIX_ATTENTION_LAYERS.md`
   - No manual code edits required
   - Saves `attentions/*.npy` files
   - Generates `attn_grid.png` preview
3. Check Phase 6 verification:
   - MLflow artifact `attentions/` folder exists
   - `attn_grid.png` is valid PNG (non-zero size, valid header)
   - run_id is queryable via API

**Expected outputs:**
- ✅ Attention maps exported without code edits
- ✅ Artifacts in MLflow: `attentions/layer_*.npy`, `attn_grid.png`
- ✅ Phase 6 passes with valid run_id
- ✅ Grid shows 12 attention layers (CLIP ViT-B/32 default)

**Task definition template:**
```json
{
  "task_id": "T01",
  "description": "Export CLIP cross-attention maps for given image/text pair",
  "acceptance_criteria": "MLflow run logged with attentions/*.npy and attn_grid.png",
  "owner": "Ops Agent",
  "deadline": "2025-10-25",
  "inputs": {
    "image_path": "TEST_IMAGE_PATH",
    "prompt": "a dog on a beach"
  },
  "required_artifacts": [
    "attentions/",
    "attn_grid.png"
  ]
}
```

---

### **T13: Minimal Visualization Export**

**What it tests:** PNG generation and linking

**Implementation checklist:**
1. Task: "Generate bar chart of fake metrics and save as `summary.png`"
2. Script produces simple matplotlib plot (e.g., 3 bars)
3. Check:
   - PNG saved to MLflow artifacts
   - Markdown report links PNG: `![Summary](summary.png)`
   - Phase 6 validates PNG is not corrupted (file size > 0, valid header)

**Expected outputs:**
- ✅ Valid PNG artifact (file size > 1KB)
- ✅ Markdown has embedded image link
- ✅ Phase 6 checks file integrity
- ✅ PNG viewable when opened

---

## TIER 3: Retry and Governance (After Tiers 1-2 validated)

### **T03: Import Error Recovery**

**What it tests:** Phase 5.5 execution retry with minimal patch

**Implementation checklist:**
1. Inject script with `ModuleNotFoundError` (e.g., missing `import sklearn`)
2. Phase 5 fails with traceback
3. Phase 5.5 detects error, applies minimal patch:
   - Option A: Add missing import
   - Option B: Replace with alternative (numpy instead of sklearn)
4. Re-run Phase 5 succeeds
5. Check `ledgers/patches/` has provenance record

**Expected outputs:**
- ✅ Phase 5 initial failure with traceback
- ✅ Phase 5.5 applies fix (logged in telemetry)
- ✅ Re-run Phase 5 succeeds
- ✅ Patch logged in `ledgers/reflections.jsonl` with confidence score
- ✅ Retry counter: `execution_retries=1`

**Task definition template:**
```json
{
  "task_id": "T03",
  "description": "Detect and fix missing import during execution",
  "acceptance_criteria": "Patch applied at Phase 5.5, execution succeeds, provenance recorded",
  "owner": "Ops Agent",
  "deadline": "2025-10-26",
  "inject_error": {
    "type": "ModuleNotFoundError",
    "missing_module": "sklearn"
  }
}
```

---

### **T06: Approval Contract Enforcement**

**What it tests:** Phase 4 gate + Phase 4.5 targeted retry

**Implementation checklist:**
1. Create agent response missing required field (e.g., `implementation.code` is empty)
2. Phase 4 gate rejects with structured reason + low confidence
3. Phase 4.5 applies targeted fix:
   - Re-prompt agent with schema reminder
   - Include example of correct format
4. Re-run Phase 4 with corrected response
5. Check retry counter incremented

**Expected outputs:**
- ✅ Phase 4 initial rejection (logged reason)
- ✅ Phase 4.5 retry with schema-aware prompt
- ✅ Phase 4 approval after fix (confidence ≥ 0.7)
- ✅ Tracker shows `approval_retries=1`
- ✅ Contract validation passed

---

### **T09: API Evidence Check**

**What it tests:** Phase 6 MLflow verification with API

**Implementation checklist:**
1. Run task that logs MLflow run
2. Phase 6 calls MLflow API to validate:
   - `run_id` exists in tracking server
   - `run_id` matches expected experiment
   - Required artifacts are present
3. **Negative test:** provide fake run_id, verify Phase 6 fails and triggers 6.5

**Expected outputs:**
- ✅ Valid run_id passes Phase 6
- ✅ Invalid run_id triggers Phase 6.5 regeneration
- ✅ Verification logs show API call details (HTTP 200 vs 404)
- ✅ Normalized run_id format (strips `mlflow://runs/` prefix if present)

---

## TIER 4: Advanced Governance (Optional for initial validation)

**T02: Evidence Regeneration** - Phase 6.5 minimal script generation
**T07: Risk-Scoped Patch Application** - Reflection confidence gating (tau threshold)
**T10: Timeout Handling** - Safe abort with partial artifacts
**T14: Retry Budget Exhaustion** - Retry cap enforcement
**T15: Owner/Deadline Compliance** - Metadata validation

---

## IMMEDIATE 3-DAY ACTION PLAN (10/24 - 10/26)

### **Day 1 (10/24): Setup & Tier 1 Validation**

**Morning (2 hours):**
1. Create task definition files:
   - `multi-agent/experiments/tier1_tasks.json` (T04, T05, T12)
   - Copy templates from above
2. Verify notebook setup:
   - Check MLflow tracking URI is set
   - Verify `TEST_IMAGE_PATH` exists
   - Confirm `reports/` and `ledgers/` directories exist

**Afternoon (3 hours):**
```python
# Run Tier 1 experiments
ids = prepare_execution_cycle(selected_task_ids=["T04", "T05", "T12"])
run_execution_cycle(ids)

# Check results
export_path = export_task_contexts('tier1_validation')
print(f"Results exported to: {export_path}")

# Verify outputs
tracker.get_summary()  # Should show 3 tasks processed
tracker.get_retry_stats()  # Should show 0 retries (these are simple tasks)
```

**Success criteria for Day 1:**
- ✅ All 3 tasks reach Phase 7 with `final_status=success`
- ✅ No crashes in Phase 0, 5, or 6
- ✅ Artifacts present in MLflow
- ✅ Telemetry logged correctly

**If anything fails:** Debug before moving to Tier 2

---

### **Day 2 (10/25): Core Capabilities**

**Morning (1 hour):**
1. Review Day 1 results
2. Create `multi-agent/experiments/tier2_tasks.json` (T01, T13)

**Afternoon (4 hours):**
```python
# Run Tier 2 experiments
ids = prepare_execution_cycle(selected_task_ids=["T01", "T13"])
run_execution_cycle(ids)

# Validate T01 (CLIP attention)
# Should see attentions/ folder with .npy files + attn_grid.png

# Validate T13 (visualization)
# Should see summary.png in artifacts
```

**Success criteria for Day 2:**
- ✅ T01: Attention maps exported **without manual code edits**
- ✅ T01: `attn_grid.png` shows 12 layers (3x4 grid)
- ✅ T13: PNG generated and linked in Markdown
- ✅ Both pass Phase 6 verification with valid run_id
- ✅ No import errors or missing dependencies

---

### **Day 3 (10/26): Retry Logic**

**Morning (1 hour):**
1. Create `multi-agent/experiments/tier3_tasks.json` (T03, T06, T09)
2. **Important:** For T03, intentionally inject import error in task definition

**Afternoon (4 hours):**
```python
# Run Tier 3 experiments
ids = prepare_execution_cycle(selected_task_ids=["T03", "T06", "T09"])
run_execution_cycle(ids)

# Check retry stats
stats = tracker.get_retry_stats()
print(f"Approval retries (T06): {stats['approval_retries']}")  # Should be ≥ 1
print(f"Execution retries (T03): {stats['execution_retries']}")  # Should be ≥ 1
print(f"Verification retries (T09): {stats['verification_retries']}")  # Should be ≥ 1 if negative test triggered

# Check ledgers
!ls -lh multi-agent/ledgers/reflections.jsonl
!tail -20 multi-agent/ledgers/reflections.jsonl  # Should show patch records
```

**Success criteria for Day 3:**
- ✅ T03: Phase 5.5 retry triggered and succeeds
- ✅ T06: Phase 4.5 retry triggered (contract enforcement working)
- ✅ T09: Valid run_id passes, invalid run_id triggers 6.5
- ✅ Retry counters logged in tracker
- ✅ Patches recorded in `ledgers/reflections.jsonl` with confidence scores

---

## EVALUATION SHEET POPULATION (After Day 3)

**File to create:** `multi-agent/experiments/evaluation_sheet.csv`

**Step 1:** Copy header row from Section 2 above

**Step 2:** For each completed task (T04, T05, T12, T01, T13, T03, T06, T09), fill:

| Task_ID | Task_Name | Domain | Acceptance_Criteria | Ours_Success | Ours_Hallucinated_Success | Ours_Num_Attempts | Ours_Human_Intervention_Minutes | Ours_Verification_Pass | Ours_Final_Run_ID | Notes |
|---------|-----------|--------|---------------------|--------------|---------------------------|-------------------|---------------------------------|------------------------|-------------------|-------|
| T04 | Data Path Validation | Infrastructure | Fallback data created and logged | Y | N | 1 | 0 | Y | mlflow://... | Fallback worked, no intervention |
| T05 | Results Report | Artifacts | JSON + MD generated with tags | Y | N | 1 | 0 | Y | mlflow://... | All artifacts present |
| T12 | Environment Pinning | Reproducibility | Env metadata in MLflow + JSON | Y | N | 1 | 0 | Y | mlflow://... | Python 3.10, torch 2.0.1 logged |
| T01 | CLIP Attention | Vision Safety | Attention maps + grid PNG | Y | N | 1 | 0 | Y | mlflow://... | 12 layers extracted, no code edits |
| T13 | Visualization | Reporting | PNG generated and linked | Y | N | 1 | 0 | Y | mlflow://... | Bar chart valid |
| T03 | Import Error Recovery | Execution Retry | Patch applied, execution succeeds | Y | N | 2 | 0 | Y | mlflow://... | Phase 5.5 fixed ModuleNotFoundError |
| T06 | Contract Enforcement | Approval Gate | Rejection → fix → approval | Y | N | 2 | 0 | Y | mlflow://... | Phase 4.5 corrected missing field |
| T09 | Evidence Check | Verification | API validation passed | Y | N | 1 | 0 | Y | mlflow://... | run_id normalized and verified |

**Step 3:** Compute aggregate metrics:
```python
import pandas as pd

df = pd.read_csv('multi-agent/experiments/evaluation_sheet.csv')

# Overall success rate
success_rate = (df['Ours_Success'] == 'Y').sum() / len(df) * 100

# Hallucination rate
hallucination_rate = (df['Ours_Hallucinated_Success'] == 'Y').sum() / len(df) * 100

# Avg attempts
avg_attempts = df['Ours_Num_Attempts'].mean()

# Avg human time
avg_human_time = df['Ours_Human_Intervention_Minutes'].mean()

# Verification pass rate
verify_pass_rate = (df['Ours_Verification_Pass'] == 'Y').sum() / len(df) * 100

print(f"Success Rate: {success_rate:.1f}%")
print(f"Hallucination Rate: {hallucination_rate:.1f}%")
print(f"Avg Attempts: {avg_attempts:.2f}")
print(f"Avg Human Intervention: {avg_human_time:.1f} min")
print(f"Verification Pass Rate: {verify_pass_rate:.1f}%")
```

**Expected baseline for "Ours" system (8 tasks):**
- Success Rate: **85-100%** (7-8 tasks succeed)
- Hallucination Rate: **0-10%** (0-1 tasks claim false success)
- Avg Attempts: **1.2-1.5** (some retries for T03, T06)
- Avg Human Intervention: **0-2 min** (minimal debugging)
- Verification Pass Rate: **85-100%** (Phase 6 catches issues)

---

## FILES TO CREATE TODAY (10/24)

### 1. Task Definition Files

**File:** `multi-agent/experiments/tier1_tasks.json`
```json
{
  "tier": 1,
  "description": "Foundation tasks - Phase 0/5/6 validation",
  "tasks": [
    {
      "task_id": "T04",
      "description": "Validate data path with invalid directory",
      "acceptance_criteria": "Fallback synthetic data created, logged in telemetry",
      "owner": "Infrastructure Agent",
      "deadline": "2025-10-24",
      "inputs": { "TEST_DATA_DIR": "/invalid/path" }
    },
    {
      "task_id": "T05",
      "description": "Generate results JSON and Markdown summary",
      "acceptance_criteria": "Files in reports/execution_summary/, MLflow tags present",
      "owner": "Ops Agent",
      "deadline": "2025-10-24"
    },
    {
      "task_id": "T12",
      "description": "Capture environment metadata",
      "acceptance_criteria": "Python version, packages logged to MLflow and JSON",
      "owner": "Infrastructure Agent",
      "deadline": "2025-10-24"
    }
  ]
}
```

**File:** `multi-agent/experiments/tier2_tasks.json`
```json
{
  "tier": 2,
  "description": "Core capabilities - CLIP attention, visualization",
  "tasks": [
    {
      "task_id": "T01",
      "description": "Export CLIP cross-attention maps",
      "acceptance_criteria": "MLflow run with attentions/*.npy and attn_grid.png",
      "owner": "Ops Agent",
      "deadline": "2025-10-25",
      "inputs": {
        "image_path": "TEST_IMAGE_PATH",
        "prompt": "a dog on a beach"
      }
    },
    {
      "task_id": "T13",
      "description": "Generate bar chart visualization",
      "acceptance_criteria": "summary.png in MLflow artifacts, linked in Markdown",
      "owner": "Ops Agent",
      "deadline": "2025-10-25"
    }
  ]
}
```

**File:** `multi-agent/experiments/tier3_tasks.json`
```json
{
  "tier": 3,
  "description": "Retry logic - Phase 4.5/5.5/6.5 validation",
  "tasks": [
    {
      "task_id": "T03",
      "description": "Recover from import error",
      "acceptance_criteria": "Patch applied at Phase 5.5, execution succeeds",
      "owner": "Ops Agent",
      "deadline": "2025-10-26",
      "inject_error": { "type": "ModuleNotFoundError", "module": "sklearn" }
    },
    {
      "task_id": "T06",
      "description": "Enforce approval contract",
      "acceptance_criteria": "Rejection at Phase 4, fix at 4.5, approval succeeds",
      "owner": "Quality Agent",
      "deadline": "2025-10-26"
    },
    {
      "task_id": "T09",
      "description": "Verify MLflow run via API",
      "acceptance_criteria": "Valid run_id passes, invalid triggers 6.5",
      "owner": "Ops Agent",
      "deadline": "2025-10-26"
    }
  ]
}
```

### 2. Evaluation Sheet Template

**File:** `multi-agent/experiments/evaluation_sheet.csv`

Copy header row from Section 2, then add empty rows for T04, T05, T12, T01, T13, T03, T06, T09.

### 3. Experiment Runner Script

**File:** `multi-agent/experiments/run_tier_experiments.py`
```python
"""
Experiment runner for tiered task validation.
Usage: python run_tier_experiments.py --tier 1
"""
import sys
sys.path.append('/content/computer-vision')  # Adjust for Colab

from system_merged_execution_v3_6 import (
    prepare_execution_cycle,
    run_execution_cycle,
    export_task_contexts,
    tracker
)
import argparse
import json

def load_tier_tasks(tier: int):
    """Load task definitions for given tier."""
    with open(f'multi-agent/experiments/tier{tier}_tasks.json') as f:
        data = json.load(f)
    return [task['task_id'] for task in data['tasks']]

def run_tier(tier: int):
    """Run all tasks for a given tier."""
    print(f"\n{'='*60}")
    print(f"RUNNING TIER {tier} EXPERIMENTS")
    print(f"{'='*60}\n")

    task_ids = load_tier_tasks(tier)
    print(f"Tasks to run: {task_ids}")

    # Prepare and run
    ids = prepare_execution_cycle(selected_task_ids=task_ids)
    run_execution_cycle(ids)

    # Export results
    export_path = export_task_contexts(f'tier{tier}_results')
    print(f"\nResults exported to: {export_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    tracker.get_summary()
    tracker.get_retry_stats()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', type=int, required=True, choices=[1,2,3,4])
    args = parser.parse_args()

    run_tier(args.tier)
```

---

## SUMMARY: What You Should Do RIGHT NOW

✅ **Today (10/24 Morning):**
1. Create the 3 task definition JSON files above
2. Create `evaluation_sheet.csv` with header row
3. Create `run_tier_experiments.py` script

✅ **Today (10/24 Afternoon):**
1. Run Tier 1 experiments (T04, T05, T12)
2. Verify all 3 tasks complete successfully
3. Fill first 3 rows of evaluation sheet

✅ **Tomorrow (10/25):**
1. Run Tier 2 experiments (T01, T13)
2. Verify CLIP attention extraction works without code edits
3. Fill next 2 rows of evaluation sheet

✅ **Day 3 (10/26):**
1. Run Tier 3 experiments (T03, T06, T09)
2. Verify retry mechanisms work correctly
3. Fill final 3 rows of evaluation sheet
4. Compute aggregate metrics

**By end of 10/26, you will have:**
- ✅ 8 validated tasks with data
- ✅ Populated evaluation sheet with metrics
- ✅ Proof that Phase 0/4/5/6 + retries work correctly
- ✅ Foundation for Table 1 in paper

**This gives you concrete evidence to write Results section by 10/27.**
