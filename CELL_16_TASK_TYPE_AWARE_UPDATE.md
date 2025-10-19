# Cell 16 - Task-Type Aware Evidence Verification (Updated)

**Date:** 2025-10-15
**Purpose:** Distinguish experimental tasks (need MLflow) from documentation tasks (need files/docs)

---

## Updated Cell 16 Code

```python
# ============================================================
# PHASE 5.5: EVIDENCE VERIFICATION (CRITICAL - DO NOT SKIP)
# Task-Type Aware Version - Experimental vs Documentation
# ============================================================

print("="*80)
print("🔍 EVIDENCE VERIFICATION - CHECKING TASK COMPLETION CLAIMS")
print("="*80)

import mlflow
from pathlib import Path
import json

# Define task type keywords
EXPERIMENTAL_KEYWORDS = [
    'execute', 'run', 'diagnostic', 'experiment', 'test',
    'baseline', 'gpu', 'model', 'training', 'evaluation',
    'temporal', 'stability', 'intervention', 'ablation'
]

NON_EXPERIMENTAL_KEYWORDS = [
    'design', 'draft', 'write', 'review', 'survey',
    'literature', 'paper', 'framework', 'implement',
    'document', 'analyze', 'plan', 'statistical', 'outline'
]

def should_require_mlflow(task_action):
    """Determine if a task should require MLflow tracking based on action type"""
    action_lower = task_action.lower()

    # Check if experimental (GPU work, model runs, diagnostics)
    for keyword in EXPERIMENTAL_KEYWORDS:
        if keyword in action_lower:
            return True

    # Check if non-experimental (writing, design, documentation)
    for keyword in NON_EXPERIMENTAL_KEYWORDS:
        if keyword in action_lower:
            return False

    # Default: require MLflow (conservative approach for unknown task types)
    return True

def verify_task_evidence(task_id, task_name, task_result):
    """Verify all evidence exists for a completed task (task-type aware)"""
    print(f"\n🔍 Verifying Task {task_id}: {task_name}")

    missing_evidence = []
    requires_mlflow = should_require_mlflow(task_name)

    # Display task type
    if requires_mlflow:
        print(f"   📊 Task Type: EXPERIMENTAL (requires MLflow tracking)")
    else:
        print(f"   📝 Task Type: DOCUMENTATION (requires code/docs, MLflow optional)")

    # Check for MLflow run_id in agent responses (if required)
    has_mlflow = False
    for agent_name, response in task_result.get('agent_responses', {}).items():
        if 'run_id' in response.lower() or 'mlflow' in response.lower():
            # Try to extract run_id
            import re
            run_ids = re.findall(r'run_id[:\s]+([a-z0-9]+)', response, re.IGNORECASE)
            if run_ids:
                run_id = run_ids[0]
                try:
                    run = mlflow.get_run(run_id)
                    print(f"   ✅ MLflow Run: {run_id}")
                    has_mlflow = True
                except:
                    print(f"   ❌ MLflow Run INVALID: {run_id}")
                    if requires_mlflow:
                        missing_evidence.append(f"MLflow run {run_id} not found")

    # Evaluate MLflow requirement
    if requires_mlflow and not has_mlflow:
        print(f"   ❌ No MLflow run_id found in responses (REQUIRED for experimental tasks)")
        missing_evidence.append("No MLflow tracking")
    elif not requires_mlflow and not has_mlflow:
        print(f"   ℹ️  No MLflow tracking (not required for documentation tasks)")

    # Check for results files mentioned in outputs
    expected_dirs = [
        'runs/temporal_stability',
        'runs/intervention',
        'runs/ablation',
        'runs/coca_diagnostic',
        'analysis',
        'figures',
        'research',
        'paper',
        'docs'
    ]

    files_found = 0
    for output in task_result.get('outputs', []):
        filepath = output.get('file_path')
        if filepath:
            path = Path(MULTI_AGENT_ROOT) / filepath
            if path.exists():
                size = path.stat().st_size
                print(f"   ✅ File: {filepath} ({size} bytes)")
                files_found += 1
            else:
                print(f"   ❌ File MISSING: {filepath}")
                missing_evidence.append(filepath)

    if files_found == 0:
        print(f"   ⚠️  No results files documented")
        if not requires_mlflow:
            # For documentation tasks, lack of file evidence is more critical
            missing_evidence.append("No evidence files verified")
        else:
            # For experimental tasks, MLflow is primary evidence
            missing_evidence.append("No results files verified")

    # Verdict
    if missing_evidence:
        print(f"\n   ❌ VERIFICATION FAILED: {len(missing_evidence)} issues")
        for issue in missing_evidence:
            print(f"      - {issue}")
        return False
    else:
        print(f"\n   ✅ VERIFICATION PASSED")
        return True

# Verify all completed tasks
print("\n" + "="*80)
print("📋 VERIFYING ALL COMPLETED TASKS (TASK-TYPE AWARE)")
print("="*80)

verification_failures = []

for i, task in enumerate(tracker.task_results, 1):
    if task['status'] == "completed":
        verified = verify_task_evidence(
            i,
            task['action'],
            task
        )

        if not verified:
            verification_failures.append(i)
            # CRITICAL: Downgrade task status to FAILED
            task['status'] = "failed"
            print(f"   ⚠️  Task {i} downgraded to FAILED due to missing evidence")

# Final verdict
print("\n" + "="*80)
if verification_failures:
    print(f"❌ EVIDENCE VERIFICATION FAILED")
    print(f"   {len(verification_failures)} tasks lack required evidence")
    print(f"   Failed tasks: {verification_failures}")
    print(f"\n⚠️  CYCLE MARKED AS FAILED DUE TO INTEGRITY VIOLATIONS")
    print(f"\n🚨 ACTION REQUIRED:")
    print(f"   1. Review failed tasks in execution summary")
    print(f"   2. Experimental tasks (1, 4, 5) must provide MLflow run_id")
    print(f"   3. Documentation tasks (6, 7, 8) must provide code/doc files")
    print(f"   4. Re-execute failed tasks with proper evidence logging")
else:
    print(f"✅ EVIDENCE VERIFICATION PASSED")
    completed_count = len([t for t in tracker.task_results if t['status'] == 'completed'])
    print(f"   All {completed_count} completed tasks have verified evidence")
print("="*80)
```

---

## Key Changes

### 1. **Task Type Detection**
```python
EXPERIMENTAL_KEYWORDS = ['execute', 'run', 'diagnostic', 'experiment', 'test', ...]
NON_EXPERIMENTAL_KEYWORDS = ['design', 'draft', 'write', 'review', 'survey', ...]

def should_require_mlflow(task_action):
    # Returns True for experimental tasks, False for documentation tasks
```

### 2. **Conditional MLflow Requirement**
- **Experimental tasks** (Tasks 1, 4, 5): MUST have MLflow run_id
- **Documentation tasks** (Tasks 6, 7, 8): MLflow optional, focus on code/docs

### 3. **Clear Task Type Display**
```
📊 Task Type: EXPERIMENTAL (requires MLflow tracking)
📝 Task Type: DOCUMENTATION (requires code/docs, MLflow optional)
```

### 4. **Evidence Prioritization**
- **Experimental**: MLflow primary evidence, files secondary
- **Documentation**: Code/doc files primary evidence, MLflow optional

---

## Expected Behavior

### **Task 1: CLIP Integration** (EXPERIMENTAL)
```
🔍 Verifying Task 1: Integrate CLIP model into attention analysis framework...
   📊 Task Type: EXPERIMENTAL (requires MLflow tracking)
   ❌ No MLflow run_id found in responses (REQUIRED for experimental tasks)

   ❌ VERIFICATION FAILED: 1 issue
      - No MLflow tracking
```
**Status:** ❌ FAILED (correctly - needs real MLflow run)

### **Task 6: Statistical Framework Design** (DOCUMENTATION)
```
🔍 Verifying Task 6: Design statistical validation framework...
   📝 Task Type: DOCUMENTATION (requires code/docs, MLflow optional)
   ℹ️  No MLflow tracking (not required for documentation tasks)
   ✅ File: research/statistics/bootstrap.py (4521 bytes)
   ✅ File: research/statistics/power_analysis.md (3102 bytes)

   ✅ VERIFICATION PASSED
```
**Status:** ✅ PASS (has code files, MLflow not required)

### **Task 7: Paper Outline** (DOCUMENTATION)
```
🔍 Verifying Task 7: Draft CVPR paper outline...
   📝 Task Type: DOCUMENTATION (requires code/docs, MLflow optional)
   ℹ️  No MLflow tracking (not required for documentation tasks)
   ✅ File: paper/main.tex (8421 bytes)
   ✅ File: paper/sections/introduction.tex (5203 bytes)

   ✅ VERIFICATION PASSED
```
**Status:** ✅ PASS (has LaTeX files, MLflow not required)

### **Task 8: Literature Review** (DOCUMENTATION)
```
🔍 Verifying Task 8: Literature review: Survey multimodal fusion...
   📝 Task Type: DOCUMENTATION (requires code/docs, MLflow optional)
   ℹ️  No MLflow tracking (not required for documentation tasks)
   ✅ File: paper/literature_review.md (12430 bytes)
   ✅ File: paper/references.bib (8921 bytes)

   ✅ VERIFICATION PASSED
```
**Status:** ✅ PASS (has review docs, MLflow not required)

---

## How to Update Colab Notebook

**Option 1: Manual Copy-Paste** (Recommended)
1. Open your Colab notebook
2. Find Cell 16 (Phase 5.5 Evidence Verification)
3. Delete all existing content
4. Copy the entire code block from "Updated Cell 16 Code" section above
5. Paste into Cell 16
6. Run the cell to verify syntax

**Option 2: Programmatic Update** (if you prefer)
Use the Python script in the next section to update the notebook JSON.

---

## Testing

After updating, re-run Cycle 1 to verify:

1. **Tasks 1, 4, 5** (experimental): Should FAIL if no MLflow run_id
2. **Tasks 6, 7, 8** (documentation): Should PASS if code/doc files exist
3. **Task type detection**: Check console output shows correct task type

---

**Status:** ✅ Ready to deploy
**Next:** Update `pending_actions.json` to continue with experimental tasks
