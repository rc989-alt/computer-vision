# Code Executor Deployment Guide - Complete Solution

**Date:** 2025-10-15
**Purpose:** Deploy automatic code execution system to fix "agents write but don't execute" problem
**Time:** 15 minutes deployment + 3 hours execution
**Status:** ✅ READY TO DEPLOY

---

## 🎯 What This Solves

**Problem:** Agents write excellent Python code but don't execute it
- Ops Commander shows: `mlflow.start_run()` (CODE)
- Phase 5.5 expects: `MLflow Run ID: abc123...` (OUTPUT)
- Result: ❌ Phase 5.5 correctly rejects tasks for lacking evidence

**Solution:** Automatic code executor (Cell 11.5)
- Extracts Python code from agent responses
- Executes code automatically in Colab
- Captures run_id and adds to task results
- Phase 5.5 now finds evidence → ✅ PASSES

---

## 📋 Two Changes Needed

### **Change 1: Add Cell 11.5 (Code Executor)**
- Extracts and executes agent code
- 80 lines of Python
- Insert between Cell 11 and Cell 12

### **Change 2: Update Cell 15 (Remove Truncation)**
- Remove 1000-char limit on agent responses
- Change 1 line: `response[:1000]` → `response`
- Allows Cell 11.5 to access full code

---

## 🚀 Quick Start (Copy-Paste Ready)

### **Step 1: Update Cell 15 (2 minutes)**

**Find this line (~line 48):**
```python
{response[:1000]}{'...' if len(response) > 1000 else ''}
```

**Change to:**
```python
{response}
```

**Full context (for finding the right location):**
```python
for agent_name, response in task_result['agent_responses'].items():
    progress_report += f"""
#### {agent_name}
```
{response}  # <-- CHANGED: removed [:1000] truncation
```
"""
```

---

### **Step 2: Add Cell 11.5 (5 minutes)**

**Insert new cell after Cell 11:**

1. Click below Cell 11 (Task Execution Loop)
2. Click "+ Code" to add new cell
3. Copy-paste the complete code from `CELL_11.5_CODE_EXECUTOR.md`

**Or use this complete cell:**

```python
# ============================================================
# CELL 11.5: AUTOMATIC CODE EXECUTOR
# Extracts and executes Python code from Ops Commander responses
# ============================================================

print("="*80)
print("⚡ AUTOMATIC CODE EXECUTOR - EXECUTING AGENT IMPLEMENTATIONS")
print("="*80)

import re
import sys
from io import StringIO
from pathlib import Path
import traceback

def extract_python_code_blocks(text):
    """Extract all Python code blocks from markdown-formatted text"""
    pattern1 = r'```python\n(.*?)```'
    pattern2 = r'```\n(.*?)```'

    blocks = []
    matches = re.findall(pattern1, text, re.DOTALL)
    blocks.extend(matches)

    if not matches:
        matches = re.findall(pattern2, text, re.DOTALL)
        blocks.extend(matches)

    return blocks

def should_execute_task(task_result):
    """Determine if a task should have code executed"""
    if task_result['status'] != 'completed':
        return False

    action = task_result.get('action', '').lower()
    experimental_keywords = [
        'execute', 'run', 'diagnostic', 'experiment', 'test',
        'baseline', 'gpu', 'model', 'integration', 'attention'
    ]

    for keyword in experimental_keywords:
        if keyword in action:
            return True

    return False

def safe_execute_code(code, task_id, task_action):
    """Execute code in controlled environment with output capture"""
    print(f"\n{'='*80}")
    print(f"🔧 Executing code for Task {task_id}")
    print(f"   Action: {task_action[:60]}...")
    print(f"{'='*80}")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    execution_result = {
        'success': False,
        'stdout': '',
        'stderr': '',
        'error': None,
        'run_id': None
    }

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        exec_globals = {
            '__name__': '__main__',
            'MULTI_AGENT_ROOT': MULTI_AGENT_ROOT,
            'Path': Path,
        }

        exec(code, exec_globals)
        execution_result['success'] = True

        if 'run_id' in exec_globals:
            execution_result['run_id'] = exec_globals['run_id']

    except Exception as e:
        execution_result['error'] = str(e)
        execution_result['traceback'] = traceback.format_exc()

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        execution_result['stdout'] = stdout_capture.getvalue()
        execution_result['stderr'] = stderr_capture.getvalue()

    return execution_result

# Execute code for each completed experimental task
execution_results = []

for i, task_result in enumerate(tracker.task_results, 1):
    if not should_execute_task(task_result):
        print(f"\n⏭️  Task {i}: Skipping execution (not an experimental task or not completed)")
        continue

    print(f"\n{'='*80}")
    print(f"📋 Task {i}: {task_result['action'][:60]}...")
    print(f"{'='*80}")

    ops_response = task_result['agent_responses'].get('ops_commander', '')

    if not ops_response:
        print(f"⚠️  No Ops Commander response found, skipping execution")
        continue

    code_blocks = extract_python_code_blocks(ops_response)

    if not code_blocks:
        print(f"⚠️  No Python code blocks found in response")
        print(f"   Response length: {len(ops_response)} chars")
        print(f"   Response preview: {ops_response[:200]}...")
        continue

    print(f"✅ Found {len(code_blocks)} code block(s)")

    task_execution_success = False
    task_run_id = None

    for j, code in enumerate(code_blocks, 1):
        print(f"\n--- Executing code block {j}/{len(code_blocks)} ---")
        print(f"Code length: {len(code)} chars")
        print(f"First 3 lines:")
        for line in code.split('\n')[:3]:
            print(f"  {line}")

        result = safe_execute_code(code, i, task_result['action'])

        if result['success']:
            print(f"\n✅ Execution successful!")
            if result['stdout']:
                print(f"\n--- Output ---")
                print(result['stdout'][:1000])
                if len(result['stdout']) > 1000:
                    print(f"... (truncated, total {len(result['stdout'])} chars)")

            if 'MLflow Run ID:' in result['stdout']:
                match = re.search(r'MLflow Run ID:\s*([a-f0-9]+)', result['stdout'])
                if match:
                    task_run_id = match.group(1)
                    print(f"\n🎯 Captured MLflow Run ID: {task_run_id}")

            task_execution_success = True

        else:
            print(f"\n❌ Execution failed!")
            print(f"Error: {result['error']}")
            if result['stderr']:
                print(f"\n--- Error Output ---")
                print(result['stderr'][:500])
            if result.get('traceback'):
                print(f"\n--- Traceback ---")
                print(result['traceback'][:1000])

        execution_results.append({
            'task_id': i,
            'block_id': j,
            'result': result
        })

    if task_execution_success:
        if 'execution' not in task_result:
            task_result['execution'] = {}

        task_result['execution']['code_executed'] = True
        task_result['execution']['run_id'] = task_run_id

        if task_run_id:
            task_result['agent_responses']['ops_commander'] += f"\n\n✅ CODE EXECUTED SUCCESSFULLY\nMLflow Run ID: {task_run_id}\n"
            print(f"\n✅ Updated task result with run_id: {task_run_id}")

print(f"\n{'='*80}")
print(f"⚡ CODE EXECUTION COMPLETE")
print(f"{'='*80}")
print(f"Tasks processed: {len(tracker.task_results)}")
print(f"Code blocks executed: {len(execution_results)}")
print(f"Successful executions: {sum(1 for r in execution_results if r['result']['success'])}")
print(f"Failed executions: {sum(1 for r in execution_results if not r['result']['success'])}")

tracker.execution_results = execution_results

print(f"{'='*80}")
```

---

### **Step 3: Test (5 minutes)**

Run this test cell after Cell 11.5:

```python
# Test code extraction and execution
test_code = """
Here's a test implementation:

```python
import mlflow
mlflow.set_tracking_uri(str(MULTI_AGENT_ROOT / "mlruns"))
mlflow.set_experiment("test_executor")

with mlflow.start_run(run_name="test") as run:
    run_id = run.info.run_id
    print(f"✅ MLflow Run ID: {run_id}")
    mlflow.log_param("test_param", "test_value")
    mlflow.log_metric("test_metric", 42.0)
```

This creates a test run.
"""

# Test extraction
blocks = extract_python_code_blocks(test_code)
print(f"✅ Extracted {len(blocks)} code blocks")
print(f"Code preview: {blocks[0][:100]}...")

# Test execution
result = safe_execute_code(blocks[0], 999, "Test execution")
print(f"\n✅ Execution result: {result['success']}")
if result['success']:
    print(f"Output: {result['stdout'][:200]}")
```

**Expected output:**
```
✅ Extracted 1 code blocks
Code preview: import mlflow
mlflow.set_tracking_uri(str(MULTI_AGENT_ROOT / "mlruns"))
...

✅ Execution result: True
Output: ✅ MLflow Run ID: abc123def456...
```

---

## 🔍 Verification Steps

After deploying, re-run Cycle 2:

### **1. Check Cell 11.5 Output**
```
================================================================================
⚡ AUTOMATIC CODE EXECUTOR - EXECUTING AGENT IMPLEMENTATIONS
================================================================================

📋 Task 1: Re-execute Task 1: CLIP Integration...
✅ Found 3 code block(s)
--- Executing code block 1/3 ---
✅ Execution successful!
🎯 Captured MLflow Run ID: abc123def456

📋 Task 2: Execute Task 4: CLIP Diagnostic...
✅ Found 1 code block(s)
--- Executing code block 1/1 ---
✅ Execution successful!
🎯 Captured MLflow Run ID: def456ghi789

⚡ CODE EXECUTION COMPLETE
Tasks processed: 3
Code blocks executed: 5
Successful executions: 5
Failed executions: 0
```

### **2. Check MLflow Directory**
```python
# Verify mlruns populated
mlflow_dir = MULTI_AGENT_ROOT / "mlruns"
print(f"MLflow experiments: {len(list(mlflow_dir.glob('*/')))} ")

# Verify run files
for exp in mlflow_dir.glob('*/'):
    if exp.name not in ['.trash', '0']:
        runs = list(exp.glob('*/'))
        print(f"  {exp.name}: {len(runs)} runs")
```

**Expected:**
```
MLflow experiments: 3
  week1_clip_integration: 1 runs
  week1_clip_diagnostic: 1 runs
  week1_align_or_fallback_diagnostic: 1 runs
```

### **3. Check Phase 5.5 Output**
```
================================================================================
📋 VERIFYING ALL COMPLETED TASKS (TASK-TYPE AWARE)
================================================================================

🔍 Verifying Task 1: Re-execute Task 1: CLIP Integration...
   📊 Task Type: EXPERIMENTAL (requires MLflow tracking)
   ✅ MLflow Run: abc123def456
   ✅ File: runs/clip_integration/baseline_attention.json (4521 bytes)

   ✅ VERIFICATION PASSED

🔍 Verifying Task 2: Execute Task 4: CLIP Diagnostic...
   📊 Task Type: EXPERIMENTAL (requires MLflow tracking)
   ✅ MLflow Run: def456ghi789
   ✅ File: runs/clip_diagnostic/mcs_results.json (3210 bytes)

   ✅ VERIFICATION PASSED

🔍 Verifying Task 3: Complete Task 5: ALIGN/CoCa...
   📊 Task Type: EXPERIMENTAL (requires MLflow tracking)
   ✅ MLflow Run: ghi789jkl012
   ✅ File: runs/align_diagnostic/mcs_results.json (2987 bytes)

   ✅ VERIFICATION PASSED

================================================================================
✅ EVIDENCE VERIFICATION PASSED
   All 3 completed tasks have verified evidence
================================================================================
```

---

## 🎯 Expected Results

### **Before Cell 11.5:**
- Agents write code ✅
- Code NOT executed ❌
- No MLflow runs created ❌
- Phase 5.5 fails ❌
- Manual execution needed ⚠️

### **After Cell 11.5:**
- Agents write code ✅
- Code automatically executed ✅
- Real MLflow runs created ✅
- Phase 5.5 passes ✅
- Fully automated ✅

---

## ⚡ Workflow After Deployment

**New execution flow:**

1. **Cell 1-10:** Setup, load tasks
2. **Cell 11:** 3-agent approval → Ops Commander writes code
3. **Cell 11.5:** ⚡ **EXECUTE code** → Create MLflow runs → Capture run_id
4. **Cell 12-15:** Generate summaries (with actual run_ids)
5. **Cell 16:** Phase 5.5 verification → ✅ PASSES (finds run_ids + files)
6. **Cell 17-21:** Final reports, handoff

**Result:** Fully automated, end-to-end execution with evidence!

---

## 🚨 Troubleshooting

### **Issue: "No Python code blocks found"**
**Cause:** Agent didn't use ````python``` blocks
**Fix:** Update `extract_python_code_blocks()` to handle more formats

### **Issue: "Execution failed: ModuleNotFoundError"**
**Cause:** Missing package imports
**Fix:** Install packages in Cell 1:
```python
!pip install open_clip_torch pycocotools scikit-learn
```

### **Issue: "MLflow Run ID not captured"**
**Cause:** Code doesn't print run_id in expected format
**Fix:** Update regex to handle variations:
```python
# Current:
match = re.search(r'MLflow Run ID:\s*([a-f0-9]+)', result['stdout'])

# More flexible:
match = re.search(r'(?:run[-_\s]*id|Run ID):\s*([a-f0-9]{16,})', result['stdout'], re.IGNORECASE)
```

### **Issue: "Execution takes too long"**
**Cause:** CLIP diagnostic processes 150 samples (30+ minutes)
**Solution:** Add progress monitoring:
```python
# In safe_execute_code, add timeout:
import signal
signal.alarm(1800)  # 30 minute timeout
```

---

## 📊 Performance

**Cell 11.5 execution time:**
- Code extraction: <1 second per task
- Task 1 (CLIP Integration): 5-10 minutes
- Task 2 (CLIP Diagnostic): 30-40 minutes (150 samples)
- Task 3 (ALIGN/CoCa): 20-30 minutes (50 samples)

**Total:** ~60-90 minutes for all 3 tasks

**This is acceptable because:**
- Runs automatically (no manual work)
- Gets real statistical results
- Creates evidence for GO/NO-GO decision

---

## ✅ Deployment Checklist

- [ ] **Step 1:** Update Cell 15 (remove `[:1000]` truncation)
- [ ] **Step 2:** Add Cell 11.5 (code executor)
- [ ] **Step 3:** Test with simple code block
- [ ] **Step 4:** Re-run Cycle 2 execution (Cells 1-21)
- [ ] **Step 5:** Verify Cell 11.5 output shows executions
- [ ] **Step 6:** Verify `mlruns/` directory populated
- [ ] **Step 7:** Verify Phase 5.5 passes all 3 tasks
- [ ] **Step 8:** Check statistical results (p-values, MCS scores)
- [ ] **Step 9:** Make GO/NO-GO decision based on results
- [ ] **Step 10:** Prepare for October 20 meeting

---

## 🎉 Success Criteria

**System is working when:**
✅ Cell 11.5 executes agent code automatically
✅ MLflow runs created (check `mlruns/` directory)
✅ Result files created (check `runs/*/` directories)
✅ Phase 5.5 passes all experimental tasks
✅ Statistical evidence ready (p-values, CI95, Cohen's d)
✅ GO/NO-GO decision can be made

---

## 📁 Files Reference

**Created:**
- `CELL_11.5_CODE_EXECUTOR.md` - Complete Cell 11.5 code with documentation
- `CELL_15_REMOVE_TRUNCATION.md` - Instructions for removing truncation
- `CODE_EXECUTOR_DEPLOYMENT_GUIDE.md` - This file (deployment guide)

**Previous:**
- `CYCLE2_EXECUTION_ANALYSIS.md` - Diagnosis of the problem
- `IMMEDIATE_ACTION_PLAN.md` - Options considered
- `MANUAL_EXECUTION_GUIDE.md` - Manual execution alternative

**All synced to Google Drive** ✅

---

## 🏆 Bottom Line

**Problem:** Agents write code but don't execute → No evidence → Phase 5.5 fails

**Solution:** Cell 11.5 automatically executes agent code → Real runs created → Phase 5.5 passes

**Deployment:** 15 minutes (2 simple changes)

**Benefit:** Fully automated execution with real evidence for GO/NO-GO decision

**Status:** ✅ Ready to deploy now!

---

**Next Action:** Update Cell 15 → Add Cell 11.5 → Re-run Cycle 2 → Get results! 🚀
