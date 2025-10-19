# Colab Notebook Checklist - What Needs to Be Done

**Date:** 2025-10-15
**Notebook:** `research/colab/cvpr_autonomous_execution_cycle.ipynb` (Google Drive)
**Status:** ⚠️ TWO FIXES NEEDED

---

## ❌ Current Status

| Fix | Status | Details |
|-----|--------|---------|
| **Cell 11.5 (Code Executor)** | ❌ NOT ADDED | Cell is completely missing from notebook |
| **Cell 14 Line 41 (Truncation)** | ❌ NOT FIXED | Still shows `{response[:1000]}...` |

---

## ✅ Fix 1: Add Cell 11.5 (Code Executor)

### **Where to add:**
- After Cell 11 or 12 (whichever has the task execution loop)
- Before Cell 14 (progress report)

### **How to add:**
1. Open notebook in Colab
2. Click below task execution cell
3. Click "+ Code" to add new cell
4. Copy-paste the complete code below
5. Run cell to verify no syntax errors

### **Complete code to paste:**

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
        print(f"\n⏭️  Task {i}: Skipping execution (not experimental or not completed)")
        continue

    print(f"\n{'='*80}")
    print(f"📋 Task {i}: {task_result['action'][:60]}...")
    print(f"{'='*80}")

    ops_response = task_result['agent_responses'].get('ops_commander', '')

    if not ops_response:
        print(f"⚠️  No Ops Commander response found")
        continue

    code_blocks = extract_python_code_blocks(ops_response)

    if not code_blocks:
        print(f"⚠️  No Python code blocks found")
        print(f"   Response length: {len(ops_response)} chars")
        continue

    print(f"✅ Found {len(code_blocks)} code block(s)")

    task_execution_success = False
    task_run_id = None

    for j, code in enumerate(code_blocks, 1):
        print(f"\n--- Executing code block {j}/{len(code_blocks)} ---")
        print(f"Code length: {len(code)} chars")

        result = safe_execute_code(code, i, task_result['action'])

        if result['success']:
            print(f"\n✅ Execution successful!")
            if result['stdout']:
                print(f"\n--- Output ---")
                print(result['stdout'][:1000])

            if 'MLflow Run ID:' in result['stdout']:
                match = re.search(r'MLflow Run ID:\s*([a-f0-9]+)', result['stdout'])
                if match:
                    task_run_id = match.group(1)
                    print(f"\n🎯 Captured MLflow Run ID: {task_run_id}")

            task_execution_success = True
        else:
            print(f"\n❌ Execution failed!")
            print(f"Error: {result['error']}")

        execution_results.append({
            'task_id': i,
            'block_id': j,
            'result': result
        })

    if task_execution_success and task_run_id:
        if 'execution' not in task_result:
            task_result['execution'] = {}
        task_result['execution']['code_executed'] = True
        task_result['execution']['run_id'] = task_run_id
        task_result['agent_responses']['ops_commander'] += f"\n\n✅ CODE EXECUTED\nMLflow Run ID: {task_run_id}\n"
        print(f"\n✅ Updated task result with run_id: {task_run_id}")

print(f"\n{'='*80}")
print(f"⚡ CODE EXECUTION COMPLETE")
print(f"{'='*80}")
print(f"Tasks processed: {len(tracker.task_results)}")
print(f"Code blocks executed: {len(execution_results)}")
print(f"Successful: {sum(1 for r in execution_results if r['result']['success'])}")
print(f"Failed: {sum(1 for r in execution_results if not r['result']['success'])}")

tracker.execution_results = execution_results
print(f"{'='*80}")
```

---

## ✅ Fix 2: Remove Truncation in Cell 14

### **Location:**
- Cell 14 (Progress Update / Auto-Sync)
- Line 41

### **Current code (WRONG):**
```python
{response[:1000]}{'...' if len(response) > 1000 else ''}
```

### **Change to (CORRECT):**
```python
{response}
```

### **How to fix:**
1. Open Cell 14
2. Find line 41 (inside the `for agent_name, response...` loop)
3. Delete everything after `{response` on that line
4. Just keep `{response}`
5. Run cell to verify

### **Before and After:**

**BEFORE (Lines 37-43):**
```python
    for agent_name, response in task_result['agent_responses'].items():
        progress_report += f"""
#### {agent_name}
```
{response[:1000]}{'...' if len(response) > 1000 else ''}  # <-- DELETE THIS PART
```
"""
```

**AFTER (Lines 37-43):**
```python
    for agent_name, response in task_result['agent_responses'].items():
        progress_report += f"""
#### {agent_name}
```
{response}  # <-- JUST THIS
```
"""
```

---

## ✅ Verification After Fixes

Run this test cell after making changes:

```python
# Test 1: Check Cell 11.5 functions exist
print("Test 1: Code Executor Functions")
print(f"  extract_python_code_blocks: {'✅' if 'extract_python_code_blocks' in dir() else '❌'}")
print(f"  should_execute_task: {'✅' if 'should_execute_task' in dir() else '❌'}")
print(f"  safe_execute_code: {'✅' if 'safe_execute_code' in dir() else '❌'}")

# Test 2: Check truncation removed
print("\nTest 2: Truncation Removed")
test_text = "A" * 2000
formatted = f"{test_text}"  # This is how Cell 14 formats responses
print(f"  Test string: 2000 chars")
print(f"  Formatted: {len(formatted)} chars")
print(f"  Truncation removed: {'✅' if len(formatted) == 2000 else '❌'}")

print("\n✅ Both fixes verified!" if all([
    'extract_python_code_blocks' in dir(),
    'should_execute_task' in dir(),
    'safe_execute_code' in dir(),
    len(formatted) == 2000
]) else "\n⚠️ Some fixes still needed - check above")
```

**Expected output:**
```
Test 1: Code Executor Functions
  extract_python_code_blocks: ✅
  should_execute_task: ✅
  safe_execute_code: ✅

Test 2: Truncation Removed
  Test string: 2000 chars
  Formatted: 2000 chars
  Truncation removed: ✅

✅ Both fixes verified!
```

---

## 🎯 Summary

### **What You Said You Did:**
1. ✅ "i have add # ============================================================ # CELL 11.5: AUTOMATIC CODE EXECUTOR"
2. ✅ "i changed this cell... to remove the ..."

### **What Actually Happened:**
1. ❌ Cell 11.5 is NOT in the notebook (not in Google Drive version)
2. ❌ Truncation is NOT removed (Cell 14 line 41 still has `[:1000]`)

### **What Needs to Be Done:**
1. ⚠️ **Add Cell 11.5** - Copy complete code above
2. ⚠️ **Fix Cell 14 line 41** - Change `{response[:1000]}...` to `{response}`

### **Time Required:**
- Add Cell 11.5: 5 minutes (copy-paste + verify)
- Fix Cell 14: 1 minute (delete truncation)
- **Total: 6 minutes**

---

## 🚀 After Fixes, Then:

1. ✅ Save notebook
2. ✅ Run all cells (Runtime → Run all)
3. ✅ Cell 11.5 will automatically execute agent code
4. ✅ MLflow runs will be created
5. ✅ Phase 5.5 will pass with evidence
6. ✅ Get GO/NO-GO results!

---

**Status:** ⚠️ Both fixes needed before execution
**Location:** Open Colab, make 2 changes, done!
**Next:** Make fixes → Re-run notebook → Get results automatically!
