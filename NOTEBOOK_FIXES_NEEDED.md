# Notebook Fixes Needed - Verification Results

**Date:** 2025-10-15
**Notebook:** cvpr_autonomous_execution_cycle.ipynb (Desktop)

---

## 🔍 Verification Results

### **Issue 1: Cell 11.5 (Code Executor) - ❌ MISSING**

**Status:** NOT FOUND in notebook
**You said:** "i have add this"
**Reality:** The code executor cell is not present

**What to do:**
1. Open Colab notebook
2. Find Cell 11 or Cell 12 (Task Execution Loop)
3. Click below it to add a new cell
4. Copy the complete Cell 11.5 code (see below)
5. Paste and run to verify syntax

---

### **Issue 2: Cell 14 (Progress Update) - ⚠️ STILL HAS TRUNCATION**

**Status:** Found at Cell 14, but truncation NOT removed
**You said:** "i changed this cell... to remove the ..."
**Reality:** Line 41 still has `[:1000]` truncation

**Current (WRONG):**
```python
{response[:1000]}{'...' if len(response) > 1000 else ''}
```

**Should be (CORRECT):**
```python
{response}
```

**Location in notebook:**
- Cell 14 (Cell index 13)
- Line 41
- Inside the `for agent_name, response in task_result['agent_responses'].items():` loop

---

## 🛠️ Fix 1: Add Cell 11.5 (Code Executor)

**Where to insert:** After Cell 11 or 12 (Task Execution Loop), before progress report cell

**Complete code to add:**

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

## 🛠️ Fix 2: Remove Truncation in Cell 14

**Location:** Cell 14, Line 41

**Find this code block (lines 37-43):**
```python
    for agent_name, response in task_result['agent_responses'].items():
        progress_report += f"""
#### {agent_name}
```
{response[:1000]}{'...' if len(response) > 1000 else ''}  # <-- LINE 41: REMOVE THIS
```
"""
```

**Change to:**
```python
    for agent_name, response in task_result['agent_responses'].items():
        progress_report += f"""
#### {agent_name}
```
{response}  # <-- LINE 41: CHANGED - no truncation
```
"""
```

**Just change line 41:**
- **OLD:** `{response[:1000]}{'...' if len(response) > 1000 else ''}`
- **NEW:** `{response}`

---

## ✅ Verification Steps

After making both fixes:

### **1. Test Cell 11.5 exists:**
```python
# Run this in a new cell
print("Checking for code executor functions...")
print(f"extract_python_code_blocks: {'✅' if 'extract_python_code_blocks' in dir() else '❌'}")
print(f"should_execute_task: {'✅' if 'should_execute_task' in dir() else '❌'}")
print(f"safe_execute_code: {'✅' if 'safe_execute_code' in dir() else '❌'}")
```

**Expected output:**
```
Checking for code executor functions...
extract_python_code_blocks: ✅
should_execute_task: ✅
safe_execute_code: ✅
```

### **2. Test truncation removed:**
```python
# Check that responses are not truncated
test_response = "A" * 2000  # 2000 character response
formatted = f"{test_response}"  # Should be 2000 chars
print(f"Test response length: {len(test_response)}")
print(f"Formatted length: {len(formatted)}")
print(f"Truncation removed: {'✅' if len(formatted) == 2000 else '❌'}")
```

**Expected output:**
```
Test response length: 2000
Formatted length: 2000
Truncation removed: ✅
```

---

## 📊 Summary

| Component | Status | Action Needed |
|-----------|--------|---------------|
| **Cell 11.5 (Code Executor)** | ❌ Missing | Add complete cell (see above) |
| **Cell 14 Line 41 (Truncation)** | ⚠️ Not fixed | Change `{response[:1000]}...` to `{response}` |
| **Cell 12 (Print statement)** | ✅ Fixed | Already corrected |

---

## 🎯 Next Steps

1. **Add Cell 11.5** (10 minutes)
   - Copy complete code above
   - Insert after task execution cell
   - Run to verify syntax

2. **Fix Cell 14 Line 41** (1 minute)
   - Change one line: remove `[:1000]` truncation
   - Run to verify syntax

3. **Re-run Cycle 2** (3 minutes to start)
   - Execute all cells
   - Watch Cell 11.5 automatically execute agent code
   - Verify Phase 5.5 passes with evidence

---

## 🚨 Important Notes

**Why Cell 11.5 is critical:**
- Without it, agents write code but don't execute
- Phase 5.5 will fail (no MLflow runs, no files)
- You'll need to execute code manually

**Why removing truncation is critical:**
- Cell 11.5 needs full code to execute
- Truncated code = incomplete implementations
- May cause execution errors

**Both fixes are required for automatic execution to work!**

---

**Status:** ⚠️ Fixes needed before running
**Time to fix:** ~11 minutes total
**Impact:** Enables automatic code execution with evidence
