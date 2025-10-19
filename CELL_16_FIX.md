# Cell 16 (Phase 5.5) - Corrected Lines

Replace lines 873-900 in Cell 16 with this corrected version:

```python
for i, task in enumerate(tracker.task_results, 1):  # ✅ FIXED: tracker.task_results
    if task['status'] == "completed":  # ✅ FIXED: task['status']
        verified = verify_task_evidence(
            i,
            task['action'],  # ✅ FIXED: task['action']
            task
        )

        if not verified:
            verification_failures.append(i)
            # CRITICAL: Downgrade task status to FAILED
            task['status'] = "failed"  # ✅ FIXED: task['status']
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
    print(f"   2. Do NOT proceed to Planning Team meeting")
    print(f"   3. Re-execute failed tasks with proper evidence logging")
else:
    print(f"✅ EVIDENCE VERIFICATION PASSED")
    completed_count = len([t for t in tracker.task_results if t['status'] == 'completed'])  # ✅ FIXED
    print(f"   All {completed_count} completed tasks have verified evidence")
print("="*80)
```

## Changes Made:

1. Line 873: `tracker.tasks` → `tracker.task_results`
2. Line 874: `task.status` → `task['status']`
3. Line 877: `task.action` → `task['action']`
4. Line 884: `task.status` → `task['status']`
5. Line 900: `tracker.tasks` → `tracker.task_results` AND `t.status` → `t['status']`

## How to Apply in Colab:

1. Open the notebook in Google Colab
2. Find **Cell 16** (the one starting with "# PHASE 5.5: EVIDENCE VERIFICATION")
3. Locate the loop starting with `for i, task in enumerate(tracker.tasks, 1):`
4. Replace the entire loop section (lines 873-900) with the corrected code above
5. **Cell 11 does NOT need changes** - it's not in the pasted content, which means it's likely already correct

## Verification:

After applying the fix, run the cell and verify:
- No `AttributeError` about `tracker.tasks`
- No `AttributeError` about `task.status` or `task.action`
- Evidence verification runs successfully
