# Cell 12 Fix - Restore Broken Print Statement

**Error Location:** Cell 12, Line 101
**Issue:** Print statement was accidentally truncated during editing

---

## The Error

**You have (BROKEN):**
```python
print(f"{response)")  # Missing closing and ternary expression!
```

**Should be (CORRECT):**
```python
print(f"{response[:500]}..." if len(response) > 500 else response)
```

---

## Complete Fixed Section (Lines 96-103)

Replace lines 96-103 in Cell 12 with:

```python
        # Log all agent responses
        for agent_name, response in responses.items():
            tracker.log_agent_response(agent_name, response)

            # Display response preview
            print(f"\n📝 Response from {agent_name}:")
            print(f"{response[:500]}..." if len(response) > 500 else response)
```

---

## What This Does

**Line 101:**
```python
print(f"\n📝 Response from {agent_name}:")
```
- Prints a header showing which agent is responding
- Uses f-string to insert agent_name

**Line 102:**
```python
print(f"{response[:500]}..." if len(response) > 500 else response)
```
- If response is longer than 500 chars → show first 500 + "..."
- If response is 500 chars or less → show full response
- This is a ternary expression (condition ? true_value : false_value)

---

## How to Fix in Colab

### **Option 1: Manual Edit (Recommended)**

1. Open Cell 12 in Colab
2. Find line ~101 (search for "Display response preview")
3. You'll see:
   ```python
   print(f"\n📝 Response from {agent_name}:")
   print(f"{response)")  # <-- BROKEN LINE
   ```
4. Replace the broken line with:
   ```python
   print(f"{response[:500]}..." if len(response) > 500 else response)
   ```
5. Run the cell to verify syntax (should show no errors)

---

### **Option 2: Replace Entire Cell 12**

If you want to be extra safe, here's the complete correct Cell 12 code:

```python
# Execute tasks in priority order
print("="*80)
print("🚀 EXECUTIVE TEAM EXECUTION - WEEK 1 TASKS")
print("="*80)

# Define 3-agent approval function BEFORE the loop
def determine_task_status(ops_response, quality_response, infra_response):
    """Determine task status from all three agent reports"""

    # Check Ops Commander claims
    ops_claims_complete = "COMPLETED" in ops_response

    # Check Quality & Safety assessment
    quality_passes = "❌ FAIL" not in quality_response and "✅ PASS" in quality_response

    # Check Infrastructure validation
    infra_valid = "❌" not in infra_response and "✅" in infra_response

    # ALL gates must pass
    if ops_claims_complete and quality_passes and infra_valid:
        return "completed"
    else:
        return "failed"

# Sort decisions by priority (HIGH → MEDIUM → LOW)
priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
sorted_decisions = sorted(
    pending_actions['decisions'],
    key=lambda x: priority_order.get(x.get('priority', 'LOW'), 3)
)

print(f"\n📋 Executing {len(sorted_decisions)} tasks in priority order")
print(f"   HIGH: {sum(1 for d in sorted_decisions if d.get('priority') == 'HIGH')}")
print(f"   MEDIUM: {sum(1 for d in sorted_decisions if d.get('priority') == 'MEDIUM')}")
print(f"   LOW: {sum(1 for d in sorted_decisions if d.get('priority') == 'LOW')}")
print()

# Execute each task
for task_id, decision in enumerate(sorted_decisions, 1):
    action = decision['action']
    priority = decision.get('priority', 'MEDIUM')
    rationale = decision.get('rationale', '')
    acceptance_criteria = decision.get('acceptance_criteria', [])

    # Start task tracking
    tracker.start_task(task_id, action, priority)

    print(f"\n{'='*80}")
    print(f"📋 TASK {task_id}/{len(sorted_decisions)}: {action[:60]}...")
    print(f"{'='*80}")
    print(f"Priority: {priority}")
    print(f"Rationale: {rationale[:100]}...")
    print(f"\nAcceptance Criteria ({len(acceptance_criteria)}):")
    for i, criterion in enumerate(acceptance_criteria[:3], 1):
        print(f"  {i}. {criterion}")
    if len(acceptance_criteria) > 3:
        print(f"  ... and {len(acceptance_criteria) - 3} more")

    # Prepare message for Executive Team (all 3 agents)
    message = {
        'task_id': task_id,
        'action': action,
        'priority': priority,
        'rationale': rationale,
        'acceptance_criteria': acceptance_criteria,
        'context': {
            'research_goal': pending_actions.get('research_goal', ''),
            'week1_objectives': pending_actions.get('week1_objectives', []),
            'mlflow_configuration': pending_actions.get('mlflow_configuration', {}),
            'evidence_requirements': decision.get('evidence_requirements', {})
        }
    }

    # Execute task with all 3 agents
    print(f"\n🎯 Routing to Executive Team (3-agent approval required)...")

    try:
        # Route to all 3 agents (Ops Commander, Quality & Safety, Infrastructure)
        responses = executive_router.route_message(message)

        # Verify all 3 agents responded
        expected_agents = ['ops_commander', 'quality_safety', 'infrastructure']
        missing_agents = [agent for agent in expected_agents if agent not in responses]

        if missing_agents:
            print(f"⚠️ Missing responses from: {', '.join(missing_agents)}")
            tracker.complete_task('failed')
            continue

        # Log all agent responses
        for agent_name, response in responses.items():
            tracker.log_agent_response(agent_name, response)

            # Display response preview
            print(f"\n📝 Response from {agent_name}:")
            print(f"{response[:500]}..." if len(response) > 500 else response)

        # Use 3-agent approval gate to determine task status
        ops_response = responses.get('ops_commander', '')
        quality_response = responses.get('quality_safety', '')
        infra_response = responses.get('infrastructure', '')

        status = determine_task_status(ops_response, quality_response, infra_response)
        tracker.complete_task(status)

        print(f"\n{'='*80}")
        print(f"✅ Task {task_id} {status.upper()}")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n❌ Error executing task {task_id}: {str(e)}")
        tracker.log_error(str(e))
        tracker.complete_task('failed')

print(f"\n{'='*80}")
print(f"🎉 EXECUTIVE TEAM EXECUTION COMPLETE")
print(f"{'='*80}")

# Summary
summary = tracker.get_summary()
print(f"\n📊 Execution Summary:")
print(f"   Total tasks: {summary['total_tasks']}")
print(f"   Completed: {summary['completed']} ✅")
print(f"   Failed: {summary['failed']} ❌")
print(f"   Duration: {summary['total_duration_seconds']:.1f}s")
```

---

## Testing After Fix

After fixing, run this verification:

```python
# Test that the print statement works
test_response = "This is a test response that is longer than 500 characters. " * 10

# Test short response
print(f"{test_response[:500]}..." if len(test_response) > 500 else test_response)

# Should print first 500 chars + "..."
```

---

## Why This Happened

**Likely scenario:**
1. You were editing Cell 12
2. Wanted to change the response display
3. Started typing: `print(f"{response)...`
4. Got interrupted or accidentally saved before finishing
5. The line got truncated

**The correct line uses a ternary expression:**
```python
value_if_true if condition else value_if_false
```

In this case:
```python
f"{response[:500]}..." if len(response) > 500 else response
```
Means: "If response is > 500 chars, show first 500 + '...', otherwise show full response"

---

## Quick Fix Summary

**Find this broken line in Cell 12, line ~101:**
```python
print(f"{response)")  # BROKEN
```

**Replace with:**
```python
print(f"{response[:500]}..." if len(response) > 500 else response)  # FIXED
```

**Then re-run the cell** to verify no syntax errors.

---

**Status:** ✅ Fix ready
**Time:** 1 minute to fix
**Impact:** Fixes agent response display

---

**After fixing this, you can proceed with adding Cell 11.5 (Code Executor) as planned!**
