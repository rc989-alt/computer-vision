# Executive Team Execution Notebook - Phase 6 Cell

**File:** `cvpr_autonomous_execution_cycle.ipynb`
**Cell:** Phase 6 - Trigger Next Planning Team Meeting (Manual Checkpoint)

---

## Cell 17 (Markdown Header)

```markdown
---
## Phase 6: Trigger Next Planning Team Meeting (Manual Checkpoint)
```

---

## Cell 18 (Code) - CORRECT VERSION

**Purpose:** Create a trigger file for the Planning Team with execution results summary

```python
# Create trigger for next Planning Team meeting
import json
from datetime import datetime
from pathlib import Path

print("="*80)
print("⏸️  MANUAL CHECKPOINT - READY FOR PLANNING TEAM REVIEW")
print("="*80)

# Get execution summary
summary = tracker.get_summary()

# Create next meeting trigger
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

next_meeting_trigger = {
    'trigger_type': 'executive_team_complete',
    'timestamp': datetime.now().isoformat(),
    'cycle_number': 1,  # Increment this for each cycle
    'executive_summary': {
        'total_tasks': summary['total_tasks'],
        'completed': summary['completed'],
        'failed': summary['failed'],
        'duration_seconds': summary['total_duration_seconds']
    },
    'next_meeting': {
        'team': 'planning',
        'purpose': 'Review Week 1 execution results and plan next cycle',
        'required_inputs': [
            f'execution_progress_update_{timestamp}.md',
            f'execution_results_{timestamp}.json'
        ],
        'agenda': [
            'Review task completion status',
            'Assess progress toward Week 1 GO/NO-GO criteria',
            'Identify blockers and risks',
            'Plan next cycle tasks (if needed)',
            'Generate new pending_actions.json'
        ]
    },
    'manual_checkpoint': True,
    'checkpoint_message': '⏸️ MANUAL REVIEW REQUIRED: Check execution results before starting next Planning Team meeting'
}

# Save trigger file with timestamp
trigger_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/next_meeting_trigger_{timestamp}.json'
trigger_file.write_text(json.dumps(next_meeting_trigger, indent=2), encoding='utf-8')

print(f"\n✅ Next meeting trigger saved:")
print(f"   📄 File: next_meeting_trigger_{timestamp}.json")
print(f"   📊 Tasks completed: {summary['completed']}/{summary['total_tasks']}")
print(f"   ⏱️  Duration: {summary['total_duration_seconds']:.1f}s ({summary['total_duration_seconds']/60:.1f} minutes)")

print(f"\n📋 Planning Team Agenda:")
for i, item in enumerate(next_meeting_trigger['next_meeting']['agenda'], 1):
    print(f"   {i}. {item}")

print(f"\n⏸️  MANUAL CHECKPOINT:")
print(f"   ⚠️  {next_meeting_trigger['checkpoint_message']}")
print(f"\n🔄 Next Steps:")
print(f"   1. Review execution results in reports/handoff/")
print(f"   2. If satisfied, run Planning Team notebook: planning_team_review_cycle.ipynb")
print(f"   3. Planning Team will read these results and plan next cycle")
print(f"   4. Execute next cycle with updated pending_actions.json")

print("\n" + "="*80)
print("✅ EXECUTIVE TEAM CYCLE COMPLETE")
print("="*80)
print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results saved to: reports/handoff/")
print(f"Ready for Planning Team review")
print("="*80)
```

---

## Alternative: Non-Timestamped Version (Original)

If you want the simpler version without timestamps:

```python
# Create trigger for next Planning Team meeting
import json
from datetime import datetime
from pathlib import Path

print("="*80)
print("⏸️  MANUAL CHECKPOINT - READY FOR PLANNING TEAM REVIEW")
print("="*80)

summary = tracker.get_summary()

next_meeting_trigger = {
    'trigger_type': 'executive_team_complete',
    'timestamp': datetime.now().isoformat(),
    'cycle_number': 1,
    'executive_summary': {
        'total_tasks': summary['total_tasks'],
        'completed': summary['completed'],
        'failed': summary['failed'],
        'duration_seconds': summary['total_duration_seconds']
    },
    'next_meeting': {
        'team': 'planning',
        'purpose': 'Review execution results and plan next cycle',
        'required_inputs': [
            'execution_progress_update.md',
            'execution_results.json'
        ],
        'agenda': [
            'Review task completion status',
            'Assess progress toward Week 1 GO/NO-GO criteria',
            'Identify blockers and risks',
            'Plan next cycle tasks',
            'Generate new pending_actions.json'
        ]
    },
    'manual_checkpoint': True,
    'checkpoint_message': '⏸️ MANUAL REVIEW REQUIRED: Check execution results before starting Planning Team'
}

# Save trigger
trigger_file = Path(MULTI_AGENT_ROOT) / 'reports/handoff/next_meeting_trigger.json'
trigger_file.write_text(json.dumps(next_meeting_trigger, indent=2), encoding='utf-8')

print(f"\n✅ Next meeting trigger saved")
print(f"   Tasks completed: {summary['completed']}/{summary['total_tasks']}")
print(f"   Duration: {summary['total_duration_seconds']:.1f}s")

print(f"\n⏸️  MANUAL CHECKPOINT:")
print(f"   Review results in reports/handoff/")
print(f"   Then run: planning_team_review_cycle.ipynb")

print("\n" + "="*80)
print("✅ EXECUTIVE TEAM CYCLE COMPLETE")
print("="*80)
```

---

## Expected Output

When Cell 18 runs, you should see:

```
================================================================================
⏸️  MANUAL CHECKPOINT - READY FOR PLANNING TEAM REVIEW
================================================================================

✅ Next meeting trigger saved:
   📄 File: next_meeting_trigger_20251015_005911.json
   📊 Tasks completed: 8/8
   ⏱️  Duration: 845.0s (14.1 minutes)

📋 Planning Team Agenda:
   1. Review task completion status
   2. Assess progress toward Week 1 GO/NO-GO criteria
   3. Identify blockers and risks
   4. Plan next cycle tasks (if needed)
   5. Generate new pending_actions.json

⏸️  MANUAL CHECKPOINT:
   ⚠️  MANUAL REVIEW REQUIRED: Check execution results before starting next Planning Team meeting

🔄 Next Steps:
   1. Review execution results in reports/handoff/
   2. If satisfied, run Planning Team notebook: planning_team_review_cycle.ipynb
   3. Planning Team will read these results and plan next cycle
   4. Execute next cycle with updated pending_actions.json

================================================================================
✅ EXECUTIVE TEAM CYCLE COMPLETE
================================================================================

Execution completed at: 2025-10-15 00:59:11
Results saved to: reports/handoff/
Ready for Planning Team review
================================================================================
```

---

## What This Cell Does

1. **Gets execution summary** from the task tracker
2. **Creates a trigger JSON file** with:
   - Execution results summary (tasks, duration)
   - Planning Team agenda
   - Required input files
   - Manual checkpoint message
3. **Saves to handoff directory** with timestamp
4. **Displays checkpoint message** for manual review
5. **Shows next steps** for Planning Team meeting

---

## Files Created by This Cell

```
reports/handoff/
└── next_meeting_trigger_20251015_005911.json
```

**Content example:**
```json
{
  "trigger_type": "executive_team_complete",
  "timestamp": "2025-10-15T00:59:11.567835",
  "cycle_number": 1,
  "executive_summary": {
    "total_tasks": 8,
    "completed": 8,
    "failed": 0,
    "duration_seconds": 845.0
  },
  "next_meeting": {
    "team": "planning",
    "purpose": "Review Week 1 execution results and plan next cycle",
    "required_inputs": [
      "execution_progress_update_20251015_005911.md",
      "execution_results_20251015_005911.json"
    ],
    "agenda": [
      "Review task completion status",
      "Assess progress toward Week 1 GO/NO-GO criteria",
      "Identify blockers and risks",
      "Plan next cycle tasks (if needed)",
      "Generate new pending_actions.json"
    ]
  },
  "manual_checkpoint": true,
  "checkpoint_message": "⏸️ MANUAL REVIEW REQUIRED: Check execution results before starting next Planning Team meeting"
}
```

---

## How to Add Back to Notebook

**In Google Colab:**

1. Navigate to where Cell 18 should be (after Cell 17: Phase 6 header)
2. If Cell 18 exists but has wrong content, delete it first
3. Click "+ Code" to add a new code cell
4. Copy and paste the code from above (choose timestamped or non-timestamped version)
5. Save the notebook (Ctrl/Cmd + S)

---

## Important Notes

- **`tracker` must exist:** This cell uses `tracker.get_summary()` from Phase 3
- **Depends on previous phases:** Make sure Phases 1-5 ran successfully
- **Manual checkpoint:** This pauses execution for you to review before Planning Team
- **Timestamp version recommended:** Keeps full history of all cycles

---

## Quick Fix Version

Minimal code if you just need it working:

```python
import json
from datetime import datetime
from pathlib import Path

summary = tracker.get_summary()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

trigger = {
    'trigger_type': 'executive_team_complete',
    'timestamp': datetime.now().isoformat(),
    'executive_summary': {
        'total_tasks': summary['total_tasks'],
        'completed': summary['completed'],
        'failed': summary['failed']
    },
    'manual_checkpoint': True
}

trigger_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/next_meeting_trigger_{timestamp}.json'
trigger_file.write_text(json.dumps(trigger, indent=2))

print("✅ Trigger saved. Ready for Planning Team review.")
print(f"   Completed: {summary['completed']}/{summary['total_tasks']} tasks")
```

---

**Choose your version:**
- **Timestamped (Recommended):** Use first code block - keeps full history
- **Non-timestamped (Original):** Use alternative version - simpler but overwrites
- **Minimal:** Use quick fix - bare minimum to work
