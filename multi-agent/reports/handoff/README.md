# Handoff Directory

**Purpose:** Bridge between Planning Team and Executive Team

## What Goes Here

### `pending_actions.json` - PRIMARY HANDOFF FILE

This file is **created by the Planning Team** after each meeting and **read by the Executive Team** to know what to execute.

**Format:**
```json
{
  "timestamp": "2025-10-14T12:00:00",
  "meeting_id": "meeting_20251014_120000",
  "actions": [
    {
      "id": 1,
      "action": "Deploy V1.0 to shadow environment",
      "owner": "ops_commander",
      "priority": "high",
      "estimated_duration": "5min",
      "success_criteria": "Model deployed, no errors, smoke tests pass"
    },
    {
      "id": 2,
      "action": "Evaluate V1.0 on validation set",
      "owner": "latency_analyst",
      "priority": "high",
      "estimated_duration": "3min",
      "success_criteria": "NDCG@10 >= 0.72, latency < 50ms"
    },
    {
      "id": 3,
      "action": "Collect baseline metrics",
      "owner": "compliance_monitor",
      "priority": "medium",
      "estimated_duration": "2min",
      "success_criteria": "All SLOs measured and logged"
    }
  ],
  "count": 3,
  "source": "planning_meeting_2"
}
```

## Required Fields

Each action MUST have:

1. **`id`** - Unique identifier (integer)
2. **`action`** - **COMPLETE description** (not a fragment!)
   - ✅ Good: "Deploy V1.0 to shadow environment"
   - ❌ Bad: "Deploy to shadow" (incomplete)
   - ❌ Bad: "tracking in MLflow" (fragment)
3. **`owner`** - Which Executive Team agent executes
   - Valid: `ops_commander`, `infra_guardian`, `latency_analyst`, `compliance_monitor`, `integration_engineer`, `rollback_officer`
4. **`priority`** - Execution order
   - Valid: `high`, `medium`, `low`
   - Executive Team executes: HIGH → MEDIUM → LOW
5. **`estimated_duration`** - Time estimate
   - Format: "Xmin" (e.g., "5min", "10min")
6. **`success_criteria`** - How to verify completion
   - Must be specific and measurable
   - Example: "NDCG@10 >= 0.72, latency < 50ms"

## Workflow

### Planning Team (Creates pending_actions.json)

1. Hold strategic meeting
2. Discuss deployment decisions
3. Create action items with complete details
4. Write `pending_actions.json` to this directory
5. Enhanced Progress Sync reads it and groups by priority

### Enhanced Progress Sync (Processes pending_actions.json)

1. Reads `pending_actions.json`
2. Groups actions by priority (HIGH/MEDIUM/LOW)
3. Formats with complete details for each action
4. Writes `reports/execution_progress_update.md`

### Executive Team (Reads execution_progress_update.md)

1. Reads `reports/execution_progress_update.md`
2. Sees actions grouped by priority with emoji indicators:
   - 🔴 HIGH PRIORITY (Execute First)
   - 🟡 MEDIUM PRIORITY (Execute After High)
   - 🟢 LOW PRIORITY (Execute After Medium)
3. Executes in strict priority order
4. Each agent knows their assigned tasks via `owner` field
5. Verifies completion using `success_criteria`

## Priority Execution Order

```
Phase 1: 🔴 HIGH Priority
├── Execute all HIGH priority actions
├── Wait for all to complete
└── Verify success criteria

Phase 2: 🟡 MEDIUM Priority
├── Execute all MEDIUM priority actions
├── Wait for all to complete
└── Verify success criteria

Phase 3: 🟢 LOW Priority
├── Execute all LOW priority actions
├── Wait for all to complete
└── Verify success criteria

✅ All actions complete
```

## File Lifecycle

1. **Created by:** Planning Team Moderator
2. **Read by:** Enhanced Progress Sync
3. **Transformed to:** `reports/execution_progress_update.md`
4. **Consumed by:** Executive Team (Ops Commander + 5 agents)
5. **Archived:** After execution cycle completes
6. **Next cycle:** New `pending_actions.json` created

## Important Notes

### ✅ DO:
- Write complete action descriptions (full sentences)
- Specify clear success criteria
- Assign explicit owners
- Use consistent priority levels (high/medium/low)
- Include realistic time estimates

### ❌ DON'T:
- Write action fragments ("tracking in MLflow", "of fusion mechanisms")
- Leave fields empty or null
- Assign non-existent owners
- Use unclear priorities
- Skip success criteria

## Directory Structure

```
reports/
├── planning/
│   ├── transcript_*.md      # Full meeting conversation
│   ├── summary_*.md         # Meeting summary
│   └── actions_*.json       # Raw actions (also here)
├── handoff/
│   └── pending_actions.json # ← PRIMARY HANDOFF FILE (THIS DIRECTORY)
├── execution/
│   ├── execution_*.json     # Tool execution results
│   └── deployments_*.log    # Deployment logs
└── execution_progress_update.md  # ← Formatted for Executive Team
```

## System Architecture

```
┌─────────────────────────────────────────────┐
│         Planning Team Meeting               │
│  (Moderator + 5 advisors)                   │
└──────────────┬──────────────────────────────┘
               │
               │ Writes
               ↓
┌─────────────────────────────────────────────┐
│   reports/handoff/pending_actions.json      │ ← You are here
│   (Complete action specifications)          │
└──────────────┬──────────────────────────────┘
               │
               │ Read by
               ↓
┌─────────────────────────────────────────────┐
│      Enhanced Progress Sync                 │
│   (Groups by priority, formats details)     │
└──────────────┬──────────────────────────────┘
               │
               │ Writes
               ↓
┌─────────────────────────────────────────────┐
│  reports/execution_progress_update.md       │
│  (Priority-grouped actions)                 │
└──────────────┬──────────────────────────────┘
               │
               │ Read by
               ↓
┌─────────────────────────────────────────────┐
│         Executive Team                      │
│  (Ops Commander + 5 executors)              │
│  Executes: HIGH → MEDIUM → LOW              │
└─────────────────────────────────────────────┘
```

## Troubleshooting

### Issue: Executive Team not executing

**Check:**
```bash
# Does pending_actions.json exist?
ls -lh reports/handoff/pending_actions.json

# Is it valid JSON?
cat reports/handoff/pending_actions.json | python3 -m json.tool

# Are all required fields present?
cat reports/handoff/pending_actions.json | jq '.actions[] | {id, action, owner, priority}'
```

### Issue: Actions have incomplete descriptions

**Symptom:** Descriptions like "tracking in MLflow", "of fusion mechanisms"

**Fix:** Planning Team must write **complete sentences**:
- ❌ "tracking in MLflow"
- ✅ "Track model training metrics in MLflow with run_id and experiment_id"

### Issue: No priority grouping visible

**Check:**
```bash
# Did Enhanced Progress Sync run?
cat reports/execution_progress_update.md | grep "HIGH PRIORITY"

# Are priorities valid?
cat reports/handoff/pending_actions.json | jq '.actions[].priority'
```

## Status

✅ **Directory structure: Created**
✅ **README: Complete**
⏳ **pending_actions.json: Will be created by Planning Team at first meeting**

---

**Last Updated:** October 14, 2025
**Version:** 1.0
**Status:** Ready for first planning meeting
