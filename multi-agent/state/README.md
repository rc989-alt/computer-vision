# State Management

## Purpose
Persist system state to enable recovery from interruptions and maintain audit trail.

---

## Directory Structure

```
state/
├── README.md (this file)
├── current_session.json          # Active session state
├── last_planning_output.json     # Most recent Planning Team output
├── last_execution_output.json    # Most recent Executive Team output
├── deployment_state.json          # Current deployment state
├── metrics_state.json             # Latest metrics
└── checkpoints/                   # Periodic snapshots
    └── checkpoint_YYYYMMDD_HHMMSS.json
```

---

## Files

### current_session.json
**Purpose:** Track active session state for crash recovery

**Schema:**
```json
{
  "session_id": "session_20251014_173045",
  "start_time": "2025-10-14T17:30:45Z",
  "last_update": "2025-10-14T17:45:00Z",
  "status": "active|paused|completed|error",
  "current_meeting": {
    "meeting_id": "planning_20251014_173045",
    "team": "planning|execution",
    "progress": "25%",
    "current_agent": "strategic_leader",
    "agents_completed": ["empirical_validation_lead"]
  },
  "pending_actions": 5,
  "active_agents": [
    "strategic_leader",
    "empirical_validation_lead",
    "critical_evaluator",
    "gemini_advisor"
  ]
}
```

**Updated:** After each agent response during meetings

---

### last_planning_output.json
**Purpose:** Store most recent Planning Team decisions for recovery

**Schema:**
```json
{
  "meeting_id": "planning_20251014_173045",
  "timestamp": "2025-10-14T17:45:00Z",
  "verdict": "GO|PAUSE_AND_FIX|REJECT",
  "actions_generated": 5,
  "priority_breakdown": {
    "HIGH": 2,
    "MEDIUM": 2,
    "LOW": 1
  },
  "acceptance_gates": {
    "visual_ablation_drop": "≥ 5%",
    "v1_v2_corr": "< 0.95",
    "ndcg@10": "≥ 0.72"
  },
  "assigned_owners": {
    "ops_commander": 3,
    "quality_safety_officer": 1,
    "infrastructure_performance_monitor": 1
  },
  "handoff_file": "reports/handoff/pending_actions.json"
}
```

**Updated:** After each Planning Team meeting

---

### last_execution_output.json
**Purpose:** Track most recent Executive Team results

**Schema:**
```json
{
  "meeting_id": "execution_20251014_180000",
  "timestamp": "2025-10-14T18:15:00Z",
  "actions_executed": 3,
  "actions_failed": 0,
  "actions_in_progress": 2,
  "results": {
    "experiments_run": 2,
    "mlflow_run_ids": ["abc123", "def456"],
    "metrics_collected": {
      "compliance@1": 0.85,
      "ndcg@10": 0.73,
      "latency_p95_ms": 42
    }
  },
  "slo_status": {
    "compliance": true,
    "latency": true,
    "error_rate": true
  },
  "handoff_file": "reports/handoff/execution_progress_update.md"
}
```

**Updated:** After each Executive Team meeting

---

### deployment_state.json
**Purpose:** Track current deployment status

**Schema:**
```json
{
  "current_version": "v1.0_lightweight",
  "stage": "shadow|5%|20%|50%|100%",
  "slo_status": {
    "compliance": true,
    "latency": true,
    "error_rate": true,
    "ndcg": true,
    "availability": true,
    "rollback_ready": true
  },
  "last_updated": "2025-10-14T18:00:00Z",
  "rollback_ready": true,
  "deployment_start": "2025-10-14T12:00:00Z",
  "time_in_stage_hours": 6.0,
  "next_stage_eligible": true,
  "metrics": {
    "compliance_current": 0.85,
    "compliance_baseline": 0.72,
    "ndcg_current": 0.73,
    "ndcg_baseline": 0.72,
    "latency_p95_ms": 42,
    "latency_baseline_ms": 45,
    "error_rate": 0.001
  }
}
```

**Updated:** By `autonomous_coordinator.py` during heartbeat cycles

---

### metrics_state.json
**Purpose:** Latest system metrics for monitoring

**Schema:**
```json
{
  "timestamp": "2025-10-14T18:00:00Z",
  "compliance_current": 0.85,
  "compliance_baseline": 0.72,
  "ndcg_current": 0.73,
  "ndcg_baseline": 0.72,
  "latency_p95_ms": 42,
  "latency_baseline_ms": 45,
  "error_rate": 0.001,
  "last_measurement": "2025-10-14T18:00:00Z",
  "ci95_bounds": {
    "compliance": [0.83, 0.87],
    "ndcg": [0.71, 0.75],
    "latency": [40, 44]
  },
  "trends": {
    "compliance": "improving",
    "ndcg": "stable",
    "latency": "improving"
  }
}
```

**Updated:** After each measurement by Quality & Safety Officer or Ops Commander

---

### checkpoints/
**Purpose:** Periodic snapshots during long-running operations

**Files:** `checkpoint_YYYYMMDD_HHMMSS.json`

**Schema:**
```json
{
  "checkpoint_id": "checkpoint_20251014_180000",
  "timestamp": "2025-10-14T18:00:00Z",
  "session_snapshot": { ... },
  "deployment_snapshot": { ... },
  "metrics_snapshot": { ... },
  "planning_snapshot": { ... },
  "execution_snapshot": { ... },
  "notes": "Periodic snapshot during heartbeat cycle"
}
```

**Created:** Every hour during autonomous cycles, or before major operations (deployment, rollback)

---

## Usage

### Python Integration

```python
from tools.state_manager import StateManager

# Initialize
state_mgr = StateManager(state_dir="multi-agent/state")

# Save session state
state_mgr.save_session({
    "session_id": "session_20251014_173045",
    "status": "active",
    "current_meeting": {...}
})

# Load session state (for recovery)
last_session = state_mgr.load_session()
if last_session and last_session["status"] == "active":
    resume_from_checkpoint(last_session)

# Save planning output
state_mgr.save_planning_output({
    "meeting_id": "planning_20251014_173045",
    "verdict": "GO",
    "actions_generated": 5
})

# Save execution output
state_mgr.save_execution_output({
    "meeting_id": "execution_20251014_180000",
    "actions_executed": 3
})

# Create checkpoint
state_mgr.create_checkpoint(notes="Before stage progression")
```

---

## Recovery Scenarios

### Scenario 1: Crash During Planning Meeting

**Problem:** Planning Team meeting interrupted mid-way

**Recovery:**
```python
# Check current session
session = StateManager.load_session()

if session["status"] == "active" and session["current_meeting"]["team"] == "planning":
    # Resume from last completed agent
    completed_agents = session["current_meeting"]["agents_completed"]
    remaining_agents = [a for a in PLANNING_AGENTS if a not in completed_agents]

    # Continue meeting with remaining agents
    resume_planning_meeting(
        meeting_id=session["current_meeting"]["meeting_id"],
        start_from=remaining_agents[0]
    )
```

---

### Scenario 2: Crash During Deployment

**Problem:** Deployment interrupted, need to rollback

**Recovery:**
```python
# Check deployment state
deployment = StateManager.load_deployment_state()

if deployment["stage"] in ["5%", "20%", "50%"]:
    # Load last checkpoint before deployment
    last_checkpoint = StateManager.get_latest_checkpoint()

    # Initiate rollback
    rollback_to_checkpoint(last_checkpoint)
```

---

### Scenario 3: Lost Handoff File

**Problem:** `pending_actions.json` deleted or corrupted

**Recovery:**
```python
# Recover from last planning output
last_planning = StateManager.load_planning_output()

if last_planning:
    # Regenerate handoff file from state
    regenerate_handoff_from_state(
        meeting_id=last_planning["meeting_id"],
        actions_count=last_planning["actions_generated"]
    )
```

---

## Cleanup Policy

### Checkpoint Retention
- **Keep last 24 checkpoints** (24 hours if hourly)
- **Keep 1 checkpoint per day for last 30 days**
- **Archive rest to reports/archive/**

### Session Files
- **Current session** - Always keep
- **Last planning/execution outputs** - Keep most recent 10
- **Older files** - Archive after 30 days

---

## Monitoring

### Health Checks

```bash
# Check if session is stale (no update in 10+ minutes)
python -c "from tools.state_manager import StateManager; StateManager.check_stale_session()"

# Verify all state files are present
ls state/*.json

# Check checkpoint count
ls state/checkpoints/ | wc -l

# View latest state
cat state/current_session.json | python -m json.tool
```

---

## Troubleshooting

### Issue: current_session.json shows status="error"
**Cause:** Previous session crashed

**Solution:**
```bash
# Review error details
cat state/current_session.json

# Load last checkpoint
cat state/checkpoints/$(ls -t state/checkpoints/ | head -1)

# Reset session manually
echo '{"session_id": null, "status": "idle"}' > state/current_session.json
```

### Issue: deployment_state.json missing
**Cause:** First time running or state corrupted

**Solution:**
```python
from tools.state_manager import StateManager

# Initialize default deployment state
StateManager.initialize_deployment_state({
    "current_version": "v1.0_lightweight",
    "stage": "shadow",
    "slo_status": {"compliance": False, "latency": False, ...},
    "rollback_ready": False
})
```

---

## Integration with Autonomous Coordinator

The `autonomous_coordinator.py` automatically manages state:

1. **On startup:** Check for active session, offer recovery
2. **During meetings:** Update `current_session.json` after each agent
3. **After meetings:** Save planning/execution outputs
4. **Hourly heartbeat:** Create checkpoint
5. **Before deployment:** Create pre-deployment checkpoint
6. **On shutdown:** Mark session as completed

---

## Version History

- **v3.0** (2025-10-14) - Added state management for consolidated 5-agent system
- **v2.0** (2025-10-12) - Added checkpoint system
- **v1.0** (2025-10-11) - Initial state management

---

**Maintained by:** Autonomous Coordinator
**Last Updated:** 2025-10-14
