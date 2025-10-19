# Multi-Agent System Scripts

## Production Scripts

### ../autonomous_coordinator.py (ROOT)
**Primary entry point** for the autonomous multi-agent system.
- Orchestrates both Planning and Executive teams
- Implements priority-based execution
- Handles handoff mechanism via `pending_actions.json`
- Uses consolidated 5-agent configuration
- **Use this for normal operations**

**Usage:**
```bash
cd /Users/guyan/computer_vision/computer-vision/multi-agent
python autonomous_coordinator.py
```

### run_planning_meeting.py
Run ONLY the Planning Team meeting (4 agents).
- **Agents:** Strategic Leader, Empirical Validation Lead, Critical Evaluator, Gemini Advisor
- Creates `pending_actions.json` for Executive Team
- Does NOT execute actions
- **Use for:** Strategic planning sessions, architecture reviews, research coordination

**Usage:**
```bash
python scripts/run_planning_meeting.py
```

### run_execution_meeting.py
Run ONLY the Executive Team meeting (3 agents).
- **Agents:** Ops Commander, Quality & Safety Officer, Infrastructure & Performance Monitor
- Reads `pending_actions.json` from Planning Team
- Executes actions by priority (HIGH → MEDIUM → LOW)
- Reports results via `execution_progress_update.md`
- **Use for:** Deployment operations, experiment execution

**Usage:**
```bash
python scripts/run_execution_meeting.py
```

### run_strategic_analysis.py
Run deep strategic analysis session.
- Extended Planning Team meeting
- No immediate execution
- Focuses on long-term strategy, research direction
- **Use for:** Quarterly planning, CVPR 2025 paper coordination, major architecture decisions

**Usage:**
```bash
python scripts/run_strategic_analysis.py
```

---

## Experimental Scripts

### experimental/run_meeting.py
Generic meeting runner (any team composition).
- Flexible agent composition
- Not optimized for production use
- **Use for:** Testing new agent combinations, prototyping

### experimental/run_with_context.py
Utility to add context to any script.
- Not a standalone entry point
- **Use for:** Development and debugging

---

## Quick Reference

| Use Case | Script | Agents | Output |
|----------|--------|--------|--------|
| Full autonomous cycle | `../autonomous_coordinator.py` | All 5 | `pending_actions.json` + `execution_progress_update.md` |
| Strategic planning only | `run_planning_meeting.py` | Planning (4) | `pending_actions.json` |
| Execute approved actions | `run_execution_meeting.py` | Executive (3) | `execution_progress_update.md` |
| Deep strategic analysis | `run_strategic_analysis.py` | Planning (4) | Extended analysis report |

---

## Configuration

All scripts use the consolidated 5-agent configuration:
- **Config file:** `configs/consolidated_5_agent_coordination.yaml`
- **Agent prompts:** `agents/prompts/planning_team/` and `agents/prompts/executive_team/`
- **Handoff directory:** `reports/handoff/`
- **State directory:** `state/`

---

## Notes

- Scripts maintain backward compatibility with 14-agent configuration
- Set `use_consolidated = True` in code to use 5-agent structure
- All scripts generate reports in `reports/` directory
- Logs saved to appropriate subdirectories based on meeting type

**Version:** 3.0 (Consolidated 5-agent system)
**Last Updated:** 2025-10-14
