# Two-System Architecture: Planning Advisors + Executive Workers

## 🏗️ System Separation

### System 1: Planning Team (Advisors)
**Location:** `multi-agent/agents/prompts/planning_team/`
**Role:** Strategic planning, analysis, recommendations
**Output:** Meeting reports → Executive team

### System 2: Executive Team (Workers)
**Location:** `multi-agent/agents/prompts/executive_team/`
**Role:** Execute plans, deploy models, run experiments
**Input:** Plans from advisors
**Output:** Execution reports, deployed artifacts

---

## 📋 Planning Team Agents (Advisors)

Located in `/multi-agent/agents/prompts/planning_team/`:

| Agent | Model | File | Role |
|-------|-------|------|------|
| Moderator | Claude Opus | `moderator.md` | Meeting orchestration |
| Pre-Architect | Claude Opus | `pre_arch_opus.md` | System architecture |
| Data Analyst | Claude Sonnet | `claude_data_analyst.md` | Data insights |
| Critic | GPT-4 | `critic_openai.md` | Critical evaluation |
| Gemini Feasibility | Gemini | `Gemini_Feasibility_Search.md` | Feasibility search |

**Meeting Frequency:** Every 15 minutes
**Output Files:**
- `reports/planning/transcript_YYYYMMDD_HHMMSS.md`
- `reports/planning/summary_YYYYMMDD_HHMMSS.md`
- `reports/planning/actions_YYYYMMDD_HHMMSS.json`
- `reports/planning/decisions_YYYYMMDD_HHMMSS.json`

---

## ⚙️ Executive Team Agents (Workers)

Located in `/multi-agent/agents/prompts/executive_team/`:

| Agent | Model | File | Role |
|-------|-------|------|------|
| Ops Commander | Claude Opus | `Ops_Commander_(V1_Executive_Lead).md` | Overall execution |
| Infra Guardian | GPT-4 | `infra_guardian.md` | Infrastructure |
| Latency Analyst | Claude Sonnet | `latency_analysis.md` | Performance |
| Compliance Monitor | GPT-4 | `compliance_monitor.md` | Metrics validation |
| Integration Engineer | Claude Sonnet | `integration_engineer.md` | Integration |
| Rollback Officer | GPT-4 | `roll_back_recovery_officer.md` | Rollback prep |

**Execution Frequency:** Every 5 minutes (between planning meetings)
**Output Files:**
- `reports/execution/execution_YYYYMMDD_HHMMSS.json`
- `reports/execution/deployments_YYYYMMDD_HHMMSS.log`
- `deployment/shadow/` - Shadow deployment artifacts
- `deployment/metrics/` - Performance metrics

---

## 🔄 Workflow: Planning → Executive Handoff

```
┌─────────────────────────┐
│  Planning Team Meeting  │
│  (Strategic Advisors)   │
└───────────┬─────────────┘
            │
            │ Output: decisions.json
            │         actions.json
            │         summary.md
            ↓
┌─────────────────────────┐
│   Handoff Directory     │
│  reports/handoff/       │
└───────────┬─────────────┘
            │
            │ Executive reads:
            │ - What to do
            │ - Priority
            │ - Success criteria
            ↓
┌─────────────────────────┐
│  Executive Team Work    │
│  (Autonomous Workers)   │
└───────────┬─────────────┘
            │
            │ Output: execution.json
            │         deployments.log
            │         metrics/
            ↓
┌─────────────────────────┐
│   Results Directory     │
│  reports/results/       │
└─────────────────────────┘
```

---

## 📊 Monitoring Dashboards

### Dashboard 1: Planning Team Monitor
**File:** `monitor_planning.ipynb`

Shows:
- Latest meeting summary
- Key decisions made
- Actions recommended
- Priority queue
- Agent participation

**Refresh:** Every minute

### Dashboard 2: Executive Team Monitor
**File:** `monitor_execution.ipynb`

Shows:
- Current execution status
- Actions in progress
- Tools being used
- Deployment stage
- Metrics collected
- Success/failure rate

**Refresh:** Every 30 seconds

### Dashboard 3: Unified Status
**File:** `monitor_unified.ipynb`

Shows both teams side-by-side:
- Planning → Executive flow
- Actions: Planned vs Executed
- Timeline of decisions and actions
- Overall system health

---

## 📁 Directory Structure

```
multi-agent/
├── agents/
│   └── prompts/
│       ├── planning_team/          # Advisors
│       │   ├── moderator.md
│       │   ├── pre_arch_opus.md
│       │   ├── claude_data_analyst.md
│       │   ├── critic_openai.md
│       │   └── Gemini_Feasibility_Search.md
│       │
│       └── executive_team/          # Workers
│           ├── Ops_Commander_(V1_Executive_Lead).md
│           ├── infra_guardian.md
│           ├── latency_analysis.md
│           ├── compliance_monitor.md
│           ├── integration_engineer.md
│           └── roll_back_recovery_officer.md
│
├── reports/
│   ├── planning/                   # Planning outputs
│   │   ├── transcript_*.md
│   │   ├── summary_*.md
│   │   ├── actions_*.json
│   │   └── decisions_*.json
│   │
│   ├── handoff/                    # Planning → Executive
│   │   ├── pending_actions.json   # Read by Executive
│   │   └── latest_summary.md
│   │
│   ├── execution/                  # Executive outputs
│   │   ├── execution_*.json
│   │   ├── deployments_*.log
│   │   └── tool_usage_*.json
│   │
│   └── results/                    # Final results
│       ├── metrics/
│       ├── evaluations/
│       └── deployment_status.json
│
└── deployment/
    ├── shadow/                     # Shadow deployment
    ├── 5_percent/                  # 5% rollout
    ├── 20_percent/                 # 20% rollout
    └── production/                 # Full deployment
```

---

## 🎯 Key Features

### 1. Clear Separation
- **Planning team** never executes code
- **Executive team** never makes strategic decisions
- Clean interface through JSON files

### 2. Proper Handoff
- Planning outputs `actions.json` with clear instructions
- Executive reads actions and executes autonomously
- Results flow back to planning team for next meeting

### 3. Full Monitoring
- Real-time view of both teams
- See planning decisions immediately
- Track execution progress live
- Alert on failures

### 4. Accountability
- Every decision logged with reasoning
- Every execution logged with results
- Full audit trail from planning → execution → results

---

## 🚀 Benefits

✅ **Separation of Concerns** - Advisors advise, workers work
✅ **Parallel Operation** - Both teams work simultaneously
✅ **Clear Accountability** - Know who decided what and who did what
✅ **Easy Monitoring** - Separate dashboards for each team
✅ **Scalable** - Can add more advisors or workers independently
✅ **Maintainable** - Clear boundaries and interfaces

---

## 📝 Next Steps

1. Create `planning_coordinator.py` - Manages planning team only
2. Create `executive_coordinator.py` - Manages execution only
3. Create handoff mechanism (`reports/handoff/`)
4. Create 3 monitoring notebooks
5. Deploy both systems to Colab
6. Watch them work together!
