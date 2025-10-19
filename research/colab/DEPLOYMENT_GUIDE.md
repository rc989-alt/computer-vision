# Updated Autonomous System - Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 PLANNING TEAM (Advisors)                │
│  - Holds strategic meetings every 30 minutes            │
│  - 5 agents: Moderator, Architect, Analyst, Critic      │
│  - Creates summaries and recommended actions            │
│  - Saves to: multi-agent/reports/planning/              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Handoff (pending_actions.json)
                   ↓
┌─────────────────────────────────────────────────────────┐
│               EXECUTIVE TEAM (Workers)                   │
│  - Executes actions autonomously                        │
│  - 6 agents: Ops Commander, Infra Guardian, etc.        │
│  - Real execution: deploy models, run code, metrics     │
│  - Saves to: multi-agent/reports/execution/             │
└─────────────────────────────────────────────────────────┘
```

## Step 1: Deploy Updated System

In Colab, run:

```python
# Option A: Run the deployment script
exec(open('/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/deploy_updated_system.py').read())
```

OR

```python
# Option B: Manual deployment
import time
import shutil
from pathlib import Path

# Wait for Drive sync
print("⏳ Waiting for Drive sync...")
time.sleep(60)

# Copy files
DRIVE = Path("/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean")
LOCAL = Path("/content/cv_project")

shutil.copy(DRIVE / "executive_coordinator.py", LOCAL / "executive_coordinator.py")
shutil.copy(DRIVE / "multi-agent/tools/execution_tools.py", LOCAL / "multi-agent/tools/execution_tools.py")

# Stop old coordinator
if 'coordinator' in globals():
    coordinator.stop()

# Clear cache
import sys
for mod in ['executive_coordinator', 'execution_tools']:
    if mod in sys.modules:
        del sys.modules[mod]

# Start new coordinator
sys.path.insert(0, str(LOCAL))
from executive_coordinator import ExecutiveCoordinator

coordinator = ExecutiveCoordinator(LOCAL, log_file="/content/executive.log")
coordinator.start()

print("✅ System deployed!")
```

## Step 2: Open Monitoring Dashboards

Open these notebooks in **separate tabs**:

1. **Planning Team Monitor**:
   - Location: `research/colab/monitor_planning.ipynb`
   - Shows: meeting summaries, recommended actions, history
   - Refresh: 60 seconds

2. **Executive Team Monitor**:
   - Location: `research/colab/monitor_execution.ipynb`
   - Shows: agent activity, tool usage, deployments
   - Refresh: 30 seconds

## Step 3: Verify System is Working

Run these commands in Colab:

```python
# Check logs
!tail -50 /content/executive.log

# Check planning reports
!ls -lh /content/cv_project/multi-agent/reports/planning/

# Check handoff file
!cat /content/cv_project/multi-agent/reports/handoff/pending_actions.json

# Check execution reports
!ls -lh /content/cv_project/multi-agent/reports/execution/

# Check deployment directories
!ls -R /content/cv_project/deployment/
```

## Expected Workflow

### Phase 1: Planning (First 30 minutes)
1. Planning Team holds initial meeting
2. 5 agents discuss V1.0 Lightweight Enhancer goals
3. Meeting summary saved to `reports/planning/summary_*.md`
4. Actions saved to `reports/planning/actions_*.json`
5. Handoff file created: `reports/handoff/pending_actions.json`

### Phase 2: Execution (Continuous)
1. Executive Team reads `pending_actions.json`
2. Routes actions to appropriate agents:
   - `deploy` → Ops Commander
   - `evaluate` → Latency Analysis
   - `monitor` → Compliance Monitor
   - `infrastructure` → Infra Guardian
3. Agents execute with real tools:
   - `run_python_script()` - Execute training/evaluation
   - `deploy_model()` - Copy models to deployment stages
   - `collect_metrics()` - Gather performance data
   - `run_bash_command()` - System operations
4. Results saved to `reports/execution/`

### Phase 3: Iteration (Every 30 minutes)
1. Planning Team reviews execution results
2. Creates new strategic recommendations
3. Updates handoff file with new actions
4. Executive Team executes new actions
5. Cycle continues → V1.0 deployment

## Key Differences from Previous Version

### Before:
- ❌ Planning and execution mixed together
- ❌ Agents only created plans, didn't execute
- ❌ No handoff mechanism
- ❌ Single monitoring approach
- ❌ Reports in one directory

### After:
- ✅ Clear separation: advisors vs workers
- ✅ Executive agents can deploy, evaluate, collect metrics
- ✅ JSON handoff file for team-to-team communication
- ✅ Separate monitors for each team
- ✅ Organized reports: planning/, execution/, handoff/

## Tool Execution Capabilities

Executive Team agents can now:

```python
# Deploy models
deploy_model(source="models/v1.0.pth", target="deployment/shadow/", stage="shadow")

# Run evaluations
run_evaluation(
    eval_script="research/evaluate_v1.py",
    model_path="models/v1.0.pth",
    dataset_path="data/test/"
)

# Collect metrics
collect_metrics(metrics_source="deployment/shadow/metrics.json")

# Execute Python
run_python_script(script_path="research/train_v1.py", args=["--epochs", "10"])

# Run bash commands
run_bash_command(command="nvidia-smi", timeout=10)
```

## Troubleshooting

### System not starting?
```python
# Check for errors
!tail -100 /content/executive.log | grep -i error

# Restart
coordinator.stop()
time.sleep(5)
coordinator.start()
```

### No handoff file?
- Planning Team needs to complete first meeting (30 minutes)
- Check: `!ls /content/cv_project/multi-agent/reports/planning/`
- If empty, wait for first meeting cycle

### Agents not executing?
- Check handoff file exists: `!cat reports/handoff/pending_actions.json`
- Check execution log: `!grep "TOOL" /content/executive.log`
- Verify tools initialized: `!grep "execution_tools" /content/executive.log`

### Drive sync issues?
```python
# Force sync
!ls -R /content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/ > /dev/null

# Wait and retry
import time
time.sleep(60)
```

## Next Steps

1. **Deploy** using Step 1 commands above
2. **Monitor** using both dashboard notebooks
3. **Wait 30 minutes** for first planning meeting
4. **Verify handoff** file created in `reports/handoff/`
5. **Watch execution** in monitor_execution.ipynb
6. **Check results** in `reports/execution/`

## Success Criteria

You'll know it's working when you see:

- ✅ Planning meeting completes → `summary_*.md` created
- ✅ Actions file appears → `actions_*.json` created
- ✅ Handoff file generated → `pending_actions.json` created
- ✅ Executive agents start executing → logs show `[TOOL]` entries
- ✅ Deployment artifacts appear → `deployment/shadow/` populated
- ✅ Metrics collected → `execution/metrics_*.json` files

---

**Ready to deploy?** Copy Step 1 commands into Colab and run!
