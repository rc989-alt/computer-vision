#  autonomous system - User Guide

**Version**: 2.0
**Date**: 2025-10-12
**Status**: ✅ Ready for Production

---

## Overview

This autonomous system enables your multi-agent team to **deploy V1.0 Lightweight and execute strategic plans without manual intervention**. It provides:

- **Hierarchy**: Clear reporting structure (Ops Commander → Executive Team → Planning Team)
- **Shared Memory**: File-based state management for cross-agent coordination
- **Triggers**: Automatic actions based on events (SLO breach → rollback, stage complete → progress)
- **Heartbeat**: Hourly synchronization cycles with 6-phase execution
- **File Editing**: Planning team can autonomously edit docs
- **Smoke Testing**: Executive team runs tests during meetings
- **Colab Integration**: Deploy to Google Colab GPU for high-performance testing

---

## Quick Start

### 1. Start the Autonomous System

```bash
cd /Users/guyan/computer_vision/computer-vision/multi-agent
python3 autonomous_coordinator.py
```

**Output**:
```
✅ Autonomous Coordinator initialized
   Project root: /Users/guyan/computer_vision/computer-vision
   Agents: 15
   Channels: 8

🚀 STARTING AUTONOMOUS COORDINATION SYSTEM
✅ Initialized deployment state
💓 Heartbeat system started
✅ AUTONOMOUS SYSTEM ACTIVE

💡 System running in background...
💡 Press Ctrl+C to stop
```

### 2. Monitor Progress

The system runs autonomously with hourly heartbeat cycles. Check status:

```bash
# View deployment state
cat multi-agent/state/deployment_state.json

# View metrics
cat multi-agent/state/metrics_state.json

# View decision log
cat multi-agent/state/decision_log.json

# View latest cycle snapshot
cat multi-agent/state/cycle_start.json
```

### 3. Trigger Manual Actions

If you need to manually trigger actions:

```python
from autonomous_coordinator import AutonomousCoordinator
from pathlib import Path

# Load coordinator
coordinator = AutonomousCoordinator(
    Path("configs/autonomous_coordination.yaml"),
    Path("/Users/guyan/computer_vision/computer-vision")
)

# Trigger planning meeting
coordinator.triggers.check_triggers()

# Force stage progression
coordinator._trigger_stage_progression()

# Emergency rollback
coordinator._trigger_emergency_rollback()
```

---

## System Architecture

### Hierarchy

```
Ops Commander (Claude Sonnet 4)
├── Infra Guardian (Claude Sonnet 4)
├── Latency Analyst (Claude Sonnet 4)
├── Compliance Monitor (Claude Sonnet 4)
├── Integration Engineer (Claude Sonnet 4)
└── Rollback Officer (Claude Sonnet 4)

Planning Moderator (Claude Opus 4)
├── Pre-Architect (Claude Opus 4)
├── Tech Analysis Team (Claude Sonnet 4)
├── Data Analyst (Claude Sonnet 4)
├── Gemini Feasibility Search (Gemini 2.0)
└── OpenAI Critic (GPT-4 Turbo)

V2 Scientific Lead (Claude Opus 4)
└── Integrity Guardian (Claude Sonnet 4)
```

### Communication Channels

**V1 Executive Internal**:
- `v1_baseline_feed` - Environment and baseline metrics
- `v1_latency_feed` - Performance profiling
- `v1_compliance_feed` - Compliance metrics
- `v1_integration_feed` - Integration status
- `v1_rollback_feed` - Rollback readiness

**Cross-Team**:
- `innovation_relay` - V1 → V2 discoveries
- `planning_to_exec` - Planning decisions → Ops
- `v2_validation_feed` - V2 scientific validation

### Shared Memory Stores

**State Files** (in `multi-agent/state/`):
- `deployment_state.json` - Current deployment stage, SLO status
- `metrics_state.json` - Real-time metrics (compliance, nDCG, latency)
- `artifact_registry.json` - Tracked files and changes
- `decision_log.json` - Historical decisions
- `innovation_queue.json` - V1 ↔ V2 innovation exchange

---

## Heartbeat Cycle (Every 60 Minutes)

### Phase 0: Initiation (5 min)
- **Responsible**: Heartbeat System
- **Actions**: Broadcast BEGIN_CYCLE, save state snapshot

### Phase 1: Monitoring (15 min)
- **Responsible**: Infra Guardian, Latency Analyst, Compliance Monitor
- **Parallel**: Yes
- **Actions**:
  - Verify environment reproducibility
  - Profile runtime latency
  - Evaluate compliance metrics
- **Output**: Publish to respective feeds

### Phase 2: Integration (10 min)
- **Responsible**: Integration Engineer
- **Wait For**: Phase 1 complete
- **Actions**: Run integration tests, validate compatibility
- **Output**: Publish to `v1_integration_feed`

### Phase 3: Rollback Check (10 min)
- **Responsible**: Rollback Officer
- **Parallel**: With Phase 2
- **Actions**: Test rollback mechanism, verify checkpoints
- **Output**: Publish to `v1_rollback_feed`

### Phase 4: Decision (5 min)
- **Responsible**: Ops Commander
- **Wait For**: Phases 2 & 3 complete
- **Actions**:
  - Collect all reports
  - Compute verdict: GO | SHADOW_ONLY | PAUSE_AND_FIX | ROLLBACK
  - Save decision to state

### Phase 5: Execution (10 min)
- **Responsible**: Ops Commander
- **Actions**:
  - If GO: Deploy next stage
  - If ROLLBACK: Initiate rollback (via Rollback Officer)
  - If PAUSE_AND_FIX: Notify planning team

### Phase 6: Innovation Relay (5 min)
- **Responsible**: Ops Commander
- **Actions**:
  - Check innovation queue
  - Publish discoveries to V2
  - Integrate V2 validated innovations

---

## Automatic Triggers

### Emergency Rollback Trigger
**Condition**:
- Compliance drops > 2 pts from baseline, OR
- Latency P95 exceeds 110% of baseline, OR
- Error rate > 1%

**Action**: Automatic rollback to shadow stage
**Priority**: CRITICAL
**Notify**: Ops Commander, Rollback Officer

### Stage Progression Trigger
**Condition**:
- All SLOs pass, AND
- Minimum time in stage elapsed:
  - Shadow: 24 hours
  - 5%: 48 hours
  - 20%: 48 hours
  - 50%: 72 hours

**Action**: Progress to next deployment stage
**Priority**: HIGH
**Notify**: Ops Commander, Planning Moderator

### Artifact Scan Trigger
**Condition**: Time since last scan > 60 minutes
**Action**: Tech Analysis Team scans research/ for new files
**Updates**: `artifact_registry.json`

### Innovation Detection Trigger
**Condition**: Significant finding (delta > threshold AND confidence > 0.95)
**Action**: Publish to `innovation_relay` channel
**Notify**: V2 Scientific Team

### Planning Meeting Trigger
**Condition**:
- New artifacts > 10, OR
- Days since last meeting > 7

**Action**: Convene planning meeting with all agents
**Participants**: Pre-Architect, Tech Analysis, Data Analyst, Gemini, Critic

---

## File Editing Permissions

### Planning Team (Autonomous Editing)
**Agents**: Pre-Architect, Tech Analysis, Data Analyst, Planning Moderator

**Allowed Paths**:
- `docs/**/*.md`
- `research/plans/**/*.json`
- `research/analysis/**/*.md`
- `multi-agent/reports/planning_*.md`
- `multi-agent/decisions/**/*.json`

**Operations**: Create, Edit, Delete
**Git Commit**: Automatic
**Commit Message**: `[Planning] {agent_id}: {action} {file_path}`

### Executive Team (Deployment Docs)
**Agents**: Ops Commander, Integration Engineer

**Allowed Paths**:
- `deploy/**/*.md`
- `ops/**/*.yaml`
- `multi-agent/reports/deployment_*.md`
- `multi-agent/state/**/*.json`

**Operations**: Create, Edit
**Git Commit**: Automatic
**Commit Message**: `[Ops] {agent_id}: {action} {file_path}`

### V2 Scientific (Research Audits)
**Agents**: V2 Scientific Lead

**Allowed Paths**:
- `research/audits/**/*.md`
- `research/audits/**/*.json`
- `multi-agent/reports/scientific_*.md`

**Operations**: Create, Edit
**Git Commit**: Automatic
**Commit Message**: `[Scientific] V2 Scientific: {action} {file_path}`

---

## Smoke Testing

### During Meetings
**Enabled**: Yes
**Trigger**: Planning proposes deployment

**Workflow**:
1. Planning Moderator proposes deployment
2. Meeting pauses
3. Ops Commander requests smoke tests
4. Tests run in parallel (max 3 minutes):
   - Sanity Check (30s)
   - Performance Smoke (60s)
   - Compliance Smoke (60s)
5. Integration Engineer reports results
6. Ops Commander makes GO/NO-GO decision
7. Meeting resumes with verdict

### Test Suites

**Sanity Check** (30s):
- Verify API endpoints reachable
- Check model loads
- Validate input/output schema
- **Pass Criteria**: All tests pass

**Performance Smoke** (60s):
- Measure P95 latency on 10 requests
- Check throughput sustained
- Verify memory under limit
- **Pass Criteria**: Within 10% of baseline

**Compliance Smoke** (60s):
- Run validation on 20 samples
- Measure Compliance@1
- Check no conflict errors
- **Pass Criteria**: Within 1pt of baseline

**Integration Smoke** (90s):
- Test backward compatibility
- Verify no API breakage
- Check dependency versions
- Run regression test suite
- **Pass Criteria**: Zero regressions

**Rollback Smoke** (120s):
- Simulate rollback
- Measure RTO
- Verify checkpoint integrity
- Test warm-start
- **Pass Criteria**: RTO < 10 min, zero data loss

---

## Google Colab Integration

### Setup

1. **Authenticate Google Account**:
   ```python
   from google.colab import auth
   auth.authenticate_user()
   ```

2. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Upload Model**:
   ```bash
   # From local machine
   cp research/models/v1_lightweight.pth \
      /content/drive/MyDrive/cv_multimodal/models/
   ```

### Deployment Workflow

**Phase 1: Preparation** (Integration Engineer)
- Package V1.0 lightweight model
- Create deployment notebook
- Prepare test dataset
- Generate run config

**Phase 2: Upload** (Integration Engineer)
- Upload to Google Drive
- Create Colab notebook
- Configure GPU runtime (A100 preferred)

**Phase 3: Deployment** (Ops Commander)
- Execute notebook cells
- Monitor deployment logs
- Capture metrics

**Phase 4: Validation** (Latency Analyst + Compliance Monitor)
- Profile Colab performance
- Validate metrics match baseline
- Check GPU utilization

**Phase 5: Results** (Ops Commander)
- Download results
- Compare to baseline
- Make deployment decision

### GPU Configuration

- **Preferred**: A100 (40GB memory, ~$2.50/hour)
- **Fallback**: V100 → T4
- **Memory Requirement**: 16GB minimum
- **Monitoring**: GPU util, memory, latency, throughput, cost

---

## V1.0 Deployment Mission

### Timeline: Week 1 (Oct 12-19, 2025)

**Day 0-1: Preparation**
- ✅ Package V1.0 model (Integration Engineer)
- ✅ Prepare 120-query test dataset (Data Analyst)
- ✅ Setup monitoring dashboards (Infra Guardian)

**Day 1-2: Shadow Deployment**
- Deploy to shadow environment
- Run smoke tests
- Collect 24hr baseline metrics

**Day 2-3: Colab GPU Testing**
- Deploy to Colab A100
- Run performance benchmarks
- Validate compliance metrics

**Day 3-5: 5% Rollout**
- Deploy to 5% production traffic
- Monitor SLOs for 48 hours
- Prepare rollback if needed

**Day 5: Executive Decision**
- Convene decision meeting
- Review all metrics
- Make GO/NO-GO for 20% rollout

**Day 5-7: 20% Rollout** (if approved)
- Deploy to 20% traffic
- Full SLO validation for 48 hours
- Continue if successful

### Success Criteria

✅ V1.0 deployed to production (5%+ traffic)
✅ All SLOs met for 48+ hours
✅ Zero P0 incidents
✅ Compliance ≥ +13% maintained
✅ Latency P95 < 500ms
✅ nDCG optimization plan defined

### Rollback Criteria

❌ Compliance drops > 2 pts from baseline
❌ Latency P95 > 550ms
❌ Error rate > 1%
❌ Any P0 production incident

---

## Parallel Mission: nDCG Optimization Sprint

**Start Trigger**: V1.0 deployed to 5%+ traffic
**Duration**: Week 2-4
**Responsible**: Planning Team

### Objectives

1. Improve nDCG from +1.14% → +8% target
2. Implement learning-to-rank features
3. Optimize result diversity
4. Test personalization approaches

### Weekly Checkpoints

**Week 2**: Feature engineering sprint
**Week 3**: Model training and validation
**Week 4**: Integration testing and shadow deployment

---

## Human Oversight

### Automatic Notifications

**Sent to Human Operator When**:
- P0 critical incident
- All automatic recovery failed
- Conflicting decisions from agents
- Week 6 strategic checkpoint
- Deployment to 50%+ traffic

### Human Approval Required For

- Rollback to previous major version
- Budget exceeds $1000/day
- Architectural change proposed
- V2 → V1 integration (new multimodal features)

### Override Capability

**Command**: `HUMAN_OVERRIDE`
**Effect**: Pause all autonomous actions, await manual instructions

### Reporting

- **Daily Summary Email**: Automatically sent
- **Weekly Detailed Report**: Comprehensive analysis
- **Real-Time Dashboard**: `http://localhost:8000/dashboard`

---

## Error Handling & Recovery

### Automatic Recovery Strategies

**Agent Timeout**:
- Recovery: Retry with extended timeout
- Max Retries: 3
- Escalate To: Ops Commander

**SLO Breach**:
- Recovery: Initiate rollback
- Handler: Rollback Officer
- Notify: Ops Commander, Planning Moderator

**File Access Error**:
- Recovery: Verify permissions and retry
- Max Retries: 2
- Fallback: Notify Ops Commander

**Colab Connection Lost**:
- Recovery: Reconnect and resume
- Max Retries: 5
- Fallback: Fail and rollback

### Incident Escalation Levels

**P3 - Minor**:
- Response Time: 60 minutes
- Notify: Responsible agent
- Auto-Resolve: Yes

**P2 - Moderate**:
- Response Time: 15 minutes
- Notify: Agent + Ops Commander
- Auto-Resolve: No

**P1 - Major**:
- Response Time: 5 minutes
- Notify: Commander + Moderator + Rollback Officer
- Auto-Action: Pause deployments

**P0 - Critical**:
- Response Time: 1 minute
- Notify: All agents + Human operator
- Auto-Action: Immediate rollback

---

## Monitoring & Dashboards

### Real-Time Metrics

**Deployment State**:
```bash
watch -n 10 cat multi-agent/state/deployment_state.json
```

**Metrics**:
```bash
watch -n 10 cat multi-agent/state/metrics_state.json
```

**Decision Log**:
```bash
tail -f multi-agent/state/decision_log.json | jq
```

### Channel Messages

**View Latest Messages**:
```python
from autonomous_coordinator import ChannelManager

channels = ChannelManager()
messages = channels.get_messages("v1_compliance_feed")
print(messages[-1])  # Latest message
```

### Heartbeat Status

**Check Last Cycle**:
```bash
cat multi-agent/state/cycle_start.json | jq .timestamp
```

**View Cycle History**:
```bash
ls -lt multi-agent/state/cycle_*.json | head -10
```

---

## Troubleshooting

### System Won't Start

**Check**:
1. Config file exists: `configs/autonomous_coordination.yaml`
2. Python dependencies installed: `pip install -r requirements.txt`
3. State directory writable: `mkdir -p multi-agent/state`

**Logs**:
```bash
python3 autonomous_coordinator.py 2>&1 | tee logs/coordinator.log
```

### Agents Not Responding

**Check Agent Status**:
```python
coordinator = AutonomousCoordinator(...)
print(coordinator.agents)
```

**Restart Heartbeat**:
```python
coordinator.heartbeat.stop()
coordinator.heartbeat.start()
```

### Triggers Not Firing

**Manually Check Conditions**:
```python
# Check SLO breach
print(coordinator.triggers.slo_breach_condition())

# Check stage progression
print(coordinator.triggers.stage_progression_condition())
```

**Force Trigger Check**:
```python
coordinator.triggers.check_triggers()
```

### Colab Connection Issues

**Re-authenticate**:
```python
from google.colab import auth
auth.authenticate_user()
```

**Check GPU Availability**:
```python
!nvidia-smi
```

### State Corruption

**Reset State**:
```bash
rm -rf multi-agent/state/*.json
python3 autonomous_coordinator.py  # Will reinitialize
```

**Backup State**:
```bash
cp -r multi-agent/state multi-agent/state_backup_$(date +%Y%m%d_%H%M%S)
```

---

## Advanced Usage

### Custom Triggers

**Add Your Own Trigger**:
```python
def my_condition():
    # Your condition logic
    return True

def my_action():
    # Your action logic
    print("Custom action executed!")

coordinator.triggers.register_trigger(
    name="my_custom_trigger",
    condition=my_condition,
    action=my_action,
    priority="HIGH"
)
```

### Custom Channels

**Create New Channel**:
```python
coordinator.channels.subscribe("my_channel", "my_agent")

# Publish message
coordinator.channels.publish(
    "my_channel",
    "sender_agent",
    {"key": "value"}
)

# Get messages
messages = coordinator.channels.get_messages("my_channel")
```

### Custom State Stores

**Add New Store**:
```python
coordinator.memory.update_state(
    "my_custom_store",
    {"data": "value"},
    merge=True
)

# Get state
state = coordinator.memory.get_state("my_custom_store")
```

---

## Best Practices

### 1. Monitor Regularly
Check dashboards at least daily, even though system is autonomous

### 2. Review Decision Logs
Understand why agents made specific decisions

### 3. Validate Smoke Tests
Ensure tests cover your critical paths

### 4. Backup State Files
Before major changes, backup `multi-agent/state/`

### 5. Test Rollback
Weekly rollback drills to ensure recovery works

### 6. Document Custom Changes
If you add custom triggers/channels, document them

### 7. Keep Configs in Sync
If you modify agent prompts, update coordination config

### 8. Monitor Colab Costs
A100 GPU can be expensive - monitor usage

### 9. Review Innovation Queue
Check V1 → V2 discoveries regularly

### 10. Human Review at Milestones
Week 6 checkpoint requires human strategic input

---

## FAQ

**Q: Can I pause the autonomous system?**
A: Yes, Ctrl+C will stop it. State is persisted, resume anytime.

**Q: How do I change deployment stage manually?**
A: Use `coordinator._trigger_stage_progression()` or edit `deployment_state.json`

**Q: Can agents edit any file?**
A: No, strict permissions defined in config. Planning team can edit docs, exec team can edit ops files.

**Q: What if two agents conflict?**
A: Hierarchy resolves: Ops Commander has final say on deployment decisions

**Q: How long does each heartbeat cycle take?**
A: ~45-50 minutes of actual work, 60 min total with buffer

**Q: Can I run multiple coordinators?**
A: No, only one coordinator per project. State would conflict.

**Q: Is the system safe for production?**
A: Yes, with proper monitoring. Automatic rollback on SLO breach.

**Q: How do I know if rollback is working?**
A: Rollback Officer tests it weekly, check `v1_rollback_feed`

**Q: Can I integrate with other tools?**
A: Yes, system is extensible. Add custom triggers and channels.

**Q: What if Colab GPU is unavailable?**
A: Fallback to V100 or T4. Config specifies fallback tiers.

---

## Support & Resources

**Documentation**:
- `STRATEGIC_MEETING_GUIDE.md` - Meeting protocols
- `COTRR_BENCHMARK_DOCUMENTATION.md` - CoTRR analysis
- `FILE_ACCESS_FIXED.md` - File access system

**Agent Prompts**:
- `agents/prompts/executive_team/` - Executive agent prompts
- `agents/prompts/planning_team/` - Planning agent prompts

**Reports**:
- `reports/STRATEGIC_MEETING_FINAL_SYNTHESIS.md` - Latest strategic analysis
- `reports/transcript_*.md` - Meeting transcripts
- `reports/weekly_summary_*.md` - Weekly summaries

**State Files**:
- `state/*.json` - Current system state
- `reports/*.md` - Generated reports
- `decisions/*.json` - Decision history

---

## Version History

**v2.0** (2025-10-12):
- Full autonomous coordination system
- Hierarchy, shared memory, triggers, heartbeat
- File editing permissions
- Smoke testing integration
- Colab GPU support

**v1.0** (2025-10-11):
- Basic multi-agent meeting system
- File access bridge
- Manual coordination

---

**Status**: ✅ Production Ready
**Next Review**: Week 6 Checkpoint (2025-10-19)
**Owner**: Ops Commander + Planning Moderator
