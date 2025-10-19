# Running Autonomous System on Google Colab GPU

Quick guide to get your autonomous multi-agent coordination system running on Google Colab GPU instead of your local laptop.

## What This Does

Deploys your V1.0 Lightweight Enhancer with full autonomous monitoring and decision-making on Google Colab GPU (A100/V100/T4), freeing up your laptop for other work.

The system will:
- 🤖 Deploy V1.0 through staged rollout (shadow → 5% → 20% → 50% → 100%)
- 📊 Monitor metrics (compliance, nDCG, latency, error rate)
- 🔄 Auto-rollback if SLOs breached
- 💓 Run hourly heartbeat cycles with 15 agents
- 📝 Make autonomous decisions via Ops Commander
- 💾 Save all results back to Google Drive

## Problem Solved: "Files Not Found" Error ✅

**Your issue**: Uploaded entire `computer-vision` folder to Drive, but notebook couldn't find files.

**Root cause**: Notebook was looking for hardcoded path `/content/drive/MyDrive/cv_multimodal/project/`

**Fix applied**: Notebook now auto-detects project location! Checks these paths:
- `/content/drive/MyDrive/computer-vision/` ✅ (your upload)
- `/content/drive/MyDrive/cv_multimodal/project/`
- `/content/drive/MyDrive/cv_project/`
- `/content/drive/MyDrive/multimodal/`

Plus added diagnostic cell to show your Drive structure and manual path entry fallback.

## Quick Start (5 Minutes)

### 1. Create .env File (On Laptop)
```bash
cd /Users/guyan/computer_vision/computer-vision
cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-proj-your-key-here
GOOGLE_API_KEY=AIzaSy-your-key-here
EOF
```



### 2. Upload to Drive
1. Open drive.google.com
2. Drag `computer-vision` folder to "My Drive"
3. Wait for upload to complete

### 3. Open Notebook in Colab
1. Go to colab.research.google.com
2. Upload: `research/colab/autonomous_system_colab.ipynb`
3. Runtime → Change runtime type → GPU (A100/V100/T4)
4. Run all cells (Runtime → Run all)

### 4. Verify Running
After ~5 minutes, you should see:
```
✅ Found project at: /content/drive/MyDrive/computer-vision
✅ All required files found!
✅ API keys loaded
✅ Coordinator initialized (15 agents, 8 channels, 4 triggers)
✅ Autonomous system is now running!
```

**Done!** System is now running on GPU. Monitor via notebook cells.

## Documentation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **COLAB_SETUP_CHECKLIST.md** | Step-by-step checklist with checkboxes | First time setup |
| **COLAB_DEPLOYMENT_GUIDE.md** | Comprehensive deployment guide | Detailed instructions |
| **COLAB_QUICK_FIX.md** | Troubleshooting guide | When you hit errors |
| **COLAB_FIXES_SUMMARY.md** | What changed in notebook | Understanding the fixes |
| **COLAB_README.md** | This file | Quick overview |

**Start here**: `COLAB_SETUP_CHECKLIST.md` ← Print and check off items!

## Key Features of Fixed Notebook

### Auto-Detection System
```python
# Automatically finds your project
possible_locations = [
    "/content/drive/MyDrive/computer-vision",  # Your upload
    "/content/drive/MyDrive/cv_multimodal/project",
    # ... more locations
]
```

### Diagnostic Cell
```python
# Shows your Drive structure
📂 Your Google Drive structure:
   📁 computer-vision
      ✅ Contains multi-agent/ - This looks like your project!
```

### Flexible API Key Loading
```python
# Checks 3 locations + manual input fallback
api_key_locations = [
    ".env",
    "research/api_keys.env",
    "multi-agent/.env",
]
```

### Better Error Messages
```
❌ Missing required files!
   Expected location: /content/drive/MyDrive/computer-vision

Please verify your Drive has this structure:
   MyDrive/
   └── computer-vision/
       └── multi-agent/
           ├── autonomous_coordinator.py
           └── configs/
```

## Cost Estimate

| GPU Type | Availability | Cost | Recommended For |
|----------|--------------|------|-----------------|
| **T4** | Free/Pro | Included | Testing (2-4 hours) |
| **V100** | Pro/Pro+ | Included | Short runs (< 24h) |
| **A100** | Pro+ only | ~$2.50/hr | Production (Week 1 deployment) |

**Week 1 deployment (168 hours)**:
- A100: ~$420/week
- V100: ~$150-200/week (included in Pro+ $50/month)
- T4: Not recommended for 24/7 (session limits)

**Recommendation**: Start with V100 (2-4 hour test), then scale to A100 if needed.

## System Architecture

```
Google Colab GPU (A100)
    ↓
Autonomous Coordinator
    ↓
15 Agents in Hierarchy
    ├── Ops Commander (Executive Lead)
    │   ├── Infra Guardian
    │   ├── Latency Analyst
    │   ├── Compliance Monitor
    │   ├── Integration Engineer
    │   └── Rollback Officer
    ├── Planning Moderator
    │   ├── 6 Planning Agents
    └── V2 Scientific Team (2 agents)
    ↓
8 Communication Channels
    ↓
5 Shared Memory Stores
    ↓
Deployment Pipeline
    shadow → 5% → 20% → 50% → 100%
    ↓
Results Saved to Google Drive
```

## Monitoring

### Real-Time Monitoring Cells (Run periodically)

**Deployment State** (every 15-30 min):
```python
# Shows: version, stage, traffic %, SLO status
```

**Performance Metrics** (every 15-30 min):
```python
# Shows: compliance, nDCG, latency, error rate
```

**GPU Usage** (every 15-30 min):
```python
# Shows: GPU utilization, memory, temperature
```

**Decision Log** (every 30-60 min):
```python
# Shows: recent agent decisions, verdicts, reasoning
```

### Automated Backups

Results automatically saved to Drive hourly:
```
MyDrive/results/colab_run_[timestamp]/
├── state/          # Deployment state, metrics
├── reports/        # Agent reports
└── logs/           # System logs
```

## Emergency Procedures

### Emergency Rollback
```python
coordinator._trigger_emergency_rollback()
```

### Force Heartbeat (Manual)
```python
coordinator.heartbeat._execute_main_cycle()
```

### Stop System
```python
coordinator.stop()
```

### View Logs
```python
!tail -n 100 /content/cv_project/multi-agent/logs/coordinator.log
```

## Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Files not found | Run diagnostic cell, verify folder name |
| GPU not available | Runtime → Change runtime type → GPU |
| Session disconnected | Reconnect, re-run cells from Step 7 |
| API keys missing | Enter manually when prompted |
| Out of memory | `torch.cuda.empty_cache()` or upgrade to A100 |

**Full troubleshooting**: See `COLAB_QUICK_FIX.md`

## What's Different From Local Run

| Aspect | Local Laptop | Google Colab |
|--------|--------------|--------------|
| **GPU** | None or limited | A100 (40GB) / V100 (16GB) |
| **Monitoring** | Terminal output | Jupyter cells |
| **Duration** | Limited by laptop usage | 24/7 possible |
| **Cost** | Free (uses your hardware) | $2.50/hr (A100) |
| **Results** | Local files | Saved to Drive |
| **Access** | Need laptop open | Access from anywhere |

## Project Files

### Required (Must Upload)
- `multi-agent/autonomous_coordinator.py` (23 KB)
- `multi-agent/configs/autonomous_coordination.yaml` (27 KB)
- `multi-agent/tools/` (file bridge, etc.)
- `.env` (API keys)

### Optional (Recommended)
- `research/` (evaluation scripts, models)
- `docs/` (documentation)
- `data/` (datasets)

### Not Needed
- `.git/` (version control - large)
- `runs/` (old results - large)
- `*.pth` (model checkpoints - large)
- `__pycache__/` (compiled Python)

**Total upload size**: ~1-10 MB (essential) or ~50-500 MB (with research)

## Deployment Timeline

### Week 1 (Current)
- **Shadow deployment** (Day 1-2): 0% traffic, monitoring only
- **5% rollout** (Day 3-4): If metrics good, progress to 5%
- **20% rollout** (Day 5-7): If stable, progress to 20%

### Week 2-4 (Future)
- 50% → 100% rollout
- nDCG optimization experiments
- V2 architectural fixes (if needed)

**Current focus**: Deploy V1.0, gather real-world metrics, ensure stability.

## Success Criteria

System is working correctly when:

✅ Notebook shows "Autonomous system is now running!"
✅ Deployment state shows version and stage
✅ Metrics being collected (compliance, nDCG, latency)
✅ GPU utilization visible in nvidia-smi
✅ Agent decisions appearing in decision log
✅ Hourly backups saving to Drive
✅ No critical errors or crashes

## Getting Help

1. **Check COLAB_QUICK_FIX.md** for troubleshooting
2. **Run diagnostic cell** to see Drive structure
3. **View logs**: `!cat /content/cv_project/multi-agent/logs/coordinator.log`
4. **Test import**: Try importing coordinator module manually
5. **Verify paths**: Print `DRIVE_PROJECT` and `PROJECT_ROOT` variables

## Next Steps After Setup

1. ✅ Verify system running (all monitoring cells show data)
2. ✅ Let shadow deployment run for 6-12 hours
3. ✅ Review first heartbeat cycle results
4. ✅ Check decision log for Ops Commander decisions
5. ✅ Monitor metrics trends (improving or stable?)
6. ✅ Prepare for 5% rollout (Day 3-4)

## File Structure Reference

```
MyDrive/computer-vision/          ← You uploaded this
├── .env                          ← API keys (create this)
├── multi-agent/                  ← Required
│   ├── autonomous_coordinator.py ← Main coordinator
│   ├── configs/
│   │   └── autonomous_coordination.yaml  ← System config
│   ├── tools/                    ← File bridge, etc.
│   └── agents/prompts/           ← Agent prompts
├── research/                     ← Optional
│   ├── colab/
│   │   └── autonomous_system_colab.ipynb  ← The notebook
│   └── 01_v1_production_line/    ← V1.0 artifacts
└── docs/                         ← Optional

MyDrive/results/                  ← Auto-created
└── colab_run_[timestamp]/        ← Hourly backups
    ├── state/
    ├── reports/
    └── logs/
```

## Summary

**Problem**: "Files not found" when running Colab notebook
**Cause**: Hardcoded path didn't match your upload location
**Solution**: Auto-detection + diagnostics + manual fallback
**Result**: Works with any folder name/location

**Setup time**: 5-10 minutes (after upload)
**Run duration**: 24/7 possible
**Monitoring**: Every 30-60 minutes via notebook cells
**Cost**: ~$2.50/hour (A100) or included in Pro+ (V100)

**You're ready to deploy!** 🚀

---

**Quick Links**:
- 📋 [Setup Checklist](COLAB_SETUP_CHECKLIST.md) ← Start here!
- 📖 [Full Deployment Guide](COLAB_DEPLOYMENT_GUIDE.md)
- 🔧 [Troubleshooting](COLAB_QUICK_FIX.md)
- 📊 [System Architecture](AUTONOMOUS_SYSTEM_GUIDE.md)

**Last Updated**: 2025-10-13
**Status**: Ready to use
**Tested**: Colab Free, Pro, Pro+ with T4, V100, A100
