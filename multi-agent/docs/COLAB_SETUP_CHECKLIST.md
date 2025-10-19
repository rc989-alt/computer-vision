# Google Colab Setup Checklist

Use this checklist to set up the autonomous system on Google Colab GPU.

## Pre-Flight Checklist (On Your Laptop)

### Step 1: Prepare Project
- [ ] Navigate to project directory: `cd /Users/guyan/computer_vision/computer-vision`
- [ ] Create `.env` file with API keys (see below)
- [ ] Verify essential files exist (see below)

### Step 2: Create .env File
```bash
cd /Users/guyan/computer_vision/computer-vision
cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-proj-your-key-here
GOOGLE_API_KEY=AIzaSy-your-key-here
EOF
```

**Verify**:
- [ ] `.env` file created
- [ ] Contains all 3 API keys
- [ ] Keys are valid (no typos)

### Step 3: Verify Essential Files
Run this on your laptop:
```bash
cd /Users/guyan/computer_vision/computer-vision

# Check required files
ls -lh multi-agent/autonomous_coordinator.py
ls -lh multi-agent/configs/autonomous_coordination.yaml
ls -lh .env

# Check sizes
du -sh multi-agent/
du -sh research/
```

**Expected**:
- [ ] `autonomous_coordinator.py` exists (~23 KB)
- [ ] `autonomous_coordination.yaml` exists (~27 KB)
- [ ] `.env` exists (~200 bytes)
- [ ] `multi-agent/` folder size: ~500 KB
- [ ] `research/` folder size: varies (optional)

## Upload to Google Drive

### Step 4: Upload Project
- [ ] Open Google Drive in browser (drive.google.com)
- [ ] Go to "My Drive"
- [ ] Drag and drop `computer-vision` folder
- [ ] Wait for upload to complete (check progress in bottom-right)

**Upload time estimate**:
- Essential files only (~1 MB): 1-2 minutes
- With research files (~50 MB): 5-10 minutes
- Full project with models (~500 MB): 30-60 minutes

**Verification**:
- [ ] See "computer-vision" folder in Drive
- [ ] Open folder → see `multi-agent/` subfolder
- [ ] Open `multi-agent/` → see Python files
- [ ] See `.env` file in root of `computer-vision/`

## Setup in Google Colab

### Step 5: Open Colab
- [ ] Go to colab.research.google.com
- [ ] Click "File" → "Upload notebook"
- [ ] Upload: `research/colab/autonomous_system_colab.ipynb`
- [ ] Or: "File" → "Open notebook" → "Google Drive" → Navigate to notebook

### Step 6: Configure Runtime
- [ ] Click "Runtime" → "Change runtime type"
- [ ] Hardware accelerator: **GPU**
- [ ] GPU type: **T4** (free/Pro) or **A100** (Pro+ recommended)
- [ ] Click "Save"

### Step 7: Run Setup Cells

#### Cell 1: Check GPU
```python
!nvidia-smi
```
- [ ] Shows GPU info (Tesla T4, V100, or A100)
- [ ] Shows GPU memory (16-40 GB)
- [ ] No errors

#### Cell 2: Mount Drive
```python
from google.colab import auth, drive
auth.authenticate_user()
drive.mount('/content/drive')
```
- [ ] Prompts for Google authentication
- [ ] Shows "Mounted at /content/drive"
- [ ] No errors

#### Cell 3: Diagnostic (NEW)
```python
# Quick diagnostic: Show Drive structure
```
**Expected output**:
- [ ] Lists folders in MyDrive
- [ ] Shows "computer-vision" folder
- [ ] Shows "✅ Contains multi-agent/"

**If not shown**:
- Wait 1 minute and re-run cell
- Check Drive web interface for folder
- Verify upload completed

#### Cell 4: Auto-Detect Project (NEW)
```python
# Setup paths
print("🔍 Auto-detecting project location...")
```
**Expected output**:
- [ ] "✅ Found project at: /content/drive/MyDrive/computer-vision"
- [ ] Shows Drive project path
- [ ] Shows local project path

**If failed**:
- [ ] Enter path manually when prompted
- [ ] Path format: `/content/drive/MyDrive/[folder-name]`
- [ ] Verify path with: `!ls [path]/multi-agent`

#### Cell 5: Install Dependencies
```python
!pip install -q anthropic openai google-generativeai pyyaml
```
- [ ] Installs packages (takes ~30 seconds)
- [ ] No error messages
- [ ] Shows package versions

#### Cell 6: Verify Files
```python
# Check if project files exist on Drive
```
**Expected output**:
- [ ] "✅ multi-agent/autonomous_coordinator.py"
- [ ] "✅ multi-agent/configs/autonomous_coordination.yaml"
- [ ] "✅ All required files found!"

**If failed**:
- [ ] Check error message for missing files
- [ ] Verify files exist in Drive web interface
- [ ] Re-upload missing files

#### Cell 7: Copy to Local
```python
# Copy project from Drive to Colab local storage
```
- [ ] Shows "✅ Project copied to /content/cv_project"
- [ ] Lists key Python files
- [ ] No errors

#### Cell 8: Load API Keys (NEW)
```python
# Load API keys from env file
```
**Expected output**:
- [ ] "🔑 Loading API keys from: .env"
- [ ] "✅ ANTHROPIC_API_KEY = sk-ant-api..."
- [ ] "✅ OPENAI_API_KEY = sk-proj-..."
- [ ] "✅ API Key Status: ANTHROPIC_API_KEY: ✅ Set"

**If failed (no .env file)**:
- [ ] Prompted to enter API keys manually
- [ ] Enter ANTHROPIC_API_KEY
- [ ] Enter OPENAI_API_KEY
- [ ] Enter GOOGLE_API_KEY (optional)
- [ ] Verify "✅ Set" status

#### Cell 9: Create Directories
```python
# Create necessary directories
```
- [ ] Creates state/, reports/, logs/ folders
- [ ] Shows "✅ Directories ready"
- [ ] No errors

## Start Autonomous System

### Step 8: Initialize Coordinator
```python
from autonomous_coordinator import AutonomousCoordinator
coordinator = AutonomousCoordinator(...)
```
**Expected output**:
- [ ] "✅ Coordinator initialized"
- [ ] Shows agent count (15 agents)
- [ ] Shows channel count (8 channels)
- [ ] Shows trigger count (4 triggers)
- [ ] No import errors

**If failed**:
- [ ] Check error message
- [ ] Verify `autonomous_coordinator.py` exists
- [ ] Try: `!ls /content/cv_project/multi-agent/autonomous_coordinator.py`
- [ ] Check Python path: `import sys; print(sys.path[:3])`

### Step 9: Start System
```python
coordinator.start()
```
**Expected output**:
- [ ] "💓 Starting heartbeat system..."
- [ ] "✅ Autonomous system is now running!"
- [ ] Shows next heartbeat time
- [ ] No errors

**System is now running!** 🎉

## Monitor System (Run Periodically)

### Step 10: Check Deployment State
```python
# Check deployment state
```
**Expected output**:
- [ ] Shows current version
- [ ] Shows deployment stage (shadow/5%/20%/etc.)
- [ ] Shows traffic percentage
- [ ] Shows SLO status (✅ or ❌)

**Run every**: 15-30 minutes

### Step 11: Check Metrics
```python
# Check metrics
```
**Expected output**:
- [ ] Shows compliance (e.g., 0.8234)
- [ ] Shows nDCG (e.g., 1.0025)
- [ ] Shows latency P95 (e.g., 0.062 ms)
- [ ] Shows error rate (e.g., 0.0001)

**Run every**: 15-30 minutes

### Step 12: Check GPU Usage
```python
!nvidia-smi
```
**Expected output**:
- [ ] GPU utilization: 0-100%
- [ ] Memory usage: < 80% is good
- [ ] Temperature: < 80°C is good

**Run every**: 15-30 minutes

### Step 13: View Decision Log
```python
# Check decision log
```
**Expected output**:
- [ ] Shows recent decisions
- [ ] Shows verdict (APPROVE/ROLLBACK/CONTINUE)
- [ ] Shows reasoning
- [ ] Shows current phase

**Run every**: 30-60 minutes

## Save Results

### Step 14: Periodic Backup (Every Hour)
```python
# Copy state and results to Drive
```
- [ ] Creates timestamped backup folder
- [ ] Copies state files
- [ ] Copies reports
- [ ] Copies logs
- [ ] Shows save location

**Run every**: 60 minutes (or after each heartbeat cycle)

## Shutdown

### Step 15: Stop System
```python
coordinator.stop()
```
- [ ] Shows "✅ System stopped"
- [ ] Final backup created
- [ ] Results saved to Drive
- [ ] Safe to disconnect

### Step 16: Download Results (Optional)
- [ ] Go to Google Drive
- [ ] Navigate to `MyDrive/results/colab_run_[timestamp]/`
- [ ] Download `state/`, `reports/`, `logs/` folders
- [ ] Review on laptop

## Troubleshooting Quick Reference

### Problem: GPU Not Available
**Fix**: Runtime → Change runtime type → GPU → Save

### Problem: Drive Not Mounted
**Fix**: Re-run mount cell, authenticate again

### Problem: Files Not Found
**Fix**:
1. Run diagnostic cell to see Drive structure
2. Verify folder name matches
3. Enter path manually if prompted

### Problem: API Keys Not Loaded
**Fix**:
1. Check .env file exists in Drive
2. Verify format: `KEY=value` (no quotes)
3. Enter manually if prompted

### Problem: Import Error (autonomous_coordinator)
**Fix**:
1. Verify file copied: `!ls /content/cv_project/multi-agent/`
2. Check Python path added
3. Try: `import sys; sys.path.insert(0, '/content/cv_project/multi-agent')`

### Problem: Session Disconnected
**Fix**:
1. Reconnect to runtime
2. Re-run cells from Step 7 (skip Steps 1-6)
3. Coordinator state is saved, can resume

### Problem: Out of Memory
**Fix**:
1. Clear GPU cache: `torch.cuda.empty_cache()`
2. Restart runtime
3. Upgrade to A100 (40GB)

## Timeline Expectations

### Initial Setup (First Time)
- Upload project to Drive: 5-30 minutes
- Run all setup cells: 5-10 minutes
- **Total**: 10-40 minutes

### Subsequent Runs (After Setup)
- Open notebook: 1 minute
- Run cells (skip setup): 2-3 minutes
- **Total**: 3-5 minutes

### Deployment Duration
- Week 1 (shadow → 20%): 7 days continuous
- Monitor every 30-60 minutes
- Backup hourly
- Cost (A100): ~$420/week
- Cost (T4/V100): ~$100-200/week

## Success Criteria

You've successfully set up the system when:

- [x] Notebook shows "✅ Autonomous system is now running!"
- [x] Deployment state cell shows version and stage
- [x] Metrics cell shows compliance, nDCG, latency
- [x] GPU usage shows activity
- [x] Decision log shows agent decisions
- [x] Backups saving to Drive hourly
- [x] No critical errors in any cell

**You're ready to deploy!** 🚀

## Quick Commands

### Force Check Triggers
```python
coordinator.triggers.check_triggers()
```

### Force Heartbeat (Manual)
```python
coordinator.heartbeat._execute_main_cycle()
```

### Emergency Rollback
```python
coordinator._trigger_emergency_rollback()
```

### View Logs
```python
!tail -n 100 /content/cv_project/multi-agent/logs/coordinator.log
```

### Clear GPU Memory
```python
import torch
torch.cuda.empty_cache()
```

## Documentation

- **Full Guide**: `COLAB_DEPLOYMENT_GUIDE.md`
- **Troubleshooting**: `COLAB_QUICK_FIX.md`
- **Fixes Summary**: `COLAB_FIXES_SUMMARY.md`
- **System Guide**: `AUTONOMOUS_SYSTEM_GUIDE.md`
- **Config Reference**: `configs/autonomous_coordination.yaml`

---

**Print this checklist and check off items as you complete them!**

**Last Updated**: 2025-10-13
