# Google Colab GPU Deployment Guide

## Overview

This guide walks you through deploying the autonomous multi-agent coordination system on Google Colab with A100 GPU support, eliminating the need to run on your local laptop.

## Prerequisites

- Google account with Google Drive access
- Google Colab access (free tier works, but A100 requires Colab Pro+)
- Anthropic API key (for Claude models)
- OpenAI API key (for GPT models)
- Google Gemini API key (optional, for Gemini models)

## Cost Estimate

- **Colab Free**: Limited GPU access, may disconnect
- **Colab Pro ($12/month)**: Better GPU access, longer sessions
- **Colab Pro+ ($50/month)**: A100 GPU access (~$2.50/hour compute units)

**Estimated cost for Week 1 deployment (24/7 monitoring)**:
- With A100: ~$420/week (168 hours × $2.50/hour)
- With V100/T4: ~$100-200/week (included in Pro+ subscription)

**Recommended approach**: Use V100/T4 for initial testing, then scale to A100 if needed.

## Step 1: Upload Project to Google Drive

### 1.1 Simple Upload (RECOMMENDED - 5 minutes)

The easiest way is to upload your entire project folder directly:

1. **Open Google Drive** in your web browser
2. **Go to "My Drive"**
3. **Drag and drop** the entire `computer-vision` folder from your laptop

That's it! The notebook will auto-detect the location.

**Your Drive will look like this**:
```
MyDrive/
└── computer-vision/          # Your entire project
    ├── multi-agent/          # Required
    ├── research/             # Optional but recommended
    ├── data/                 # Optional
    ├── docs/                 # Optional
    └── .env                  # Required (create this)
```

### 1.2 Alternative: Compress and Upload (for slow connections)

If you have a slow internet connection, compress first:

```bash
# On your laptop
cd /Users/guyan/computer_vision

# Compress (excludes large/unnecessary files)
tar -czf computer-vision.tar.gz \
    computer-vision/multi-agent/ \
    computer-vision/research/ \
    computer-vision/docs/ \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pth' \
    --exclude='runs/'

# Upload computer-vision.tar.gz to MyDrive
# Then extract in Colab: !tar -xzf /content/drive/MyDrive/computer-vision.tar.gz
```

### 1.3 Create .env File

Create a file named `.env` in `MyDrive/computer-vision/` with your API keys:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

**Security Note**: Google Drive files are private by default, but ensure the .env file is NOT shared with anyone.

**How to create .env on your laptop**:
```bash
cd /Users/guyan/computer_vision/computer-vision
cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-proj-your-key-here
GOOGLE_API_KEY=AIzaSy-your-key-here
EOF
```

Then the .env file will be included when you upload the folder to Drive.

## Step 2: Upload Colab Notebook

### 2.1 Locate Notebook
The notebook is at:
```
/Users/guyan/computer_vision/computer-vision/research/colab/autonomous_system_colab.ipynb
```

### 2.2 Upload to Colab
1. Go to https://colab.research.google.com
2. Click "File" → "Upload notebook"
3. Upload `autonomous_system_colab.ipynb`
4. Or: Upload to `MyDrive/computer-vision/` and open from Drive

## Step 3: Run the Notebook

### 3.1 Select GPU Runtime

**IMPORTANT**: Before running any cells:
1. Click "Runtime" → "Change runtime type"
2. Select "Hardware accelerator" → "GPU"
3. For A100 (recommended): Select "GPU type" → "A100" (requires Colab Pro+)
4. For testing: Use T4 or V100 (available in free/Pro tier)
5. Click "Save"

### 3.2 Check GPU Availability

Run the first cell:
```python
!nvidia-smi
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Expected output**:
```
GPU: Tesla V100-SXM2-16GB  (or Tesla A100-SXM4-40GB)
GPU Memory: 16.00 GB (or 40.00 GB for A100)
```

**If GPU not available**:
- Click "Runtime" → "Change runtime type" → "Hardware accelerator" → "GPU" → "Save"
- For A100: Select "GPU" → "Premium" → "A100"

### 3.3 Authenticate and Mount Drive

Run the authentication cell:
```python
from google.colab import auth, drive
auth.authenticate_user()
drive.mount('/content/drive')
```

Follow the prompts to authenticate with your Google account.

**Expected output**:
```
Mounted at /content/drive
```

### 3.4 Check Drive Structure (NEW - Diagnostic Cell)

Run the diagnostic cell to see your Drive structure:
```python
# Quick diagnostic: Show Drive structure
print("📂 Your Google Drive structure:\n")
...
```

**Expected output**:
```
Top-level folders in MyDrive:
   📁 computer-vision
      ✅ Contains multi-agent/ - This looks like your project!
   📁 Documents
   ...
```

This confirms the notebook can see your uploaded project.

### 3.5 Auto-Detect Project Location (NEW)

The notebook will automatically find your project:

**Expected output**:
```
🔍 Auto-detecting project location on Google Drive...

✅ Found project at: /content/drive/MyDrive/computer-vision

📁 Drive project: /content/drive/MyDrive/computer-vision
📁 Local project: /content/cv_project
📁 Results save to: /content/drive/MyDrive/results
```

If auto-detection fails, you'll be prompted to enter the path manually.

### 3.6 Verify Required Files

The notebook will check for required files:

**Expected output**:
```
🔍 Checking for required files...

✅ multi-agent/autonomous_coordinator.py (23.4 KB)
✅ multi-agent/configs/autonomous_coordination.yaml (26.8 KB)

Optional files:
✅ .env
⚠️ research/api_keys.env - not found (will need manual input)

✅ All required files found!
```

### 3.7 Install Dependencies

Run the setup cells in order:

**Cell 1: Install dependencies**
```python
!pip install -q anthropic openai google-generativeai pyyaml
```

**Cell 2: Setup paths**
```python
import os
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/cv_multimodal')
PROJECT_ROOT = Path('/content/cv_multimodal')
```

**Cell 3: Extract project files**
```python
# If you uploaded tar.gz
!mkdir -p /content/cv_multimodal
!tar -xzf /content/drive/MyDrive/cv_multimodal/cv_multimodal.tar.gz -C /content/cv_multimodal

# Or if you uploaded zip
!unzip -q /content/drive/MyDrive/cv_multimodal/cv_minimal.zip -d /content/cv_multimodal
```

**Cell 4: Load API keys**
```python
from dotenv import load_dotenv
load_dotenv(DRIVE_ROOT / '.env')

# Verify keys loaded
print("ANTHROPIC_API_KEY:", "✓" if os.getenv('ANTHROPIC_API_KEY') else "✗")
print("OPENAI_API_KEY:", "✓" if os.getenv('OPENAI_API_KEY') else "✗")
print("GOOGLE_API_KEY:", "✓" if os.getenv('GOOGLE_API_KEY') else "✗")
```

**Expected output**:
```
ANTHROPIC_API_KEY: ✓
OPENAI_API_KEY: ✓
GOOGLE_API_KEY: ✓
```

### 3.4 Start Autonomous Coordinator

Run the coordinator startup cell:
```python
import sys
sys.path.insert(0, str(PROJECT_ROOT / 'multi-agent'))

from autonomous_coordinator import AutonomousCoordinator

coordinator = AutonomousCoordinator(
    config_path=PROJECT_ROOT / "multi-agent/configs/autonomous_coordination.yaml",
    project_root=PROJECT_ROOT
)

print("Starting autonomous coordination system...")
coordinator.start()
```

**Expected output**:
```
Starting autonomous coordination system...
[2025-10-12 14:30:00] Autonomous Coordinator initialized
[2025-10-12 14:30:00] Shared memory stores created: 5
[2025-10-12 14:30:00] Communication channels created: 8
[2025-10-12 14:30:00] Triggers armed: 4
[2025-10-12 14:30:00] Heartbeat system started (60 min cycle)
[2025-10-12 14:30:00] System status: RUNNING
```

## Step 4: Monitor Deployment

### 4.1 Real-Time Monitoring

Run monitoring cells (these can run in parallel notebooks or sequentially):

**Monitor Deployment State**:
```python
import json
import time

deployment_file = PROJECT_ROOT / 'multi-agent/state/deployment_state.json'

while True:
    if deployment_file.exists():
        with open(deployment_file, 'r') as f:
            state = json.load(f)

        print(f"\n{'='*60}")
        print(f"Deployment Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Version: {state.get('current_version', 'N/A')}")
        print(f"Stage: {state.get('stage', 'N/A')}")
        print(f"Traffic: {state.get('traffic_percentage', 0)}%")
        print(f"Status: {state.get('status', 'N/A')}")
        print(f"Last Update: {state.get('last_updated', 'N/A')}")

    time.sleep(60)  # Update every minute
```

**Monitor Metrics**:
```python
metrics_file = PROJECT_ROOT / 'multi-agent/state/metrics_state.json'

while True:
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        print(f"\n{'='*60}")
        print(f"Performance Metrics - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Compliance: {metrics.get('compliance_current', 0):.4f} (Δ {metrics.get('compliance_improvement', 0):+.4f})")
        print(f"nDCG: {metrics.get('ndcg_current', 0):.4f} (Δ {metrics.get('ndcg_improvement', 0):+.4f})")
        print(f"Latency P95: {metrics.get('latency_p95_ms', 0):.3f} ms")
        print(f"Error Rate: {metrics.get('error_rate', 0):.4f}")

        # Check SLO status
        compliance_ok = metrics.get('compliance_current', 0) >= metrics.get('compliance_baseline', 0) - 0.02
        latency_ok = metrics.get('latency_p95_ms', 999) <= metrics.get('latency_baseline_ms', 0) * 1.1
        error_ok = metrics.get('error_rate', 1) <= 0.01

        print(f"\nSLO Status:")
        print(f"  Compliance: {'✓' if compliance_ok else '✗ BREACH'}")
        print(f"  Latency: {'✓' if latency_ok else '✗ BREACH'}")
        print(f"  Error Rate: {'✓' if error_ok else '✗ BREACH'}")

    time.sleep(60)
```

**Monitor GPU Usage**:
```python
import subprocess

while True:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True)
    gpu_util, mem_used, mem_total = result.stdout.strip().split(',')

    print(f"\n{'='*60}")
    print(f"GPU Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"GPU Utilization: {gpu_util.strip()}%")
    print(f"Memory: {mem_used.strip()} MB / {mem_total.strip()} MB")
    print(f"Memory %: {float(mem_used) / float(mem_total) * 100:.1f}%")

    time.sleep(30)
```

### 4.2 View Decision Log

```python
decision_file = PROJECT_ROOT / 'multi-agent/state/decision_log.json'

if decision_file.exists():
    with open(decision_file, 'r') as f:
        decisions = json.load(f)

    print(f"\n{'='*60}")
    print(f"Recent Decisions (Last 10)")
    print(f"{'='*60}")

    for decision in decisions[-10:]:
        print(f"\n[{decision['timestamp']}]")
        print(f"Decider: {decision['decider']}")
        print(f"Type: {decision['decision_type']}")
        print(f"Decision: {decision['decision']}")
        print(f"Rationale: {decision['rationale'][:100]}...")
```

## Step 5: Run Smoke Tests

Execute smoke tests to verify system health:

```python
# Run all smoke tests
coordinator.run_smoke_tests()

# Or run specific test
coordinator.run_smoke_test('sanity')
coordinator.run_smoke_test('performance')
coordinator.run_smoke_test('compliance')
coordinator.run_smoke_test('integration')
coordinator.run_smoke_test('rollback')
```

**Expected output**:
```
Running smoke test: sanity
[✓] V1 baseline metrics loaded
[✓] V1.0 artifacts present
[✓] Deployment state initialized
[✓] All agents registered

Running smoke test: performance
[✓] Latency < 5ms: 0.062ms
[✓] Throughput > 1000 QPS
[✓] GPU memory < 80%: 45%

Running smoke test: compliance
[✓] Compliance improvement: +13.82%
[✓] nDCG improvement: +1.14%
[✓] Low margin rate: 98%

All smoke tests passed: 5/5
```

## Step 6: Save Results

Results are automatically saved to Drive every heartbeat cycle:

```python
import shutil

# Define save interval
SAVE_INTERVAL = 3600  # Save every hour

while coordinator.is_running():
    time.sleep(SAVE_INTERVAL)

    # Copy state files to Drive
    state_dir = PROJECT_ROOT / 'multi-agent/state'
    drive_backup = DRIVE_ROOT / 'results' / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
    drive_backup.mkdir(parents=True, exist_ok=True)

    shutil.copytree(state_dir, drive_backup / 'state', dirs_exist_ok=True)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Backup saved to Drive: {drive_backup}")
```

## Step 7: Stop Coordinator

When deployment is complete or you need to stop:

```python
print("Stopping autonomous coordinator...")
coordinator.stop()

# Save final state to Drive
final_backup = DRIVE_ROOT / 'results' / f"final_{time.strftime('%Y%m%d_%H%M%S')}"
shutil.copytree(PROJECT_ROOT / 'multi-agent/state', final_backup / 'state', dirs_exist_ok=True)

print(f"Final state saved to: {final_backup}")
print("System stopped successfully.")
```

## Troubleshooting

### GPU Not Available
**Symptom**: Cell shows "No GPU available"
**Solution**:
1. Click "Runtime" → "Change runtime type"
2. Select "GPU" as hardware accelerator
3. For A100: Select "Premium" tier (requires Colab Pro+)
4. Restart runtime

### Session Disconnected
**Symptom**: Colab disconnects after 12 hours (free tier) or 24 hours (Pro)
**Solution**:
1. Free tier: Use keep-alive script:
```python
import time
from IPython.display import display, Javascript

while True:
    display(Javascript('window.keepAlive = setInterval(function(){console.log("keepalive")}, 60000)'))
    time.sleep(300)
```
2. Pro+: Longer sessions, but still limited to 24 hours
3. Long deployments: Set up checkpoint saving every hour to Drive

### API Rate Limits
**Symptom**: "Rate limit exceeded" errors
**Solution**:
1. Check `autonomous_coordination.yaml` for request rates
2. Reduce heartbeat frequency (e.g., 120 min instead of 60 min)
3. Implement exponential backoff in coordinator

### Out of Memory
**Symptom**: CUDA out of memory errors
**Solution**:
1. Use smaller batch sizes in evaluation
2. Clear GPU cache: `torch.cuda.empty_cache()`
3. Upgrade to A100 (40GB vs 16GB)

### Files Not Found
**Symptom**: "FileNotFoundError" when coordinator starts
**Solution**:
1. Verify Drive mount: `!ls /content/drive/MyDrive/cv_multimodal`
2. Check extraction: `!ls /content/cv_multimodal/multi-agent`
3. Verify paths in notebook match your Drive structure

## Emergency Procedures

### Emergency Rollback
```python
# Force immediate rollback
coordinator.trigger_emergency_rollback(
    reason="Manual emergency rollback requested",
    triggered_by="user"
)
```

### Force Heartbeat
```python
# Trigger immediate heartbeat cycle
coordinator.trigger_immediate_heartbeat()
```

### View System Logs
```python
# View coordinator logs
!tail -n 100 /content/cv_multimodal/multi-agent/logs/coordinator.log

# View agent logs
!tail -n 100 /content/cv_multimodal/multi-agent/logs/ops_commander.log
```

## Best Practices

1. **Start with short test**: Run 1-2 heartbeat cycles (2-4 hours) before committing to full week
2. **Monitor costs**: Check Colab compute units usage in account settings
3. **Regular backups**: Save state to Drive every hour
4. **Use checkpoints**: Enable checkpoint saving in coordinator config
5. **Set SLO alerts**: Configure email/webhook alerts for SLO breaches
6. **Keep notebook open**: Don't close browser tab to prevent disconnection
7. **Use V100/T4 first**: Test on cheaper GPU before scaling to A100

## Next Steps

After successful Colab deployment:

1. Monitor Week 1 deployment progress (shadow → 5% → 20%)
2. Review decision log daily for insights
3. Analyze metrics trends (compliance, nDCG, latency)
4. Prepare Week 2 optimization based on Week 1 findings
5. Consider setting up automated reporting to email/Slack

## Support

- **Colab Issues**: https://research.google.com/colaboratory/faq.html
- **Coordinator Issues**: Check `multi-agent/logs/coordinator.log`
- **Agent Issues**: Check individual agent logs in `multi-agent/logs/`
- **API Issues**: Check rate limits and billing on respective platforms

## Summary Checklist

- [ ] Google Drive folder created (`MyDrive/cv_multimodal/`)
- [ ] Project files uploaded (tar.gz or zip)
- [ ] `.env` file created with API keys
- [ ] `requirements.txt` uploaded
- [ ] Colab notebook uploaded and opened
- [ ] GPU enabled (V100/T4 or A100)
- [ ] Drive mounted and authenticated
- [ ] Dependencies installed
- [ ] Project files extracted
- [ ] API keys loaded successfully
- [ ] Coordinator started successfully
- [ ] Smoke tests passed (5/5)
- [ ] Monitoring cells running
- [ ] Backup schedule configured
- [ ] Emergency procedures understood

**You're now ready to run the autonomous multi-agent system on Google Colab GPU!**
