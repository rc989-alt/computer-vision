# Colab Quick Fix Guide

## Problem: "Files Not Found" Error

If you see this error in Colab:
```
❌ multi-agent/autonomous_coordinator.py - NOT FOUND
❌ multi-agent/configs/autonomous_coordination.yaml - NOT FOUND
```

## Solution: The notebook now auto-detects your project location!

### What Changed

**Before**: The notebook looked for files at `/content/drive/MyDrive/cv_multimodal/project/`

**Now**: The notebook automatically searches these locations:
1. `/content/drive/MyDrive/computer-vision/` ← Most common (full folder upload)
2. `/content/drive/MyDrive/cv_multimodal/project/`
3. `/content/drive/MyDrive/cv_project/`
4. `/content/drive/MyDrive/multimodal/`

### Step-by-Step Fix

#### Step 1: Run the diagnostic cell (NEW!)

After mounting Drive, run this cell:
```python
# Quick diagnostic: Show Drive structure
print("📂 Your Google Drive structure:\n")
...
```

**Output will show**:
```
Top-level folders in MyDrive:
   📁 computer-vision
      ✅ Contains multi-agent/ - This looks like your project!
   📁 Documents
   📁 Photos
   ...
```

This confirms where your project is located.

#### Step 2: Run the auto-detection cell

The notebook will automatically find your project:
```python
# Setup paths
print("🔍 Auto-detecting project location on Google Drive...")
```

**Expected output**:
```
✅ Found project at: /content/drive/MyDrive/computer-vision

📁 Drive project: /content/drive/MyDrive/computer-vision
📁 Local project: /content/cv_project
📁 Results save to: /content/drive/MyDrive/results
```

#### Step 3: Verify files found

The next cell will check for required files:
```
🔍 Checking for required files...

✅ multi-agent/autonomous_coordinator.py (23.4 KB)
✅ multi-agent/configs/autonomous_coordination.yaml (26.8 KB)
```

If you see ✅ for both, you're good to go!

### If Auto-Detection Fails

If the notebook still can't find your project, it will prompt:
```
❌ Could not auto-detect project location

Please enter the full path to your project on Drive:
Example: /content/drive/MyDrive/computer-vision

Path: _
```

**How to find the correct path**:

1. Look at the diagnostic output from Step 1
2. Find the folder that contains `multi-agent/`
3. Enter the full path: `/content/drive/MyDrive/[your-folder-name]`

**Examples**:
- If you uploaded as "computer-vision": `/content/drive/MyDrive/computer-vision`
- If you renamed it to "cv_project": `/content/drive/MyDrive/cv_project`
- If it's in a subfolder: `/content/drive/MyDrive/Projects/computer-vision`

### Common Folder Structures

#### Option A: Uploaded entire "computer-vision" folder (RECOMMENDED)
```
MyDrive/
└── computer-vision/          ← Upload this from your laptop
    ├── multi-agent/
    │   ├── autonomous_coordinator.py
    │   ├── configs/
    │   │   └── autonomous_coordination.yaml
    │   └── tools/
    ├── research/
    ├── data/
    └── docs/
```
**Path to enter**: `/content/drive/MyDrive/computer-vision`

#### Option B: Created custom folder structure
```
MyDrive/
└── cv_multimodal/
    └── project/              ← Manually created
        ├── multi-agent/
        ├── research/
        └── data/
```
**Path to enter**: `/content/drive/MyDrive/cv_multimodal/project`

#### Option C: Renamed folder
```
MyDrive/
└── my_cv_project/           ← Custom name
    ├── multi-agent/
    ├── research/
    └── data/
```
**Path to enter**: `/content/drive/MyDrive/my_cv_project`

## API Keys Issue

### Problem: No API keys file found

The notebook now checks multiple locations:
1. `research/api_keys.env`
2. `.env`
3. `multi-agent/.env`

### Solution Options

#### Option A: Upload .env file (RECOMMENDED)

1. On your laptop, create `.env` file:
   ```bash
   cd /Users/guyan/computer_vision/computer-vision
   cat > .env <<EOF
   ANTHROPIC_API_KEY=sk-ant-api03-...
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=...
   EOF
   ```

2. Upload to Google Drive root of your project folder:
   ```
   MyDrive/computer-vision/.env
   ```

3. Re-run the API keys cell in Colab

**Expected output**:
```
🔑 Loading API keys from: .env

✅ ANTHROPIC_API_KEY = sk-ant-api...
✅ OPENAI_API_KEY = sk-proj-...
✅ GOOGLE_API_KEY = AIzaSy...

✅ API keys loaded
```

#### Option B: Manual entry

If no .env file found, the notebook will prompt:
```
⚠️ No API keys file found in any location

📝 Please enter API keys manually:

ANTHROPIC_API_KEY: _
```

Just paste your API keys when prompted.

**Security Note**: Keys entered manually are only stored in Colab runtime memory, not saved to Drive.

## Verification Checklist

Before starting the autonomous system, verify:

```
✅ Drive mounted
✅ Project folder detected (shows path)
✅ multi-agent/autonomous_coordinator.py found
✅ multi-agent/configs/autonomous_coordination.yaml found
✅ API keys loaded (shows ANTHROPIC_API_KEY set)
✅ Dependencies installed
✅ Directories created
```

If all show ✅, you're ready to start the coordinator!

## Quick Test

Run this in a Colab cell to test your setup:

```python
# Quick setup test
from pathlib import Path
import os

# 1. Check Drive mount
assert Path("/content/drive/MyDrive").exists(), "Drive not mounted!"
print("✅ Drive mounted")

# 2. Check project exists
project_path = Path("/content/drive/MyDrive/computer-vision")  # Change if needed
assert project_path.exists(), f"Project not found at {project_path}"
print(f"✅ Project found: {project_path}")

# 3. Check multi-agent folder
assert (project_path / "multi-agent").exists(), "multi-agent folder missing!"
print("✅ multi-agent folder found")

# 4. Check coordinator file
coordinator_file = project_path / "multi-agent/autonomous_coordinator.py"
assert coordinator_file.exists(), "autonomous_coordinator.py missing!"
print(f"✅ Coordinator file found ({coordinator_file.stat().st_size / 1024:.1f} KB)")

# 5. Check config file
config_file = project_path / "multi-agent/configs/autonomous_coordination.yaml"
assert config_file.exists(), "autonomous_coordination.yaml missing!"
print(f"✅ Config file found ({config_file.stat().st_size / 1024:.1f} KB)")

# 6. Check API keys
assert os.getenv('ANTHROPIC_API_KEY'), "ANTHROPIC_API_KEY not set!"
print("✅ ANTHROPIC_API_KEY set")

assert os.getenv('OPENAI_API_KEY'), "OPENAI_API_KEY not set!"
print("✅ OPENAI_API_KEY set")

print("\n🎉 All checks passed! Ready to start coordinator.")
```

## Still Having Issues?

### Issue: "FileNotFoundError: No such file or directory"

**Cause**: Path is incorrect or files weren't uploaded completely

**Fix**:
1. In Google Drive web interface, verify you can see:
   - `MyDrive/computer-vision/multi-agent/autonomous_coordinator.py`
2. In Colab, run: `!ls -la /content/drive/MyDrive/computer-vision/multi-agent/`
3. If files are missing, re-upload from your laptop

### Issue: "No module named 'autonomous_coordinator'"

**Cause**: Python can't find the module

**Fix**:
```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/computer-vision")  # Your path
sys.path.insert(0, str(PROJECT_ROOT / "multi-agent"))

# Verify it's in path
print("Python path:")
for p in sys.path[:3]:
    print(f"  {p}")

# Try import again
from autonomous_coordinator import AutonomousCoordinator
print("✅ Import successful")
```

### Issue: "API rate limit exceeded"

**Cause**: Too many API calls

**Fix**:
1. Check your API usage on provider websites
2. Increase heartbeat interval in config:
   ```yaml
   heartbeat:
     main_cycle_minutes: 120  # Changed from 60
   ```
3. Restart coordinator

### Issue: Colab disconnects after a few hours

**Cause**: Colab free tier has session limits

**Fix**:
1. Upgrade to Colab Pro ($12/month) for longer sessions
2. Use the keep-alive script (in notebook)
3. Set up hourly backups to Drive (already configured)

## Summary

The updated notebook now:
- ✅ Auto-detects project location on Drive
- ✅ Shows diagnostic info about your Drive structure
- ✅ Checks multiple locations for API keys
- ✅ Provides clear error messages with solutions
- ✅ Allows manual path/key entry as fallback

**No more "files not found" errors!**

## Need More Help?

1. **View your Drive structure**: Run the diagnostic cell (after mounting Drive)
2. **Check logs**: `!tail -n 50 /content/cv_project/multi-agent/logs/coordinator.log`
3. **Verify paths**: Print `DRIVE_PROJECT` and `PROJECT_ROOT` variables
4. **Test import**: Try importing coordinator module manually
5. **Check permissions**: Ensure Google Drive allows Colab to read your files

---

**Last Updated**: 2025-10-13
**Works with**: Colab Free, Pro, Pro+
**Tested on**: V100, A100 GPUs
