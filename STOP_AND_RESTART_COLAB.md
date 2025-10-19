# How to Stop and Restart Colab System

**Issue:** Cannot connect to GPU in Colab
**Cause:** System running in background using resources
**Solution:** Stop current session and restart with GPU

---

## 🛑 Step 1: Stop Current System (In Colab)

### Option 1: Stop Monitoring Cell (Gentle)

**If you're running Cell #16 (Monitoring Dashboard):**
1. Click the **Stop button** (⏹️) in Colab
2. This stops monitoring but meetings may continue in background

### Option 2: Interrupt All Execution (Recommended)

**To stop all background threads:**
1. In Colab menu: **Runtime** → **Interrupt execution**
2. Or press: `Ctrl+M I` (or `⌘+M I` on Mac)
3. This stops all Python execution including background threads

### Option 3: Restart Runtime (Clean Slate)

**To completely restart and free all resources:**
1. In Colab menu: **Runtime** → **Restart runtime**
2. Or press: `Ctrl+M .` (or `⌘+M .` on Mac)
3. **Warning:** This will clear all variables and stop the system

---

## 🔌 Step 2: Reconnect to GPU

### Check Current Runtime Type

1. In Colab: **Runtime** → **Change runtime type**
2. Check **Hardware accelerator** setting
3. Should be: **GPU** (T4, V100, or A100)

### If Not Connected:

**Option A: Simple Reconnect**
```python
# Run this in a new Colab cell
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Or with PyTorch
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

**Option B: Force Reconnect**
1. **Runtime** → **Disconnect and delete runtime**
2. Wait 10 seconds
3. **Runtime** → **Connect**
4. Check GPU: **Runtime** → **View resources**

---

## 🔄 Step 3: Restart Unified System

### After GPU is connected:

**Run these cells in order:**

1. **Cell #1: Mount Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Cell #2: Install Dependencies**
   ```python
   !pip install -q anthropic pyyaml
   ```

3. **Cell #3: Configure API Keys**
   ```python
   # Auto-loads from .env file
   ```

4. **Cell #4: Initialize System**
   ```python
   # Initializes coordinator
   # Creates NEW session ID
   ```

5. **Cell #7: Start Autonomous System**
   ```python
   # Starts background execution
   # NEW session will be created (session_YYYYMMDD_HHMMSS)
   ```

6. **Cell #8: Monitoring Dashboard**
   ```python
   # Start monitoring (optional)
   ```

---

## 📊 Verify New Session Started

**Check session directory:**
```bash
# On local machine or in Colab
!ls -la /content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/sessions/

# Should see NEW session directory like:
# session_20251014_171530 (new time)
```

**Check GPU access:**
```python
# In Colab
import torch
print("GPU:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("Memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
```

---

## 🆘 If GPU Still Not Available

### Issue: "Cannot connect to GPU backend"

**Causes:**
1. Colab usage limit reached (12 hours continuous use)
2. GPU resources exhausted
3. Colab session expired

**Solutions:**

**Solution 1: Wait and Retry**
- Colab has usage quotas
- Wait 30-60 minutes
- Try reconnecting

**Solution 2: Use Different Account**
- If you have multiple Google accounts
- Switch to different account with available quota

**Solution 3: Colab Pro**
- Colab Free: ~12 hours GPU/day
- Colab Pro: ~24 hours GPU/day
- Colab Pro+: ~unlimited with priority access

**Solution 4: Run Without GPU (For Now)**
- System can run on CPU for planning meetings
- Only execution meetings need GPU (for model experiments)
- Planning meetings (like first meeting) don't need GPU

---

## 🔍 Check What's Using Resources

### In Colab:

**Check Runtime Resources:**
```python
# View resources
# Runtime → View resources
# Shows: RAM, Disk, GPU usage
```

**Check Background Threads:**
```python
import threading
print("Active threads:", threading.active_count())
for thread in threading.enumerate():
    print(f"  - {thread.name}: {thread.is_alive()}")
```

**Check Memory Usage:**
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"Memory: {memory_info.rss / 1024**3:.2f} GB")

# GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    print(f"GPU Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
```

---

## 💡 Clean Restart Procedure (Recommended)

**Complete clean restart with GPU:**

### 1. Stop Everything
```python
# In Colab, run this in a new cell:
import threading
import sys

# Stop all background threads
print("Stopping background threads...")
main_thread = threading.main_thread()
for thread in threading.enumerate():
    if thread is not main_thread:
        print(f"  Stopping: {thread.name}")
        # Note: daemon threads will stop when runtime restarts

print("✅ Ready to restart runtime")
print("👉 Runtime → Restart runtime")
```

### 2. Restart Runtime
- **Runtime** → **Restart runtime**
- Clears all memory
- Releases all resources

### 3. Verify GPU
```python
# First cell after restart
import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ No GPU - Check Runtime → Change runtime type")
```

### 4. Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Initialize System
```python
# Run cells 2-7 from notebook
# System will create NEW session
```

---

## 📋 Session Management

### Old Session (session_20251014_162454)

**Created:** 12:24 PM
**Status:** May still have background threads on Colab
**Action:** Stop by restarting Colab runtime

**Preserved data:**
- Trajectory files in Google Drive
- Meeting transcripts in reports/
- All work saved automatically

### New Session (After Restart)

**Will create:** session_20251014_HHMMSS (current time)
**Status:** Fresh start
**Benefits:**
- Clean GPU access
- No background thread conflicts
- Fresh resource allocation

**Continuity maintained:**
- Can still read old transcripts
- Can generate topics from previous meetings
- History preserved in Google Drive

---

## 🎯 Quick Command Reference

**In Colab:**

**Stop system:**
```
Runtime → Interrupt execution
or
Ctrl+M I (⌘+M I on Mac)
```

**Restart runtime:**
```
Runtime → Restart runtime
or
Ctrl+M . (⌘+M . on Mac)
```

**Check GPU:**
```
Runtime → View resources
or
Runtime → Change runtime type
```

**Force disconnect:**
```
Runtime → Disconnect and delete runtime
(Wait 10 seconds)
Runtime → Connect
```

---

## ✅ After Restart Checklist

- [ ] Runtime restarted (Runtime → Restart runtime)
- [ ] GPU available (check with torch.cuda.is_available())
- [ ] Drive mounted (drive.mount('/content/drive'))
- [ ] Dependencies installed (pip install anthropic pyyaml)
- [ ] API keys loaded (from .env file)
- [ ] Coordinator initialized (Cell #4)
- [ ] System started (Cell #7)
- [ ] New session created (check sessions/ directory)
- [ ] Monitoring working (Cell #8)

---

## 💭 What Happens to Previous Work?

**All work is preserved!**

✅ **Saved in Google Drive:**
- All meeting transcripts (reports/)
- All action items (actions_*.json)
- All summaries (summary_*.md)
- Previous session trajectory (sessions/session_20251014_162454/)

✅ **Can continue from where you left off:**
- New session can read old transcripts
- Topic generator uses latest meeting
- No work lost

❌ **Not preserved:**
- In-memory variables (cleared on restart)
- Background thread state (stopped)
- GPU reservations (released)

**This is good!** Clean restart with all history intact.

---

## 🎓 Summary

**To restart system:**
1. **Stop:** Runtime → Restart runtime (in Colab)
2. **Verify GPU:** Runtime → View resources
3. **Restart:** Run cells 1-7 in notebook
4. **Monitor:** Run cell 8 to see progress

**Current session:** session_20251014_162454 (will stop)
**New session:** session_20251014_HHMMSS (will create)
**Work preserved:** ✅ All in Google Drive

**GPU issue:** Restart runtime to release resources and reconnect

---

**Status: 🟢 READY TO RESTART**
**Action: Go to Colab → Runtime → Restart runtime**
