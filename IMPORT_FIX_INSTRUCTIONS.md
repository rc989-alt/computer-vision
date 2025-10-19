# 🔧 Import Issue Fix - ModuleNotFoundError: run_meeting

**Error:** `ModuleNotFoundError: No module named 'run_meeting'`

**Location:** `executive_coordinator.py` line 19

---

## 🎯 Root Cause

The `executive_coordinator.py` file tries to import `run_meeting` module, but the Python path isn't set correctly in Colab before the import happens.

**Problematic code:**
```python
# Line 17-19 in executive_coordinator.py
sys.path.insert(0, str(Path(__file__).parent / "multi-agent"))
from run_meeting import MeetingOrchestrator
```

**Issue:** `Path(__file__)` doesn't work correctly when file is executed via `exec(open(...).read())` in Colab

---

## ✅ Solution 1: Add Path Setup in Notebook (RECOMMENDED)

**Add this cell BEFORE running deploy script:**

```python
import sys
from pathlib import Path

# Setup paths for imports
PROJECT = Path("/content/cv_project")
MULTI_AGENT = PROJECT / "multi-agent"

# Add to Python path
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(MULTI_AGENT))

print(f"✅ Added to sys.path:")
print(f"   - {PROJECT}")
print(f"   - {MULTI_AGENT}")

# Verify run_meeting.py exists
if (MULTI_AGENT / "run_meeting.py").exists():
    print(f"✅ run_meeting.py found")
else:
    print(f"❌ run_meeting.py NOT FOUND at {MULTI_AGENT}")
```

Then run deploy script as normal.

---

## ✅ Solution 2: Automatic Fix Script

**Run this in notebook:**

```python
# Fix import issue automatically
exec(open('/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/fix_import_issue.py').read())
```

This will patch `executive_coordinator.py` to handle the import correctly.

---

## ✅ Solution 3: Manual Edit (If needed)

**Edit `/content/cv_project/executive_coordinator.py`:**

Replace lines 15-19:

```python
# OLD (lines 15-19):
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "multi-agent"))

from run_meeting import MeetingOrchestrator
```

With:

```python
# NEW:
import sys
import os

# Add paths - handle both normal Python and Colab exec()
if '__file__' in globals():
    project_root = Path(__file__).parent
else:
    # In Colab with exec(), __file__ isn't defined
    project_root = Path("/content/cv_project")

# Add multi-agent to path
sys.path.insert(0, str(project_root / "multi-agent"))
sys.path.insert(0, str(project_root))

# Now import
from run_meeting import MeetingOrchestrator
```

---

## 📊 Verification

**After applying fix, verify:**

```python
import sys
print("Python path:")
for p in sys.path:
    if 'cv_project' in p or 'multi-agent' in p:
        print(f"  ✅ {p}")

# Test import
try:
    sys.path.insert(0, "/content/cv_project/multi-agent")
    from run_meeting import MeetingOrchestrator
    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

---

## 🚀 Updated Notebook Cell (Step 3)

**Replace Step 3 in `executive_system_improved.ipynb` with:**

```python
# Install packages first
print("📦 Installing required packages...")
!pip install -q anthropic openai google-generativeai pyyaml

# Load API keys
import os
env_file = "/content/cv_project/.env"
with open(env_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

print("✅ API keys loaded")

# ===== FIX: Add paths before deploying =====
import sys
from pathlib import Path

PROJECT = Path("/content/cv_project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "multi-agent"))

print(f"✅ Added to sys.path:")
print(f"   - {PROJECT}")
print(f"   - {PROJECT / 'multi-agent'}")

# Verify run_meeting.py exists
if (PROJECT / "multi-agent/run_meeting.py").exists():
    print("✅ run_meeting.py found")

    # Test import
    try:
        from run_meeting import MeetingOrchestrator
        print("✅ Import test successful")
    except ImportError as e:
        print(f"⚠️  Import test failed: {e}")
else:
    print("❌ run_meeting.py NOT FOUND")
# ===== END FIX =====

# Deploy system
print("\n🚀 Deploying autonomous system...")
exec(open('/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/deploy_updated_system.py').read())
```

---

## 🔍 Why This Happens

**In normal Python:**
- `__file__` contains the path to the currently executing script
- `Path(__file__).parent` correctly resolves to project root

**In Colab with `exec(open(...).read())`:**
- `__file__` is NOT defined or points to wrong location
- `Path(__file__).parent` fails or returns incorrect path
- Module imports fail

**Solution:**
- Explicitly set sys.path before running exec()
- Use absolute paths instead of relative paths
- Handle both cases (normal Python and Colab exec)

---

## ✅ Quick Fix for Running NOW

**Just add this cell before Step 3 in your notebook:**

```python
# QUICK FIX: Setup Python path
import sys
from pathlib import Path

PROJECT = Path("/content/cv_project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "multi-agent"))

print("✅ Python path configured for imports")
```

Then proceed with Step 3 as normal. The import will work!

---

**Created:** October 13, 2025
**Status:** Fix Available - Choose Solution 1 (Recommended)
