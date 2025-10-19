# Timestamped Filename System - Implementation Guide

**Date:** 2025-10-15
**Issue:** Files like `execution_progress_update.md` and `pending_actions.json` are hard to locate without timestamps

---

## 🎯 Problem

User reported:
> "execution_progress_update.md this made it hard to locate and cannot find it, execution_summary_20251015_005911.md this one is better"

**Current system:**
- ❌ `execution_progress_update.md` - no timestamp, gets overwritten
- ❌ `pending_actions.json` - no timestamp, gets overwritten
- ✅ `execution_summary_20251015_005911.md` - timestamped, easy to find
- ✅ `execution_results_20251015_005911.json` - timestamped, easy to find

**Why it's a problem:**
1. Can't distinguish between different execution cycles
2. Hard to find specific results from a particular run
3. Files get overwritten, losing history
4. Planning Team can't easily find the latest results

---

## ✅ Solution: Use Timestamps for ALL Handoff Files

### **New Naming Convention:**

**Executive Team outputs (after each cycle):**
```
reports/handoff/execution_progress_update_YYYYMMDD_HHMMSS.md
reports/handoff/execution_results_YYYYMMDD_HHMMSS.json  (already timestamped ✅)
reports/handoff/execution_summary_YYYYMMDD_HHMMSS.md    (already timestamped ✅)
reports/handoff/next_meeting_trigger_YYYYMMDD_HHMMSS.json
```

**Planning Team outputs (after each meeting):**
```
reports/handoff/pending_actions_YYYYMMDD_HHMMSS.json
reports/planning/planning_meeting_summary_YYYYMMDD_HHMMSS.md
```

**Symlinks for "latest" (optional):**
```
reports/handoff/latest_execution_progress.md → execution_progress_update_20251015_005911.md
reports/handoff/latest_pending_actions.json → pending_actions_20251015_001500.json
```

---

## 📝 Files That Need Updating

### **1. Colab Notebooks (2 files)**

#### **A. `cvpr_autonomous_execution_cycle.ipynb`**
**Location:** `research/colab/cvpr_autonomous_execution_cycle.ipynb`

**Cell 13 - Save execution progress:**
```python
# Generate execution progress update with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
summary = tracker.get_summary()

progress_report = f"""# Executive Team Progress Update

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Cycle Timestamp:** {timestamp}
**Mission:** CVPR 2025 Week 1 - Cross-architecture attention collapse validation
...
"""

# Save with timestamp
progress_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/execution_progress_update_{timestamp}.md'
progress_file.write_text(progress_report, encoding='utf-8')
print(f"✅ Progress update saved: execution_progress_update_{timestamp}.md")

# Also save to backup location
backup_file = Path(MULTI_AGENT_ROOT) / f'reports/execution/execution_progress_{timestamp}.md'
backup_file.parent.mkdir(parents=True, exist_ok=True)
backup_file.write_text(progress_report, encoding='utf-8')
```

**Cell 14 - Save execution results:**
```python
# Save execution results with timestamp (already using timestamp ✅)
results_file = Path(MULTI_AGENT_ROOT) / f'reports/execution/execution_results_{timestamp}.json'
summary_file = Path(MULTI_AGENT_ROOT) / f'reports/execution/execution_summary_{timestamp}.md'
# ... rest of code
```

**Cell 17 - Save next meeting trigger:**
```python
# Save trigger with timestamp
trigger_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/next_meeting_trigger_{timestamp}.json'
trigger_file.write_text(json.dumps(next_meeting_trigger, indent=2), encoding='utf-8')
print(f"✅ Next meeting trigger saved: next_meeting_trigger_{timestamp}.json")
```

#### **B. `planning_team_review_cycle.ipynb`**
**Location:** `research/colab/planning_team_review_cycle.ipynb`

**Cell for finding latest execution results:**
```python
# Find latest execution results (by timestamp in filename)
import re
from pathlib import Path

handoff_dir = Path(MULTI_AGENT_ROOT) / 'reports/handoff'
execution_dir = Path(MULTI_AGENT_ROOT) / 'reports/execution'

# Find all execution progress files
progress_files = list(handoff_dir.glob('execution_progress_update_*.md'))
if not progress_files:
    # Fallback to execution directory
    progress_files = list(execution_dir.glob('execution_progress_*.md'))

if progress_files:
    # Sort by timestamp in filename
    latest_progress = sorted(progress_files, key=lambda p: p.stem.split('_')[-2:])[-1]
    print(f"✅ Found latest execution progress: {latest_progress.name}")
else:
    print("❌ No execution progress files found")
    raise FileNotFoundError("No execution results to review")

# Read the latest progress file
with open(latest_progress, 'r') as f:
    progress_content = f.read()

print(f"📄 Progress file: {latest_progress.name}")
print(f"📊 Size: {len(progress_content)} chars")
```

**Cell for saving pending actions:**
```python
# Save new pending_actions with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save to handoff directory with timestamp
pending_actions_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/pending_actions_{timestamp}.json'
pending_actions_file.write_text(next_cycle_json, encoding='utf-8')
print(f"✅ Pending actions saved: pending_actions_{timestamp}.json")

# Also save to planning history
history_file = Path(MULTI_AGENT_ROOT) / f'reports/planning/pending_actions_history/pending_actions_{timestamp}.json'
history_file.parent.mkdir(parents=True, exist_ok=True)
history_file.write_text(next_cycle_json, encoding='utf-8')
print(f"✅ Backup saved: pending_actions_history/pending_actions_{timestamp}.json")
```

### **2. Python Scripts**

#### **A. `autonomous_cycle_coordinator.py`**
**Location:** `multi-agent/autonomous_cycle_coordinator.py`

**Lines to update (~115-120):**
```python
# OLD:
self.progress_update_file = self.handoff_dir / 'execution_progress_update.md'

# NEW:
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
self.progress_update_file = self.handoff_dir / f'execution_progress_update_{timestamp}.md'
self.execution_timestamp = timestamp  # Store for later use
```

**Lines to update (~200-210):**
```python
# When saving progress update:
progress_file = self.handoff_dir / f'execution_progress_update_{self.execution_timestamp}.md'
with open(progress_file, 'w') as f:
    f.write(progress_report)

print(f"✅ Progress saved: execution_progress_update_{self.execution_timestamp}.md")
```

#### **B. `run_planning_review_meeting.py`**
**Location:** `multi-agent/scripts/run_planning_review_meeting.py`

**Lines to update (~50-60):**
```python
# Find latest execution progress file
handoff_dir = Path(__file__).parent.parent / 'reports/handoff'
progress_files = list(handoff_dir.glob('execution_progress_update_*.md'))

if not progress_files:
    # Fallback to execution directory
    execution_dir = Path(__file__).parent.parent / 'reports/execution'
    progress_files = list(execution_dir.glob('execution_progress_*.md'))

if not progress_files:
    raise FileNotFoundError("No execution results found - Executive Team must run first")

# Get latest by filename timestamp
latest_progress = sorted(progress_files, key=lambda p: p.stem.split('_')[-2:])[-1]
print(f"📄 Reading: {latest_progress.name}")

with open(latest_progress, 'r') as f:
    progress_content = f.read()
```

**Lines to update (~150-160):**
```python
# Save new pending_actions with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

pending_file = handoff_dir / f'pending_actions_{timestamp}.json'
with open(pending_file, 'w') as f:
    json.dump(pending_actions, f, indent=2)

print(f"✅ Saved: pending_actions_{timestamp}.json")
```

---

## 🔍 How to Find Latest Files

### **Method 1: Glob + Sort**
```python
from pathlib import Path

def find_latest_file(pattern, directory):
    """Find latest file matching pattern by timestamp in filename."""
    files = list(Path(directory).glob(pattern))
    if not files:
        return None
    # Sort by timestamp in filename (assumes YYYYMMDD_HHMMSS format)
    return sorted(files, key=lambda p: p.stem.split('_')[-2:])[-1]

# Usage:
latest_progress = find_latest_file('execution_progress_update_*.md', 'reports/handoff')
latest_pending = find_latest_file('pending_actions_*.json', 'reports/handoff')
```

### **Method 2: File Modification Time (Fallback)**
```python
def find_latest_file_by_mtime(pattern, directory):
    """Find latest file by modification time."""
    files = list(Path(directory).glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)
```

---

## 📂 Directory Structure After Update

```
reports/
├── handoff/
│   ├── execution_progress_update_20251014_234612.md
│   ├── execution_progress_update_20251015_005911.md  ← Cycle 2
│   ├── execution_results_20251014_234612.json
│   ├── execution_results_20251015_005911.json
│   ├── execution_summary_20251014_234612.md
│   ├── execution_summary_20251015_005911.md
│   ├── next_meeting_trigger_20251014_234612.json
│   ├── next_meeting_trigger_20251015_005911.json
│   ├── pending_actions_20251015_001500.json          ← From Planning Cycle 2
│   └── pending_actions_20251015_123000.json          ← From Planning Cycle 3
│
├── execution/
│   ├── execution_progress_20251014_234612.md
│   ├── execution_progress_20251015_005911.md
│   ├── execution_results_20251014_234612.json
│   └── execution_results_20251015_005911.json
│
└── planning/
    └── pending_actions_history/
        ├── pending_actions_20251015_001500.json
        └── pending_actions_20251015_123000.json
```

---

## ✅ Benefits

1. **Easy to Find:** Each cycle has unique timestamp
2. **Complete History:** Nothing gets overwritten
3. **Easy to Debug:** Can trace back to specific cycles
4. **Sortable:** Alphabetical sort = chronological order
5. **Clear Communication:** Agents can reference specific files by timestamp

---

## 🚀 Implementation Priority

**HIGH PRIORITY:**
1. Update `cvpr_autonomous_execution_cycle.ipynb` (Executive Team)
2. Update `planning_team_review_cycle.ipynb` (Planning Team)

**MEDIUM PRIORITY:**
3. Update `autonomous_cycle_coordinator.py`
4. Update `run_planning_review_meeting.py`

**LOW PRIORITY:**
5. Update documentation
6. Add symlinks for "latest" files (optional)

---

## 🧪 Testing

After updating, verify:
```bash
# Check that files are created with timestamps
ls -la reports/handoff/execution_progress_update_*.md
ls -la reports/handoff/pending_actions_*.json

# Find latest files
ls -t reports/handoff/execution_progress_update_*.md | head -1
ls -t reports/handoff/pending_actions_*.json | head -1
```

---

## 📋 Next Steps

1. ✅ Update both Colab notebooks with timestamped filenames
2. ✅ Update Python scripts to find latest files by timestamp
3. ✅ Test with a full cycle (Planning → Executive → Planning)
4. ✅ Verify all files are created with correct timestamps
5. ✅ Update documentation with new filename patterns

**After implementation, you'll be able to easily find and reference any cycle's results!**
