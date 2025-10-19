# Timestamped Filenames Fix - Quick Implementation

**Date:** 2025-10-15
**Issue:** `execution_progress_update.md` and `pending_actions.json` need timestamps like `execution_summary_20251015_005911.md`

---

## 🎯 Quick Summary

**Problem:** Files without timestamps are hard to locate across cycles
**Solution:** Add timestamps to ALL handoff files

**Files to update:**
1. ✅ `cvpr_autonomous_execution_cycle.ipynb` (Executive Team)
2. ✅ `planning_team_review_cycle.ipynb` (Planning Team)

---

## 📝 Exact Code Changes

### **1. Executive Team Notebook (`cvpr_autonomous_execution_cycle.ipynb`)**

**Find Cell ~13 (Phase 4: Generate Report and Save Results)**

**REPLACE THIS CODE:**
```python
# Generate execution progress update
summary = tracker.get_summary()

progress_report = f"""# Executive Team Progress Update
...
"""

# Save progress update
progress_file = Path(MULTI_AGENT_ROOT) / 'reports/handoff/execution_progress_update.md'
progress_file.write_text(progress_report, encoding='utf-8')
print("✅ Progress update saved: execution_progress_update.md")
```

**WITH THIS CODE:**
```python
# Generate execution progress update with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
summary = tracker.get_summary()

progress_report = f"""# Executive Team Progress Update

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Cycle Timestamp:** {timestamp}
**Mission:** CVPR 2025 Week 1 - Cross-architecture attention collapse validation
**Meeting ID:** {pending_actions.get('meeting_id', 'unknown')}

---

## 📊 Execution Summary

**Total Tasks:** {summary['total_tasks']}
**Completed:** {summary['completed']} ✅
**Failed:** {summary['failed']} ❌
**Execution Time:** {summary['total_duration_seconds']:.1f} seconds

---

## 📋 Task Details

"""

for task in summary['task_results']:
    progress_report += f"""
### Task {task['task_id']}: {task['action']}

**Priority:** {task['priority']}
**Status:** {task['status']}
**Duration:** {(datetime.fromisoformat(task['end_time']) - datetime.fromisoformat(task['start_time'])).total_seconds():.1f}s

**Agent Responses:**

"""
    for agent, response in task['agent_responses'].items():
        progress_report += f"""
#### {agent}:
{response[:1000]}{'...' if len(response) > 1000 else ''}

---
"""

# Save progress update with timestamp
progress_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/execution_progress_update_{timestamp}.md'
progress_file.write_text(progress_report, encoding='utf-8')
print(f"✅ Progress update saved: execution_progress_update_{timestamp}.md")

# Also save to execution directory for backup
backup_file = Path(MULTI_AGENT_ROOT) / f'reports/execution/execution_progress_{timestamp}.md'
backup_file.parent.mkdir(parents=True, exist_ok=True)
backup_file.write_text(progress_report, encoding='utf-8')
print(f"✅ Backup saved: reports/execution/execution_progress_{timestamp}.md")
```

**Find Cell ~14 (Save execution results JSON)**

**ADD THIS LINE at the start if not present:**
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Reuse timestamp from previous cell
```

**Find Cell ~17 (Create next meeting trigger)**

**REPLACE THIS CODE:**
```python
# Save trigger
trigger_file = Path(MULTI_AGENT_ROOT) / 'reports/handoff/next_meeting_trigger.json'
trigger_file.write_text(json.dumps(next_meeting_trigger, indent=2), encoding='utf-8')
print("✅ Next meeting trigger saved")
```

**WITH THIS CODE:**
```python
# Save trigger with timestamp
trigger_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/next_meeting_trigger_{timestamp}.json'
trigger_file.write_text(json.dumps(next_meeting_trigger, indent=2), encoding='utf-8')
print(f"✅ Next meeting trigger saved: next_meeting_trigger_{timestamp}.json")
```

---

### **2. Planning Team Notebook (`planning_team_review_cycle.ipynb`)**

**Find Cell that reads execution results (Phase 1)**

**REPLACE THIS CODE:**
```python
# Read execution results
progress_file = handoff_dir / 'execution_progress_update.md'
with open(progress_file, 'r') as f:
    progress_content = f.read()
```

**WITH THIS CODE:**
```python
# Find latest execution progress file by timestamp
from pathlib import Path

handoff_dir = Path(MULTI_AGENT_ROOT) / 'reports/handoff'
execution_dir = Path(MULTI_AGENT_ROOT) / 'reports/execution'

# Find all execution progress files
progress_files = list(handoff_dir.glob('execution_progress_update_*.md'))
if not progress_files:
    # Fallback to execution directory
    progress_files = list(execution_dir.glob('execution_progress_*.md'))

if not progress_files:
    print("❌ No execution progress files found")
    print(f"   Searched: {handoff_dir}")
    print(f"   Searched: {execution_dir}")
    raise FileNotFoundError("No execution results to review - Executive Team must run first")

# Sort by timestamp in filename (YYYYMMDD_HHMMSS format)
latest_progress = sorted(progress_files, key=lambda p: p.stem.split('_')[-2:])[-1]

print(f"✅ Found latest execution progress: {latest_progress.name}")
print(f"📁 Location: {latest_progress.parent}")
print(f"📊 Size: {latest_progress.stat().st_size} bytes\n")

# Read the latest progress file
with open(latest_progress, 'r', encoding='utf-8') as f:
    progress_content = f.read()

print("📄 Progress Content Preview:")
print("=" * 80)
print(progress_content[:500] + "..." if len(progress_content) > 500 else progress_content)
print("=" * 80)
```

**Find Cell that saves pending_actions (Phase 4 or final phase)**

**REPLACE THIS CODE:**
```python
# Save new pending_actions.json
pending_file = handoff_dir / 'pending_actions.json'
pending_file.write_text(next_cycle_json, encoding='utf-8')
print("✅ New pending_actions.json saved")
```

**WITH THIS CODE:**
```python
# Save new pending_actions with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save to handoff directory with timestamp
pending_file = handoff_dir / f'pending_actions_{timestamp}.json'
pending_file.write_text(next_cycle_json, encoding='utf-8')
print(f"✅ Pending actions saved: pending_actions_{timestamp}.json")

# Also save to planning history for backup
history_dir = Path(MULTI_AGENT_ROOT) / 'reports/planning/pending_actions_history'
history_dir.mkdir(parents=True, exist_ok=True)
history_file = history_dir / f'pending_actions_{timestamp}.json'
history_file.write_text(next_cycle_json, encoding='utf-8')
print(f"✅ Backup saved: pending_actions_history/pending_actions_{timestamp}.json")
```

---

## 🔧 Alternative: Manual Update in Colab

If you prefer to update directly in Colab instead of downloading/uploading:

**1. Open `cvpr_autonomous_execution_cycle.ipynb` in Colab**
   - Find the cells mentioned above
   - Edit code directly in Colab
   - Run "File" → "Save" (Ctrl/Cmd+S)

**2. Open `planning_team_review_cycle.ipynb` in Colab**
   - Find the cells mentioned above
   - Edit code directly in Colab
   - Run "File" → "Save"

**3. Test with a quick cycle:**
   - Run Executive Team notebook
   - Check that `execution_progress_update_YYYYMMDD_HHMMSS.md` is created
   - Run Planning Team notebook
   - Check that it finds the latest file
   - Check that `pending_actions_YYYYMMDD_HHMMSS.json` is created

---

## ✅ Expected Results After Update

**After running Executive Team (Cycle 2):**
```
reports/handoff/
├── execution_progress_update_20251015_005911.md  ← NEW FORMAT
├── execution_results_20251015_005911.json         ← ALREADY GOOD
├── execution_summary_20251015_005911.md           ← ALREADY GOOD
└── next_meeting_trigger_20251015_005911.json      ← NEW FORMAT
```

**After running Planning Team:**
```
reports/handoff/
└── pending_actions_20251015_123000.json           ← NEW FORMAT

reports/planning/pending_actions_history/
└── pending_actions_20251015_123000.json           ← BACKUP
```

**Verification:**
```bash
# List all execution progress files (sorted by time)
ls -lt reports/handoff/execution_progress_update_*.md

# List all pending actions files (sorted by time)
ls -lt reports/handoff/pending_actions_*.json

# Find latest (most recent) files
ls -t reports/handoff/execution_progress_update_*.md | head -1
ls -t reports/handoff/pending_actions_*.json | head -1
```

---

## 🚀 Next Steps

**Option 1: Quick Manual Update (Recommended)**
1. Open both notebooks in Google Colab
2. Copy-paste the code changes from this document
3. Save both notebooks
4. Run a test cycle to verify

**Option 2: Automated Script Update**
1. Run `update_notebooks_with_timestamps.py` script
2. Upload updated notebooks to Google Colab
3. Run a test cycle to verify

**After Update:**
- ✅ All files will have timestamps
- ✅ Easy to find specific cycles
- ✅ Complete history preserved
- ✅ Planning Team can automatically find latest results

---

## 📋 Checklist

- [ ] Update `cvpr_autonomous_execution_cycle.ipynb` Cell ~13 (progress file)
- [ ] Update `cvpr_autonomous_execution_cycle.ipynb` Cell ~17 (trigger file)
- [ ] Update `planning_team_review_cycle.ipynb` - find latest progress file
- [ ] Update `planning_team_review_cycle.ipynb` - save pending_actions with timestamp
- [ ] Test full cycle: Planning → Executive → Planning
- [ ] Verify all files created with timestamps
- [ ] Verify Planning Team finds latest files correctly

**After completing checklist, all handoff files will use timestamped filenames!**
