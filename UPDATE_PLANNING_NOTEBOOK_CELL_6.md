# Planning Team Notebook - Cell 6 Update

**File:** `planning_team_review_cycle.ipynb`
**Cell to Update:** Cell 6 (Phase 1: Read Execution Results)

---

## 🎯 Current Problem

**Current code (Cell 6):**
```python
# Read progress update (main handoff file)
progress_file = Path(MULTI_AGENT_ROOT) / 'reports/handoff/execution_progress_update.md'

if not progress_file.exists():
    print(f"❌ No execution results found at {progress_file}")
    raise FileNotFoundError(f"Missing: {progress_file}")
```

**Issue:** Looks for non-timestamped file that won't exist after Executive Team implements timestamps.

---

## ✅ Updated Code (Replace entire Cell 6)

```python
import json
from datetime import datetime
from pathlib import Path

print("="*80)
print("📥 READING EXECUTIVE TEAM EXECUTION RESULTS")
print("="*80)

# Find latest execution progress file by timestamp
handoff_dir = Path(MULTI_AGENT_ROOT) / 'reports/handoff'
execution_dir = Path(MULTI_AGENT_ROOT) / 'reports/execution'

# Search for timestamped execution progress files
print("\n🔍 Searching for latest execution results...")

# Method 1: Look in handoff directory for execution_progress_update_*.md
progress_files = list(handoff_dir.glob('execution_progress_update_*.md'))

if not progress_files:
    # Method 2: Fallback to execution/summaries directory
    summaries_dir = execution_dir / 'summaries'
    if summaries_dir.exists():
        progress_files = list(summaries_dir.glob('execution_summary_*.md'))

if not progress_files:
    # Method 3: Fallback to execution directory
    progress_files = list(execution_dir.glob('execution_progress_*.md'))

if not progress_files:
    print(f"\n❌ No execution progress files found!")
    print(f"   Searched locations:")
    print(f"   - {handoff_dir}/execution_progress_update_*.md")
    print(f"   - {execution_dir}/summaries/execution_summary_*.md")
    print(f"   - {execution_dir}/execution_progress_*.md")
    print(f"\n⚠️ Executive Team must complete execution first")
    raise FileNotFoundError("No execution results found to review")

# Sort by timestamp in filename (YYYYMMDD_HHMMSS format)
# Extract timestamp from filename like: execution_progress_update_20251015_005911.md
def extract_timestamp(filepath):
    """Extract timestamp from filename for sorting."""
    import re
    match = re.search(r'(\d{8}_\d{6})', filepath.name)
    if match:
        return match.group(1)
    return ''

latest_progress = sorted(progress_files, key=extract_timestamp)[-1]

print(f"\n✅ Found latest execution progress file")
print(f"   📄 File: {latest_progress.name}")
print(f"   📁 Location: {latest_progress.parent}")
print(f"   📊 Size: {latest_progress.stat().st_size:,} bytes")
print(f"   🕒 Modified: {datetime.fromtimestamp(latest_progress.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

# Read the latest progress file
with open(latest_progress, 'r', encoding='utf-8') as f:
    progress_text = f.read()

print(f"\n📖 Content loaded: {len(progress_text):,} characters")

# Find most recent JSON results (using same timestamp search)
results_dir = Path(MULTI_AGENT_ROOT) / 'reports/execution/results'
json_files = list(results_dir.glob('execution_results_*.json'))

if json_files:
    # Get most recent by timestamp in filename
    latest_json = sorted(json_files, key=extract_timestamp)[-1]

    print(f"\n✅ Found execution results JSON")
    print(f"   📄 File: {latest_json.name}")
    print(f"   📁 Location: {latest_json.parent}")

    try:
        with open(latest_json, 'r', encoding='utf-8') as f:
            results_data = json.load(f)

        print(f"   📊 Tasks: {results_data.get('total_tasks', 'N/A')}")
        print(f"   ✅ Completed: {results_data.get('completed', 'N/A')}")
        print(f"   ❌ Failed: {results_data.get('failed', 'N/A')}")
        print(f"   ⏱️  Duration: {results_data.get('total_duration_seconds', 0):.1f}s")
    except json.JSONDecodeError as e:
        print(f"   ⚠️ Could not parse JSON: {e}")
        results_data = {}
else:
    results_data = {}
    latest_json = None
    print("\n⚠️ No JSON results found, using markdown only")

print("\n" + "="*80)
print("📊 EXECUTION SUMMARY FOR PLANNING TEAM REVIEW")
print("="*80)

if results_data:
    print(f"\n📋 Executive Team Results:")
    print(f"   📅 Execution Date: {latest_progress.name.split('_')[3][:8]}")  # Extract YYYYMMDD
    print(f"   ⏰ Execution Time: {latest_progress.name.split('_')[-1].replace('.md', '')}")  # Extract HHMMSS
    print(f"   📦 Total tasks: {results_data.get('total_tasks', 'N/A')}")
    print(f"   ✅ Completed: {results_data.get('completed', 'N/A')}")
    print(f"   ❌ Failed: {results_data.get('failed', 'N/A')}")
    print(f"   ⏱️  Duration: {results_data.get('total_duration_seconds', 0):.1f}s ({results_data.get('total_duration_seconds', 0)/60:.1f} minutes)")

print(f"\n📄 Progress Report Preview (first 500 chars):")
print("-" * 80)
print(progress_text[:500])
print("..." if len(progress_text) > 500 else "")
print("-" * 80)

print(f"\n📊 Full report loaded for Planning Team analysis")
print(f"   Total size: {len(progress_text):,} characters")
print(f"   Ready for agent review")

print("\n" + "="*80)
```

---

## 🔍 How It Works

**Smart Search Strategy:**

1. **Primary:** Search `reports/handoff/execution_progress_update_*.md`
2. **Fallback 1:** Search `reports/execution/summaries/execution_summary_*.md`
3. **Fallback 2:** Search `reports/execution/execution_progress_*.md`

**Timestamp Extraction:**
- Uses regex to find `YYYYMMDD_HHMMSS` pattern in filename
- Sorts files by extracted timestamp
- Takes the latest (most recent) file

**Benefits:**
- ✅ Automatically finds latest execution results
- ✅ Works with any timestamp format
- ✅ Shows clear information about which file was found
- ✅ Displays execution date/time from filename
- ✅ Provides helpful error messages if no files found

---

## 📋 Example Output

After update, you'll see:

```
================================================================================
📥 READING EXECUTIVE TEAM EXECUTION RESULTS
================================================================================

🔍 Searching for latest execution results...

✅ Found latest execution progress file
   📄 File: execution_progress_update_20251015_005911.md
   📁 Location: /content/drive/.../reports/handoff
   📊 Size: 27,092 bytes
   🕒 Modified: 2025-10-15 00:59:11

📖 Content loaded: 27,092 characters

✅ Found execution results JSON
   📄 File: execution_results_20251015_005911.json
   📁 Location: /content/drive/.../reports/execution/results
   📊 Tasks: 8
   ✅ Completed: 8
   ❌ Failed: 0
   ⏱️  Duration: 845.0s

================================================================================
📊 EXECUTION SUMMARY FOR PLANNING TEAM REVIEW
================================================================================

📋 Executive Team Results:
   📅 Execution Date: 20251015
   ⏰ Execution Time: 005911
   📦 Total tasks: 8
   ✅ Completed: 8
   ❌ Failed: 0
   ⏱️  Duration: 845.0s (14.1 minutes)

📄 Progress Report Preview (first 500 chars):
--------------------------------------------------------------------------------
# Executive Team Progress Update

**Date:** 2025-10-15 00:59:11
**Cycle Timestamp:** 20251015_005911
**Mission:** CVPR 2025 Week 1 - Cross-architecture attention collapse validation
...
--------------------------------------------------------------------------------

📊 Full report loaded for Planning Team analysis
   Total size: 27,092 characters
   Ready for agent review

================================================================================
```

---

## 🚀 How to Update

**Option 1: Direct Edit in Colab (Recommended)**

1. Open `planning_team_review_cycle.ipynb` in Google Colab
2. Find Cell 6 (should be in "Phase 1: Read Execution Results")
3. Select all code in the cell and delete it
4. Copy the "Updated Code" from above
5. Paste into the cell
6. Save the notebook (Ctrl/Cmd + S)

**Option 2: Download/Upload**

1. Download notebook from Google Drive
2. Edit cell 6 locally with the updated code
3. Upload back to Google Drive
4. Open in Colab

---

## ✅ Verification

After updating, test by running the notebook:

1. Cell 6 should find the latest timestamped file
2. Should display clear information about which execution it found
3. Should show execution date and time from the filename
4. Should work across multiple execution cycles

**What to check:**
- ✅ Does it find `execution_progress_update_20251015_005911.md`?
- ✅ Does it display the correct timestamp?
- ✅ Does it load the full content?
- ✅ Does it find the corresponding JSON file?

---

## 🎯 Result

After this update:
- ✅ Planning Team automatically finds latest execution results
- ✅ No more hardcoded filename issues
- ✅ Works with timestamped files from Executive Team
- ✅ Clear visibility into which cycle is being reviewed
- ✅ Robust fallback search across multiple directories

**Planning Team will now correctly read the latest timestamped execution results every time!**
