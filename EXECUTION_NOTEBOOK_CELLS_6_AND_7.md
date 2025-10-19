# Executive Team Execution Notebook - Cells 6 & 7

**File:** `cvpr_autonomous_execution_cycle.ipynb`
**Location:** `research/colab/cvpr_autonomous_execution_cycle.ipynb`

---

## Cell 6 (Markdown)

**Type:** Markdown
**Purpose:** Section header for Phase 1

```markdown
---
## Phase 1: Read Pending Actions from Planning Team
```

---

## Cell 7 (Code)

**Type:** Code
**Purpose:** Read pending_actions.json from Planning Team and display task list

```python
import json
from datetime import datetime
from pathlib import Path

# Read pending actions from Planning Team
pending_actions_file = Path(MULTI_AGENT_ROOT) / 'reports/handoff/pending_actions.json'

if not pending_actions_file.exists():
    print(f"❌ No pending actions found at {pending_actions_file}")
    print("⚠️ Planning Team must generate pending_actions.json first")
    raise FileNotFoundError(f"Missing: {pending_actions_file}")

with open(pending_actions_file, 'r') as f:
    pending_actions = json.load(f)

print("="*80)
print("📥 PENDING ACTIONS FROM PLANNING TEAM")
print("="*80)
print(f"\n📋 Meeting ID: {pending_actions.get('meeting_id')}")
print(f"🗓️  Generated: {pending_actions.get('generated_at')}")
print(f"🎯 Context: {pending_actions.get('context')}")

decisions = pending_actions.get('decisions', [])
print(f"\n📊 Total tasks: {len(decisions)}")

# Group by priority
high_priority = [d for d in decisions if d.get('priority') == 'HIGH']
medium_priority = [d for d in decisions if d.get('priority') == 'MEDIUM']
low_priority = [d for d in decisions if d.get('priority') == 'LOW']

print(f"   ⭐ HIGH: {len(high_priority)}")
print(f"   🟠 MEDIUM: {len(medium_priority)}")
print(f"   🔵 LOW: {len(low_priority)}")

print("\n📋 Task List:")
for i, decision in enumerate(decisions, 1):
    priority = decision.get('priority', 'UNKNOWN')
    action = decision.get('action', 'No action specified')
    owner = decision.get('owner', 'unassigned')
    deadline = decision.get('deadline', 'No deadline')

    priority_icon = '⭐' if priority == 'HIGH' else '🟠' if priority == 'MEDIUM' else '🔵'
    print(f"\n{i}. {priority_icon} [{priority}] {action}")
    print(f"   👤 Owner: {owner}")
    print(f"   ⏰ Deadline: {deadline}")

print("\n" + "="*80)
```

---

## Expected Output

When Cell 7 runs successfully, you should see:

```
================================================================================
📥 PENDING ACTIONS FROM PLANNING TEAM
================================================================================

📋 Meeting ID: cvpr_planning_cycle2_20251015_010000
🗓️  Generated: 2025-10-15T01:00:00Z
🎯 Context: Week 1 Cycle 2: Memory optimization, temporal stability validation...

📊 Total tasks: 8
   ⭐ HIGH: 4
   🟠 MEDIUM: 3
   🔵 LOW: 1

📋 Task List:

1. ⭐ [HIGH] Implement memory optimization for Flamingo and LLaVA models
   👤 Owner: infrastructure
   ⏰ Deadline: 2025-10-16T12:00:00Z

2. ⭐ [HIGH] Execute temporal stability analysis with 5 independent runs
   👤 Owner: ops_commander
   ⏰ Deadline: 2025-10-17T23:59:59Z

3. ⭐ [HIGH] Design and implement attention intervention experiment
   👤 Owner: ops_commander
   ⏰ Deadline: 2025-10-18T23:59:59Z

...

================================================================================
```

---

## How to Add Back to Notebook

**In Google Colab:**

1. **Add Markdown Cell (Cell 6):**
   - Click "+ Code" dropdown → Select "Text"
   - Paste the markdown content:
   ```markdown
   ---
   ## Phase 1: Read Pending Actions from Planning Team
   ```

2. **Add Code Cell (Cell 7):**
   - Click "+ Code" (or use Insert → Code cell)
   - Paste the entire Python code from above

3. **Verify Position:**
   - Cell 6 should come after the "Install dependencies" cell
   - Cell 7 should come right after Cell 6 (the markdown header)
   - Cell 8 should be the "Phase 2: Initialize Executive Team Agents" markdown

4. **Save:**
   - File → Save (or Ctrl/Cmd + S)

---

## Cell Context (Where it fits)

**Before Cell 6:**
- Cell 1-2: Mount Google Drive and set paths
- Cell 3: Load API keys
- Cell 4: Install dependencies
- Cell 5: (Usually another setup cell)

**Cell 6-7:** Read pending actions ← **YOU ARE HERE**

**After Cell 7:**
- Cell 8: Phase 2 header (Initialize Executive Team)
- Cell 9: Initialize Executive Team agents
- Cell 10+: Execute tasks

---

## Alternative: If you're updating for timestamps

If you want to update this cell to read **timestamped** `pending_actions.json` files (recommended):

```python
import json
from datetime import datetime
from pathlib import Path
import re

# Find latest pending_actions file by timestamp
handoff_dir = Path(MULTI_AGENT_ROOT) / 'reports/handoff'

# Search for timestamped pending_actions files
pending_files = list(handoff_dir.glob('pending_actions_*.json'))

if not pending_files:
    # Fallback to non-timestamped file
    pending_actions_file = handoff_dir / 'pending_actions.json'
    if not pending_actions_file.exists():
        print(f"❌ No pending actions found in {handoff_dir}")
        print("⚠️ Planning Team must generate pending_actions.json first")
        raise FileNotFoundError("No pending actions found")
else:
    # Extract timestamp from filename for sorting
    def extract_timestamp(filepath):
        match = re.search(r'(\d{8}_\d{6})', filepath.name)
        return match.group(1) if match else ''

    # Get latest by timestamp
    pending_actions_file = sorted(pending_files, key=extract_timestamp)[-1]
    print(f"✅ Found latest pending actions: {pending_actions_file.name}")

with open(pending_actions_file, 'r') as f:
    pending_actions = json.load(f)

print("="*80)
print("📥 PENDING ACTIONS FROM PLANNING TEAM")
print("="*80)
print(f"\n📋 Meeting ID: {pending_actions.get('meeting_id')}")
print(f"🗓️  Generated: {pending_actions.get('generated_at')}")
print(f"🎯 Context: {pending_actions.get('context')}")

decisions = pending_actions.get('decisions', [])
print(f"\n📊 Total tasks: {len(decisions)}")

# Group by priority
high_priority = [d for d in decisions if d.get('priority') == 'HIGH']
medium_priority = [d for d in decisions if d.get('priority') == 'MEDIUM']
low_priority = [d for d in decisions if d.get('priority') == 'LOW']

print(f"   ⭐ HIGH: {len(high_priority)}")
print(f"   🟠 MEDIUM: {len(medium_priority)}")
print(f"   🔵 LOW: {len(low_priority)}")

print("\n📋 Task List:")
for i, decision in enumerate(decisions, 1):
    priority = decision.get('priority', 'UNKNOWN')
    action = decision.get('action', 'No action specified')
    owner = decision.get('owner', 'unassigned')
    deadline = decision.get('deadline', 'No deadline')

    priority_icon = '⭐' if priority == 'HIGH' else '🟠' if priority == 'MEDIUM' else '🔵'
    print(f"\n{i}. {priority_icon} [{priority}] {action}")
    print(f"   👤 Owner: {owner}")
    print(f"   ⏰ Deadline: {deadline}")

print("\n" + "="*80)
```

This updated version:
- ✅ Looks for timestamped files first: `pending_actions_20251015_001500.json`
- ✅ Falls back to non-timestamped file if needed: `pending_actions.json`
- ✅ Automatically finds latest by timestamp
- ✅ Future-proof for timestamp system

---

## Quick Fix

**Just need the basic version back?**

Copy this minimal code:

```python
import json
from pathlib import Path

pending_actions_file = Path(MULTI_AGENT_ROOT) / 'reports/handoff/pending_actions.json'
with open(pending_actions_file, 'r') as f:
    pending_actions = json.load(f)

print("="*80)
print("📥 PENDING ACTIONS FROM PLANNING TEAM")
print("="*80)

for i, decision in enumerate(pending_actions.get('decisions', []), 1):
    print(f"{i}. [{decision.get('priority')}] {decision.get('action')}")
    print(f"   Owner: {decision.get('owner')}")

print("="*80)
```

This gives you the essential functionality to get the notebook working again.

---

**Choose your version:**
- **Original:** Use the first code block (simple, reads `pending_actions.json`)
- **Timestamp-aware:** Use the alternative version (finds latest timestamped file)
- **Minimal:** Use the quick fix (bare minimum to work)
