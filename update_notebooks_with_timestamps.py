#!/usr/bin/env python3
"""
Update Colab notebooks to use timestamped filenames for all handoff files.

This script modifies:
1. cvpr_autonomous_execution_cycle.ipynb - Executive Team execution notebook
2. planning_team_review_cycle.ipynb - Planning Team review notebook

Changes:
- execution_progress_update.md → execution_progress_update_YYYYMMDD_HHMMSS.md
- pending_actions.json → pending_actions_YYYYMMDD_HHMMSS.json
- next_meeting_trigger.json → next_meeting_trigger_YYYYMMDD_HHMMSS.json
"""

import json
import sys
from pathlib import Path

# Paths
GDRIVE_COLAB_DIR = Path('/Users/guyan/Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/computer-vision-clean/research/colab')
EXEC_NOTEBOOK = GDRIVE_COLAB_DIR / 'cvpr_autonomous_execution_cycle.ipynb'
PLANNING_NOTEBOOK = GDRIVE_COLAB_DIR / 'planning_team_review_cycle.ipynb'

def update_executive_notebook(notebook_path):
    """Update Executive Team notebook with timestamped filenames."""

    print(f"\n📝 Updating: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell.get('source', []))
        original = source

        # Cell 13: Save execution progress update with timestamp
        if 'execution_progress_update' in source and ('write_text' in source or 'with open' in source):
            # Add timestamp generation
            if 'timestamp = datetime.now()' not in source:
                source = source.replace(
                    'summary = tracker.get_summary()',
                    'timestamp = datetime.now().strftime(\'%Y%m%d_%H%M%S\')\nsummary = tracker.get_summary()'
                )

            # Update progress file path
            source = source.replace(
                "reports/handoff/execution_progress_update.md",
                "reports/handoff/execution_progress_update_{timestamp}.md"
            )
            source = source.replace(
                "'reports/handoff/execution_progress_update.md'",
                "f'reports/handoff/execution_progress_update_{timestamp}.md'"
            )

            # Update print statements
            source = source.replace(
                'execution_progress_update.md',
                'execution_progress_update_{timestamp}.md'
            )

            if source != original:
                changes.append(f"Cell {i}: Updated execution_progress_update to use timestamp")

        # Cell 14/15: Save execution results (already timestamped, but ensure consistency)
        if 'execution_results' in source and 'timestamp' not in source:
            # Ensure timestamp variable exists
            if 'timestamp = datetime.now()' not in source:
                source = 'timestamp = datetime.now().strftime(\'%Y%m%d_%H%M%S\')\n' + source
                changes.append(f"Cell {i}: Added timestamp variable for execution_results")

        # Cell 17: Save next meeting trigger with timestamp
        if 'next_meeting_trigger' in source and ('write_text' in source or 'with open' in source):
            # Update trigger file path
            source = source.replace(
                "reports/handoff/next_meeting_trigger.json",
                "reports/handoff/next_meeting_trigger_{timestamp}.json"
            )
            source = source.replace(
                "'reports/handoff/next_meeting_trigger.json'",
                "f'reports/handoff/next_meeting_trigger_{timestamp}.json'"
            )

            if source != original:
                changes.append(f"Cell {i}: Updated next_meeting_trigger to use timestamp")

        # Update cell source if changed
        if source != original:
            cell['source'] = source.split('\n')
            # Add newline to each line except the last
            cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                             for i, line in enumerate(cell['source'])]

    if changes:
        # Save updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print(f"✅ Updated {len(changes)} cells:")
        for change in changes:
            print(f"   - {change}")
        return True
    else:
        print(f"⚠️  No changes needed")
        return False

def update_planning_notebook(notebook_path):
    """Update Planning Team notebook to find latest timestamped files."""

    print(f"\n📝 Updating: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell.get('source', []))
        original = source

        # Find cell that reads execution progress
        if 'execution_progress_update' in source and 'read' in source.lower():
            # Replace with code to find latest timestamped file
            new_code = """# Find latest execution progress file by timestamp
from pathlib import Path
import re

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
print(f"📊 Size: {latest_progress.stat().st_size} bytes")

# Read the latest progress file
with open(latest_progress, 'r', encoding='utf-8') as f:
    progress_content = f.read()

print(f"\\n📄 Progress Content ({len(progress_content)} chars):")
print("=" * 80)
print(progress_content[:500] + "..." if len(progress_content) > 500 else progress_content)
print("=" * 80)
"""

            if 'glob(' in source or 'Find latest' in source:
                # Already updated
                pass
            else:
                source = new_code
                changes.append(f"Cell {i}: Updated to find latest execution_progress_update by timestamp")

        # Find cell that saves pending_actions
        if 'pending_actions.json' in source and ('write' in source or 'dump' in source):
            # Ensure timestamp is generated
            if 'timestamp = datetime.now()' not in source:
                source = 'timestamp = datetime.now().strftime(\'%Y%m%d_%H%M%S\')\n' + source

            # Update file path
            source = source.replace(
                "reports/handoff/pending_actions.json",
                "reports/handoff/pending_actions_{timestamp}.json"
            )
            source = source.replace(
                "'reports/handoff/pending_actions.json'",
                "f'reports/handoff/pending_actions_{timestamp}.json'"
            )

            # Update print statements
            source = source.replace(
                'pending_actions.json saved',
                'pending_actions_{timestamp}.json saved'
            )

            if source != original:
                changes.append(f"Cell {i}: Updated pending_actions to use timestamp")

        # Update cell source if changed
        if source != original:
            cell['source'] = source.split('\n')
            # Add newline to each line except the last
            cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                             for i, line in enumerate(cell['source'])]

    if changes:
        # Save updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print(f"✅ Updated {len(changes)} cells:")
        for change in changes:
            print(f"   - {change}")
        return True
    else:
        print(f"⚠️  No changes needed")
        return False

def main():
    """Main function to update both notebooks."""

    print("=" * 80)
    print("🔧 UPDATING COLAB NOTEBOOKS WITH TIMESTAMPED FILENAMES")
    print("=" * 80)

    # Check if notebooks exist
    if not EXEC_NOTEBOOK.exists():
        print(f"❌ Executive notebook not found: {EXEC_NOTEBOOK}")
        return 1

    if not PLANNING_NOTEBOOK.exists():
        print(f"❌ Planning notebook not found: {PLANNING_NOTEBOOK}")
        return 1

    # Update Executive Team notebook
    exec_updated = update_executive_notebook(EXEC_NOTEBOOK)

    # Update Planning Team notebook
    planning_updated = update_planning_notebook(PLANNING_NOTEBOOK)

    print("\n" + "=" * 80)
    if exec_updated or planning_updated:
        print("✅ NOTEBOOKS UPDATED SUCCESSFULLY")
        print("\nUpdated files:")
        if exec_updated:
            print(f"   ✅ {EXEC_NOTEBOOK.name}")
        if planning_updated:
            print(f"   ✅ {PLANNING_NOTEBOOK.name}")
        print("\n📋 Next steps:")
        print("   1. Upload updated notebooks to Google Colab")
        print("   2. Run a test cycle to verify timestamped files are created")
        print("   3. Check that Planning Team can find latest files")
    else:
        print("⚠️  NO CHANGES NEEDED - Notebooks may already be updated")
    print("=" * 80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
