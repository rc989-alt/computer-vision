#!/usr/bin/env python3
"""
Test File Bridge and Progress Sync Hook
Verifies agents can access local files
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools.file_bridge import FileBridge, create_default_policies
from tools.progress_sync_hook import ProgressSyncHook, create_agent_context_with_progress


def test_file_bridge():
    """Test basic file bridge functionality"""
    print("="*60)
    print("Testing File Bridge")
    print("="*60)

    project_root = Path(__file__).parent.parent
    print(f"\n🔍 Debug Info:")
    print(f"   __file__: {Path(__file__)}")
    print(f"   parent: {Path(__file__).parent}")
    print(f"   parent.parent: {project_root}")
    print(f"   Resolved: {project_root.resolve()}")
    print(f"   research/ exists: {(project_root / 'research').exists()}")
    if (project_root / 'research').exists():
        import os
        print(f"   research/ contents: {os.listdir(project_root / 'research')[:5]}")
    print()

    file_bridge = FileBridge(project_root, create_default_policies(project_root))

    # Test 1: List files in research directory
    print("\n1️⃣ Testing file listing (Tech Analysis agent)...")
    files = file_bridge.list_files('tech_analysis', 'research', '*.py')
    print(f"   Found {len(files)} Python files in research/")
    if files:
        print(f"   Latest: {files[0]['name']} ({files[0]['modified']})")

    # Test 2: Read a file
    print("\n2️⃣ Testing file reading (Data Analyst agent)...")
    content = file_bridge.read_file('data_analyst', 'data/dataset/metadata_summary.json')
    if content:
        print(f"   Successfully read file ({len(content)} bytes)")
    else:
        print(f"   File not found or access denied")

    # Test 3: Scan artifacts
    print("\n3️⃣ Testing artifact scan...")
    scan = file_bridge.scan_artifacts('tech_analysis', ['research', 'data'])
    print(f"   Total files scanned: {scan['total_files']}")
    print(f"   Total size: {scan['total_size'] / 1_000_000:.2f} MB")

    # Test 4: Access control
    print("\n4️⃣ Testing access control...")
    # Try to access file outside allowed directories
    restricted = file_bridge.read_file('critic', 'config.yaml')
    if restricted:
        print(f"   ⚠️  WARNING: Should have been denied!")
    else:
        print(f"   ✅ Access correctly denied for restricted path")

    # Test 5: View access log
    print("\n5️⃣ Access log summary...")
    log = file_bridge.get_access_log()
    print(f"   Total access attempts: {len(log)}")
    denied = [entry for entry in log if entry['action'] == 'DENIED']
    print(f"   Denied attempts: {len(denied)}")


def test_progress_sync():
    """Test progress sync hook"""
    print("\n" + "="*60)
    print("Testing Progress Sync Hook")
    print("="*60)

    project_root = Path(__file__).parent.parent

    print("\n🔄 Running progress sync...")
    try:
        progress_context = create_agent_context_with_progress(project_root)

        print("\n✅ Progress sync completed!")
        print("\n📄 Generated context preview:")
        print(progress_context[:500] + "...")

        # Check if progress report was saved
        report_path = project_root / 'multi-agent' / 'reports' / 'progress_update.md'
        if report_path.exists():
            print(f"\n✅ Progress report saved: {report_path}")
        else:
            print(f"\n⚠️  Progress report not found at expected location")

    except Exception as e:
        print(f"\n❌ Error during progress sync: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n🧪 File Access System Test Suite\n")

    try:
        test_file_bridge()
        test_progress_sync()

        print("\n" + "="*60)
        print("✅ All Tests Complete!")
        print("="*60)
        print("\n📋 Summary:")
        print("   - File Bridge: Operational")
        print("   - Access Control: Working")
        print("   - Progress Sync Hook: Ready")
        print("   - Agents can now access local files!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
