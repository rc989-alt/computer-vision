#!/usr/bin/env python3
"""
Test script to see where coordinator initialization hangs
"""

import sys
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path("/content/cv_project")  # Colab path
sys.path.insert(0, str(PROJECT_ROOT / "multi-agent"))

print("="*60)
print("COORDINATOR INITIALIZATION TEST")
print("="*60)
print()

# Step 1: Test imports
print("Step 1: Testing imports...")
try:
    import yaml
    print("  ✅ yaml imported")

    from run_meeting import MeetingOrchestrator
    print("  ✅ MeetingOrchestrator imported")

    from tools.file_bridge import FileBridge, create_default_policies
    print("  ✅ FileBridge imported")

    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print()

# Step 2: Test config loading
print("Step 2: Loading configuration...")
config_path = PROJECT_ROOT / "multi-agent/configs/autonomous_coordination.yaml"
try:
    start = time.time()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    elapsed = time.time() - start
    print(f"  ✅ Config loaded in {elapsed:.2f}s")
    print(f"  📊 Config size: {len(str(config))} chars")
except Exception as e:
    print(f"  ❌ Config load failed: {e}")
    sys.exit(1)

print()

# Step 3: Test SharedMemoryManager
print("Step 3: Creating SharedMemoryManager...")
try:
    start = time.time()
    from autonomous_coordinator import SharedMemoryManager
    memory = SharedMemoryManager(PROJECT_ROOT / "multi-agent/state")
    elapsed = time.time() - start
    print(f"  ✅ SharedMemoryManager created in {elapsed:.2f}s")
except Exception as e:
    print(f"  ❌ SharedMemoryManager failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 4: Test ChannelManager
print("Step 4: Creating ChannelManager...")
try:
    start = time.time()
    from autonomous_coordinator import ChannelManager
    channels = ChannelManager()
    elapsed = time.time() - start
    print(f"  ✅ ChannelManager created in {elapsed:.2f}s")
except Exception as e:
    print(f"  ❌ ChannelManager failed: {e}")
    sys.exit(1)

print()

# Step 5: Test TriggerSystem (without coordinator)
print("Step 5: Testing TriggerSystem...")
print("  ⏭️  Skipping (needs coordinator)")

print()

# Step 6: Test FileBridge
print("Step 6: Creating FileBridge...")
try:
    start = time.time()
    policies = create_default_policies(PROJECT_ROOT)
    elapsed = time.time() - start
    print(f"  ✅ Policies created in {elapsed:.2f}s")
    print(f"  📊 Policies: {len(policies)} agent policies")

    start = time.time()
    file_bridge = FileBridge(PROJECT_ROOT, policies)
    elapsed = time.time() - start
    print(f"  ✅ FileBridge created in {elapsed:.2f}s")
except Exception as e:
    print(f"  ❌ FileBridge failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 7: Try full initialization
print("Step 7: Full coordinator initialization...")
print("  ⚠️  This is where it might hang...")
try:
    from autonomous_coordinator import AutonomousCoordinator

    print("  → Creating coordinator...")
    start = time.time()
    coordinator = AutonomousCoordinator(
        config_path=config_path,
        project_root=PROJECT_ROOT
    )
    elapsed = time.time() - start
    print(f"  ✅ Coordinator initialized in {elapsed:.2f}s")
    print(f"  📊 Agents: {len(coordinator.agents)}")
    print(f"  📊 Channels: {len(coordinator.channels._subscribers)}")
except Exception as e:
    print(f"  ❌ Coordinator initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
print()
print("If you see this, the coordinator CAN initialize successfully!")
print("The hang might be in the .start() method or heartbeat thread.")
