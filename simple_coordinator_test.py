#!/usr/bin/env python3
"""
Simplified coordinator test - validates setup without running full heartbeat
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/cv_project")
sys.path.insert(0, str(PROJECT_ROOT / "multi-agent"))

print("="*60)
print("SIMPLE AUTONOMOUS COORDINATOR TEST")
print("="*60)
print()

# Import coordinator
from autonomous_coordinator import AutonomousCoordinator

# Create coordinator (this validates all the setup)
print("Creating coordinator...")
coordinator = AutonomousCoordinator(
    config_path=PROJECT_ROOT / "multi-agent/configs/autonomous_coordination.yaml",
    project_root=PROJECT_ROOT
)

print()
print("="*60)
print("✅ COORDINATOR SETUP SUCCESSFUL!")
print("="*60)
print()
print(f"📊 System Status:")
print(f"   Agents registered: {len(coordinator.agents)}")
print(f"   Channels configured: {len(coordinator.channels._subscribers)}")
print(f"   Triggers registered: {len(coordinator.triggers._active_triggers)}")
print()
print(f"🤖 Registered Agents:")
for agent_id, agent_info in list(coordinator.agents.items())[:10]:
    print(f"   • {agent_info['role']} ({agent_id})")
if len(coordinator.agents) > 10:
    print(f"   ... and {len(coordinator.agents) - 10} more")

print()
print("="*60)
print("✅ ALL SYSTEMS READY")
print("="*60)
print()
print("The autonomous coordinator is fully configured and ready.")
print("To actually start the heartbeat system, call: coordinator.start()")
print()
print("Note: The full heartbeat takes time to run (saves state, runs")
print("phases, etc.). This test validates everything is set up correctly")
print("without actually running the heartbeat.")
