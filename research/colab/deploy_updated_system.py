"""
Deploy Updated Autonomous System to Colab
Includes: separated reports, handoff mechanism, tool execution
"""

import time
import shutil
from pathlib import Path

print("="*60)
print("🚀 DEPLOYING UPDATED AUTONOMOUS SYSTEM")
print("="*60)

# Wait for Drive sync
print("\n⏳ Waiting 60 seconds for Google Drive sync...")
time.sleep(60)

# Define paths
DRIVE_ROOT = Path("/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean")
LOCAL_PROJECT = Path("/content/cv_project")

print("\n📦 Copying updated files from Drive...")

# Copy main coordinator
print("   - executive_coordinator.py")
shutil.copy(
    DRIVE_ROOT / "executive_coordinator.py",
    LOCAL_PROJECT / "executive_coordinator.py"
)

# Copy execution tools
print("   - multi-agent/tools/execution_tools.py")
tools_dir = LOCAL_PROJECT / "multi-agent/tools"
tools_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(
    DRIVE_ROOT / "multi-agent/tools/execution_tools.py",
    tools_dir / "execution_tools.py"
)

# Copy meeting config with enhanced file access
print("   - multi-agent/configs/meeting.yaml")
configs_dir = LOCAL_PROJECT / "multi-agent/configs"
configs_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(
    DRIVE_ROOT / "multi-agent/configs/meeting.yaml",
    configs_dir / "meeting.yaml"
)

# Copy research context for agents
print("   - research/RESEARCH_CONTEXT.md")
research_dir = LOCAL_PROJECT / "research"
research_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(
    DRIVE_ROOT / "research/RESEARCH_CONTEXT.md",
    research_dir / "RESEARCH_CONTEXT.md"
)

# Create report directories
print("\n📁 Creating report directory structure...")
report_dirs = [
    LOCAL_PROJECT / "multi-agent/reports/planning",
    LOCAL_PROJECT / "multi-agent/reports/execution",
    LOCAL_PROJECT / "multi-agent/reports/handoff"
]
for report_dir in report_dirs:
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ {report_dir.relative_to(LOCAL_PROJECT)}")

print("\n✅ Files updated")

# Stop existing coordinator if running
print("\n🛑 Stopping existing system if running...")
if 'coordinator' in globals():
    try:
        coordinator.stop()
        print("✅ Previous coordinator stopped")
    except:
        print("⚠️  No coordinator was running")
else:
    print("ℹ️  No coordinator to stop")

# Clean module cache
print("\n🧹 Clearing module cache...")
import sys
modules_to_clear = [
    'executive_coordinator',
    'execution_tools',
    'autonomous_coordinator'
]
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"   ✅ Cleared {mod}")

# Import and initialize new coordinator
print("\n🔄 Initializing updated coordinator...")
sys.path.insert(0, str(LOCAL_PROJECT))

from executive_coordinator import ExecutiveCoordinator

coordinator = ExecutiveCoordinator(
    LOCAL_PROJECT,
    log_file="/content/executive.log"
)

print("\n🚀 Starting autonomous system...")
coordinator.start()

print("\n" + "="*60)
print("✅ DEPLOYMENT COMPLETE")
print("="*60)
print("\n📊 New Features Active:")
print("   ✓ Separated planning/execution reports")
print("   ✓ Handoff mechanism (pending_actions.json)")
print("   ✓ Tool execution enabled for Executive Team")
print("   ✓ Execution capabilities: deploy, evaluate, collect metrics")
print("   ✓ Enhanced file access: research summaries, reports, experiments")
print("   ✓ Research context available: RESEARCH_CONTEXT.md")
print("   ✓ All 6 agents active (including Gemini CoTRR Team)")
print("\n📂 Report Structure:")
print("   - multi-agent/reports/planning/    (Meeting summaries & actions)")
print("   - multi-agent/reports/execution/   (Tool usage & deployments)")
print("   - multi-agent/reports/handoff/     (Actions from planning → execution)")
print("\n📊 Monitoring Notebooks:")
print("   - Open: research/colab/monitor_planning.ipynb")
print("   - Open: research/colab/monitor_execution.ipynb")
print("\n⏱️  System Status:")
print("   - Heartbeat cycle: 5 minutes")
print("   - Planning meetings: Every 30 minutes")
print("   - Executive actions: Continuous")
print("\n🔍 Check logs:")
print("   !tail -50 /content/executive.log")
print("="*60)
