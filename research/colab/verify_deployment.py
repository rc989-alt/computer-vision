"""
Verify that all components are in place before deployment
Run this in Colab to check everything is ready
"""

from pathlib import Path
import sys

print("="*60)
print("🔍 DEPLOYMENT VERIFICATION")
print("="*60)

DRIVE_ROOT = Path("/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean")
LOCAL_PROJECT = Path("/content/cv_project")

checks_passed = 0
checks_failed = 0

def check(name, condition, path=None):
    global checks_passed, checks_failed
    if condition:
        checks_passed += 1
        print(f"✅ {name}")
        if path:
            print(f"   {path}")
    else:
        checks_failed += 1
        print(f"❌ {name}")
        if path:
            print(f"   Expected: {path}")
    return condition

print("\n📁 GOOGLE DRIVE FILES")
print("-"*60)
check("executive_coordinator.py",
      (DRIVE_ROOT / "executive_coordinator.py").exists(),
      DRIVE_ROOT / "executive_coordinator.py")
check("execution_tools.py",
      (DRIVE_ROOT / "multi-agent/tools/execution_tools.py").exists(),
      DRIVE_ROOT / "multi-agent/tools/execution_tools.py")
check("monitor_planning.ipynb",
      (DRIVE_ROOT / "research/colab/monitor_planning.ipynb").exists(),
      DRIVE_ROOT / "research/colab/monitor_planning.ipynb")
check("monitor_execution.ipynb",
      (DRIVE_ROOT / "research/colab/monitor_execution.ipynb").exists(),
      DRIVE_ROOT / "research/colab/monitor_execution.ipynb")
check("deployment_guide",
      (DRIVE_ROOT / "research/colab/DEPLOYMENT_GUIDE.md").exists(),
      DRIVE_ROOT / "research/colab/DEPLOYMENT_GUIDE.md")
check(".env file",
      (DRIVE_ROOT / ".env").exists(),
      DRIVE_ROOT / ".env")

print("\n📦 LOCAL PROJECT STRUCTURE")
print("-"*60)
check("Project root exists",
      LOCAL_PROJECT.exists(),
      LOCAL_PROJECT)
check("Multi-agent directory",
      (LOCAL_PROJECT / "multi-agent").exists(),
      LOCAL_PROJECT / "multi-agent")
check("Tools directory",
      (LOCAL_PROJECT / "multi-agent/tools").exists(),
      LOCAL_PROJECT / "multi-agent/tools")
check("Configs directory",
      (LOCAL_PROJECT / "multi-agent/configs").exists(),
      LOCAL_PROJECT / "multi-agent/configs")

print("\n🔑 ENVIRONMENT")
print("-"*60)
import os
check("ANTHROPIC_API_KEY set",
      bool(os.getenv("ANTHROPIC_API_KEY")),
      "Check with: print(os.getenv('ANTHROPIC_API_KEY', 'NOT SET')[:20] + '...')")
check("OPENAI_API_KEY set",
      bool(os.getenv("OPENAI_API_KEY")),
      "Check with: print(os.getenv('OPENAI_API_KEY', 'NOT SET')[:20] + '...')")

print("\n📋 REPORT DIRECTORIES")
print("-"*60)
report_dirs = [
    "multi-agent/reports/planning",
    "multi-agent/reports/execution",
    "multi-agent/reports/handoff"
]
for report_dir in report_dirs:
    full_path = LOCAL_PROJECT / report_dir
    exists = full_path.exists()
    if not exists:
        full_path.mkdir(parents=True, exist_ok=True)
    check(f"{report_dir}/",
          True,  # We just created it if it didn't exist
          full_path)

print("\n🐍 PYTHON ENVIRONMENT")
print("-"*60)
try:
    import anthropic
    check("anthropic package", True, f"Version: {anthropic.__version__}")
except ImportError:
    check("anthropic package", False, "Run: !pip install anthropic")

try:
    import openai
    check("openai package", True, f"Version: {openai.__version__}")
except ImportError:
    check("openai package", False, "Run: !pip install openai")

try:
    import yaml
    check("yaml package", True)
except ImportError:
    check("yaml package", False, "Run: !pip install pyyaml")

print("\n" + "="*60)
print(f"📊 RESULTS: {checks_passed} passed, {checks_failed} failed")
print("="*60)

if checks_failed == 0:
    print("\n✅ ALL CHECKS PASSED - READY TO DEPLOY!")
    print("\n🚀 Next steps:")
    print("   1. Run the deployment script:")
    print("      exec(open('/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/deploy_updated_system.py').read())")
    print("\n   2. Open monitoring notebooks:")
    print("      - research/colab/monitor_planning.ipynb")
    print("      - research/colab/monitor_execution.ipynb")
    print("\n   3. Wait ~30 minutes for first planning meeting")
    print("\n   4. Watch executive team execute actions!")
else:
    print(f"\n⚠️  {checks_failed} CHECKS FAILED")
    print("\n📝 Fix the failed checks above before deploying.")
    if checks_failed <= 3:
        print("\n💡 Most issues can be fixed by:")
        print("   - Waiting 60s for Drive sync")
        print("   - Loading API keys from .env file")
        print("   - Creating missing directories")
