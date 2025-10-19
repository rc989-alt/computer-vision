from pathlib import Path

# Check project location
project = Path("/content/drive/MyDrive/cv_multimodal/project/computer-vision")

print("🔍 Checking files on Google Drive:\n")
print(f"Project folder exists: {project.exists()}")
print(f"multi-agent folder exists: {(project / 'multi-agent').exists()}")
print(f".env exists: {(project / '.env').exists()}")
print(f"autonomous_coordinator.py exists: {(project / 'multi-agent/autonomous_coordinator.py').exists()}")
print(f"autonomous_coordination.yaml exists: {(project / 'multi-agent/configs/autonomous_coordination.yaml').exists()}")

if (project / 'multi-agent/autonomous_coordinator.py').exists():
    size = (project / 'multi-agent/autonomous_coordinator.py').stat().st_size / 1024
    print(f"\nautonomous_coordinator.py size: {size:.1f} KB")
    if size > 20:
        print("✅ Looks like the correct file (new version)")
    else:
        print("⚠️ File seems small, might be old version")

if (project / 'multi-agent/configs/autonomous_coordination.yaml').exists():
    size = (project / 'multi-agent/configs/autonomous_coordination.yaml').stat().st_size / 1024
    print(f"autonomous_coordination.yaml size: {size:.1f} KB")
    if size > 30:
        print("✅ Looks like the correct file (new version with 15 agents)")
    else:
        print("⚠️ File seems small, might be old version")

if (project / '.env').exists():
    print("\n✅ .env file found")

print("\n" + "="*60)
if all([
    project.exists(),
    (project / 'multi-agent').exists(),
    (project / '.env').exists(),
    (project / 'multi-agent/autonomous_coordinator.py').exists(),
    (project / 'multi-agent/configs/autonomous_coordination.yaml').exists(),
]):
    print("✅ ALL FILES PRESENT - Ready to start!")
    print("\nYou can now run the autonomous_system_colab.ipynb notebook!")
else:
    print("❌ Some files missing - check above")
