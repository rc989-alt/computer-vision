#!/usr/bin/env python3
"""
Quick diagnostic script to run in Google Colab
Helps identify why files aren't being found

Copy this entire cell and run it in Colab after mounting Drive
"""

from pathlib import Path
import os

print("="*60)
print("COLAB DRIVE DIAGNOSTIC TOOL")
print("="*60)

# Step 1: Check if Drive is mounted
print("\n1. Checking Drive mount...")
drive_path = Path("/content/drive")
if drive_path.exists():
    print("   ✅ Drive is mounted at /content/drive")
else:
    print("   ❌ Drive is NOT mounted!")
    print("   Run this first:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    exit(1)

# Step 2: Check MyDrive
print("\n2. Checking MyDrive...")
mydrive = Path("/content/drive/MyDrive")
if mydrive.exists():
    print(f"   ✅ MyDrive accessible")

    # Count folders
    folders = [f for f in mydrive.iterdir() if f.is_dir()]
    print(f"   📊 Total folders in MyDrive: {len(folders)}")
else:
    print("   ❌ MyDrive not found!")
    exit(1)

# Step 3: Look for project folders
print("\n3. Searching for project folders...")
print("   Looking for folders that might contain your project:\n")

project_candidates = []
for folder in folders:
    folder_name = folder.name.lower()

    # Check if folder name suggests it's the project
    if any(keyword in folder_name for keyword in ['computer', 'vision', 'cv', 'multi', 'project']):
        print(f"   📁 Found candidate: {folder.name}")

        # Check if it has multi-agent subfolder
        has_multiagent = (folder / "multi-agent").exists()
        if has_multiagent:
            print(f"      ✅ Has multi-agent/ subfolder - THIS IS YOUR PROJECT!")
            project_candidates.append(folder)
        else:
            print(f"      ⚠️ No multi-agent/ subfolder found")

            # Show what's inside
            print(f"      Contents:")
            try:
                for item in list(folder.iterdir())[:10]:
                    prefix = "📁" if item.is_dir() else "📄"
                    print(f"         {prefix} {item.name}")
            except Exception as e:
                print(f"         Error: {e}")

# Step 4: Detailed inspection of computer-vision folder
print("\n4. Detailed inspection of 'computer-vision' folder...")
cv_folder = mydrive / "computer-vision"

if cv_folder.exists():
    print(f"   ✅ Found: {cv_folder}")
    print(f"\n   📂 Full contents of computer-vision/:\n")

    try:
        all_items = list(cv_folder.iterdir())

        # Separate folders and files
        folders_in_cv = sorted([f for f in all_items if f.is_dir()], key=lambda x: x.name)
        files_in_cv = sorted([f for f in all_items if not f.is_dir()], key=lambda x: x.name)

        print(f"   Folders ({len(folders_in_cv)}):")
        for folder in folders_in_cv:
            print(f"      📁 {folder.name}/")

            # If this is multi-agent, show what's inside
            if folder.name == "multi-agent":
                print(f"         ✅ FOUND MULTI-AGENT!")
                try:
                    ma_items = list(folder.iterdir())[:10]
                    for item in ma_items:
                        prefix = "📁" if item.is_dir() else "📄"
                        size = ""
                        if item.is_file():
                            size = f" ({item.stat().st_size / 1024:.1f} KB)"
                        print(f"            {prefix} {item.name}{size}")
                except Exception as e:
                    print(f"            Error reading: {e}")

        print(f"\n   Files ({len(files_in_cv)}):")
        for file in files_in_cv:
            size = file.stat().st_size / 1024
            print(f"      📄 {file.name} ({size:.1f} KB)")

        # Check for nested computer-vision
        nested_cv = cv_folder / "computer-vision"
        if nested_cv.exists():
            print(f"\n   ⚠️ FOUND NESTED FOLDER: computer-vision/computer-vision/")
            print(f"      This is likely a mistake from how the folder was uploaded")

            if (nested_cv / "multi-agent").exists():
                print(f"      ✅ multi-agent/ is in the NESTED folder!")
                print(f"\n   🔧 SOLUTION: Use this path in notebook:")
                print(f"      /content/drive/MyDrive/computer-vision/computer-vision")

    except Exception as e:
        print(f"   ❌ Error reading folder: {e}")
else:
    print(f"   ❌ Not found: {cv_folder}")

# Step 5: Check for required files
print("\n5. Checking for required files...")

paths_to_check = [
    "computer-vision/multi-agent/autonomous_coordinator.py",
    "computer-vision/multi-agent/configs/autonomous_coordination.yaml",
    "computer-vision/.env",
    "computer-vision/research/api_keys.env",
]

for path_str in paths_to_check:
    full_path = mydrive / path_str
    if full_path.exists():
        size = full_path.stat().st_size / 1024
        print(f"   ✅ {path_str} ({size:.1f} KB)")
    else:
        print(f"   ❌ {path_str} - NOT FOUND")

# Step 6: Summary and recommendations
print("\n" + "="*60)
print("SUMMARY & RECOMMENDATIONS")
print("="*60)

if project_candidates:
    print(f"\n✅ Found {len(project_candidates)} potential project location(s):")
    for candidate in project_candidates:
        print(f"\n   Path to use in notebook:")
        print(f"   {candidate}")
        print(f"\n   Or copy/paste this:")
        print(f"   DRIVE_PROJECT = Path('{candidate}')")
else:
    print("\n❌ Could not find project with multi-agent/ folder")
    print("\n🔧 Possible issues:")
    print("   1. Upload still in progress (wait a few minutes)")
    print("   2. Uploaded wrong folder (should contain multi-agent/)")
    print("   3. Files uploaded separately (need to recreate folder structure)")

    print("\n📋 What to upload:")
    print("   Your laptop: /Users/guyan/computer_vision/computer-vision/")
    print("   Should contain: multi-agent/, research/, docs/, .env")

    print("\n🎯 Quick fix:")
    print("   1. Go to drive.google.com")
    print("   2. Delete the 'computer-vision' folder if it exists")
    print("   3. Drag the ENTIRE 'computer-vision' folder from your laptop")
    print("   4. Wait for upload to complete (check bottom-right corner)")
    print("   5. Re-run this diagnostic")

# Step 7: Test import (if project found)
if project_candidates:
    print("\n7. Testing Python import...")
    test_path = project_candidates[0]

    import sys
    sys.path.insert(0, str(test_path / "multi-agent"))

    try:
        # Try importing
        import autonomous_coordinator
        print(f"   ✅ Successfully imported autonomous_coordinator module!")
        print(f"   Module location: {autonomous_coordinator.__file__}")
    except ImportError as e:
        print(f"   ⚠️ Import failed: {e}")
        print(f"   This might be okay - may have missing dependencies")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\n💡 Next steps:")
print("   1. If project was found: Copy the path shown above")
print("   2. If project NOT found: Follow the 'Quick fix' instructions")
print("   3. Re-run the notebook setup cell with the correct path")
print("\n")
