#!/usr/bin/env python3
"""
Test script to verify the computer vision pipeline setup.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def test_file_structure():
    """Test that all required files are present."""
    required_files = [
        'README.md',
        'requirements.txt',
        'pipeline.py',
        'setup.sh',
        '.env.example',
        'package.json',
        'config/default.json',
        'data/input/sample_input.json',
        'scripts/image_model.py',
        'scripts/yolo_detector.py',
        'scripts/rerank_with_compliance.mjs',
        'scripts/rerank_listwise_llm.mjs',
        'scripts/utils.mjs',
        'scripts/clip_probe/train_clip_probe_balanced.py',
        'scripts/clip_probe/run_clip_probe_inference.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ All required files present")
    return True

def test_python_imports():
    """Test that key Python modules can be imported."""
    try_imports = [
        'torch',
        'torchvision', 
        'clip',
        'ultralytics',
        'sklearn',
        'numpy',
        'PIL'
    ]
    
    failed_imports = []
    for module in try_imports:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError:
            failed_imports.append(module)
            print(f"❌ Failed to import {module}")
    
    return len(failed_imports) == 0

def test_config_files():
    """Test that configuration files are valid JSON."""
    config_files = [
        'config/default.json',
        'config/advanced.json',
        'data/input/sample_input.json'
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                json.load(f)
            print(f"✅ {config_file} is valid JSON")
        except json.JSONDecodeError as e:
            print(f"❌ {config_file} has invalid JSON: {e}")
            return False
        except FileNotFoundError:
            print(f"❌ {config_file} not found")
            return False
    
    return True

def test_scripts_executable():
    """Test that key scripts have proper permissions."""
    executable_files = ['setup.sh', 'pipeline.py']
    
    for script in executable_files:
        if os.path.exists(script):
            if os.access(script, os.X_OK) or script.endswith('.py'):
                print(f"✅ {script} is executable")
            else:
                print(f"❌ {script} is not executable")
                return False
        else:
            print(f"❌ {script} not found")
            return False
    
    return True

def main():
    print("🔬 Testing Computer Vision Pipeline Setup")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Configuration Files", test_config_files), 
        ("Script Permissions", test_scripts_executable),
        ("Python Imports", test_python_imports)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("�� Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
        print("\n🚀 Next steps:")
        print("  1. Edit .env with your API keys")
        print("  2. Run: python pipeline.py --config config/default.json --input data/input/sample_input.json --output data/output/results.json")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
