#!/usr/bin/env python3
"""
Test the fixed action parser against known bad examples
"""

import sys
import os

# Add the Google Drive project to path
gdrive_path = '/Users/guyan/Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/computer-vision-clean'
sys.path.insert(0, gdrive_path)

# Change to that directory to ensure imports work
os.chdir(gdrive_path)

from multi_agent.tools.parse_actions import ActionParser

# Test cases from actual meeting output
test_cases = [
    # BAD examples that should be REJECTED
    ("│  ✓   │  ✗   │    ✓     │  ✓   │", False, "Table content"),
    ("(AEM)                      │", False, "Table row fragment"),
    ("analyzer                      │", False, "Table cell"),
    ("4 architectures minimum", False, "Number + noun fragment"),
    ("### **Phase 2 (Days 8-14): Empirical Validation & Solution Testing**", False, "Markdown header"),
    ("**Success Probability Matrix:**", False, "Bold header"),
    ("on All Architectures)", False, "Preposition fragment"),
    ("on our V2 + 1 external model", False, "Preposition fragment"),
    ("on progress, challenges, and insights from ongoing tests", False, "Preposition fragment"),
    ("(backup): 95% success probability", False, "Parenthetical note"),

    # GOOD examples that should be ACCEPTED
    ("Implement diagnostic test suite for attention collapse detection across multiple architectures", True, "Complete action with verb"),
    ("Validate the mitigation strategies on at least two architectures to confirm effectiveness", True, "Complete action with verb"),
    ("Test on 3-4 diverse architectures from the literature or previous internal projects", True, "Complete action with verb"),
    ("The diagnostic framework should be tested on V2 and CLIP models for generalizability", True, "Action with subject"),
    ("We need to analyze gradient magnitudes per modality during training to identify imbalance patterns", True, "Action with subject"),
    ("Run baseline attention analysis on V2 model to establish current visual contribution metrics", True, "Complete action with verb"),
    ("Create a comprehensive diagnostic tool that works across different multimodal fusion architectures", True, "Complete action with verb"),
]

print("=" * 80)
print("ACTION PARSER FIX VALIDATION TEST")
print("=" * 80)
print()

passed = 0
failed = 0

for description, should_pass, category in test_cases:
    result = ActionParser._is_valid_action(description)

    if result == should_pass:
        status = "✅ PASS"
        passed += 1
    else:
        status = "❌ FAIL"
        failed += 1

    print(f"{status} [{category}]")
    print(f"   Input: \"{description[:70]}{'...' if len(description) > 70 else ''}\"")
    print(f"   Expected: {'ACCEPT' if should_pass else 'REJECT'}, Got: {'ACCEPT' if result else 'REJECT'}")
    print()

print("=" * 80)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 80)

if failed == 0:
    print("\n🎉 ALL TESTS PASSED! Action parser fix is working correctly.")
    sys.exit(0)
else:
    print(f"\n⚠️  {failed} test(s) failed. Review the logic in _is_valid_action().")
    sys.exit(1)
