#!/usr/bin/env python3
import sys
sys.path.append('scripts')
from demo_candidate_library import CandidateLibraryDemo

demo = CandidateLibraryDemo(gallery_dir='pilot_gallery')
stats = demo.get_library_stats()

print("🧪 Gallery Test Results:")
print(f"   • Total candidates: {stats['total']:,}")
print(f"   • By domain: {stats['by_domain']}")
print(f"   • By provider: {stats['by_provider']}")

if stats['total'] >= 200:
    print("   ✅ Gallery operational for RA-Guard testing")
else:
    print("   ❌ Gallery too small for reliable testing")
