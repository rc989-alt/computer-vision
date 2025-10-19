#!/usr/bin/env python3
"""
Quick Production Gallery Builder - Scale from 25 to 1,000+ images
Uses existing candidate_library_setup.py infrastructure
"""

import argparse
import subprocess
import sys
from pathlib import Path

def build_target_gallery(domains: list, target_per_domain: int, gallery_dir: str):
    """Build production gallery using existing tools"""
    
    print("🎯 SCALING RA-GUARD GALLERY TO PRODUCTION")
    print("=" * 50)
    
    total_target = len(domains) * target_per_domain
    print(f"📊 Target: {total_target:,} images ({target_per_domain} × {len(domains)} domains)")
    
    # Assessment thresholds
    if total_target >= 3000:
        print("🎯 Scale: VALIDATION READY (credible 300-query evaluation)")
    elif total_target >= 1000:
        print("🎯 Scale: PILOT READY (reliable nDCG@10 measurement)")
    elif total_target >= 200:
        print("🎯 Scale: FLOOR MET (end-to-end testing)")
    else:
        print("⚠️  Scale: INSUFFICIENT (<200 minimum)")
    
    print(f"\nQuality Gates:")
    print(f"   • Resolution: ≥512px shorter side")
    print(f"   • Deduplication: pHash filtering")  
    print(f"   • Sources: Pexels + Unsplash APIs")
    print(f"   • Precompute: CLIP + detection features")
    
    # Build each domain
    results = {}
    
    for domain in domains:
        print(f"\n🔄 Building {domain} domain ({target_per_domain} images)...")
        
        # Build from Pexels first
        pexels_target = target_per_domain // 2
        print(f"   🌐 Pexels: collecting {pexels_target} images...")
        
        try:
            cmd = [
                sys.executable, 
                "scripts/candidate_library_setup.py",
                "--source", "pexels",
                "--domains", domain,
                "--target-per-domain", str(pexels_target), 
                "--gallery-dir", gallery_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(f"   ✅ Pexels: collected successfully")
            else:
                print(f"   ⚠️  Pexels: {result.stderr.strip()[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Pexels error: {e}")
        
        # Build from Unsplash second  
        unsplash_target = target_per_domain - pexels_target
        print(f"   🌐 Unsplash: collecting {unsplash_target} images...")
        
        try:
            cmd = [
                sys.executable,
                "scripts/candidate_library_setup.py", 
                "--source", "unsplash",
                "--domains", domain,
                "--target-per-domain", str(unsplash_target),
                "--gallery-dir", gallery_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(f"   ✅ Unsplash: collected successfully")
            else:
                print(f"   ⚠️  Unsplash: {result.stderr.strip()[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Unsplash error: {e}")
    
    # Final validation
    print(f"\n📊 VALIDATING FINAL GALLERY...")
    
    gallery_path = Path(gallery_dir)
    if gallery_path.exists():
        # Count actual images
        total_images = 0
        for domain in domains:
            domain_path = gallery_path / domain
            if domain_path.exists():
                domain_count = len(list(domain_path.glob("*.jpg")))
                total_images += domain_count
                print(f"   • {domain}: {domain_count} images")
                results[domain] = domain_count
        
        print(f"\n📈 FINAL RESULTS:")
        print(f"   • Total collected: {total_images:,} images")
        print(f"   • Target achievement: {total_images/total_target*100:.1f}%")
        
        # Readiness assessment
        if total_images >= 3000:
            print(f"   🚀 VALIDATION READY: {total_images} images for credible evaluation")
        elif total_images >= 1000:
            print(f"   🎯 PILOT READY: {total_images} images for reliable nDCG@10")
        elif total_images >= 200:
            print(f"   ✅ FLOOR MET: {total_images} images for end-to-end testing")
        else:
            print(f"   ⚠️  INSUFFICIENT: {total_images} < 200 minimum")
        
        # Test the gallery
        print(f"\n🧪 TESTING GALLERY FUNCTIONALITY...")
        try:
            cmd = [
                sys.executable,
                "test_real_images.py"
            ]
            
            # Temporarily modify test script to use new gallery
            test_script_content = f'''#!/usr/bin/env python3
import sys
sys.path.append('scripts')
from demo_candidate_library import CandidateLibraryDemo

demo = CandidateLibraryDemo(gallery_dir='{gallery_dir}')
stats = demo.get_library_stats()

print("🧪 Gallery Test Results:")
print(f"   • Total candidates: {{stats['total']:,}}")
print(f"   • By domain: {{stats['by_domain']}}")
print(f"   • By provider: {{stats['by_provider']}}")

if stats['total'] >= 200:
    print("   ✅ Gallery operational for RA-Guard testing")
else:
    print("   ❌ Gallery too small for reliable testing")
'''
            
            with open("test_production_gallery.py", "w") as f:
                f.write(test_script_content)
                
            result = subprocess.run([sys.executable, "test_production_gallery.py"], 
                                  capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"   ❌ Test failed: {result.stderr}")
                
        except Exception as e:
            print(f"   ⚠️  Test error: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Build Production RA-Guard Gallery")
    parser.add_argument("--scale", choices=['floor', 'pilot', 'validation'], 
                       default='pilot', help="Target scale")
    parser.add_argument("--domains", nargs='+', 
                       default=['cocktails'],
                       help="Domains to collect") 
    parser.add_argument("--gallery-dir", default="production_gallery",
                       help="Output gallery directory")
    
    args = parser.parse_args()
    
    # Scale configurations
    scale_targets = {
        'floor': 200,      # End-to-end testing
        'pilot': 1000,     # Reliable nDCG@10
        'validation': 3000 # Credible evaluation
    }
    
    total_target = scale_targets[args.scale]
    per_domain = total_target // len(args.domains)
    
    print(f"🎯 Building {args.scale.upper()} scale gallery:")
    print(f"   • {per_domain} images per domain")
    print(f"   • {len(args.domains)} domains: {args.domains}")
    print(f"   • Output: {args.gallery_dir}/")
    
    results = build_target_gallery(args.domains, per_domain, args.gallery_dir)
    
    total_built = sum(results.values())
    print(f"\n🎉 BUILD COMPLETE: {total_built:,} images ready for RA-Guard!")

if __name__ == "__main__":
    main()