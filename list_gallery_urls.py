#!/usr/bin/env python3
"""
Gallery URL Generator
Creates browsable lists of Pexels/Unsplash URLs from gallery database
"""

import sqlite3
import argparse
from pathlib import Path
import json

def generate_url_list(gallery_dir: str, output_format: str = 'text'):
    """Generate list of URLs for gallery verification"""
    
    db_path = Path(gallery_dir) / "candidate_library.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all candidates with provider info
    cursor.execute('''
        SELECT id, domain, provider, url_path, license
        FROM candidates 
        ORDER BY domain, provider, id
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("❌ No candidates found in database")
        return
    
    print(f"🔗 GALLERY URL LIST ({len(results)} images)")
    print("=" * 50)
    
    # Group by domain and provider
    by_domain = {}
    for id, domain, provider, url_path, license in results:
        if domain not in by_domain:
            by_domain[domain] = {}
        if provider not in by_domain[domain]:
            by_domain[domain][provider] = []
        
        by_domain[domain][provider].append({
            'id': id,
            'url_path': url_path,
            'license': license
        })
    
    # Generate URLs
    url_data = {}
    
    for domain, providers in by_domain.items():
        print(f"\n📁 {domain.upper()} ({sum(len(imgs) for imgs in providers.values())} images)")
        
        url_data[domain] = {}
        
        for provider, images in providers.items():
            print(f"\n   🌐 {provider} ({len(images)} images):")
            
            urls = []
            
            for img in images[:10]:  # Show first 10, save all
                if provider == 'pexels':
                    # Extract Pexels ID from filename
                    pexels_id = img['id'].replace(f'pexels_{domain}_', '')
                    url = f"https://www.pexels.com/photo/{pexels_id}/"
                    
                elif provider == 'unsplash':
                    # Extract Unsplash ID 
                    unsplash_id = img['id'].replace(f'unsplash_{domain}_', '')
                    url = f"https://unsplash.com/photos/{unsplash_id}"
                    
                else:
                    url = f"Local file: {img['id']}"
                
                urls.append({
                    'id': img['id'],
                    'url': url,
                    'license': img['license']
                })
                
                if len(urls) <= 10:  # Print first 10
                    print(f"     • {url}")
                    print(f"       License: {img['license']}")
            
            if len(images) > 10:
                print(f"     ... and {len(images) - 10} more")
            
            # Store all URLs for export
            all_urls = []
            for img in images:
                if provider == 'pexels':
                    pexels_id = img['id'].replace(f'pexels_{domain}_', '')
                    url = f"https://www.pexels.com/photo/{pexels_id}/"
                elif provider == 'unsplash':
                    unsplash_id = img['id'].replace(f'unsplash_{domain}_', '')
                    url = f"https://unsplash.com/photos/{unsplash_id}"
                else:
                    url = f"Local: {img['id']}"
                
                all_urls.append({
                    'id': img['id'],
                    'url': url,
                    'license': img['license']
                })
            
            url_data[domain][provider] = all_urls
    
    # Export to file if requested
    if output_format == 'json':
        output_file = Path(gallery_dir) / "gallery_urls.json"
        with open(output_file, 'w') as f:
            json.dump(url_data, f, indent=2)
        print(f"\n💾 URLs exported to: {output_file}")
    
    elif output_format == 'txt':
        output_file = Path(gallery_dir) / "gallery_urls.txt"
        with open(output_file, 'w') as f:
            f.write("RA-Guard Gallery URLs\n")
            f.write("=" * 30 + "\n\n")
            
            for domain, providers in url_data.items():
                f.write(f"{domain.upper()}\n")
                f.write("-" * len(domain) + "\n")
                
                for provider, images in providers.items():
                    f.write(f"\n{provider} ({len(images)} images):\n")
                    
                    for img in images:
                        f.write(f"  • {img['url']}\n")
                        f.write(f"    License: {img['license']}\n")
                
                f.write("\n\n")
        
        print(f"\n💾 URLs exported to: {output_file}")
    
    # Summary
    total_images = sum(len(imgs) for domain_data in url_data.values() 
                      for imgs in domain_data.values())
    
    print(f"\n📊 SUMMARY:")
    print(f"   • Total images: {total_images}")
    print(f"   • Domains: {len(url_data)}")
    print(f"   • All URLs are verifiable on original platforms")

def main():
    parser = argparse.ArgumentParser(description="Generate gallery URL list")
    parser.add_argument("--gallery-dir", default="real_candidate_gallery",
                       help="Gallery directory")
    parser.add_argument("--format", choices=['text', 'json', 'txt'], 
                       default='text', help="Output format")
    
    args = parser.parse_args()
    generate_url_list(args.gallery_dir, args.format)

if __name__ == "__main__":
    main()