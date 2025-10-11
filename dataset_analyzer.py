#!/usr/bin/env python3
"""
Dataset Query and Analysis Utilities

Simple tools to query and analyze the managed dataset.
"""

import sqlite3
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

class DatasetAnalyzer:
    """Analyze and query the managed dataset."""
    
    def __init__(self, dataset_path: str = "data/dataset"):
        self.dataset_path = Path(dataset_path)
        self.db_path = self.dataset_path / "metadata" / "dataset.db"
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Dataset database not found: {self.db_path}")
    
    def query_by_domain(self, domain: str, limit: Optional[int] = None) -> List[Dict]:
        """Query images by domain."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM images WHERE domain = ?"
        params = [domain]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def query_by_split(self, split: str) -> List[Dict]:
        """Query images by split (train/val/test)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM images WHERE split = ?", [split])
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of images across domains."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT domain, sub_domain, COUNT(*) as count
            FROM images 
            GROUP BY domain, sub_domain
            ORDER BY domain, count DESC
        """)
        
        results = {}
        for domain, sub_domain, count in cursor.fetchall():
            if domain not in results:
                results[domain] = {}
            results[domain][sub_domain] = count
        
        conn.close()
        return results
    
    def get_score_analysis(self) -> Dict[str, Any]:
        """Analyze baseline vs enhanced scores."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT domain, 
                   AVG(score_baseline) as avg_baseline,
                   AVG(score_enhanced) as avg_enhanced,
                   AVG(score_enhanced - score_baseline) as avg_improvement,
                   COUNT(*) as count
            FROM images 
            WHERE score_baseline > 0 AND score_enhanced > 0
            GROUP BY domain
            ORDER BY avg_improvement DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'domain': row[0],
                'avg_baseline': round(row[1], 3),
                'avg_enhanced': round(row[2], 3), 
                'avg_improvement': round(row[3], 3),
                'count': row[4]
            })
        
        conn.close()
        return results
    
    def find_duplicates(self, hash_type: str = 'phash') -> List[Dict]:
        """Find potential duplicate images by hash."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT {hash_type}, COUNT(*) as count, GROUP_CONCAT(image_id) as image_ids
            FROM images 
            WHERE {hash_type} != ''
            GROUP BY {hash_type}
            HAVING count > 1
        """)
        
        results = []
        for hash_val, count, image_ids in cursor.fetchall():
            results.append({
                'hash': hash_val,
                'hash_type': hash_type,
                'count': count,
                'image_ids': image_ids.split(',')
            })
        
        conn.close()
        return results
    
    def export_split_files(self):
        """Export train/val/test split files."""
        splits_dir = self.dataset_path / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            images = self.query_by_split(split)
            
            split_data = {
                'split': split,
                'count': len(images),
                'images': [
                    {
                        'image_id': img['image_id'],
                        'url': img['url'],
                        'local_path': img['local_path'],
                        'domain': img['domain'],
                        'sub_domain': img['sub_domain']
                    }
                    for img in images
                ]
            }
            
            split_file = splits_dir / f"{split}.json"
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            print(f"Exported {len(images)} images to {split_file}")
    
    def generate_report(self) -> str:
        """Generate comprehensive dataset report."""
        report = []
        report.append("# Dataset Analysis Report")
        report.append(f"Generated: {pd.Timestamp.now()}")
        report.append("")
        
        # Basic stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT domain) FROM images")
        total_domains = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM images WHERE local_path IS NOT NULL")
        downloaded_images = cursor.fetchone()[0]
        
        report.append("## Overview")
        report.append(f"- Total images: {total_images}")
        report.append(f"- Unique domains: {total_domains}")
        report.append(f"- Downloaded images: {downloaded_images}")
        report.append(f"- Download rate: {downloaded_images/total_images:.1%}")
        report.append("")
        
        # Domain distribution
        report.append("## Domain Distribution")
        domain_dist = self.get_domain_distribution()
        for domain, sub_domains in domain_dist.items():
            total_domain = sum(sub_domains.values())
            report.append(f"### {domain} ({total_domain} images)")
            for sub_domain, count in sub_domains.items():
                report.append(f"- {sub_domain}: {count}")
        report.append("")
        
        # Score analysis
        report.append("## Score Analysis")
        score_analysis = self.get_score_analysis()
        if score_analysis:
            report.append("| Domain | Baseline | Enhanced | Improvement | Count |")
            report.append("|--------|----------|----------|-------------|-------|")
            for analysis in score_analysis:
                report.append(f"| {analysis['domain']} | {analysis['avg_baseline']:.3f} | {analysis['avg_enhanced']:.3f} | +{analysis['avg_improvement']:.3f} | {analysis['count']} |")
        report.append("")
        
        # Split distribution
        report.append("## Split Distribution")
        cursor.execute("SELECT split, COUNT(*) FROM images GROUP BY split")
        for split, count in cursor.fetchall():
            percentage = count / total_images * 100
            report.append(f"- {split}: {count} ({percentage:.1f}%)")
        
        conn.close()
        
        return "\n".join(report)

def main():
    """CLI interface for dataset analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Analysis Tools')
    parser.add_argument('--domain', help='Query by domain')
    parser.add_argument('--split', help='Query by split (train/val/test)')
    parser.add_argument('--duplicates', action='store_true', help='Find duplicate images')
    parser.add_argument('--report', action='store_true', help='Generate full report')
    parser.add_argument('--export-splits', action='store_true', help='Export split files')
    parser.add_argument('--limit', type=int, help='Limit number of results')
    
    args = parser.parse_args()
    
    try:
        analyzer = DatasetAnalyzer()
        
        if args.domain:
            results = analyzer.query_by_domain(args.domain, args.limit)
            print(f"Found {len(results)} images in domain '{args.domain}':")
            for result in results:
                print(f"  {result['image_id']}: {result['alt_description'][:50]}...")
        
        elif args.split:
            results = analyzer.query_by_split(args.split)
            print(f"Found {len(results)} images in {args.split} split:")
            for result in results:
                print(f"  {result['image_id']} ({result['domain']})")
        
        elif args.duplicates:
            duplicates = analyzer.find_duplicates('phash')
            if duplicates:
                print(f"Found {len(duplicates)} potential duplicate groups:")
                for dup in duplicates:
                    print(f"  Hash {dup['hash']}: {dup['image_ids']}")
            else:
                print("No duplicates found")
        
        elif args.export_splits:
            analyzer.export_split_files()
        
        elif args.report:
            report = analyzer.generate_report()
            
            report_file = Path("data/dataset/metadata/analysis_report.md")
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(report)
            print(f"\nReport saved to: {report_file}")
        
        else:
            # Default: show domain distribution
            print("Domain Distribution:")
            domain_dist = analyzer.get_domain_distribution()
            for domain, sub_domains in domain_dist.items():
                total = sum(sub_domains.values())
                print(f"  {domain}: {total} images")
                for sub_domain, count in sub_domains.items():
                    print(f"    └─ {sub_domain}: {count}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the dataset manager first to create the database.")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())