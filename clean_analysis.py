#!/usr/bin/env python3
"""
Clean Dataset Analysis & Metrics

Comprehensive analysis of the clean dataset after overlay corrections,
with full provenance tracking and quality metrics.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import numpy as np

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CleanDatasetAnalyzer:
    """Analyze the clean dataset with provenance tracking."""
    
    def __init__(self, dataset_path: str = "data/dataset"):
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "metadata"
        
        # Load clean dataset
        from overlay_loader import OverlayDatasetLoader
        self.loader = OverlayDatasetLoader(str(self.dataset_path))
        self.clean_dataset = self.loader.load_clean_dataset()
        
        # Convert to DataFrame for analysis
        self.df = self._create_dataframe()
        
        logger.info(f"Initialized analyzer with {len(self.df)} clean items")
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert clean dataset to pandas DataFrame."""
        data = []
        
        for inspiration in self.clean_dataset['inspirations']:
            query = inspiration['query']
            description = inspiration.get('description', '')
            
            for candidate in inspiration['candidates']:
                item = {
                    'id': candidate['id'],
                    'url': candidate['regular'],
                    'query': query,
                    'description': description,
                    'alt_description': candidate.get('alt_description', ''),
                    'score_baseline': candidate.get('baseline_score', 0.0),
                    'score_enhanced': candidate.get('score', 0.0),
                    'domain': self._classify_domain(query, candidate.get('alt_description', ''))
                }
                data.append(item)
        
        return pd.DataFrame(data)
    
    def _classify_domain(self, query: str, description: str) -> str:
        """Classify domain based on query and description."""
        text = f"{query} {description}".lower()
        
        # Color-based classification (matches original approach)
        color_patterns = {
            'color_pink': ['pink', 'rose', 'blush'],
            'color_golden': ['golden', 'yellow', 'amber', 'gold'],
            'color_blue': ['blue', 'azure', 'cobalt'],
            'color_green': ['green', 'lime', 'mint'],
            'color_red': ['red', 'ruby', 'crimson', 'strawberry'],
            'color_clear': ['clear', 'transparent', 'crystal'],
            'color_purple': ['purple', 'violet', 'lavender'],
            'color_orange': ['orange', 'tangerine', 'peach'],
            'color_black': ['black', 'charcoal', 'dark'],
            'color_white': ['white', 'cream', 'ivory'],
            'color_silver': ['silver', 'metallic', 'chrome']
        }
        
        for domain, patterns in color_patterns.items():
            if any(pattern in text for pattern in patterns):
                return domain
        
        return 'unclassified'
    
    def get_dataset_overview(self) -> Dict[str, Any]:
        """Get comprehensive dataset overview."""
        
        # Basic statistics
        total_items = len(self.df)
        unique_queries = self.df['query'].nunique()
        
        # Score analysis
        baseline_scores = self.df['score_baseline']
        enhanced_scores = self.df['score_enhanced']
        improvements = enhanced_scores - baseline_scores
        
        # Domain distribution
        domain_dist = self.df['domain'].value_counts().to_dict()
        
        # Query distribution  
        query_dist = self.df['query'].value_counts().to_dict()
        
        return {
            'basic_stats': {
                'total_items': total_items,
                'unique_queries': unique_queries,
                'avg_candidates_per_query': total_items / unique_queries if unique_queries > 0 else 0
            },
            'score_analysis': {
                'baseline_mean': float(baseline_scores.mean()),
                'baseline_std': float(baseline_scores.std()),
                'enhanced_mean': float(enhanced_scores.mean()),
                'enhanced_std': float(enhanced_scores.std()),
                'improvement_mean': float(improvements.mean()),
                'improvement_std': float(improvements.std()),
                'improvement_median': float(improvements.median()),
                'items_improved': int((improvements > 0).sum()),
                'items_degraded': int((improvements < 0).sum()),
                'items_unchanged': int((improvements == 0).sum())
            },
            'domain_distribution': domain_dist,
            'query_distribution': query_dist
        }
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality metrics for the clean dataset."""
        
        # URL quality
        valid_urls = self.df['url'].str.startswith('https://').sum()
        unique_urls = self.df['url'].nunique()
        
        # Description quality
        has_alt_desc = self.df['alt_description'].str.len() > 0
        avg_desc_length = self.df['alt_description'].str.len().mean()
        
        # Score quality
        has_both_scores = (self.df['score_baseline'] > 0) & (self.df['score_enhanced'] > 0)
        
        return {
            'url_quality': {
                'valid_https_urls': int(valid_urls),
                'total_urls': len(self.df),
                'url_validity_rate': float(valid_urls / len(self.df)),
                'unique_urls': int(unique_urls),
                'url_uniqueness_rate': float(unique_urls / len(self.df))
            },
            'description_quality': {
                'items_with_alt_description': int(has_alt_desc.sum()),
                'alt_description_coverage': float(has_alt_desc.mean()),
                'avg_alt_description_length': float(avg_desc_length)
            },
            'score_quality': {
                'items_with_both_scores': int(has_both_scores.sum()),
                'score_completeness': float(has_both_scores.mean())
            }
        }
    
    def get_provenance_info(self) -> Dict[str, Any]:
        """Get provenance and data lineage information."""
        
        # Load original snapshot for comparison
        frozen_path = self.metadata_path / "frozen_snapshot.json"
        with open(frozen_path, 'r') as f:
            original_snapshot = json.load(f)
        
        # Count original items
        original_count = 0
        for insp in original_snapshot['data']['inspirations']:
            original_count += len(insp['candidates'])
        
        # Overlay information
        overlays_applied = self.clean_dataset.get('applied_overlays', [])
        
        return {
            'data_lineage': {
                'original_snapshot_items': original_count,
                'clean_dataset_items': len(self.df),
                'items_excluded': original_count - len(self.df),
                'exclusion_rate': float((original_count - len(self.df)) / original_count) if original_count > 0 else 0
            },
            'overlay_provenance': {
                'overlays_applied': len(overlays_applied),
                'overlay_details': overlays_applied
            },
            'validation_metadata': self.clean_dataset.get('metadata', {})
        }
    
    def analyze_improvements(self) -> Dict[str, Any]:
        """Analyze score improvements by domain and query."""
        
        improvements = self.df['score_enhanced'] - self.df['score_baseline']
        self.df['improvement'] = improvements
        
        # By domain - flatten MultiIndex columns
        domain_grouped = self.df.groupby('domain').agg({
            'improvement': ['mean', 'std', 'median', 'count'],
            'score_baseline': 'mean',
            'score_enhanced': 'mean'
        }).round(3)
        
        domain_analysis = {}
        for domain in domain_grouped.index:
            domain_analysis[domain] = {
                'improvement_mean': float(domain_grouped.loc[domain, ('improvement', 'mean')]),
                'improvement_std': float(domain_grouped.loc[domain, ('improvement', 'std')]),
                'improvement_median': float(domain_grouped.loc[domain, ('improvement', 'median')]),
                'count': int(domain_grouped.loc[domain, ('improvement', 'count')]),
                'baseline_mean': float(domain_grouped.loc[domain, ('score_baseline', 'mean')]),
                'enhanced_mean': float(domain_grouped.loc[domain, ('score_enhanced', 'mean')])
            }
        
        # By query - flatten MultiIndex columns  
        query_grouped = self.df.groupby('query').agg({
            'improvement': ['mean', 'std', 'median', 'count'],
            'score_baseline': 'mean',
            'score_enhanced': 'mean'
        }).round(3)
        
        query_analysis = {}
        for query in query_grouped.index:
            query_analysis[query] = {
                'improvement_mean': float(query_grouped.loc[query, ('improvement', 'mean')]),
                'improvement_std': float(query_grouped.loc[query, ('improvement', 'std')]),
                'improvement_median': float(query_grouped.loc[query, ('improvement', 'median')]),
                'count': int(query_grouped.loc[query, ('improvement', 'count')]),
                'baseline_mean': float(query_grouped.loc[query, ('score_baseline', 'mean')]),
                'enhanced_mean': float(query_grouped.loc[query, ('score_enhanced', 'mean')])
            }
        
        return {
            'domain_improvements': domain_analysis,
            'query_improvements': query_analysis,
            'overall_improvement': {
                'mean': float(improvements.mean()),
                'median': float(improvements.median()),
                'std': float(improvements.std()),
                'best_improvement': float(improvements.max()),
                'worst_improvement': float(improvements.min())
            }
        }
    
    def export_comprehensive_report(self, output_path: str = None) -> str:
        """Export comprehensive analysis report."""
        if output_path is None:
            output_path = str(self.metadata_path / "clean_dataset_analysis.json")
        
        # Collect all analysis
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset_path': str(self.dataset_path),
                'analyzer_version': '1.0.0'
            },
            'dataset_overview': self.get_dataset_overview(),
            'quality_metrics': self.get_quality_metrics(),
            'provenance_info': self.get_provenance_info(),
            'improvement_analysis': self.analyze_improvements()
        }
        
        # Export to JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exported comprehensive report to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a formatted summary of the clean dataset."""
        overview = self.get_dataset_overview()
        quality = self.get_quality_metrics()
        provenance = self.get_provenance_info()
        
        print("=" * 60)
        print("🧹 CLEAN DATASET ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Basic stats
        basic = overview['basic_stats']
        print(f"\n📊 Dataset Overview:")
        print(f"   Total items: {basic['total_items']}")
        print(f"   Unique queries: {basic['unique_queries']}")
        print(f"   Avg candidates per query: {basic['avg_candidates_per_query']:.1f}")
        
        # Provenance
        lineage = provenance['data_lineage']
        print(f"\n🔍 Data Provenance:")
        print(f"   Original → Clean: {lineage['original_snapshot_items']} → {lineage['clean_dataset_items']}")
        print(f"   Exclusion rate: {lineage['exclusion_rate']:.1%}")
        print(f"   Overlays applied: {provenance['overlay_provenance']['overlays_applied']}")
        
        # Quality metrics
        url_qual = quality['url_quality']
        desc_qual = quality['description_quality']
        print(f"\n✅ Quality Metrics:")
        print(f"   URL validity: {url_qual['url_validity_rate']:.1%}")
        print(f"   URL uniqueness: {url_qual['url_uniqueness_rate']:.1%}")
        print(f"   Alt description coverage: {desc_qual['alt_description_coverage']:.1%}")
        
        # Scores
        scores = overview['score_analysis']
        print(f"\n📈 Score Analysis:")
        print(f"   Baseline mean: {scores['baseline_mean']:.3f} ± {scores['baseline_std']:.3f}")
        print(f"   Enhanced mean: {scores['enhanced_mean']:.3f} ± {scores['enhanced_std']:.3f}")
        print(f"   Improvement: {scores['improvement_mean']:.3f} ± {scores['improvement_std']:.3f}")
        print(f"   Items improved: {scores['items_improved']}/{basic['total_items']} ({scores['items_improved']/basic['total_items']:.1%})")
        
        # Domain distribution
        print(f"\n🎯 Domain Distribution:")
        for domain, count in sorted(overview['domain_distribution'].items()):
            print(f"   {domain}: {count}")
        
        print("=" * 60)

def main():
    """CLI interface for clean dataset analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean Dataset Analysis')
    parser.add_argument('--summary', action='store_true', help='Print dataset summary')
    parser.add_argument('--export', help='Export comprehensive report to JSON file')
    parser.add_argument('--csv', help='Export clean data to CSV file')
    
    args = parser.parse_args()
    
    analyzer = CleanDatasetAnalyzer()
    
    if args.summary:
        analyzer.print_summary()
    
    if args.export:
        report_path = analyzer.export_comprehensive_report(args.export)
        print(f"✅ Exported comprehensive report to {report_path}")
    
    if args.csv:
        analyzer.df.to_csv(args.csv, index=False)
        print(f"✅ Exported clean data to {args.csv}")
    
    if not any([args.summary, args.export, args.csv]):
        # Default: print summary
        analyzer.print_summary()

if __name__ == "__main__":
    main()