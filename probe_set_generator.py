#!/usr/bin/env python3
"""
Canary Probe Set Generator

Creates a stable, balanced probe set for monitoring dataset quality drift.
"""

import json
import csv
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import logging
from datetime import datetime
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProbeSetGenerator:
    """Generate balanced probe sets for canary monitoring."""
    
    def __init__(self, output_path: str = "data/probe"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Fixed seed for reproducible probe set generation
        self.random_seed = 42
        random.seed(self.random_seed)
        
        logger.info(f"Initialized probe generator at {self.output_path}")
    
    def generate_positive_examples(self) -> List[Dict[str, Any]]:
        """Generate ~50 positive cocktail examples with domain balance."""
        
        # Domain-balanced positive examples
        positive_templates = {
            # Classic cocktails (martini/old fashioned family)
            'classic_clear': [
                "crystal clear martini with olive garnish in chilled glass",
                "gin martini with lemon twist in coupe glass", 
                "vodka martini with three olives on pick",
                "dry vermouth martini in frosted glass",
                "classic gin martini with cocktail onion",
            ],
            'classic_amber': [
                "golden old fashioned with orange peel and ice sphere",
                "whiskey old fashioned with muddled cherry",
                "bourbon old fashioned with expressed orange oils",
                "rye whiskey old fashioned with sugar cube",
                "aged whiskey cocktail with large ice cube",
            ],
            
            # Colored cocktails (vibrant/Instagram-worthy)
            'color_pink': [
                "bright pink cosmopolitan with lime wheel garnish",
                "rose-colored cocktail with dried flowers",
                "pink gin fizz with raspberry garnish",
                "fuchsia-colored martini with sugar rim",
                "blush pink cocktail with edible glitter",
            ],
            'color_blue': [
                "electric blue tropical cocktail with pineapple",
                "sapphire blue martini with silver rim",
                "ocean blue cocktail with umbrella garnish",
                "blue curacao cocktail with ice and mint",
                "azure blue drink with coconut rim",
            ],
            'color_green': [
                "emerald green cocktail with mint sprig",
                "lime green margarita with salt rim",
                "forest green cocktail with herbs",
                "jade colored drink with cucumber garnish",
                "bright green cocktail with basil leaves",
            ],
            
            # Tricky/edge cases (challenging but valid)
            'tricky_dark': [
                "black charcoal cocktail with silver garnish",
                "dark purple cocktail with blackberry",
                "midnight black drink with gold rim",
                "charcoal-filtered cocktail in coupe glass",
                "deep black cocktail with activated carbon",
            ],
            'tricky_cream': [
                "creamy white cocktail with cinnamon dusting",
                "ivory colored cocktail with vanilla foam",
                "cream-colored drink with nutmeg garnish",
                "milky white cocktail with coffee beans",
                "pale cream cocktail with caramel drizzle",
            ],
            'tricky_layered': [
                "layered cocktail with distinct color bands",
                "rainbow cocktail with multiple layers",
                "gradient cocktail from red to yellow",
                "ombre cocktail with color transition",
                "striped cocktail with density layers",
            ]
        }
        
        # Generate examples with Unsplash URLs (placeholder approach)
        positive_examples = []
        probe_id = 1
        
        for domain, descriptions in positive_templates.items():
            for desc in descriptions:
                # Generate stable hash-based IDs
                content_hash = hashlib.md5(f"{domain}_{desc}".encode()).hexdigest()[:8]
                
                example = {
                    'id': f"probe_pos_{probe_id:03d}",
                    'url': f"https://images.unsplash.com/photo-{content_hash}-placeholder?cocktail=true",
                    'description': desc,
                    'domain': domain,
                    'label': 'positive',
                    'category': 'cocktail',
                    'difficulty': 'easy' if 'classic' in domain or 'color' in domain else 'hard'
                }
                positive_examples.append(example)
                probe_id += 1
        
        return positive_examples[:50]  # Limit to 50
    
    def generate_negative_examples(self) -> List[Dict[str, Any]]:
        """Generate ~50 hard negative examples (non-cocktails in glass)."""
        
        # Hard negative categories
        negative_templates = {
            # Beverages that aren't cocktails
            'beverage_tea': [
                "green tea in clear glass teacup with steam",
                "black tea with lemon slice in glass mug",
                "herbal tea with mint leaves in transparent cup",
                "iced tea with ice cubes in tall glass",
                "bubble tea with tapioca pearls in clear cup",
            ],
            'beverage_soda': [
                "cola with ice in clear glass with straw",
                "sparkling water with lime in wine glass",
                "orange soda with bubbles in tall glass",
                "clear soda with lemon twist in tumbler",
                "ginger ale with ice in highball glass",
            ],
            'beverage_juice': [
                "fresh orange juice in clear glass pitcher",
                "apple juice in wine glass with apple slice",
                "cranberry juice in tall clear glass",
                "tomato juice with celery stick garnish",
                "grape juice in champagne flute",
            ],
            'beverage_water': [
                "sparkling mineral water with lime wedge",
                "infused water with cucumber slices",
                "plain water with ice in clear glass",
                "flavored water with berry garnish",
                "tonic water with bubbles in glass",
            ],
            
            # Non-beverage liquids in glassware
            'liquid_perfume': [
                "perfume bottle with golden liquid inside",
                "cologne in crystal decanter with stopper",
                "essential oil in glass dropper bottle",
                "fragrance in ornate glass bottle",
                "amber perfume in vintage glass container",
            ],
            'liquid_soup': [
                "clear broth soup in glass bowl with spoon",
                "gazpacho soup in wine glass with herbs",
                "tomato soup in clear glass mug",
                "consommé in transparent bowl",
                "cold soup served in glass vessel",
            ],
            'liquid_sauce': [
                "olive oil in glass cruet with herbs",
                "vinegar in crystal bottle with cork",
                "salad dressing in glass container",
                "honey in glass jar with dipper",
                "syrup in clear glass bottle",
            ],
            
            # Decorative/functional glass items
            'decorative_vase': [
                "clear glass vase with water and flowers",
                "colored glass vase with single stem",
                "crystal vase with arrangement",
                "tall glass vase with bamboo",
                "round glass bowl with floating candles",
            ],
            'decorative_candle': [
                "glass hurricane with pillar candle",
                "votive candle in clear glass holder",
                "floating candles in glass bowl",
                "glass lantern with tea light",
                "glass chimney with oil lamp",
            ]
        }
        
        # Generate negative examples
        negative_examples = []
        probe_id = 1
        
        for domain, descriptions in negative_templates.items():
            for desc in descriptions:
                content_hash = hashlib.md5(f"{domain}_{desc}".encode()).hexdigest()[:8]
                
                example = {
                    'id': f"probe_neg_{probe_id:03d}",
                    'url': f"https://images.unsplash.com/photo-{content_hash}-placeholder?cocktail=false",
                    'description': desc,
                    'domain': domain,
                    'label': 'negative',
                    'category': domain.split('_')[0],  # beverage, liquid, decorative
                    'difficulty': 'hard'  # All negatives are hard since they're in glassware
                }
                negative_examples.append(example)
                probe_id += 1
        
        return negative_examples[:50]  # Limit to 50
    
    def create_probe_set(self, version: str = "v1.0") -> Dict[str, Any]:
        """Create balanced probe set with metadata."""
        
        positive_examples = self.generate_positive_examples()
        negative_examples = self.generate_negative_examples()
        
        # Combine and shuffle for random ordering
        all_examples = positive_examples + negative_examples
        random.shuffle(all_examples)
        
        # Create probe set metadata
        probe_set = {
            'metadata': {
                'version': version,
                'created_at': datetime.now().isoformat(),
                'total_items': len(all_examples),
                'positive_count': len(positive_examples),
                'negative_count': len(negative_examples),
                'random_seed': self.random_seed,
                'description': 'Balanced probe set for canary monitoring of cocktail classification'
            },
            'balance_report': self._analyze_balance(all_examples),
            'examples': all_examples
        }
        
        return probe_set
    
    def _analyze_balance(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze balance across different dimensions."""
        
        # Label balance
        labels = {}
        domains = {}
        categories = {}
        difficulties = {}
        
        for example in examples:
            # Count labels
            label = example['label']
            labels[label] = labels.get(label, 0) + 1
            
            # Count domains
            domain = example['domain']
            domains[domain] = domains.get(domain, 0) + 1
            
            # Count categories
            category = example['category']
            categories[category] = categories.get(category, 0) + 1
            
            # Count difficulties
            difficulty = example['difficulty']
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        return {
            'label_distribution': labels,
            'domain_distribution': domains,
            'category_distribution': categories,
            'difficulty_distribution': difficulties
        }
    
    def export_probe_set(self, probe_set: Dict[str, Any], format: str = 'json') -> str:
        """Export probe set to file."""
        
        version = probe_set['metadata']['version']
        
        if format == 'json':
            output_file = self.output_path / f"probe_set_{version}.json"
            with open(output_file, 'w') as f:
                json.dump(probe_set, f, indent=2)
        
        elif format == 'csv':
            output_file = self.output_path / f"probe_set_{version}.csv"
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'id', 'url', 'description', 'domain', 'label', 
                    'category', 'difficulty'
                ])
                
                for example in probe_set['examples']:
                    writer.writerow([
                        example['id'],
                        example['url'],
                        example['description'],
                        example['domain'],
                        example['label'],
                        example['category'],
                        example['difficulty']
                    ])
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported probe set to {output_file}")
        return str(output_file)
    
    def validate_probe_set(self, probe_set: Dict[str, Any]) -> Dict[str, Any]:
        """Validate probe set meets requirements."""
        
        examples = probe_set['examples']
        balance = probe_set['balance_report']
        
        issues = []
        warnings = []
        
        # Check size requirements (50-100 items)
        total = len(examples)
        if not (50 <= total <= 100):
            issues.append(f"Total size {total} outside range [50, 100]")
        
        # Check positive/negative balance
        pos_count = balance['label_distribution'].get('positive', 0)
        neg_count = balance['label_distribution'].get('negative', 0)
        
        if abs(pos_count - 50) > 5:
            warnings.append(f"Positive count {pos_count} not close to 50")
        if abs(neg_count - 50) > 5:
            warnings.append(f"Negative count {neg_count} not close to 50")
        
        # Check domain diversity
        domains = balance['domain_distribution']
        if len(domains) < 8:
            warnings.append(f"Only {len(domains)} domains, expected 8+")
        
        # Check difficulty balance
        difficulties = balance['difficulty_distribution']
        easy_count = difficulties.get('easy', 0)
        hard_count = difficulties.get('hard', 0)
        
        if hard_count < 20:
            warnings.append(f"Only {hard_count} hard examples, expected 20+")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'stats': {
                'total_items': total,
                'positive_items': pos_count,
                'negative_items': neg_count,
                'domain_count': len(domains),
                'hard_examples': hard_count
            }
        }

def main():
    """CLI interface for probe set generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Canary Probe Set Generator')
    parser.add_argument('--version', default='v1.0', help='Probe set version')
    parser.add_argument('--output', default='data/probe', help='Output directory')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Export format')
    parser.add_argument('--validate', action='store_true', help='Validate probe set')
    
    args = parser.parse_args()
    
    # Generate probe set
    generator = ProbeSetGenerator(args.output)
    probe_set = generator.create_probe_set(args.version)
    
    # Export
    output_file = generator.export_probe_set(probe_set, args.format)
    
    # Validate
    if args.validate:
        validation = generator.validate_probe_set(probe_set)
        
        print("\n🔍 Probe Set Validation:")
        print(f"   Valid: {validation['is_valid']}")
        print(f"   Total items: {validation['stats']['total_items']}")
        print(f"   Positive: {validation['stats']['positive_items']}")
        print(f"   Negative: {validation['stats']['negative_items']}")
        print(f"   Domains: {validation['stats']['domain_count']}")
        print(f"   Hard examples: {validation['stats']['hard_examples']}")
        
        if validation['issues']:
            print("\n❌ Issues:")
            for issue in validation['issues']:
                print(f"   - {issue}")
        
        if validation['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   - {warning}")
    
    # Print summary
    balance = probe_set['balance_report']
    print(f"\n📊 Probe Set Generated: {output_file}")
    print(f"   Version: {args.version}")
    print(f"   Total: {probe_set['metadata']['total_items']} items")
    print(f"   Labels: {balance['label_distribution']}")
    print(f"   Domains: {len(balance['domain_distribution'])} types")
    print(f"   Difficulties: {balance['difficulty_distribution']}")

if __name__ == "__main__":
    main()