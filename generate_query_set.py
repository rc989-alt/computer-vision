#!/usr/bin/env python3
"""
RA-Guard Query Dataset Generator
Creates balanced 100-query evaluation set with easy/medium/hard distribution
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

@dataclass
class EvalQuery:
    """Evaluation query with difficulty and domain metadata"""
    id: str
    text: str
    domain: str
    difficulty: str  # easy, medium, hard
    expected_subjects: List[str]  # What should be found
    conflict_types: List[str]     # Expected conflicts to test
    notes: str = ""

class QuerySetGenerator:
    """Generate balanced evaluation queries"""
    
    def __init__(self):
        # Query templates by difficulty and domain
        self.cocktail_queries = {
            'easy': [
                ("refreshing cocktail", ["cocktail", "drink"], ["watermark"], "Simple drink request"),
                ("colorful drink", ["cocktail", "beverage"], ["text_overlay"], "Color-focused"),
                ("summer cocktail", ["cocktail", "glass"], ["season_mismatch"], "Seasonal basic"),
                ("martini glass", ["martini", "glass"], ["wrong_glassware"], "Specific glassware"),
                ("tropical drink", ["cocktail", "fruit"], ["indoor_outdoor"], "Tropical theme"),
            ],
            'medium': [
                ("elegant evening cocktail with garnish", ["cocktail", "garnish", "elegant"], ["casual_formal", "lighting"], "Multi-attribute"),
                ("craft cocktail with citrus and herbs", ["cocktail", "citrus", "herbs"], ["ingredient_mismatch"], "Ingredient specific"),
                ("whiskey cocktail in vintage setting", ["whiskey", "cocktail", "vintage"], ["style_period", "atmosphere"], "Style + ingredient"),
                ("frozen margarita with lime wheel", ["margarita", "lime", "frozen"], ["temperature", "garnish"], "Specific preparation"),
                ("artisanal cocktail with smoke effect", ["cocktail", "smoke", "artisanal"], ["technique", "presentation"], "Advanced technique"),
            ],
            'hard': [
                ("award-winning molecular gastronomy cocktail with spherification", ["molecular", "cocktail", "sphere"], ["technique_complexity", "presentation"], "High technique"),
                ("prohibition-era inspired rye whiskey cocktail in speakeasy ambiance", ["rye", "cocktail", "prohibition", "speakeasy"], ["historical", "atmosphere", "ingredient"], "Historical complexity"),
                ("temperature-layered cocktail with contrasting textures and complementary flavor profiles", ["layered", "cocktail", "temperature"], ["technique", "complexity"], "Multi-constraint"),
                ("sustainable zero-waste cocktail using upcycled ingredients and biodegradable garnishes", ["sustainable", "cocktail", "zero-waste"], ["ethical", "ingredient"], "Conceptual complexity"),
            ]
        }
        
        self.flower_queries = {
            'easy': [
                ("beautiful flowers", ["flower", "bloom"], ["artificial"], "Basic flower"),
                ("red roses", ["rose", "red"], ["color_mismatch"], "Color + type"),
                ("spring flowers", ["flower", "spring"], ["season"], "Seasonal"),
                ("wedding bouquet", ["bouquet", "wedding"], ["occasion"], "Event specific"),
                ("garden flowers", ["flower", "garden"], ["indoor_outdoor"], "Setting based"),
            ],
            'medium': [
                ("delicate pastel flowers in morning light", ["flower", "pastel", "morning"], ["lighting", "color"], "Light + color"),
                ("wild meadow flowers with dewdrops", ["wildflower", "meadow", "dew"], ["natural", "detail"], "Natural detail"),
                ("exotic orchids in greenhouse setting", ["orchid", "exotic", "greenhouse"], ["species", "environment"], "Specific environment"),
                ("vintage roses in antique vase arrangement", ["rose", "vintage", "vase"], ["style", "container"], "Style + arrangement"),
                ("seasonal tulips in Dutch countryside", ["tulip", "seasonal", "countryside"], ["location", "season"], "Geographic + time"),
            ],
            'hard': [
                ("rare endemic alpine flowers blooming at high altitude during brief summer window", ["alpine", "endemic", "rare"], ["rarity", "season", "location"], "Ecological complexity"),
                ("botanically accurate heirloom flower varieties preserving genetic diversity", ["heirloom", "genetic", "botanical"], ["accuracy", "conservation"], "Scientific accuracy"),
                ("permaculture flower garden demonstrating companion planting and natural pest management", ["permaculture", "companion", "natural"], ["sustainability", "technique"], "Agricultural complexity"),
            ]
        }
        
        self.professional_queries = {
            'easy': [
                ("business headshot", ["headshot", "business"], ["casual_formal"], "Basic professional"),
                ("office meeting", ["office", "meeting"], ["setting"], "Workplace"),
                ("professional woman", ["professional", "woman"], ["gender", "attire"], "Gender + role"),
                ("corporate team", ["corporate", "team"], ["group", "business"], "Team context"),
                ("conference presentation", ["conference", "presentation"], ["event"], "Event type"),
            ],
            'medium': [
                ("executive portrait with confident expression in modern office", ["executive", "portrait", "confident"], ["role", "expression", "setting"], "Multi-attribute professional"),
                ("diverse tech team collaborating on innovative project", ["diverse", "tech", "collaboration"], ["diversity", "industry"], "Industry + diversity"),
                ("senior consultant presenting data insights to stakeholders", ["consultant", "presenting", "data"], ["role", "activity"], "Specific role + action"),
                ("professional woman in STEM field working with advanced equipment", ["woman", "STEM", "equipment"], ["gender", "field", "technology"], "Gender + specialization"),
            ],
            'hard': [
                ("C-suite executive demonstrating authentic leadership during crisis management", ["executive", "leadership", "crisis"], ["authenticity", "situation"], "Leadership complexity"),
                ("interdisciplinary research team achieving breakthrough innovation through collaborative methodology", ["research", "breakthrough", "collaborative"], ["innovation", "methodology"], "Innovation complexity"),
            ]
        }
    
    def generate_query_set(self, total_queries: int = 100, domains: List[str] = None) -> List[EvalQuery]:
        """Generate balanced query set"""
        
        if domains is None:
            domains = ['cocktails']  # Default to single domain
        
        queries_per_domain = total_queries // len(domains)
        
        # Difficulty distribution: easy=30%, medium=50%, hard=20%
        easy_count = int(queries_per_domain * 0.30)
        medium_count = int(queries_per_domain * 0.50) 
        hard_count = queries_per_domain - easy_count - medium_count
        
        all_queries = []
        query_id = 1
        
        for domain in domains:
            if domain == 'cocktails':
                domain_templates = self.cocktail_queries
            elif domain == 'flowers':
                domain_templates = self.flower_queries
            elif domain == 'professional':
                domain_templates = self.professional_queries
            else:
                continue
            
            # Generate queries for each difficulty
            for difficulty, count in [('easy', easy_count), ('medium', medium_count), ('hard', hard_count)]:
                templates = domain_templates[difficulty]
                
                # Sample with replacement if needed
                selected = random.choices(templates, k=count)
                
                for text, subjects, conflicts, notes in selected:
                    query = EvalQuery(
                        id=f"q_{query_id:03d}",
                        text=text,
                        domain=domain,
                        difficulty=difficulty,
                        expected_subjects=subjects,
                        conflict_types=conflicts,
                        notes=notes
                    )
                    all_queries.append(query)
                    query_id += 1
        
        # Shuffle to randomize order
        random.shuffle(all_queries)
        
        return all_queries
    
    def save_query_set(self, queries: List[EvalQuery], output_path: str = "datasets/mini_100q.json"):
        """Save query set to JSON file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        # Convert to serializable format
        query_data = {
            'metadata': {
                'total_queries': len(queries),
                'domains': list(set(q.domain for q in queries)),
                'difficulty_distribution': {
                    'easy': sum(1 for q in queries if q.difficulty == 'easy'),
                    'medium': sum(1 for q in queries if q.difficulty == 'medium'), 
                    'hard': sum(1 for q in queries if q.difficulty == 'hard')
                },
                'created_for': 'RA-Guard evaluation pipeline'
            },
            'queries': [asdict(q) for q in queries]
        }
        
        with open(output_file, 'w') as f:
            json.dump(query_data, f, indent=2)
        
        return output_file

def main():
    print("🎯 GENERATING RA-GUARD EVALUATION QUERY SET")
    print("=" * 50)
    
    generator = QuerySetGenerator()
    
    # Generate 100 queries for cocktails domain (pilot)
    queries = generator.generate_query_set(
        total_queries=100,
        domains=['cocktails']
    )
    
    # Save query set
    output_file = generator.save_query_set(queries)
    
    # Show statistics
    difficulty_counts = {}
    for q in queries:
        difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
    
    print(f"📊 Generated {len(queries)} queries:")
    print(f"   • Easy: {difficulty_counts.get('easy', 0)} (30%)")
    print(f"   • Medium: {difficulty_counts.get('medium', 0)} (50%)")
    print(f"   • Hard: {difficulty_counts.get('hard', 0)} (20%)")
    print(f"   • Domain: cocktails")
    print(f"   • Saved to: {output_file}")
    
    # Show sample queries
    print(f"\n📋 SAMPLE QUERIES:")
    for difficulty in ['easy', 'medium', 'hard']:
        sample = next((q for q in queries if q.difficulty == difficulty), None)
        if sample:
            print(f"   {difficulty.upper()}: \"{sample.text}\"")
            print(f"      Expected: {sample.expected_subjects}")
            print(f"      Conflicts: {sample.conflict_types}")

if __name__ == "__main__":
    main()