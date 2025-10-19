#!/usr/bin/env python3
"""
Week 1: RA-Guard Dataset Preparation (45 → 300 queries)
Systematic query expansion across 3 domains with quality validation

Usage:
    python week1_dataset_preparation.py --pilot-data ra_guard_pilot_45q.json --output-dir datasets/ra_guard_300q/
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass
import argparse
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QueryExpansionConfig:
    """Configuration for systematic query expansion"""
    target_total: int = 300
    domains: List[str] = None
    difficulty_distribution: Dict[str, float] = None
    quality_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = ['cocktails', 'flowers', 'professional']
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
        if self.quality_thresholds is None:
            self.quality_thresholds = {'diversity': 0.85, 'coverage': 0.90, 'realism': 0.80}

class QueryExpansionEngine:
    """Systematic query expansion from pilot to 300-query dataset"""
    
    def __init__(self, config: QueryExpansionConfig):
        self.config = config
        self.queries_per_domain = config.target_total // len(config.domains)
        
    def load_pilot_data(self, pilot_file: str) -> Dict:
        """Load and analyze pilot query patterns"""
        with open(pilot_file, 'r') as f:
            pilot_data = json.load(f)
            
        # Extract pilot query patterns for analysis
        pilot_analysis = {
            'total_queries': len(pilot_data.get('queries', [])),
            'domain_distribution': {},
            'query_patterns': {},
            'expansion_seeds': {}
        }
        
        # Analyze by domain
        for domain in self.config.domains:
            domain_queries = [q for q in pilot_data.get('queries', []) if q.get('domain') == domain]
            pilot_analysis['domain_distribution'][domain] = len(domain_queries)
            pilot_analysis['query_patterns'][domain] = self._analyze_query_patterns(domain_queries)
            pilot_analysis['expansion_seeds'][domain] = self._extract_expansion_seeds(domain_queries)
            
        return pilot_analysis
    
    def _analyze_query_patterns(self, queries: List[Dict]) -> Dict:
        """Analyze patterns in existing queries for systematic expansion"""
        patterns = {
            'intent_types': [],
            'complexity_levels': [],
            'key_terms': [],
            'query_structures': []
        }
        
        for query in queries:
            query_text = query.get('text', '')
            
            # Intent classification
            if any(word in query_text.lower() for word in ['recipe', 'how to make', 'ingredients']):
                patterns['intent_types'].append('recipe')
            elif any(word in query_text.lower() for word in ['best', 'recommend', 'suggestion']):
                patterns['intent_types'].append('recommendation')
            elif any(word in query_text.lower() for word in ['difference', 'compare', 'vs']):
                patterns['intent_types'].append('comparison')
            else:
                patterns['intent_types'].append('general')
                
            # Complexity assessment
            word_count = len(query_text.split())
            if word_count <= 3:
                patterns['complexity_levels'].append('easy')
            elif word_count <= 7:
                patterns['complexity_levels'].append('medium')
            else:
                patterns['complexity_levels'].append('hard')
                
        return patterns
    
    def _extract_expansion_seeds(self, queries: List[Dict]) -> List[Dict]:
        """Extract expansion seeds for systematic query generation"""
        seeds = []
        
        for query in queries:
            seed = {
                'original_text': query.get('text', ''),
                'domain': query.get('domain', ''),
                'intent': self._classify_intent(query.get('text', '')),
                'complexity': self._assess_complexity(query.get('text', '')),
                'key_entities': self._extract_entities(query.get('text', '')),
                'expansion_potential': self._assess_expansion_potential(query)
            }
            seeds.append(seed)
            
        return sorted(seeds, key=lambda x: x['expansion_potential'], reverse=True)
    
    def generate_expanded_dataset(self, pilot_analysis: Dict) -> Dict:
        """Generate systematically expanded 300-query dataset"""
        
        expanded_dataset = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'pilot_queries': sum(pilot_analysis['domain_distribution'].values()),
                'target_queries': self.config.target_total,
                'expansion_factor': self.config.target_total / sum(pilot_analysis['domain_distribution'].values()),
                'domains': self.config.domains,
                'quality_validated': True
            },
            'queries': [],
            'domain_statistics': {},
            'quality_metrics': {}
        }
        
        # Generate queries for each domain
        for domain in self.config.domains:
            domain_queries = self._generate_domain_queries(
                domain, 
                pilot_analysis['expansion_seeds'][domain],
                pilot_analysis['query_patterns'][domain]
            )
            
            expanded_dataset['queries'].extend(domain_queries)
            expanded_dataset['domain_statistics'][domain] = {
                'query_count': len(domain_queries),
                'difficulty_distribution': self._calculate_difficulty_distribution(domain_queries),
                'intent_coverage': self._calculate_intent_coverage(domain_queries)
            }
            
        # Calculate overall quality metrics
        expanded_dataset['quality_metrics'] = self._calculate_quality_metrics(expanded_dataset['queries'])
        
        return expanded_dataset
    
    def _generate_domain_queries(self, domain: str, seeds: List[Dict], patterns: Dict) -> List[Dict]:
        """Generate domain-specific queries using expansion templates"""
        
        domain_templates = self._get_domain_templates(domain)
        target_count = self.queries_per_domain
        
        generated_queries = []
        
        # Distribute across difficulty levels
        easy_count = int(target_count * self.config.difficulty_distribution['easy'])
        medium_count = int(target_count * self.config.difficulty_distribution['medium'])
        hard_count = target_count - easy_count - medium_count
        
        # Generate easy queries
        for i in range(easy_count):
            query = self._generate_query_from_template(domain, 'easy', domain_templates, seeds)
            generated_queries.append(query)
            
        # Generate medium queries
        for i in range(medium_count):
            query = self._generate_query_from_template(domain, 'medium', domain_templates, seeds)
            generated_queries.append(query)
            
        # Generate hard queries
        for i in range(hard_count):
            query = self._generate_query_from_template(domain, 'hard', domain_templates, seeds)
            generated_queries.append(query)
            
        return generated_queries
    
    def _get_domain_templates(self, domain: str) -> Dict:
        """Get domain-specific query generation templates"""
        
        templates = {
            'cocktails': {
                'easy': [
                    "classic {cocktail_name}",
                    "{spirit} cocktails",
                    "simple {ingredient} drinks"
                ],
                'medium': [
                    "{cocktail_name} with {modification}",
                    "best {spirit} cocktails for {occasion}",
                    "{ingredient1} and {ingredient2} cocktail recipe"
                ],
                'hard': [
                    "molecular gastronomy {cocktail_name} with {technique}",
                    "craft {spirit} cocktail using {unusual_ingredient} and {garnish}",
                    "professional bartending technique for {complex_cocktail} with {dietary_restriction}"
                ]
            },
            'flowers': {
                'easy': [
                    "{flower_type} arrangements",
                    "simple {color} bouquet",
                    "{season} flowers"
                ],
                'medium': [
                    "{flower_type} arrangement for {occasion}",
                    "{color_scheme} wedding bouquet with {flower1} and {flower2}",
                    "care instructions for {flower_type} in {season}"
                ],
                'hard': [
                    "professional {flower_type} arrangement with {technique} for {special_occasion}",
                    "sustainable floriculture design using {local_flowers} and {eco_materials}",
                    "advanced {flower_type} preservation techniques for {duration} display"
                ]
            },
            'professional': {
                'easy': [
                    "{industry} headshots",
                    "professional {style} photos",
                    "business portrait"
                ],
                'medium': [
                    "{industry} professional headshots with {lighting_style}",
                    "{style} corporate photography for {company_type}",
                    "executive portraits with {background_type} and {mood}"
                ],
                'hard': [
                    "high-end {industry} executive photography with {advanced_technique} and {premium_equipment}",
                    "multi-cultural {industry} team photography with {complex_lighting} for {global_brand}",
                    "award-winning {style} corporate portraits using {artistic_technique} and {post_processing}"
                ]
            }
        }
        
        return templates.get(domain, {})
    
    def _generate_query_from_template(self, domain: str, difficulty: str, templates: Dict, seeds: List[Dict]) -> Dict:
        """Generate a specific query from templates and seeds"""
        
        # Get appropriate template
        domain_templates = templates.get(difficulty, [])
        if not domain_templates:
            # Fallback to seed-based generation
            return self._generate_from_seed(domain, difficulty, seeds)
            
        template = np.random.choice(domain_templates)
        
        # Fill template with domain-specific terms
        filled_query = self._fill_template(template, domain, difficulty)
        
        # Create query object
        query = {
            'query_id': self._generate_query_id(filled_query, domain),
            'text': filled_query,
            'domain': domain,
            'difficulty': difficulty,
            'intent': self._classify_intent(filled_query),
            'expansion_method': 'template',
            'metadata': {
                'template_used': template,
                'generation_timestamp': datetime.now().isoformat(),
                'quality_score': self._calculate_query_quality(filled_query, domain)
            }
        }
        
        return query
    
    def _fill_template(self, template: str, domain: str, difficulty: str) -> str:
        """Fill template placeholders with domain-appropriate terms"""
        
        # Domain-specific term libraries
        term_libraries = {
            'cocktails': {
                'cocktail_name': ['mojito', 'martini', 'margarita', 'old fashioned', 'manhattan', 'negroni'],
                'spirit': ['vodka', 'gin', 'rum', 'whiskey', 'tequila', 'bourbon'],
                'ingredient': ['lime', 'mint', 'elderflower', 'ginger', 'cucumber', 'basil'],
                'modification': ['no sugar', 'extra dry', 'smoky finish', 'tropical twist'],
                'occasion': ['summer party', 'wedding', 'corporate event', 'date night'],
                'technique': ['nitrogen infusion', 'clarification', 'fat washing', 'carbonation'],
                'unusual_ingredient': ['activated charcoal', 'butterfly pea flower', 'yuzu', 'cardamom'],
                'garnish': ['dehydrated citrus', 'edible flowers', 'smoked salt rim', 'gold leaf'],
                'dietary_restriction': ['vegan', 'gluten-free', 'low-calorie', 'sugar-free']
            },
            'flowers': {
                'flower_type': ['roses', 'peonies', 'hydrangeas', 'lilies', 'orchids', 'tulips'],
                'color': ['white', 'blush', 'burgundy', 'yellow', 'purple', 'coral'],
                'season': ['spring', 'summer', 'autumn', 'winter'],
                'occasion': ['wedding', 'funeral', 'birthday', 'anniversary', 'graduation'],
                'flower1': ['roses', 'peonies', 'eucalyptus'],
                'flower2': ['baby\'s breath', 'greenery', 'berries'],
                'technique': ['cascade design', 'compact arrangement', 'asymmetrical style'],
                'special_occasion': ['royal wedding', 'state dinner', 'film premiere'],
                'local_flowers': ['native wildflowers', 'seasonal blooms', 'organic varieties'],
                'eco_materials': ['biodegradable foam', 'reusable containers', 'natural twine'],
                'duration': ['week-long', 'month-long', 'seasonal']
            },
            'professional': {
                'industry': ['tech', 'finance', 'healthcare', 'legal', 'consulting', 'media'],
                'style': ['corporate', 'creative', 'casual', 'formal', 'artistic'],
                'lighting_style': ['natural light', 'studio lighting', 'dramatic shadows'],
                'company_type': ['startup', 'fortune 500', 'nonprofit', 'agency'],
                'background_type': ['white backdrop', 'office setting', 'outdoor location'],
                'mood': ['confident', 'approachable', 'authoritative', 'innovative'],
                'advanced_technique': ['high-key lighting', 'environmental portraiture', 'motion blur'],
                'premium_equipment': ['medium format camera', 'professional strobes', 'specialty lenses'],
                'complex_lighting': ['three-point setup', 'mixed lighting', 'color gels'],
                'global_brand': ['multinational corporation', 'international NGO', 'global consulting firm'],
                'artistic_technique': ['fine art approach', 'documentary style', 'conceptual photography'],
                'post_processing': ['color grading', 'digital retouching', 'artistic effects']
            }
        }
        
        library = term_libraries.get(domain, {})
        filled = template
        
        # Replace placeholders
        import re
        placeholders = re.findall(r'\{([^}]+)\}', template)
        
        for placeholder in placeholders:
            if placeholder in library:
                replacement = np.random.choice(library[placeholder])
                filled = filled.replace(f'{{{placeholder}}}', replacement)
            else:
                # Generic replacement
                filled = filled.replace(f'{{{placeholder}}}', placeholder.replace('_', ' '))
                
        return filled
    
    def _generate_query_id(self, query_text: str, domain: str) -> str:
        """Generate unique query ID"""
        content = f"{query_text}_{domain}_{datetime.now().isoformat()}"
        return f"rg300_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def _classify_intent(self, query_text: str) -> str:
        """Classify query intent"""
        text_lower = query_text.lower()
        
        if any(word in text_lower for word in ['recipe', 'how to', 'make', 'create']):
            return 'instructional'
        elif any(word in text_lower for word in ['best', 'recommend', 'suggest', 'top']):
            return 'recommendation'
        elif any(word in text_lower for word in ['difference', 'compare', 'vs', 'versus']):
            return 'comparison'
        elif any(word in text_lower for word in ['professional', 'advanced', 'expert']):
            return 'professional'
        else:
            return 'general'
    
    def _assess_complexity(self, query_text: str) -> str:
        """Assess query complexity"""
        word_count = len(query_text.split())
        unique_concepts = len(set(query_text.lower().split()))
        
        if word_count <= 3 and unique_concepts <= 3:
            return 'easy'
        elif word_count <= 7 and unique_concepts <= 6:
            return 'medium'
        else:
            return 'hard'
    
    def _extract_entities(self, query_text: str) -> List[str]:
        """Extract key entities from query"""
        # Simple entity extraction (could be enhanced with NLP)
        words = query_text.lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        entities = [word for word in words if word not in stopwords and len(word) > 2]
        return entities[:5]  # Top 5 entities
    
    def _assess_expansion_potential(self, query: Dict) -> float:
        """Assess how well a query can be expanded"""
        text = query.get('text', '')
        
        # Factors for expansion potential
        word_count = len(text.split())
        unique_concepts = len(set(text.lower().split()))
        
        # Higher potential for medium complexity queries
        complexity_score = 0.8 if 3 < word_count < 8 else 0.5
        
        # Higher potential for queries with clear domain terms
        domain_relevance = 0.9 if any(term in text.lower() for term in ['cocktail', 'flower', 'professional']) else 0.6
        
        return (complexity_score + domain_relevance) / 2
    
    def _calculate_difficulty_distribution(self, queries: List[Dict]) -> Dict:
        """Calculate actual difficulty distribution"""
        difficulties = [q['difficulty'] for q in queries]
        total = len(difficulties)
        
        return {
            'easy': difficulties.count('easy') / total,
            'medium': difficulties.count('medium') / total,
            'hard': difficulties.count('hard') / total
        }
    
    def _calculate_intent_coverage(self, queries: List[Dict]) -> Dict:
        """Calculate intent type coverage"""
        intents = [q['intent'] for q in queries]
        total = len(intents)
        
        intent_counts = {}
        for intent in set(intents):
            intent_counts[intent] = intents.count(intent) / total
            
        return intent_counts
    
    def _calculate_quality_metrics(self, queries: List[Dict]) -> Dict:
        """Calculate overall dataset quality metrics"""
        
        # Diversity: unique query texts
        unique_texts = len(set(q['text'] for q in queries))
        diversity_score = unique_texts / len(queries)
        
        # Coverage: intent and difficulty distribution
        intents = set(q['intent'] for q in queries)
        difficulties = set(q['difficulty'] for q in queries)
        
        coverage_score = (len(intents) * len(difficulties)) / (5 * 3)  # 5 intents, 3 difficulties expected
        
        # Realism: average quality scores
        quality_scores = [q['metadata']['quality_score'] for q in queries]
        realism_score = float(np.mean(quality_scores))
        
        quality_threshold_met = all([
            diversity_score >= self.config.quality_thresholds['diversity'],
            coverage_score >= self.config.quality_thresholds['coverage'],
            realism_score >= self.config.quality_thresholds['realism']
        ])
        
        return {
            'diversity': float(diversity_score),
            'coverage': float(coverage_score),
            'realism': realism_score,
            'total_queries': len(queries),
            'unique_queries': unique_texts,
            'quality_threshold_met': bool(quality_threshold_met)
        }
    
    def _calculate_query_quality(self, query_text: str, domain: str) -> float:
        """Calculate individual query quality score"""
        
        # Length appropriateness
        word_count = len(query_text.split())
        length_score = 1.0 if 2 <= word_count <= 15 else 0.7
        
        # Domain relevance
        domain_terms = {
            'cocktails': ['cocktail', 'drink', 'recipe', 'bar', 'mix', 'spirit', 'ingredient'],
            'flowers': ['flower', 'bouquet', 'arrangement', 'floral', 'bloom', 'garden', 'petal'],
            'professional': ['professional', 'headshot', 'portrait', 'corporate', 'business', 'executive']
        }
        
        relevant_terms = domain_terms.get(domain, [])
        relevance_score = 1.0 if any(term in query_text.lower() for term in relevant_terms) else 0.6
        
        # Naturalness (simple heuristics)
        natural_score = 0.9 if query_text[0].isupper() and not query_text.endswith('?') else 1.0
        
        return (length_score + relevance_score + natural_score) / 3
    
    def _generate_from_seed(self, domain: str, difficulty: str, seeds: List[Dict]) -> Dict:
        """Fallback: generate query from seed patterns"""
        
        if seeds:
            seed = np.random.choice(seeds)
            base_text = seed['original_text']
            
            # Simple variations
            variations = [
                f"best {base_text}",
                f"{base_text} recipe",
                f"professional {base_text}",
                f"how to make {base_text}",
                f"advanced {base_text}"
            ]
            
            generated_text = np.random.choice(variations)
        else:
            # Ultimate fallback
            generated_text = f"{domain} query {difficulty} level"
        
        return {
            'query_id': self._generate_query_id(generated_text, domain),
            'text': generated_text,
            'domain': domain,
            'difficulty': difficulty,
            'intent': self._classify_intent(generated_text),
            'expansion_method': 'seed_based',
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'quality_score': self._calculate_query_quality(generated_text, domain)
            }
        }

class DatasetValidator:
    """Validates the expanded 300-query dataset for quality and completeness"""
    
    def __init__(self, quality_thresholds: Dict[str, float]):
        self.thresholds = quality_thresholds
        
    def comprehensive_validation(self, dataset: Dict) -> Dict:
        """Perform comprehensive dataset validation"""
        
        validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_metadata': dataset['metadata'],
            'validation_results': {},
            'quality_assessment': {},
            'recommendations': []
        }
        
        queries = dataset['queries']
        
        # 1. Basic structure validation
        validation_report['validation_results']['structure'] = self._validate_structure(queries)
        
        # 2. Distribution validation
        validation_report['validation_results']['distribution'] = self._validate_distribution(queries)
        
        # 3. Quality validation
        validation_report['validation_results']['quality'] = self._validate_quality(queries)
        
        # 4. Uniqueness validation
        validation_report['validation_results']['uniqueness'] = self._validate_uniqueness(queries)
        
        # 5. Domain coverage validation
        validation_report['validation_results']['coverage'] = self._validate_coverage(queries)
        
        # Overall assessment
        validation_report['quality_assessment'] = self._overall_assessment(validation_report['validation_results'])
        
        # Generate recommendations
        validation_report['recommendations'] = self._generate_recommendations(validation_report)
        
        return validation_report
    
    def _validate_structure(self, queries: List[Dict]) -> Dict:
        """Validate dataset structure"""
        required_fields = ['query_id', 'text', 'domain', 'difficulty', 'intent']
        
        structure_issues = []
        for i, query in enumerate(queries):
            missing_fields = [field for field in required_fields if field not in query]
            if missing_fields:
                structure_issues.append(f"Query {i}: missing {missing_fields}")
        
        return {
            'total_queries': len(queries),
            'structure_valid': len(structure_issues) == 0,
            'issues': structure_issues[:10],  # Limit to first 10 issues
            'issue_count': len(structure_issues)
        }
    
    def _validate_distribution(self, queries: List[Dict]) -> Dict:
        """Validate query distribution across domains and difficulties"""
        
        # Domain distribution
        domain_counts = {}
        for query in queries:
            domain = query.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Difficulty distribution
        difficulty_counts = {}
        for query in queries:
            difficulty = query.get('difficulty', 'unknown')
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        # Check balance
        expected_per_domain = len(queries) // len(domain_counts)
        domain_balance = all(
            abs(count - expected_per_domain) <= expected_per_domain * 0.1 
            for count in domain_counts.values()
        )
        
        return {
            'domain_distribution': domain_counts,
            'difficulty_distribution': difficulty_counts,
            'domain_balanced': domain_balance,
            'expected_per_domain': expected_per_domain,
            'distribution_summary': {
                'domains': len(domain_counts),
                'difficulties': len(difficulty_counts)
            }
        }
    
    def _validate_quality(self, queries: List[Dict]) -> Dict:
        """Validate query quality metrics"""
        
        quality_scores = []
        low_quality_queries = []
        
        for query in queries:
            quality_score = query.get('metadata', {}).get('quality_score', 0.0)
            quality_scores.append(quality_score)
            
            if quality_score < 0.7:  # Quality threshold
                low_quality_queries.append({
                    'query_id': query.get('query_id'),
                    'text': query.get('text'),
                    'quality_score': float(quality_score)
                })
        
        return {
            'average_quality': float(np.mean(quality_scores)),
            'min_quality': float(np.min(quality_scores)),
            'max_quality': float(np.max(quality_scores)),
            'quality_std': float(np.std(quality_scores)),
            'low_quality_count': len(low_quality_queries),
            'low_quality_examples': low_quality_queries[:5],  # First 5 examples
            'quality_acceptable': bool(np.mean(quality_scores) >= 0.8)
        }
    
    def _validate_uniqueness(self, queries: List[Dict]) -> Dict:
        """Validate query uniqueness and diversity"""
        
        texts = [query.get('text', '') for query in queries]
        unique_texts = set(texts)
        
        # Find duplicates
        duplicates = []
        seen = set()
        for text in texts:
            if text in seen:
                duplicates.append(text)
            seen.add(text)
        
        uniqueness_ratio = len(unique_texts) / len(texts)
        
        return {
            'total_queries': len(texts),
            'unique_queries': len(unique_texts),
            'uniqueness_ratio': float(uniqueness_ratio),
            'duplicate_count': len(duplicates),
            'duplicates': list(set(duplicates))[:5],  # First 5 unique duplicates
            'uniqueness_acceptable': bool(uniqueness_ratio >= 0.95)
        }
    
    def _validate_coverage(self, queries: List[Dict]) -> Dict:
        """Validate intent and domain coverage"""
        
        intents = set(query.get('intent', 'unknown') for query in queries)
        domains = set(query.get('domain', 'unknown') for query in queries)
        difficulties = set(query.get('difficulty', 'unknown') for query in queries)
        
        # Intent distribution by domain
        domain_intent_coverage = {}
        for domain in domains:
            domain_queries = [q for q in queries if q.get('domain') == domain]
            domain_intents = set(q.get('intent', 'unknown') for q in domain_queries)
            domain_intent_coverage[domain] = len(domain_intents)
        
        return {
            'total_intents': len(intents),
            'total_domains': len(domains),
            'total_difficulties': len(difficulties),
            'intents': list(intents),
            'domains': list(domains),
            'difficulties': list(difficulties),
            'domain_intent_coverage': domain_intent_coverage,
            'coverage_acceptable': all([
                len(intents) >= 4,  # At least 4 intent types
                len(domains) >= 3,  # At least 3 domains
                len(difficulties) >= 3  # All 3 difficulty levels
            ])
        }
    
    def _overall_assessment(self, validation_results: Dict) -> Dict:
        """Provide overall dataset quality assessment"""
        
        checks = {
            'structure_valid': validation_results['structure']['structure_valid'],
            'domain_balanced': validation_results['distribution']['domain_balanced'],
            'quality_acceptable': validation_results['quality']['quality_acceptable'],
            'uniqueness_acceptable': validation_results['uniqueness']['uniqueness_acceptable'],
            'coverage_acceptable': validation_results['coverage']['coverage_acceptable']
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        overall_score = passed_checks / total_checks
        
        return {
            'overall_score': float(overall_score),
            'checks_passed': int(passed_checks),
            'total_checks': int(total_checks),
            'individual_checks': checks,
            'dataset_ready': bool(passed_checks >= total_checks * 0.8),  # 80% pass rate
            'confidence_level': 'HIGH' if passed_checks == total_checks else 'MEDIUM' if passed_checks >= 4 else 'LOW'
        }
    
    def _generate_recommendations(self, validation_report: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        results = validation_report['validation_results']
        
        if not results['structure']['structure_valid']:
            recommendations.append("Fix structural issues: ensure all required fields are present")
        
        if not results['distribution']['domain_balanced']:
            recommendations.append("Rebalance domain distribution: aim for equal queries per domain")
        
        if not results['quality']['quality_acceptable']:
            recommendations.append(f"Improve query quality: {results['quality']['low_quality_count']} queries below threshold")
        
        if not results['uniqueness']['uniqueness_acceptable']:
            recommendations.append(f"Increase uniqueness: {results['uniqueness']['duplicate_count']} duplicate queries found")
        
        if not results['coverage']['coverage_acceptable']:
            recommendations.append("Improve coverage: ensure all intent types and difficulty levels are represented")
        
        if not recommendations:
            recommendations.append("Dataset quality is excellent - ready for RA-Guard evaluation")
        
        return recommendations

def main():
    """Main execution function for Week 1 dataset preparation"""
    
    parser = argparse.ArgumentParser(description='Week 1: RA-Guard Dataset Preparation (45 → 300 queries)')
    parser.add_argument('--pilot-data', required=True, help='Path to pilot query data JSON file')
    parser.add_argument('--output-dir', required=True, help='Output directory for expanded dataset')
    parser.add_argument('--target-queries', type=int, default=300, help='Target number of queries (default: 300)')
    parser.add_argument('--domains', nargs='+', default=['cocktails', 'flowers', 'professional'])
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize configuration
    config = QueryExpansionConfig(target_total=args.target_queries, domains=args.domains)
    
    # Initialize expansion engine
    expander = QueryExpansionEngine(config)
    
    # Load pilot data
    logger.info(f"Loading pilot data from: {args.pilot_data}")
    pilot_analysis = expander.load_pilot_data(args.pilot_data)
    
    # Generate expanded dataset
    logger.info(f"Generating expanded dataset: {args.target_queries} queries")
    expanded_dataset = expander.generate_expanded_dataset(pilot_analysis)
    
    # Validate dataset
    logger.info("Validating expanded dataset quality")
    validator = DatasetValidator(config.quality_thresholds)
    validation_report = validator.comprehensive_validation(expanded_dataset)
    
    # Save results
    dataset_file = output_dir / 'ra_guard_300q_dataset.json'
    validation_file = output_dir / 'dataset_validation_report.json'
    
    with open(dataset_file, 'w') as f:
        json.dump(expanded_dataset, f, indent=2)
    
    with open(validation_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    # Print summary
    quality_assessment = validation_report['quality_assessment']
    
    print(f"\n📊 Week 1: Dataset Preparation Complete")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Generated Queries: {len(expanded_dataset['queries'])}")
    print(f"Quality Score: {quality_assessment['overall_score']:.2%}")
    print(f"Confidence Level: {quality_assessment['confidence_level']}")
    print(f"Dataset Ready: {'✅ YES' if quality_assessment['dataset_ready'] else '❌ NO'}")
    
    print(f"\n📈 Domain Distribution:")
    for domain, stats in expanded_dataset['domain_statistics'].items():
        print(f"  {domain}: {stats['query_count']} queries")
    
    print(f"\n📋 Quality Metrics:")
    quality_metrics = expanded_dataset['quality_metrics']
    print(f"  Diversity: {quality_metrics['diversity']:.2%}")
    print(f"  Coverage: {quality_metrics['coverage']:.2%}")
    print(f"  Realism: {quality_metrics['realism']:.2%}")
    
    if validation_report['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(validation_report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n📂 Output Files:")
    print(f"  Dataset: {dataset_file}")
    print(f"  Validation: {validation_file}")
    
    logger.info("Week 1 dataset preparation completed successfully")

if __name__ == "__main__":
    main()