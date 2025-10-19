#!/usr/bin/env python3
"""
Integrity Rules and Validation
Ensures consistency and quality in agent responses
"""

from typing import Dict, List, Any
import re


class IntegrityRules:
    """Validation rules for agent outputs"""

    @staticmethod
    def check_consistency(responses: Dict[str, str]) -> Dict[str, Any]:
        """Check for consistency across agent responses"""
        issues = []
        metrics = {}

        # Check for contradictions
        contradictions = IntegrityRules._find_contradictions(responses)
        if contradictions:
            issues.append({
                'type': 'contradiction',
                'items': contradictions
            })

        # Check for missing critical elements
        missing = IntegrityRules._check_completeness(responses)
        if missing:
            issues.append({
                'type': 'incomplete',
                'items': missing
            })

        # Calculate agreement metrics
        metrics['agent_count'] = len(responses)
        metrics['avg_response_length'] = sum(len(r) for r in responses.values()) / len(responses)
        metrics['consensus_score'] = IntegrityRules._calculate_consensus(responses)

        return {
            'issues': issues,
            'metrics': metrics,
            'passed': len(issues) == 0
        }

    @staticmethod
    def _find_contradictions(responses: Dict[str, str]) -> List[str]:
        """Identify contradictory statements"""
        contradictions = []

        # Look for opposing terms
        opposites = [
            ('better', 'worse'),
            ('increase', 'decrease'),
            ('succeed', 'fail'),
            ('recommend', 'not recommend'),
            ('should', 'should not')
        ]

        response_list = list(responses.items())
        for i, (agent1, resp1) in enumerate(response_list):
            for agent2, resp2 in response_list[i+1:]:
                for word1, word2 in opposites:
                    if word1 in resp1.lower() and word2 in resp2.lower():
                        contradictions.append(
                            f"{agent1} mentions '{word1}' while {agent2} mentions '{word2}'"
                        )

        return contradictions

    @staticmethod
    def _check_completeness(responses: Dict[str, str]) -> List[str]:
        """Check if responses cover critical elements"""
        missing = []
        critical_topics = ['performance', 'data', 'implementation', 'testing']

        for topic in critical_topics:
            covered = any(topic.lower() in resp.lower() for resp in responses.values())
            if not covered:
                missing.append(f"No agent addressed: {topic}")

        return missing

    @staticmethod
    def _calculate_consensus(responses: Dict[str, str]) -> float:
        """Calculate consensus score (0-1)"""
        if len(responses) < 2:
            return 1.0

        # Simple keyword overlap metric
        all_words = []
        for resp in responses.values():
            words = set(re.findall(r'\b\w+\b', resp.lower()))
            all_words.append(words)

        # Calculate pairwise similarity
        similarities = []
        for i, words1 in enumerate(all_words):
            for words2 in all_words[i+1:]:
                if words1 and words2:
                    overlap = len(words1 & words2)
                    union = len(words1 | words2)
                    similarities.append(overlap / union if union > 0 else 0)

        return sum(similarities) / len(similarities) if similarities else 0.0

    @staticmethod
    def validate_decision(decision: str, reasoning: str) -> Dict[str, Any]:
        """Validate a decision has proper reasoning"""
        validation = {
            'has_reasoning': len(reasoning) > 50,
            'has_evidence': 'data' in reasoning.lower() or 'result' in reasoning.lower(),
            'has_alternatives': 'alternatively' in reasoning.lower() or 'other option' in reasoning.lower(),
            'decision_clear': bool(re.search(r'\b(recommend|suggest|choose|select|decide)\b', decision.lower()))
        }

        validation['score'] = sum(validation.values()) / len(validation)
        validation['passed'] = validation['score'] >= 0.75

        return validation
