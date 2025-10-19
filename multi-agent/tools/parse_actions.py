#!/usr/bin/env python3
"""
Action Parser
Extracts actionable items from agent responses
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Action:
    """Represents an actionable item"""
    action_type: str  # 'code_change', 'experiment', 'analysis', 'decision'
    description: str
    agent: str
    priority: str = 'medium'  # 'high', 'medium', 'low'
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ActionParser:
    """Parse and extract actions from agent responses"""

    @staticmethod
    def parse_response(agent_name: str, response: str) -> List[Action]:
        """Extract actions from agent response"""
        actions = []

        # Look for action keywords
        action_patterns = {
            'code_change': r'(?:implement|code|write|modify|update|refactor)\s+([^.!?\n]+)',
            'experiment': r'(?:test|experiment|try|validate|measure)\s+([^.!?\n]+)',
            'analysis': r'(?:analyze|investigate|examine|study|review)\s+([^.!?\n]+)',
            'decision': r'(?:decide|choose|select|recommend)\s+([^.!?\n]+)'
        }

        for action_type, pattern in action_patterns.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                description = match.group(1).strip()
                if len(description) > 10:  # Filter out too short
                    actions.append(Action(
                        action_type=action_type,
                        description=description,
                        agent=agent_name,
                        priority=ActionParser._determine_priority(description)
                    ))

        return actions

    @staticmethod
    def _determine_priority(description: str) -> str:
        """Determine priority based on keywords"""
        high_priority_words = ['critical', 'urgent', 'immediately', 'must', 'required']
        low_priority_words = ['later', 'optional', 'consider', 'maybe', 'could']

        desc_lower = description.lower()

        if any(word in desc_lower for word in high_priority_words):
            return 'high'
        elif any(word in desc_lower for word in low_priority_words):
            return 'low'
        else:
            return 'medium'

    @staticmethod
    def parse_all_responses(responses: Dict[str, str]) -> List[Action]:
        """Parse actions from all agent responses"""
        all_actions = []
        for agent_name, response in responses.items():
            actions = ActionParser.parse_response(agent_name, response)
            all_actions.extend(actions)
        return all_actions

    @staticmethod
    def group_by_type(actions: List[Action]) -> Dict[str, List[Action]]:
        """Group actions by type"""
        grouped = {}
        for action in actions:
            if action.action_type not in grouped:
                grouped[action.action_type] = []
            grouped[action.action_type].append(action)
        return grouped

    @staticmethod
    def prioritize(actions: List[Action]) -> List[Action]:
        """Sort actions by priority"""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        return sorted(actions, key=lambda a: priority_order.get(a.priority, 1))

    @staticmethod
    def format_action_report(actions: List[Action]) -> str:
        """Format actions into readable report"""
        if not actions:
            return "No actionable items identified."

        report = ["# Action Items\n"]

        # Group by priority
        by_priority = {}
        for action in actions:
            if action.priority not in by_priority:
                by_priority[action.priority] = []
            by_priority[action.priority].append(action)

        # Format each priority level
        for priority in ['high', 'medium', 'low']:
            if priority in by_priority:
                report.append(f"\n## {priority.upper()} Priority\n")
                for action in by_priority[priority]:
                    report.append(f"- [{action.action_type}] {action.description} (from {action.agent})")

        return "\n".join(report)
