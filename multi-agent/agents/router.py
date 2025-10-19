#!/usr/bin/env python3
"""
Agent Router and Orchestration
Handles message routing and agent coordination
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class RoutingStrategy(Enum):
    """Different routing strategies for agent communication"""
    ROUND_ROBIN = "round_robin"
    BROADCAST = "broadcast"
    TARGETED = "targeted"
    HIERARCHICAL = "hierarchical"
    DEBATE = "debate"


@dataclass
class Message:
    """Message structure for agent communication"""
    sender: str
    recipients: List[str]
    content: str
    message_type: str = "standard"  # standard, question, proposal, critique
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentRouter:
    """Routes messages between agents based on strategy"""

    def __init__(self, agent_team, strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN):
        self.agent_team = agent_team
        self.strategy = strategy
        self.message_log = []

    def route_message(self, message: Message) -> Dict[str, str]:
        """Route message to appropriate agents"""
        responses = {}

        if self.strategy == RoutingStrategy.BROADCAST:
            responses = self._broadcast(message)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            responses = self._round_robin(message)
        elif self.strategy == RoutingStrategy.TARGETED:
            responses = self._targeted(message)
        elif self.strategy == RoutingStrategy.HIERARCHICAL:
            responses = self._hierarchical(message)
        elif self.strategy == RoutingStrategy.DEBATE:
            responses = self._debate(message)

        self.message_log.append({
            'message': message,
            'responses': responses
        })

        return responses

    def _broadcast(self, message: Message) -> Dict[str, str]:
        """Broadcast message to all agents"""
        responses = {}
        for agent_name in self.agent_team.list_agents():
            if agent_name != message.sender and agent_name != 'moderator':
                agent = self.agent_team.get_agent(agent_name)
                context = f"[From {message.sender}]: {message.content}"
                responses[agent_name] = agent.respond(context)
        return responses

    def _round_robin(self, message: Message) -> Dict[str, str]:
        """Route to agents in round-robin fashion"""
        responses = {}
        agent_names = [n for n in self.agent_team.list_agents()
                      if n != message.sender and n != 'moderator']

        for i, agent_name in enumerate(agent_names):
            agent = self.agent_team.get_agent(agent_name)
            context = f"[From {message.sender}]: {message.content}"

            # Add context from previous responses
            if i > 0:
                prev_responses = "\n\n".join([
                    f"[{name}]: {resp}"
                    for name, resp in list(responses.items())[-2:]
                ])
                context = f"{context}\n\nRecent discussion:\n{prev_responses}"

            responses[agent_name] = agent.respond(context)

        return responses

    def _targeted(self, message: Message) -> Dict[str, str]:
        """Route to specific recipients only"""
        responses = {}
        for recipient in message.recipients:
            if recipient != message.sender:
                agent = self.agent_team.get_agent(recipient)
                if agent:
                    context = f"[From {message.sender}]: {message.content}"
                    responses[recipient] = agent.respond(context)
        return responses

    def _hierarchical(self, message: Message) -> Dict[str, str]:
        """Route through hierarchy: Planning Team (4 agents) -> Executive Team (3 agents)"""
        responses = {}

        # Planning Team (strategic decision-making)
        planning_agents = ['empirical_validation_lead', 'critical_evaluator', 'gemini_advisor']
        for agent_name in planning_agents:
            agent = self.agent_team.get_agent(agent_name)
            if agent:
                responses[agent_name] = agent.respond(message.content)

        # Strategic Leader synthesizes (moderator role)
        strategic_leader = self.agent_team.get_agent('strategic_leader')
        if strategic_leader:
            planning_summary = "\n\n".join([f"[{k}]: {v}" for k, v in responses.items()])
            leader_context = f"Original question: {message.content}\n\nTeam responses:\n{planning_summary}"
            responses['strategic_leader'] = strategic_leader.respond(leader_context)

        # Executive Team (if needed for execution feedback)
        executive_agents = ['ops_commander', 'quality_safety_officer', 'infrastructure_performance_monitor']
        for agent_name in executive_agents:
            agent = self.agent_team.get_agent(agent_name)
            if agent:
                exec_context = f"Planning Team decision: {responses.get('strategic_leader', 'N/A')}\n\nOriginal question: {message.content}"
                responses[agent_name] = agent.respond(exec_context)

        return responses

    def _debate(self, message: Message) -> Dict[str, str]:
        """Facilitate debate between Planning Team members"""
        responses = {}

        # Get initial positions from Planning Team
        proponents = ['empirical_validation_lead', 'gemini_advisor']
        for agent_name in proponents:
            agent = self.agent_team.get_agent(agent_name)
            if agent:
                responses[agent_name] = agent.respond(message.content)

        # Critical Evaluator challenges each position
        critical_evaluator = self.agent_team.get_agent('critical_evaluator')
        if critical_evaluator:
            for agent_name, response in list(responses.items()):
                critique_context = f"Challenge this position:\n[{agent_name}]: {response}"
                responses[f'critical_evaluator_on_{agent_name}'] = critical_evaluator.respond(critique_context)

        # Strategic Leader synthesizes
        strategic_leader = self.agent_team.get_agent('strategic_leader')
        if strategic_leader:
            debate_summary = "\n\n".join([f"[{k}]: {v}" for k, v in responses.items()])
            synthesis_context = f"Synthesize this debate:\n{debate_summary}"
            responses['strategic_leader_synthesis'] = strategic_leader.respond(synthesis_context)

        return responses

    def get_conversation_summary(self) -> str:
        """Get summary of all messages"""
        summary = []
        for entry in self.message_log:
            msg = entry['message']
            summary.append(f"\n[{msg.sender}] -> {msg.recipients}: {msg.content}")
            for agent, response in entry['responses'].items():
                summary.append(f"  [{agent}]: {response[:100]}...")
        return "\n".join(summary)

    def clear_log(self):
        """Clear message log"""
        self.message_log = []
