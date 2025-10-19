#!/usr/bin/env python3
"""
Agent Roles and Configuration
Defines agent classes and their interaction patterns
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for an individual agent"""
    name: str
    model: str
    provider: str  # 'openai', 'anthropic', 'google'
    role: str
    prompt_file: str
    temperature: float = 0.7
    max_tokens: int = 2000


class Agent:
    """Base Agent class for multi-agent system"""

    def __init__(self, config: AgentConfig, prompt_dir: Path):
        self.config = config
        self.prompt_dir = prompt_dir
        self.system_prompt = self._load_prompt()
        self.conversation_history = []

    def _load_prompt(self) -> str:
        """Load agent's system prompt from file"""
        prompt_path = self.prompt_dir / self.config.prompt_file
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                return f.read()
        return f"You are {self.config.name}, a {self.config.role}."

    def get_client(self):
        """Get the appropriate API client based on provider"""
        if self.config.provider == "openai":
            from openai import OpenAI
            return OpenAI()
        elif self.config.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic()
        elif self.config.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            return genai
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def respond(self, context: str, tools: Optional[List[Dict]] = None) -> str:
        """Generate agent response based on context"""
        client = self.get_client()

        if self.config.provider == "openai":
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history,
                {"role": "user", "content": context}
            ]
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            reply = response.choices[0].message.content

        elif self.config.provider == "anthropic":
            messages = self.conversation_history + [
                {"role": "user", "content": context}
            ]
            response = client.messages.create(
                model=self.config.model,
                system=self.system_prompt,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            reply = response.content[0].text

        elif self.config.provider == "google":
            model = client.GenerativeModel(self.config.model)
            full_prompt = f"{self.system_prompt}\n\n{context}"
            response = model.generate_content(full_prompt)
            reply = response.text

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": context})
        self.conversation_history.append({"role": "assistant", "content": reply})

        return reply

    def reset_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class AgentTeam:
    """Manages a team of agents"""

    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.moderator = agents.get('moderator')

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get agent by name"""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all agent names"""
        return list(self.agents.keys())

    def broadcast(self, message: str) -> Dict[str, str]:
        """Send message to all agents and collect responses"""
        responses = {}
        for name, agent in self.agents.items():
            if name != 'moderator':  # Moderator doesn't respond to broadcasts
                responses[name] = agent.respond(message)
        return responses

    def reset_all(self):
        """Reset all agent histories"""
        for agent in self.agents.values():
            agent.reset_history()
