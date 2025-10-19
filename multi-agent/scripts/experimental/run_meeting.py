#!/usr/bin/env python3
"""
Multi-Agent Meeting Orchestrator
Main script to run multi-agent deliberation sessions
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Load environment variables from api_keys.env
env_file = Path(__file__).parent.parent / 'research' / 'api_keys.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.roles import Agent, AgentConfig, AgentTeam
from agents.router import AgentRouter, RoutingStrategy, Message
from tools.gemini_search import GeminiSearch
from tools.integrity_rules import IntegrityRules
from tools.parse_actions import ActionParser
from tools.collect_artifacts import ArtifactCollector
from tools.io_utils import load_yaml, ProjectContext
from tools.file_bridge import FileBridge, create_default_policies
from tools.progress_sync_hook import ProgressSyncHook, create_agent_context_with_progress


class MeetingOrchestrator:
    """Orchestrates multi-agent meetings"""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = load_yaml(config_path)
        # Fix: Use Path(__file__).parent.parent to get actual project root
        self.project_root = Path(__file__).parent.parent

        # Initialize file access system
        self.file_bridge = FileBridge(
            self.project_root,
            create_default_policies(self.project_root)
        )
        self.progress_hook = ProgressSyncHook(self.project_root)

        # Initialize components
        self.agents = self._initialize_agents()
        self.agent_team = AgentTeam(self.agents)
        self.router = AgentRouter(self.agent_team)
        self.search = GeminiSearch()
        self.integrity = IntegrityRules()
        self.action_parser = ActionParser()
        self.artifact_collector = ArtifactCollector(
            self.project_root / 'multi-agent' / 'reports'
        )

        # Load project context
        self.context = ProjectContext(self.project_root / 'multi-agent' / 'data')
        self.context.load_all()

    def _initialize_agents(self) -> dict:
        """Initialize all agents from config"""
        agents = {}
        prompt_dir = self.project_root / 'multi-agent' / 'agents' / 'prompts'

        for agent_id, agent_config in self.config['agents'].items():
            config = AgentConfig(
                name=agent_config['name'],
                model=agent_config['model'],
                provider=agent_config['provider'],
                role=agent_config['role'],
                prompt_file=agent_config['prompt_file']
            )
            agents[agent_id] = Agent(config, prompt_dir)

        return agents

    def run_meeting(self, topic: str, strategy: str = "hierarchical", rounds: int = 3):
        """Run a multi-agent meeting on a topic"""
        print(f"\n{'='*60}")
        print(f"🎯 Multi-Agent Meeting: {topic}")
        print(f"{'='*60}\n")

        # Run Progress Sync Hook at meeting start
        progress_context = create_agent_context_with_progress(self.project_root)

        # Set routing strategy
        strategy_map = {
            'broadcast': RoutingStrategy.BROADCAST,
            'round_robin': RoutingStrategy.ROUND_ROBIN,
            'hierarchical': RoutingStrategy.HIERARCHICAL,
            'debate': RoutingStrategy.DEBATE
        }
        self.router.strategy = strategy_map.get(strategy, RoutingStrategy.HIERARCHICAL)

        print(f"📋 Strategy: {strategy}")
        print(f"🔄 Rounds: {rounds}")
        print(f"👥 Agents: {', '.join(self.agents.keys())}\n")

        all_responses = {}
        messages = []

        # Run meeting rounds
        for round_num in range(1, rounds + 1):
            print(f"\n--- Round {round_num}/{rounds} ---\n")

            # Add context from project data + progress update
            context_summary = self.context.summary()
            round_topic = f"{topic}\n\nProject Context:\n{context_summary}\n\n{progress_context}"

            # Create message
            message = Message(
                sender="moderator",
                recipients=list(self.agents.keys()),
                content=round_topic,
                message_type="question"
            )
            messages.append(message.__dict__)

            # Get responses
            responses = self.router.route_message(message)
            all_responses.update(responses)

            # Show responses
            for agent_name, response in responses.items():
                print(f"\n[{agent_name}]:")
                print(f"{response[:300]}..." if len(response) > 300 else response)

            # If moderator exists, get synthesis
            if 'moderator' in self.agents and round_num < rounds:
                moderator = self.agents['moderator']
                synthesis_context = self._create_synthesis_context(responses)
                synthesis = moderator.respond(synthesis_context)
                print(f"\n[Moderator Synthesis]:")
                print(synthesis[:300] + "..." if len(synthesis) > 300 else synthesis)

                # Use synthesis for next round
                topic = f"Building on previous round:\n{synthesis}\n\nContinue analysis..."

        # Final analysis
        print(f"\n{'='*60}")
        print("📊 Final Analysis")
        print(f"{'='*60}\n")

        # Parse actions
        actions = self.action_parser.parse_all_responses(all_responses)
        print(f"\n✅ Actions identified: {len(actions)}")
        if actions:
            action_report = self.action_parser.format_action_report(actions)
            print(action_report)

        # Integrity check
        integrity_result = self.integrity.check_consistency(all_responses)
        print(f"\n🔍 Integrity Check: {'✅ PASSED' if integrity_result['passed'] else '⚠️ ISSUES'}")
        if not integrity_result['passed']:
            for issue in integrity_result['issues']:
                print(f"   - {issue['type']}: {len(issue['items'])} items")

        print(f"\n📈 Consensus Score: {integrity_result['metrics']['consensus_score']:.2f}")

        # Collect artifacts
        print(f"\n{'='*60}")
        print("💾 Saving Artifacts")
        print(f"{'='*60}\n")

        transcript = self.artifact_collector.collect_transcript(messages, all_responses)
        artifacts = self.artifact_collector.save_all_artifacts(
            transcript=transcript,
            responses=all_responses,
            actions=actions,
            integrity_check=integrity_result
        )

        return {
            'responses': all_responses,
            'actions': actions,
            'integrity': integrity_result,
            'artifacts': artifacts
        }

    def _create_synthesis_context(self, responses: dict) -> str:
        """Create context for moderator synthesis"""
        context = ["Please synthesize the following perspectives:\n"]
        for agent, response in responses.items():
            context.append(f"\n{agent}: {response}")
        context.append("\n\nProvide a balanced synthesis highlighting key agreements and disagreements.")
        return "\n".join(context)

    def quick_query(self, question: str) -> dict:
        """Quick single-round query to all agents"""
        message = Message(
            sender="user",
            recipients=list(self.agents.keys()),
            content=question
        )
        return self.router.route_message(message)


def main():
    """Main entry point"""
    # Configuration
    config_path = Path(__file__).parent / 'configs' / 'meeting.yaml'

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    # Initialize orchestrator
    print("🚀 Initializing Multi-Agent System...")
    orchestrator = MeetingOrchestrator(config_path)

    # Example meeting topics
    topics = [
        "Should we deploy V1.0 optimizations or pursue V2.0 research?",
        "What are the top 3 priorities for improving the computer vision pipeline?",
        "How should we balance performance vs. interpretability in our models?"
    ]

    # Run meeting (you can change the topic)
    topic = topics[0]  # Default topic
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])

    result = orchestrator.run_meeting(
        topic=topic,
        strategy="hierarchical",
        rounds=2
    )

    print(f"\n{'='*60}")
    print("✨ Meeting Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
