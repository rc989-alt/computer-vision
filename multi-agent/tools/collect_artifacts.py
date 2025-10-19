#!/usr/bin/env python3
"""
Artifact Collection
Collects and organizes outputs from agent meetings
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class ArtifactCollector:
    """Collects and saves meeting artifacts"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def collect_transcript(self, messages: List[Dict], responses: Dict[str, str]) -> str:
        """Create meeting transcript"""
        transcript = [
            "# Multi-Agent Meeting Transcript",
            f"Session ID: {self.session_id}",
            f"Timestamp: {datetime.now().isoformat()}",
            "\n---\n"
        ]

        for msg in messages:
            transcript.append(f"\n## Message from {msg.get('sender', 'Unknown')}")
            transcript.append(f"{msg.get('content', '')}\n")

        transcript.append("\n## Agent Responses\n")
        for agent, response in responses.items():
            transcript.append(f"\n### {agent}")
            transcript.append(f"{response}\n")

        return "\n".join(transcript)

    def save_transcript(self, content: str, filename: str = None):
        """Save transcript to file"""
        if filename is None:
            filename = f"transcript_{self.session_id}.md"

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)

        return filepath

    def save_json_artifact(self, data: Dict[str, Any], artifact_name: str):
        """Save JSON artifact"""
        filename = f"{artifact_name}_{self.session_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def create_summary_report(self,
                            responses: Dict[str, str],
                            actions: List[Any],
                            integrity_check: Dict) -> str:
        """Create summary report"""
        report = [
            "# Meeting Summary Report",
            f"Session: {self.session_id}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---\n"
        ]

        # Executive Summary
        report.append("\n## Executive Summary\n")
        report.append(f"- Agents participated: {len(responses)}")
        report.append(f"- Action items identified: {len(actions)}")
        report.append(f"- Integrity check: {'✅ PASSED' if integrity_check.get('passed') else '⚠️ ISSUES FOUND'}")

        # Agent Responses Summary
        report.append("\n## Agent Positions\n")
        for agent, response in responses.items():
            # Get first 200 chars as summary
            summary = response[:200] + "..." if len(response) > 200 else response
            report.append(f"\n### {agent}")
            report.append(summary)

        # Action Items
        if actions:
            report.append("\n## Action Items\n")
            for action in actions:
                report.append(f"- [{action.priority.upper()}] {action.description} ({action.agent})")

        # Integrity Check
        report.append("\n## Integrity Check\n")
        if integrity_check.get('issues'):
            report.append("\n⚠️ Issues found:")
            for issue in integrity_check['issues']:
                report.append(f"- {issue.get('type', 'unknown')}: {len(issue.get('items', []))} items")

        metrics = integrity_check.get('metrics', {})
        if metrics:
            report.append("\n### Metrics")
            report.append(f"- Consensus score: {metrics.get('consensus_score', 0):.2f}")
            report.append(f"- Average response length: {metrics.get('avg_response_length', 0):.0f} chars")

        return "\n".join(report)

    def save_all_artifacts(self,
                          transcript: str,
                          responses: Dict[str, str],
                          actions: List,
                          integrity_check: Dict):
        """Save all meeting artifacts"""
        artifacts = {}

        # Save transcript
        artifacts['transcript'] = self.save_transcript(transcript)

        # Save raw responses
        artifacts['responses'] = self.save_json_artifact(responses, 'responses')

        # Save actions
        action_data = [
            {
                'type': a.action_type,
                'description': a.description,
                'agent': a.agent,
                'priority': a.priority
            } for a in actions
        ]
        artifacts['actions'] = self.save_json_artifact(action_data, 'actions')

        # Save integrity report
        artifacts['integrity'] = self.save_json_artifact(integrity_check, 'integrity')

        # Save summary
        summary = self.create_summary_report(responses, actions, integrity_check)
        artifacts['summary'] = self.save_transcript(summary, f'summary_{self.session_id}.md')

        print(f"\n✅ Artifacts saved to: {self.output_dir}")
        for name, path in artifacts.items():
            print(f"   - {name}: {path.name}")

        return artifacts
