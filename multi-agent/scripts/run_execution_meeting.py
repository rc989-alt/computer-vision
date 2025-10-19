#!/usr/bin/env python3
"""
Execution Team Meeting Runner
Mirrors planning team meeting structure with:
- Full transcript of execution discussion
- Summary with decisions and results
- Action items (prioritized)
- Tool usage log
- Deployment status

Just like planning meetings, but for execution team
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from run_meeting import MeetingOrchestrator
from tools.file_bridge import FileBridge, create_default_policies
from tools.enhanced_progress_sync import EnhancedProgressSync


class ExecutionMeetingOrchestrator(MeetingOrchestrator):
    """
    Execution team meeting orchestrator
    Creates same trajectory as planning meetings:
    - transcript_*.md
    - summary_*.md
    - actions_*.json
    - responses_*.json
    - integrity_*.json
    """

    def __init__(self, config_path: Path, project_root: Path):
        super().__init__(config_path, project_root)

        # Override to use execution team config
        self.team_type = "execution"

        # Enhanced progress sync
        self.progress_sync = EnhancedProgressSync(project_root)

        # Execution-specific report directory
        self.execution_reports_dir = project_root / "multi-agent/reports/execution"
        self.execution_reports_dir.mkdir(parents=True, exist_ok=True)

    def prepare_execution_context(self) -> str:
        """
        Prepare context for execution team meeting
        Similar to planning team but focused on:
        - Latest planning decisions
        - Pending actions from handoff
        - Deployment status
        - System metrics
        - Tool usage history
        """
        print("\n🔄 Preparing execution team context...")

        # Get comprehensive execution context
        sync_result = self.progress_sync.sync_for_execution_team()

        # Build execution meeting context
        context_parts = [
            "# EXECUTION TEAM STRATEGIC MEETING",
            "",
            "## Meeting Purpose",
            "Review deployment progress, discuss execution challenges,",
            "coordinate next deployment actions, and report results to planning team.",
            "",
            "---",
            "",
            "## Current Deployment Status",
            ""
        ]

        # Deployment status
        deployment = sync_result.get('deployment_status', {})
        context_parts.extend([
            f"**Version:** {deployment.get('version', 'N/A')}",
            f"**Stage:** {deployment.get('current_stage', 'N/A')}",
            f"**Last Updated:** {deployment.get('last_updated', 'N/A')}",
            "",
            "**SLO Status:**"
        ])

        for metric, status in deployment.get('slo_status', {}).items():
            emoji = "✅" if status else "❌"
            context_parts.append(f"- {emoji} {metric}: {status}")

        # System metrics
        metrics = sync_result.get('system_metrics', {})
        context_parts.extend([
            "",
            "## Current System Metrics",
            ""
        ])

        if metrics.get('compliance_current'):
            context_parts.append(f"- **Compliance:** {metrics['compliance_current']:.4f}")
        if metrics.get('ndcg_current'):
            context_parts.append(f"- **nDCG:** {metrics['ndcg_current']:.4f}")
        if metrics.get('latency_p95_ms'):
            context_parts.append(f"- **Latency P95:** {metrics['latency_p95_ms']:.2f} ms")
        if metrics.get('error_rate'):
            context_parts.append(f"- **Error Rate:** {metrics['error_rate']:.4f}")

        # Pending actions from planning team
        pending = sync_result.get('pending_actions', [])
        context_parts.extend([
            "",
            f"## Pending Actions from Planning Team ({len(pending)} items)",
            ""
        ])

        for i, action in enumerate(pending[:10], 1):
            priority = action.get('priority', 'medium').upper()
            owner = action.get('owner', 'N/A')
            description = action.get('action', action.get('description', 'N/A'))
            context_parts.append(f"{i}. **[{priority}]** {description} (Owner: {owner})")

        # Recent planning decisions
        decisions = sync_result.get('planning_decisions', [])
        if decisions:
            context_parts.extend([
                "",
                "## Recent Planning Decisions",
                ""
            ])

            for decision in decisions[:3]:
                context_parts.append(f"- **{decision['file']}** (Modified: {decision['modified']})")

        # Execution progress
        context_parts.extend([
            "",
            "## Execution Progress Since Last Meeting",
            "",
            sync_result.get('execution_progress', 'No progress data available'),
            "",
            "---",
            "",
            "## Meeting Agenda",
            "",
            "### For Each Agent:",
            "",
            "1. **Status Update** - What have you accomplished since last meeting?",
            "2. **Current Challenges** - Any blockers or issues?",
            "3. **Next Actions** - What will you work on next?",
            "4. **Metrics** - Any performance data to report?",
            "5. **Handoff Needs** - Need input from other agents or planning team?",
            "",
            "### Meeting Goals:",
            "",
            "1. ✅ Review progress on pending actions",
            "2. ✅ Identify and resolve blockers",
            "3. ✅ Coordinate next deployment phase",
            "4. ✅ Collect metrics for planning team",
            "5. ✅ Make GO/NO-GO decisions for stage progression",
            "",
            "---",
            "",
            f"**Full execution data:** `multi-agent/reports/execution_progress_update.json`",
            ""
        ])

        return "\n".join(context_parts)

    def run_execution_meeting(self, num_rounds: int = 3) -> Dict[str, Any]:
        """
        Run execution team meeting
        Generates same artifacts as planning meetings:
        - transcript
        - summary
        - actions
        - responses
        - integrity
        """
        print("\n" + "="*70)
        print("🔧 EXECUTION TEAM STRATEGIC MEETING")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Rounds: {num_rounds}")
        print(f"Reports: {self.execution_reports_dir}")
        print("="*70 + "\n")

        # Prepare execution context
        execution_context = self.prepare_execution_context()

        # Run meeting with execution context
        # This uses the base MeetingOrchestrator but with execution-specific context
        results = self.run_meeting(
            initial_context=execution_context,
            num_rounds=num_rounds,
            save_artifacts=True
        )

        # Move artifacts to execution directory and add execution prefix
        self._organize_execution_artifacts(results)

        # Create execution-specific summary
        self._create_execution_summary(results)

        return results

    def _organize_execution_artifacts(self, results: Dict):
        """
        Move and rename artifacts to execution directory
        transcript.md → execution_transcript_TIMESTAMP.md
        """
        timestamp = results.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))

        artifact_mapping = {
            'transcript': f'execution_transcript_{timestamp}.md',
            'summary': f'execution_summary_{timestamp}.md',
            'actions': f'execution_actions_{timestamp}.json',
            'responses': f'execution_responses_{timestamp}.json',
            'integrity': f'execution_integrity_{timestamp}.json'
        }

        # Move files from reports/ to reports/execution/
        reports_dir = self.project_root / "multi-agent/reports"

        for original_key, new_filename in artifact_mapping.items():
            # Find original file
            original_pattern = f"{original_key}_{timestamp}.*"
            original_files = list(reports_dir.glob(original_pattern))

            if original_files:
                original_file = original_files[0]
                new_file = self.execution_reports_dir / new_filename

                # Copy (don't move, in case planning needs it)
                import shutil
                shutil.copy2(original_file, new_file)

                print(f"✅ Saved: {new_filename}")

    def _create_execution_summary(self, results: Dict):
        """
        Create execution-specific summary with focus on:
        - Deployment progress
        - Actions completed
        - Blockers identified
        - Metrics collected
        - Next deployment phase
        """
        timestamp = results.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))

        summary_lines = [
            "# Execution Team Meeting Summary",
            f"**Session:** {timestamp}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"- **Agents Participated:** {len(results.get('agent_responses', []))}",
            f"- **Actions Identified:** {len(results.get('actions', []))}",
            f"- **Tools Used:** {results.get('tools_used', 0)}",
            f"- **Deployments:** {results.get('deployments_made', 0)}",
            "",
            "## Deployment Status",
            ""
        ]

        # Add deployment info
        deployment_status = self._get_current_deployment_status()
        summary_lines.extend([
            f"- **Current Stage:** {deployment_status.get('stage', 'N/A')}",
            f"- **Version:** {deployment_status.get('version', 'N/A')}",
            f"- **SLO Status:** {'✅ Pass' if all(deployment_status.get('slo_status', {}).values()) else '❌ Fail'}",
            "",
            "## Actions Completed This Meeting",
            ""
        ])

        # Add actions
        for i, action in enumerate(results.get('actions', [])[:10], 1):
            description = action.get('description', action.get('action', 'N/A'))
            priority = action.get('priority', 'medium').upper()
            summary_lines.append(f"{i}. **[{priority}]** {description}")

        summary_lines.extend([
            "",
            "## Key Decisions",
            "",
            "*(Extracted from meeting transcript)*",
            "",
            "## Metrics Collected",
            ""
        ])

        # Add metrics if available
        metrics = self._get_current_metrics()
        for key, value in metrics.items():
            if value is not None:
                summary_lines.append(f"- **{key}:** {value}")

        summary_lines.extend([
            "",
            "## Blockers & Challenges",
            "",
            "*(To be reviewed by planning team)*",
            "",
            "## Next Phase",
            "",
            "*(Next deployment actions)*",
            "",
            "---",
            "",
            f"**Full transcript:** `execution_transcript_{timestamp}.md`",
            f"**Actions list:** `execution_actions_{timestamp}.json`",
            ""
        ])

        # Save summary
        summary_file = self.execution_reports_dir / f"execution_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_lines))

        print(f"✅ Summary saved: execution_summary_{timestamp}.md")

    def _get_current_deployment_status(self) -> Dict:
        """Get current deployment status"""
        state_file = self.project_root / "multi-agent/state/deployment_state.json"

        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {}

    def _get_current_metrics(self) -> Dict:
        """Get current system metrics"""
        metrics_file = self.project_root / "multi-agent/state/metrics_state.json"

        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {}


def main():
    """Main entry point for execution meetings"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "multi-agent/configs/execution_coordination.yaml"

    # Check if execution config exists, fallback to autonomous config
    if not config_path.exists():
        print("⚠️  Using autonomous_coordination.yaml (execution_coordination.yaml not found)")
        config_path = project_root / "multi-agent/configs/autonomous_coordination.yaml"

    # Create orchestrator
    orchestrator = ExecutionMeetingOrchestrator(config_path, project_root)

    # Run meeting
    results = orchestrator.run_execution_meeting(num_rounds=3)

    print("\n" + "="*70)
    print("✅ EXECUTION MEETING COMPLETE")
    print("="*70)
    print(f"\n📁 Artifacts saved to:")
    print(f"   {orchestrator.execution_reports_dir}")
    print(f"\n📊 Results:")
    print(f"   Agents: {len(results.get('agent_responses', []))}")
    print(f"   Actions: {len(results.get('actions', []))}")
    print(f"   Duration: {results.get('duration', 0):.1f}s")
    print()


if __name__ == "__main__":
    main()
