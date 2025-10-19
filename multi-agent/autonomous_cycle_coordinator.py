#!/usr/bin/env python3
"""
Autonomous Cycle Coordinator for CVPR 2025 Research

Manages Planning-Executive cycles with manual checkpoints:
1. Planning Team generates pending_actions.json
2. Executive Team executes tasks (in Colab with GPU)
3. Progress reported to execution_progress_update.md
4. Manual checkpoint for review
5. Cycle repeats with new pending_actions.json
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'multi-agent'))

class AutonomousCycleCoordinator:
    """Coordinates Planning-Executive cycles with manual checkpoints"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.multi_agent_root = self.project_root / 'multi-agent'
        self.handoff_dir = self.multi_agent_root / 'reports/handoff'
        self.cycle_number = 0
        self.cycle_history = []

        # File paths
        self.pending_actions_file = self.handoff_dir / 'pending_actions.json'
        self.progress_update_file = self.handoff_dir / 'execution_progress_update.md'
        self.trigger_file = self.handoff_dir / 'next_meeting_trigger.json'
        self.cycle_state_file = self.multi_agent_root / 'state/cycle_state.json'

        # Ensure directories exist
        self.handoff_dir.mkdir(parents=True, exist_ok=True)
        (self.multi_agent_root / 'state').mkdir(parents=True, exist_ok=True)

        # Load cycle state
        self._load_cycle_state()

    def _load_cycle_state(self):
        """Load cycle state from disk"""
        if self.cycle_state_file.exists():
            with open(self.cycle_state_file, 'r') as f:
                state = json.load(f)
                self.cycle_number = state.get('cycle_number', 0)
                self.cycle_history = state.get('cycle_history', [])
                print(f"✅ Loaded cycle state: Cycle {self.cycle_number}")
        else:
            print("✅ Initialized new cycle state")

    def _save_cycle_state(self):
        """Save cycle state to disk"""
        state = {
            'cycle_number': self.cycle_number,
            'cycle_history': self.cycle_history,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.cycle_state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def check_pending_actions_exist(self) -> bool:
        """Check if Planning Team has generated pending_actions.json"""
        if not self.pending_actions_file.exists():
            print(f"❌ No pending_actions.json found at {self.pending_actions_file}")
            print("⚠️ Planning Team must generate pending_actions.json first")
            return False

        print(f"✅ Found pending_actions.json")
        return True

    def read_pending_actions(self) -> Dict[str, Any]:
        """Read pending actions from Planning Team"""
        with open(self.pending_actions_file, 'r') as f:
            actions = json.load(f)

        decisions = actions.get('decisions', [])
        print(f"📋 Pending actions: {len(decisions)} tasks")

        # Group by priority
        high = len([d for d in decisions if d.get('priority') == 'HIGH'])
        medium = len([d for d in decisions if d.get('priority') == 'MEDIUM'])
        low = len([d for d in decisions if d.get('priority') == 'LOW'])

        print(f"   ⭐ HIGH: {high}")
        print(f"   🟠 MEDIUM: {medium}")
        print(f"   🔵 LOW: {low}")

        return actions

    def execute_executive_team_cycle(self):
        """Execute Executive Team tasks (via Colab notebook)"""
        print("\n" + "="*80)
        print("🚀 EXECUTIVE TEAM EXECUTION CYCLE")
        print("="*80)

        print(f"\n📝 Instructions:")
        print(f"1. Open Google Colab: https://colab.research.google.com/")
        print(f"2. Upload notebook: research/colab/cvpr_autonomous_execution_cycle.ipynb")
        print(f"3. Run all cells in order")
        print(f"4. Notebook will:")
        print(f"   - Read pending_actions.json")
        print(f"   - Execute tasks in priority order (HIGH → MEDIUM → LOW)")
        print(f"   - Use real deployment tools (Python, MLflow, GPU)")
        print(f"   - Generate execution_progress_update.md")
        print(f"   - Save results to Google Drive")
        print(f"   - Create next_meeting_trigger.json")
        print(f"5. Return here when Colab execution completes")

        print(f"\n⏸️ Waiting for Colab execution to complete...")
        print(f"\n" + "="*80)

        # Wait for execution completion (user will confirm)
        return self._wait_for_execution_completion()

    def _wait_for_execution_completion(self) -> bool:
        """Wait for user to confirm Colab execution completed"""
        while True:
            response = input("\n✅ Has Colab execution completed? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                # Check if progress update exists
                if self.progress_update_file.exists():
                    print("✅ Found execution_progress_update.md")
                    return True
                else:
                    print(f"⚠️ execution_progress_update.md not found at {self.progress_update_file}")
                    retry = input("Retry? (yes/no): ").lower().strip()
                    if retry not in ['yes', 'y']:
                        return False
            elif response in ['no', 'n']:
                print("⏸️ Waiting for Colab execution...")
                time.sleep(5)
            else:
                print("Please answer 'yes' or 'no'")

    def read_execution_progress(self) -> Dict[str, Any]:
        """Read execution progress from Executive Team"""
        if not self.progress_update_file.exists():
            print(f"❌ No progress update found at {self.progress_update_file}")
            return {}

        # Read markdown report
        with open(self.progress_update_file, 'r') as f:
            progress_md = f.read()

        # Try to find JSON results
        results_dir = self.multi_agent_root / 'reports/execution/results'
        json_files = list(results_dir.glob('execution_results_*.json'))

        if json_files:
            # Get most recent
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest_json, 'r') as f:
                results = json.load(f)
            print(f"✅ Found execution results: {latest_json}")
        else:
            results = {}
            print("⚠️ No JSON results found, using markdown only")

        return {
            'markdown_report': progress_md,
            'json_results': results,
            'timestamp': datetime.now().isoformat()
        }

    def display_execution_summary(self, progress: Dict[str, Any]):
        """Display execution summary for manual review"""
        print("\n" + "="*80)
        print("📊 EXECUTION SUMMARY - MANUAL REVIEW")
        print("="*80)

        results = progress.get('json_results', {})

        print(f"\n✅ Cycle {self.cycle_number + 1} execution complete")
        print(f"📊 Total tasks: {results.get('total_tasks', 'N/A')}")
        print(f"✅ Completed: {results.get('completed', 'N/A')}")
        print(f"❌ Failed: {results.get('failed', 'N/A')}")
        print(f"⏱️ Duration: {results.get('total_duration_seconds', 0):.1f}s")

        # Display markdown report preview
        md_report = progress.get('markdown_report', '')
        if md_report:
            print(f"\n📄 Progress Report Preview:")
            print(f"{md_report[:1000]}...")

        print(f"\n📁 Full report: {self.progress_update_file}")
        print("\n" + "="*80)

    def trigger_planning_team_meeting(self) -> bool:
        """Trigger next Planning Team meeting"""
        print("\n" + "="*80)
        print("📋 TRIGGER PLANNING TEAM MEETING")
        print("="*80)

        if not self.trigger_file.exists():
            print(f"⚠️ No trigger file found at {self.trigger_file}")
            return False

        with open(self.trigger_file, 'r') as f:
            trigger = json.load(f)

        print(f"\n🎯 Next Meeting: {trigger['next_meeting']['team'].upper()} TEAM")
        print(f"📋 Purpose: {trigger['next_meeting']['purpose']}")
        print(f"\n📋 Agenda:")
        for item in trigger['next_meeting']['agenda']:
            print(f"   - {item}")

        print(f"\n📝 Required Inputs:")
        for inp in trigger['next_meeting']['required_inputs']:
            print(f"   - {inp}")

        print(f"\n" + "="*80)
        return True

    def run_planning_team_meeting(self):
        """Run Planning Team meeting to review results and generate next pending_actions.json"""
        print("\n" + "="*80)
        print("📋 PLANNING TEAM REVIEW MEETING")
        print("="*80)

        print(f"\n📝 What Planning Team will do:")
        print(f"1. Review execution_progress_update.md")
        print(f"2. Assess progress toward Week 1 GO/NO-GO criteria")
        print(f"3. Identify blockers and risks")
        print(f"4. Plan next cycle tasks based on results")
        print(f"5. Generate new pending_actions.json")

        print(f"\n🤖 Running Planning Team review meeting...")

        # Call the planning review script
        review_script = self.multi_agent_root / 'scripts/run_planning_review_meeting.py'

        if review_script.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(review_script)],
                cwd=str(self.multi_agent_root),
                capture_output=True,
                text=True
            )

            print(result.stdout)
            if result.returncode != 0:
                print(f"⚠️ Planning meeting had issues:")
                print(result.stderr)
        else:
            print(f"⚠️ Review script not found at {review_script}")
            print(f"   Manually run: python scripts/run_planning_review_meeting.py")

        print(f"\n" + "="*80)
        print(f"✅ Planning Team review complete")
        print(f"📋 Check reports/handoff/pending_actions.json for next cycle tasks")
        print(f"\n" + "="*80)

    def complete_cycle(self, progress: Dict[str, Any]):
        """Complete current cycle and update state"""
        self.cycle_number += 1

        cycle_record = {
            'cycle_number': self.cycle_number,
            'timestamp': datetime.now().isoformat(),
            'tasks_completed': progress.get('json_results', {}).get('completed', 0),
            'tasks_failed': progress.get('json_results', {}).get('failed', 0),
            'duration_seconds': progress.get('json_results', {}).get('total_duration_seconds', 0)
        }

        self.cycle_history.append(cycle_record)
        self._save_cycle_state()

        print(f"\n✅ Cycle {self.cycle_number} complete")
        print(f"📊 Cycle history: {len(self.cycle_history)} cycles")

    def run_full_cycle(self):
        """Run one complete Planning-Executive cycle"""
        print("\n" + "="*80)
        print(f"🔄 AUTONOMOUS CYCLE {self.cycle_number + 1}")
        print("="*80)

        # Step 1: Check for pending actions
        if not self.check_pending_actions_exist():
            print("\n❌ Cannot start cycle: No pending actions")
            print("   Run Planning Team meeting first")
            return False

        # Step 2: Read pending actions
        actions = self.read_pending_actions()

        # Step 3: Execute Executive Team tasks (Colab)
        if not self.execute_executive_team_cycle():
            print("\n❌ Execution cycle failed or cancelled")
            return False

        # Step 4: Read execution progress
        progress = self.read_execution_progress()

        # Step 5: Display summary for manual review
        self.display_execution_summary(progress)

        # Step 6: Manual checkpoint
        print("\n⏸️ MANUAL CHECKPOINT")
        approve = input("Approve results and continue to Planning Team meeting? (yes/no): ").lower().strip()

        if approve not in ['yes', 'y']:
            print("⏸️ Cycle paused for manual review")
            return False

        # Step 7: Trigger Planning Team meeting
        if not self.trigger_planning_team_meeting():
            print("\n⚠️ Could not trigger Planning Team meeting")

        # Step 8: Run Planning Team meeting (manual for now)
        self.run_planning_team_meeting()

        # Step 9: Complete cycle
        self.complete_cycle(progress)

        print("\n" + "="*80)
        print(f"✅ CYCLE {self.cycle_number} COMPLETE")
        print("="*80)
        print(f"\n🔄 Ready for next cycle")
        print(f"   Waiting for new pending_actions.json from Planning Team")

        return True

    def start_autonomous_system(self):
        """Start autonomous system with continuous cycles"""
        print("\n" + "="*80)
        print("🚀 AUTONOMOUS CVPR 2025 RESEARCH SYSTEM")
        print("="*80)

        print(f"\n📊 System Status:")
        print(f"   Cycle number: {self.cycle_number}")
        print(f"   Cycle history: {len(self.cycle_history)} cycles")
        print(f"   Project root: {self.project_root}")

        print(f"\n🔄 Cycle Flow:")
        print(f"   1. Planning Team → pending_actions.json")
        print(f"   2. Executive Team → Execute tasks (Colab)")
        print(f"   3. Executive Team → execution_progress_update.md")
        print(f"   4. Manual checkpoint for review")
        print(f"   5. Planning Team → Review and next cycle")

        print(f"\n" + "="*80)

        while True:
            print(f"\n\n{'='*80}")
            print(f"🔄 CYCLE {self.cycle_number + 1}")
            print(f"{'='*80}")

            # Run one cycle
            success = self.run_full_cycle()

            if not success:
                print("\n⏸️ Cycle incomplete or paused")
                break

            # Ask to continue
            print(f"\n🔄 Cycle {self.cycle_number} complete")
            continue_cycle = input("Start next cycle? (yes/no): ").lower().strip()

            if continue_cycle not in ['yes', 'y']:
                print("\n⏸️ Autonomous system paused")
                break

        print("\n" + "="*80)
        print("✅ AUTONOMOUS SYSTEM STOPPED")
        print("="*80)
        print(f"\n📊 Final Status:")
        print(f"   Total cycles: {self.cycle_number}")
        print(f"   State saved to: {self.cycle_state_file}")


def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent

    print("="*80)
    print("🚀 CVPR 2025 AUTONOMOUS CYCLE COORDINATOR")
    print("="*80)

    coordinator = AutonomousCycleCoordinator(project_root)

    print("\n📋 Options:")
    print("1. Run single cycle (with manual checkpoints)")
    print("2. Start continuous autonomous system")
    print("3. Check system status")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == '1':
        coordinator.run_full_cycle()
    elif choice == '2':
        coordinator.start_autonomous_system()
    elif choice == '3':
        print(f"\n📊 System Status:")
        print(f"   Cycle number: {coordinator.cycle_number}")
        print(f"   Cycle history: {len(coordinator.cycle_history)} cycles")
        print(f"   Pending actions: {'✅ Found' if coordinator.pending_actions_file.exists() else '❌ Not found'}")
        print(f"   Progress update: {'✅ Found' if coordinator.progress_update_file.exists() else '❌ Not found'}")
    else:
        print("❌ Invalid option")


if __name__ == "__main__":
    main()
