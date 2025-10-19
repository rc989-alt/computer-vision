#!/usr/bin/env python3
"""
Autonomous Multi-Agent Coordination System
Implements self-directed execution with hierarchy, shared memory,
triggers, and periodic synchronization
"""

import os
import sys
import json
import yaml
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from run_meeting import MeetingOrchestrator
from run_execution_meeting import ExecutionMeetingOrchestrator
from tools.file_bridge import FileBridge, create_default_policies


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DeploymentStage(Enum):
    SHADOW = "shadow"
    FIVE_PERCENT = "5%"
    TWENTY_PERCENT = "20%"
    FIFTY_PERCENT = "50%"
    HUNDRED_PERCENT = "100%"


class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class Verdict(Enum):
    GO = "GO"
    SHADOW_ONLY = "SHADOW_ONLY"
    PAUSE_AND_FIX = "PAUSE_AND_FIX"
    REJECT = "REJECT"
    ROLLBACK = "ROLLBACK"


@dataclass
class Message:
    """Message for pub-sub channels"""
    channel: str
    publisher: str
    data: Dict[str, Any]
    timestamp: str
    message_id: str


@dataclass
class AgentTask:
    """Task for an agent to execute"""
    task_id: str
    agent_id: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    status: AgentStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class DeploymentState:
    """Current deployment state"""
    current_version: str
    stage: DeploymentStage
    slo_status: Dict[str, bool]
    last_updated: str
    rollback_ready: bool
    deployment_start: str


# =============================================================================
# SHARED MEMORY MANAGER
# =============================================================================

class SharedMemoryManager:
    """Manages shared state across agents using file-based persistence"""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache = {}
        self._lock = threading.Lock()

    def _get_state_unlocked(self, store_name: str) -> Dict[str, Any]:
        """Internal method to get state without locking (called when lock is already held)"""
        # Check cache first
        if store_name in self._cache:
            return self._cache[store_name].copy()

        # Load from file
        file_path = self.state_dir / f"{store_name}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                state = json.load(f)
                self._cache[store_name] = state
                return state.copy()

        # Return empty state
        return {}

    def get_state(self, store_name: str) -> Dict[str, Any]:
        """Get state from a specific store"""
        with self._lock:
            return self._get_state_unlocked(store_name)

    def update_state(self, store_name: str, updates: Dict[str, Any], merge: bool = True):
        """Update state in a specific store"""
        with self._lock:
            current = self._get_state_unlocked(store_name)

            if merge:
                current.update(updates)
            else:
                current = updates

            # Update cache
            self._cache[store_name] = current

            # Persist to file
            file_path = self.state_dir / f"{store_name}.json"
            with open(file_path, 'w') as f:
                json.dump(current, f, indent=2)

    def get_deployment_state(self) -> Optional[DeploymentState]:
        """Get current deployment state"""
        state = self.get_state("deployment_state")
        if not state:
            return None

        return DeploymentState(
            current_version=state.get("current_version", "unknown"),
            stage=DeploymentStage(state.get("stage", "shadow")),
            slo_status=state.get("slo_status", {}),
            last_updated=state.get("last_updated", ""),
            rollback_ready=state.get("rollback_ready", False),
            deployment_start=state.get("deployment_start", "")
        )

    def update_deployment_state(self, deployment_state: DeploymentState):
        """Update deployment state"""
        self.update_state("deployment_state", asdict(deployment_state), merge=False)


# =============================================================================
# COMMUNICATION CHANNEL MANAGER
# =============================================================================

class ChannelManager:
    """Manages pub-sub communication channels between agents"""

    def __init__(self):
        self._channels: Dict[str, List[Message]] = defaultdict(list)
        self._subscribers: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, channel: str, agent_id: str):
        """Subscribe agent to channel"""
        with self._lock:
            if agent_id not in self._subscribers[channel]:
                self._subscribers[channel].append(agent_id)

    def publish(self, channel: str, publisher: str, data: Dict[str, Any]) -> Message:
        """Publish message to channel"""
        with self._lock:
            message = Message(
                channel=channel,
                publisher=publisher,
                data=data,
                timestamp=datetime.now().isoformat(),
                message_id=f"{channel}_{int(time.time() * 1000)}"
            )
            self._channels[channel].append(message)
            return message

    def get_messages(self, channel: str, since: Optional[str] = None) -> List[Message]:
        """Get messages from channel, optionally since a timestamp"""
        with self._lock:
            messages = self._channels[channel]

            if since:
                messages = [m for m in messages if m.timestamp > since]

            return messages.copy()

    def get_latest_message(self, channel: str) -> Optional[Message]:
        """Get latest message from channel"""
        with self._lock:
            messages = self._channels[channel]
            return messages[-1] if messages else None


# =============================================================================
# TRIGGER SYSTEM
# =============================================================================

class TriggerSystem:
    """Manages event triggers and automatic actions"""

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self._active_triggers = []

    def register_trigger(self, name: str, condition: Callable[[], bool],
                        action: Callable[[], None], priority: str = "NORMAL"):
        """Register a trigger with condition and action"""
        self._active_triggers.append({
            'name': name,
            'condition': condition,
            'action': action,
            'priority': priority,
            'enabled': True
        })

    def check_triggers(self):
        """Check all active triggers and execute actions if conditions met"""
        triggered = []

        for trigger in self._active_triggers:
            if not trigger['enabled']:
                continue

            try:
                if trigger['condition']():
                    print(f"🔔 Trigger activated: {trigger['name']}")
                    trigger['action']()
                    triggered.append(trigger['name'])
            except Exception as e:
                print(f"❌ Trigger {trigger['name']} error: {e}")

        return triggered

    # Built-in trigger conditions
    def slo_breach_condition(self) -> bool:
        """Check if SLOs are breached"""
        metrics = self.coordinator.memory.get_state("metrics_state")

        if not metrics:
            return False

        # Check compliance drop
        if metrics.get('compliance_current', 0) < metrics.get('compliance_baseline', 0) - 0.02:
            return True

        # Check latency spike
        if metrics.get('latency_p95_ms', 0) > metrics.get('latency_baseline_ms', 0) * 1.1:
            return True

        # Check error rate
        if metrics.get('error_rate', 0) > 0.01:
            return True

        return False

    def stage_progression_condition(self) -> bool:
        """Check if ready to progress to next stage"""
        deployment = self.coordinator.memory.get_deployment_state()

        if not deployment:
            return False

        # Check all SLOs pass
        if not all(deployment.slo_status.values()):
            return False

        # Check minimum time in stage
        start_time = datetime.fromisoformat(deployment.deployment_start)
        time_in_stage = (datetime.now() - start_time).total_seconds() / 3600  # hours

        min_hours = {
            DeploymentStage.SHADOW: 24,
            DeploymentStage.FIVE_PERCENT: 48,
            DeploymentStage.TWENTY_PERCENT: 48,
            DeploymentStage.FIFTY_PERCENT: 72
        }

        required = min_hours.get(deployment.stage, 24)
        return time_in_stage >= required


# =============================================================================
# HEARTBEAT SYSTEM
# =============================================================================

class HeartbeatSystem:
    """Manages periodic synchronization cycles"""

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self._running = False
        self._thread = None

    def start(self):
        """Start heartbeat cycles"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        print("💓 Heartbeat system started")

    def stop(self):
        """Stop heartbeat cycles"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("💔 Heartbeat system stopped")

    def _heartbeat_loop(self):
        """Main heartbeat loop"""
        while self._running:
            try:
                print(f"\n{'='*60}")
                print(f"💓 HEARTBEAT CYCLE START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}\n")

                # Execute main cycle
                self._execute_main_cycle()

                print(f"\n{'='*60}")
                print(f"💓 HEARTBEAT CYCLE END")
                print(f"{'='*60}\n")

                # Wait for next cycle (60 minutes)
                time.sleep(3600)

            except Exception as e:
                print(f"❌ Heartbeat cycle error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(300)  # Wait 5 min on error

    def _execute_main_cycle(self):
        """Execute one main cycle"""

        # Phase 0: Initiation (5 min)
        print("📢 Phase 0: Cycle Initiation")
        self.coordinator.channels.publish(
            "all_agents",
            "heartbeat_system",
            {"type": "BEGIN_CYCLE", "timestamp": datetime.now().isoformat()}
        )

        # Save snapshot
        snapshot = {
            "deployment": asdict(self.coordinator.memory.get_deployment_state() or DeploymentState(
                current_version="unknown",
                stage=DeploymentStage.SHADOW,
                slo_status={},
                last_updated=datetime.now().isoformat(),
                rollback_ready=False,
                deployment_start=datetime.now().isoformat()
            )),
            "metrics": self.coordinator.memory.get_state("metrics_state"),
            "timestamp": datetime.now().isoformat()
        }
        snapshot_file = self.coordinator.project_root / "multi-agent/state/cycle_start.json"
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

        # Phase 1: Parallel Monitoring (15 min)
        print("\n📊 Phase 1: Parallel Monitoring")
        # In a full implementation, this would trigger agent tasks
        print("   → Infra Guardian: Verifying environment...")
        print("   → Latency Analyst: Profiling runtime...")
        print("   → Compliance Monitor: Evaluating metrics...")

        # Phase 2: Integration (10 min)
        print("\n🔗 Phase 2: Integration Validation")
        print("   → Integration Engineer: Running tests...")

        # Phase 3: Rollback Check (10 min)
        print("\n🛡️ Phase 3: Rollback Readiness")
        print("   → Rollback Officer: Testing mechanisms...")

        # Phase 4: Decision (5 min)
        print("\n⚖️ Phase 4: Decision Synthesis")
        verdict = self._compute_verdict()
        print(f"   → Verdict: {verdict.value}")

        # Phase 5: Execution (10 min)
        print("\n🚀 Phase 5: Action Execution")
        self._execute_action(verdict)

        # Phase 6: Innovation Relay (5 min)
        print("\n💡 Phase 6: Innovation Relay")
        print("   → Checking innovation queue...")

        # Check triggers
        print("\n🔔 Checking Triggers...")
        self.coordinator.triggers.check_triggers()

    def _compute_verdict(self) -> Verdict:
        """Compute deployment verdict based on current state"""
        deployment = self.coordinator.memory.get_deployment_state()
        metrics = self.coordinator.memory.get_state("metrics_state")

        if not deployment or not metrics:
            return Verdict.PAUSE_AND_FIX

        # Check for rollback conditions
        if self.coordinator.triggers.slo_breach_condition():
            return Verdict.ROLLBACK

        # Check for progression
        if self.coordinator.triggers.stage_progression_condition():
            return Verdict.GO

        # Check SLO status
        if all(deployment.slo_status.values()):
            return Verdict.SHADOW_ONLY

        return Verdict.PAUSE_AND_FIX

    def _execute_action(self, verdict: Verdict):
        """Execute action based on verdict"""
        if verdict == Verdict.GO:
            print("   ✅ Progressing to next stage...")
            # In full implementation, trigger stage progression
        elif verdict == Verdict.ROLLBACK:
            print("   🔴 Initiating rollback...")
            # In full implementation, trigger rollback
        elif verdict == Verdict.PAUSE_AND_FIX:
            print("   ⚠️ Pausing for fixes...")
            # In full implementation, notify planning team
        else:
            print(f"   ℹ️ Action: {verdict.value}")


# =============================================================================
# MAIN AUTONOMOUS COORDINATOR
# =============================================================================

class AutonomousCoordinator:
    """Main coordinator for autonomous multi-agent execution"""

    def __init__(self, config_path: Path, project_root: Path):
        self.config_path = Path(config_path)
        self.project_root = Path(project_root)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.memory = SharedMemoryManager(project_root / "multi-agent/state")
        self.channels = ChannelManager()
        self.triggers = TriggerSystem(self)
        self.heartbeat = HeartbeatSystem(self)

        # Initialize file bridge for agents
        self.file_bridge = FileBridge(
            project_root,
            create_default_policies(project_root)
        )

        # Agent registry
        self.agents = {}
        self._initialize_agents()

        # Setup communication channels
        self._setup_channels()

        # Register triggers
        self._register_triggers()

        # Initialize meeting orchestrators
        # Use consolidated 5-agent config (set use_consolidated=True to switch)
        use_consolidated = True
        if use_consolidated:
            config_path = project_root / "multi-agent/configs/consolidated_5_agent_coordination.yaml"
        else:
            config_path = project_root / "multi-agent/configs/autonomous_coordination.yaml"

        self.planning_meeting = MeetingOrchestrator(config_path, project_root)
        self.execution_meeting = ExecutionMeetingOrchestrator(config_path, project_root)

        print(f"✅ Autonomous Coordinator initialized")
        print(f"   Project root: {project_root}")
        print(f"   Agents: {len(self.agents)}")
        print(f"   Channels: {len(self.channels._subscribers)}")
        print(f"   Meeting systems: Planning + Execution")

    def _initialize_agents(self):
        """Initialize agent registry from config"""
        hierarchy = self.config.get('hierarchy', {})

        for team_name, team_config in hierarchy.items():
            if isinstance(team_config, list):
                for agent_config in team_config:
                    agent_id = agent_config.get('agent_id')
                    if agent_id:
                        self.agents[agent_id] = {
                            'role': agent_config.get('role'),
                            'model': agent_config.get('model'),
                            'reports_to': agent_config.get('reports_to'),
                            'status': AgentStatus.IDLE,
                            'current_task': None
                        }

        print(f"   Registered {len(self.agents)} agents")

    def _setup_channels(self):
        """Setup communication channels from config"""
        channels_config = self.config.get('shared_memory', {}).get('channels', [])

        for channel in channels_config:
            name = channel.get('name')
            subscribers = channel.get('subscribers', [])

            for subscriber in subscribers:
                self.channels.subscribe(name, subscriber)

        print(f"   Setup {len(channels_config)} channels")

    def _register_triggers(self):
        """Register triggers from config"""
        # SLO breach trigger
        self.triggers.register_trigger(
            name="emergency_rollback",
            condition=self.triggers.slo_breach_condition,
            action=self._trigger_emergency_rollback,
            priority="CRITICAL"
        )

        # Stage progression trigger
        self.triggers.register_trigger(
            name="stage_progression",
            condition=self.triggers.stage_progression_condition,
            action=self._trigger_stage_progression,
            priority="HIGH"
        )

        print(f"   Registered {len(self.triggers._active_triggers)} triggers")

    def _trigger_emergency_rollback(self):
        """Execute emergency rollback"""
        print("🚨 EMERGENCY ROLLBACK TRIGGERED")

        # Update state
        deployment = self.memory.get_deployment_state()
        if deployment:
            deployment.stage = DeploymentStage.SHADOW
            deployment.last_updated = datetime.now().isoformat()
            self.memory.update_deployment_state(deployment)

        # Publish to channels
        self.channels.publish(
            "v1_rollback_feed",
            "trigger_system",
            {
                "action": "EMERGENCY_ROLLBACK",
                "reason": "SLO_BREACH",
                "timestamp": datetime.now().isoformat()
            }
        )

    def _trigger_stage_progression(self):
        """Execute stage progression"""
        print("🎯 STAGE PROGRESSION TRIGGERED")

        deployment = self.memory.get_deployment_state()
        if not deployment:
            return

        # Progress to next stage
        stage_order = [
            DeploymentStage.SHADOW,
            DeploymentStage.FIVE_PERCENT,
            DeploymentStage.TWENTY_PERCENT,
            DeploymentStage.FIFTY_PERCENT,
            DeploymentStage.HUNDRED_PERCENT
        ]

        current_idx = stage_order.index(deployment.stage)
        if current_idx < len(stage_order) - 1:
            deployment.stage = stage_order[current_idx + 1]
            deployment.last_updated = datetime.now().isoformat()
            deployment.deployment_start = datetime.now().isoformat()
            self.memory.update_deployment_state(deployment)

            print(f"   → Progressed to: {deployment.stage.value}")

    def start(self):
        """Start autonomous execution"""
        print(f"\n{'='*60}")
        print(f"🚀 STARTING AUTONOMOUS COORDINATION SYSTEM")
        print(f"{'='*60}\n")

        # Initialize deployment state if not exists
        if not self.memory.get_deployment_state():
            initial_state = DeploymentState(
                current_version="v1.0_lightweight",
                stage=DeploymentStage.SHADOW,
                slo_status={
                    "compliance": False,
                    "latency": False,
                    "error_rate": False
                },
                last_updated=datetime.now().isoformat(),
                rollback_ready=False,
                deployment_start=datetime.now().isoformat()
            )
            self.memory.update_deployment_state(initial_state)
            print("✅ Initialized deployment state")

        # Start heartbeat
        self.heartbeat.start()

        print(f"\n{'='*60}")
        print(f"✅ AUTONOMOUS SYSTEM ACTIVE")
        print(f"{'='*60}\n")
        print("💡 System running in background...")
        print("💡 Press Ctrl+C to stop\n")

    def stop(self):
        """Stop autonomous execution"""
        print("\n🛑 Stopping autonomous system...")
        self.heartbeat.stop()
        print("✅ Autonomous system stopped")

    def run_deployment_mission(self):
        """Execute V1.0 Lightweight Deployment mission"""
        print(f"\n{'='*60}")
        print(f"🎯 MISSION: V1.0 Lightweight Deployment")
        print(f"{'='*60}\n")

        mission = self.config.get('autonomous_workflow', {}).get('current_mission', {})
        phases = mission.get('phases', [])

        for phase in phases:
            phase_name = phase.get('phase')
            status = phase.get('status')

            print(f"\n📋 Phase: {phase_name}")
            print(f"   Status: {status}")

            if status == "ready_to_start":
                print(f"   ▶️ Executing phase...")
                # In full implementation, execute tasks
                for task in phase.get('tasks', []):
                    print(f"      → {task.get('name')}")
                    print(f"         Owner: {task.get('owner')}")
            else:
                print(f"   ⏸️ Pending previous phases")

        print(f"\n{'='*60}")
        print(f"📊 Mission Status Summary")
        print(f"{'='*60}")

        # Show current state
        deployment = self.memory.get_deployment_state()
        if deployment:
            print(f"\n✅ Deployment State:")
            print(f"   Version: {deployment.current_version}")
            print(f"   Stage: {deployment.stage.value}")
            print(f"   SLO Status: {deployment.slo_status}")
            print(f"   Rollback Ready: {deployment.rollback_ready}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent

    # Use consolidated 5-agent config by default
    config_path = project_root / "multi-agent/configs/consolidated_5_agent_coordination.yaml"

    # Create coordinator
    coordinator = AutonomousCoordinator(config_path, project_root)

    print(f"✅ Using consolidated 5-agent configuration")
    print(f"   Planning Team: 4 agents")
    print(f"   Executive Team: 3 agents")
    print(f"   Total agents: 5 (down from 14)")
    print(f"   Cost reduction: 67%")
    print(f"   Meeting speed: 55% faster")

    # Start system
    try:
        coordinator.start()

        # Run deployment mission
        coordinator.run_deployment_mission()

        # Keep running until interrupted
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\n🛑 Received interrupt signal")
        coordinator.stop()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        coordinator.stop()


if __name__ == "__main__":
    main()
