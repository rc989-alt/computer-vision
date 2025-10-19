#!/usr/bin/env python3
"""
Enhanced Progress Sync System
Provides real-time access to ALL research data and execution progress
for both Planning Team and Execution Team
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from tools.file_bridge import FileBridge, ProgressTracker, create_default_policies


class EnhancedProgressSync:
    """
    Enhanced progress sync that gives teams access to:
    - All research directories (01-04 + docs)
    - Execution team progress and deployments
    - Real-time file changes
    - Meeting artifacts and trajectories
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.file_bridge = FileBridge(
            project_root,
            create_default_policies(project_root)
        )

        # ALL directories to track for planning team
        self.planning_tracked_dirs = [
            # Research lines
            'research/01_v1_production_line',
            'research/02_v2_research_line',
            'research/03_cotrr_lightweight_line',
            'research/04_final_analysis',
            'research/05_archive',

            # Documentation
            'docs/01_foundation',
            'docs/02_data_governance',
            'docs/03_research_exploration',
            'docs/04_production_deployment',
            'docs/05_analysis_reports',
            'docs/06_governance',
            'docs/06_guides_references',
            'docs/07_colab_gpu_batch_processing',

            # Data and results
            'research/data',
            'research/evaluation',
            'research/day3_results',
            'research/day4_results',

            # Legacy
            'research/stage1_progress',
        ]

        # Execution team directories to track
        self.execution_tracked_dirs = [
            # Execution reports
            'multi-agent/reports',
            'multi-agent/reports/execution',
            'multi-agent/reports/planning',
            'multi-agent/reports/handoff',

            # Deployment artifacts
            'deployment/shadow',
            'deployment/5_percent',
            'deployment/20_percent',
            'deployment/production',

            # State and logs
            'multi-agent/state',
            'multi-agent/logs',

            # Tools and agents
            'multi-agent/tools',
            'multi-agent/agents',
        ]

        # Initialize trackers
        self.planning_tracker = ProgressTracker(self.file_bridge, self.planning_tracked_dirs)
        self.execution_tracker = ProgressTracker(self.file_bridge, self.execution_tracked_dirs)

    def sync_for_planning_team(self) -> Dict[str, Any]:
        """
        Complete sync for planning team
        Returns comprehensive view of research + execution progress
        """
        print("\n🔄 Running Enhanced Progress Sync for Planning Team...")

        # 1. Scan ALL research directories
        research_scan = self.file_bridge.scan_artifacts('tech_analysis', self.planning_tracked_dirs)

        # 2. Scan execution progress
        execution_scan = self.file_bridge.scan_artifacts('system', self.execution_tracked_dirs)

        # 3. Generate progress reports
        research_progress = self.planning_tracker.generate_progress_report('tech_analysis')

        # 4. Extract latest files from all directories
        research_latest = self._get_latest_files(research_scan, top_n=30)

        # 5. Extract execution updates
        execution_latest = self._get_latest_files(execution_scan, top_n=20)

        # 6. Extract key metrics
        key_metrics = self._extract_comprehensive_metrics(research_latest)

        # 7. Get execution team's latest actions
        execution_status = self._get_execution_status()

        sync_result = {
            'sync_time': datetime.now().isoformat(),
            'sync_type': 'planning_team_comprehensive',

            # Research data
            'research_scan': research_scan,
            'research_progress': research_progress,
            'research_latest_files': research_latest,
            'key_metrics': key_metrics,

            # Execution data
            'execution_scan': execution_scan,
            'execution_latest_files': execution_latest,
            'execution_status': execution_status,

            # Combined summary
            'summary': self._create_comprehensive_summary(
                research_scan,
                execution_scan,
                research_latest,
                execution_latest
            )
        }

        # Save for agents
        self._save_planning_team_update(sync_result)

        print(f"✅ Planning Team Sync Complete")
        print(f"   Research files scanned: {research_scan['total_files']}")
        print(f"   Execution files scanned: {execution_scan['total_files']}")
        print(f"   Latest research artifacts: {len(research_latest)}")
        print(f"   Latest execution artifacts: {len(execution_latest)}\n")

        return sync_result

    def sync_for_execution_team(self) -> Dict[str, Any]:
        """
        Sync for execution team
        Returns planning decisions + deployment status + performance metrics
        """
        print("\n🔄 Running Enhanced Progress Sync for Execution Team...")

        # 1. Get latest planning decisions
        planning_decisions = self._get_latest_planning_decisions()

        # 2. Get pending actions from handoff
        pending_actions = self._get_pending_actions()

        # 3. Scan deployment status
        deployment_status = self._get_deployment_status()

        # 4. Get system metrics
        system_metrics = self._get_system_metrics()

        # 5. Generate execution progress report
        execution_progress = self.execution_tracker.generate_progress_report('system')

        sync_result = {
            'sync_time': datetime.now().isoformat(),
            'sync_type': 'execution_team_comprehensive',

            # Planning inputs
            'planning_decisions': planning_decisions,
            'pending_actions': pending_actions,

            # Deployment status
            'deployment_status': deployment_status,
            'system_metrics': system_metrics,

            # Progress tracking
            'execution_progress': execution_progress,

            # Summary
            'summary': self._create_execution_summary(
                planning_decisions,
                pending_actions,
                deployment_status
            )
        }

        # Save for agents
        self._save_execution_team_update(sync_result)

        print(f"✅ Execution Team Sync Complete")
        print(f"   Pending actions: {len(pending_actions)}")
        print(f"   Planning decisions: {len(planning_decisions)}")
        print(f"   Deployment stage: {deployment_status.get('current_stage', 'N/A')}\n")

        return sync_result

    def _get_latest_files(self, scan: Dict, top_n: int = 20) -> List[Dict]:
        """Extract most recent files from scan"""
        all_files = []

        for dir_data in scan['directories'].values():
            all_files.extend(dir_data.get('files', []))

        # Sort by modification time
        all_files.sort(key=lambda x: x['modified'], reverse=True)

        return all_files[:top_n]

    def _extract_comprehensive_metrics(self, files: List[Dict]) -> Dict[str, Any]:
        """Extract metrics from research files"""
        metrics = {
            'experiments_found': 0,
            'latest_experiment': None,
            'v1_metrics': {},
            'v2_metrics': {},
            'cotrr_metrics': {},
            'analysis_files': []
        }

        for file_info in files:
            # JSON files with metrics
            if file_info['name'].endswith('.json'):
                try:
                    content = self.file_bridge.read_file('system', file_info['path'])
                    if content and len(content) < 100_000:
                        data = json.loads(content)

                        # Check for experiment data
                        if any(key in str(data).lower() for key in ['ndcg', 'compliance', 'latency']):
                            metrics['experiments_found'] += 1

                            if not metrics['latest_experiment']:
                                metrics['latest_experiment'] = {
                                    'file': file_info['path'],
                                    'modified': file_info['modified'],
                                    'preview': str(data)[:300]
                                }

                            # Categorize by research line
                            if '01_v1' in file_info['path']:
                                metrics['v1_metrics'] = self._extract_specific_metrics(data)
                            elif '02_v2' in file_info['path']:
                                metrics['v2_metrics'] = self._extract_specific_metrics(data)
                            elif '03_cotrr' in file_info['path']:
                                metrics['cotrr_metrics'] = self._extract_specific_metrics(data)
                except:
                    pass

            # Markdown analysis files
            elif file_info['name'].endswith('.md'):
                if '04_final_analysis' in file_info['path'] or 'analysis' in file_info['name'].lower():
                    metrics['analysis_files'].append({
                        'name': file_info['name'],
                        'path': file_info['path'],
                        'modified': file_info['modified']
                    })

        return metrics

    def _extract_specific_metrics(self, data: Dict) -> Dict:
        """Extract specific performance metrics from data"""
        metrics = {}

        # Common metric keys
        for key in ['ndcg', 'compliance', 'latency', 'error_rate', 'throughput',
                    'accuracy', 'precision', 'recall', 'f1']:
            if key in data:
                metrics[key] = data[key]

        return metrics

    def _get_execution_status(self) -> Dict[str, Any]:
        """Get current execution team status"""
        status = {
            'latest_meeting': None,
            'latest_actions': [],
            'pending_count': 0,
            'completed_count': 0
        }

        # Get latest meeting summary
        reports_dir = self.project_root / 'multi-agent/reports'
        if reports_dir.exists():
            summaries = sorted(reports_dir.glob('summary_*.md'),
                             key=lambda x: x.stat().st_mtime, reverse=True)
            if summaries:
                latest = summaries[0]
                status['latest_meeting'] = {
                    'file': latest.name,
                    'modified': datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
                    'age_minutes': (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).total_seconds() / 60
                }

        # Get latest actions
        actions_files = sorted((reports_dir / 'planning').glob('actions_*.json') if (reports_dir / 'planning').exists() else [],
                             key=lambda x: x.stat().st_mtime, reverse=True)
        if actions_files:
            try:
                with open(actions_files[0], 'r') as f:
                    actions = json.load(f)
                    status['latest_actions'] = actions[:10] if isinstance(actions, list) else []
                    status['pending_count'] = len(actions) if isinstance(actions, list) else 0
            except:
                pass

        return status

    def _get_latest_planning_decisions(self) -> List[Dict]:
        """Get latest planning team decisions"""
        decisions = []

        reports_dir = self.project_root / 'multi-agent/reports/planning'
        if reports_dir.exists():
            # Get latest summary
            summaries = sorted(reports_dir.glob('summary_*.md'),
                             key=lambda x: x.stat().st_mtime, reverse=True)

            for summary_file in summaries[:3]:  # Last 3 meetings
                decisions.append({
                    'file': summary_file.name,
                    'timestamp': summary_file.stem.split('_')[-2] + '_' + summary_file.stem.split('_')[-1],
                    'modified': datetime.fromtimestamp(summary_file.stat().st_mtime).isoformat()
                })

        return decisions

    def _get_pending_actions(self) -> List[Dict]:
        """Get pending actions from handoff directory"""
        handoff_file = self.project_root / 'multi-agent/reports/handoff/pending_actions.json'

        if handoff_file.exists():
            try:
                with open(handoff_file, 'r') as f:
                    data = json.load(f)
                    return data.get('actions', [])
            except:
                pass

        return []

    def _get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        status = {
            'current_stage': 'unknown',
            'version': 'unknown',
            'slo_status': {},
            'last_updated': None
        }

        state_file = self.project_root / 'multi-agent/state/deployment_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    status.update({
                        'current_stage': state.get('stage', 'unknown'),
                        'version': state.get('current_version', 'unknown'),
                        'slo_status': state.get('slo_status', {}),
                        'last_updated': state.get('last_updated')
                    })
            except:
                pass

        return status

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        metrics = {
            'compliance_current': None,
            'ndcg_current': None,
            'latency_p95_ms': None,
            'error_rate': None,
            'last_measurement': None
        }

        metrics_file = self.project_root / 'multi-agent/state/metrics_state.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    metrics.update(data)
            except:
                pass

        return metrics

    def _create_comprehensive_summary(self, research_scan, execution_scan,
                                     research_latest, execution_latest) -> str:
        """Create comprehensive summary for planning team"""
        lines = [
            "# Comprehensive Progress Update",
            f"**Sync Time**: {datetime.now().isoformat()}",
            f"**For**: Planning Team",
            "",
            "## Research Data Summary",
            f"- Total files scanned: {research_scan['total_files']}",
            f"- Total size: {research_scan['total_size'] / 1_000_000:.2f} MB",
            f"- Latest artifacts: {len(research_latest)}",
            "",
            "## Recent Research Files",
        ]

        for file in research_latest[:10]:
            lines.append(f"- `{file['name']}` ({file['path']}) - {file['modified']}")

        lines.extend([
            "",
            "## Execution Team Progress",
            f"- Execution files tracked: {execution_scan['total_files']}",
            f"- Latest execution updates: {len(execution_latest)}",
            "",
            "## Recent Execution Activity",
        ])

        for file in execution_latest[:5]:
            lines.append(f"- `{file['name']}` - {file['modified']}")

        return "\n".join(lines)

    def _create_execution_summary(self, planning_decisions, pending_actions,
                                  deployment_status) -> str:
        """Create summary for execution team"""
        lines = [
            "# Execution Team Progress Update",
            f"**Sync Time**: {datetime.now().isoformat()}",
            f"**For**: Execution Team",
            "",
            "## Planning Decisions",
            f"- Recent meetings: {len(planning_decisions)}",
            f"- Pending actions: {len(pending_actions)}",
            "",
            "## Deployment Status",
            f"- Current stage: {deployment_status['current_stage']}",
            f"- Version: {deployment_status['version']}",
            f"- SLO status: {deployment_status['slo_status']}",
        ]

        if pending_actions:
            lines.extend([
                "",
                "## Next Actions to Execute",
            ])
            for i, action in enumerate(pending_actions[:5], 1):
                lines.append(f"{i}. {action.get('action', 'N/A')} (Priority: {action.get('priority', 'medium')})")

        return "\n".join(lines)

    def _save_planning_team_update(self, sync_result: Dict):
        """Save comprehensive update for planning team"""
        report_path = 'multi-agent/reports/planning_progress_update.md'

        self.file_bridge.write_file('system', report_path, sync_result['summary'])

        # Save full JSON
        json_path = 'multi-agent/reports/planning_progress_update.json'
        self.file_bridge.write_file(
            'system',
            json_path,
            json.dumps(sync_result, indent=2)
        )

    def _save_execution_team_update(self, sync_result: Dict):
        """Save comprehensive update for execution team"""
        report_path = 'multi-agent/reports/execution_progress_update.md'

        self.file_bridge.write_file('system', report_path, sync_result['summary'])

        # Save full JSON
        json_path = 'multi-agent/reports/execution_progress_update.json'
        self.file_bridge.write_file(
            'system',
            json_path,
            json.dumps(sync_result, indent=2)
        )


def create_planning_team_context(project_root: Path) -> str:
    """
    Generate comprehensive context for planning team
    Includes ALL research data + execution progress
    """
    sync = EnhancedProgressSync(project_root)
    result = sync.sync_for_planning_team()

    context = f"""
## COMPREHENSIVE PROGRESS UPDATE FOR PLANNING TEAM

{result['summary']}

### Key Metrics from Research:
- Experiments Found: {result['key_metrics']['experiments_found']}
- V1 Metrics: {len(result['key_metrics']['v1_metrics'])} data points
- V2 Metrics: {len(result['key_metrics']['v2_metrics'])} data points
- CoTRR Metrics: {len(result['key_metrics']['cotrr_metrics'])} data points
- Analysis Files: {len(result['key_metrics']['analysis_files'])}

### Execution Team Status:
- Latest Meeting: {result['execution_status'].get('latest_meeting', {}).get('file', 'None')}
- Pending Actions: {result['execution_status']['pending_count']}

**Full data available in**:
- `multi-agent/reports/planning_progress_update.md`
- `multi-agent/reports/planning_progress_update.json`

"""
    return context


def create_execution_team_context(project_root: Path) -> str:
    """
    Generate comprehensive context for execution team
    Includes planning decisions + deployment status
    """
    sync = EnhancedProgressSync(project_root)
    result = sync.sync_for_execution_team()

    context = f"""
## COMPREHENSIVE PROGRESS UPDATE FOR EXECUTION TEAM

{result['summary']}

**Full data available in**:
- `multi-agent/reports/execution_progress_update.md`
- `multi-agent/reports/execution_progress_update.json`

"""
    return context
