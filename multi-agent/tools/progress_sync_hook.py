#!/usr/bin/env python3
"""
Progress Sync Hook
Auto-runs at meeting start to scan artifacts and generate progress updates
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from tools.file_bridge import FileBridge, ProgressTracker, create_default_policies


class ProgressSyncHook:
    """
    Runs automatically at meeting start
    Scans project files and generates progress update for agents
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.file_bridge = FileBridge(
            project_root,
            create_default_policies(project_root)
        )

        # Directories to track for progress
        self.tracked_dirs = [
            'research/day3_results',
            'research/day4_results',
            'research/01_v1_production_line',
            'research/02_v2_research_line',
            'research/03_cotrr_lightweight_line',
            'research/04_final_analysis',
            'research/stage1_progress'
        ]

        self.progress_tracker = ProgressTracker(self.file_bridge, self.tracked_dirs)

    def run_sync(self) -> Dict[str, Any]:
        """
        Execute progress sync at meeting start
        Returns structured progress data for agents
        """
        print("\n🔄 Running Progress Sync Hook...")

        # 1. Scan all artifact directories
        artifact_scan = self.file_bridge.scan_artifacts('tech_analysis', self.tracked_dirs)

        # 2. Generate progress report
        progress_report = self.progress_tracker.generate_progress_report('tech_analysis')

        # 3. Identify latest files
        latest_files = self._get_latest_files(artifact_scan)

        # 4. Extract key metrics from latest files
        key_metrics = self._extract_key_metrics(latest_files)

        sync_result = {
            'sync_time': datetime.now().isoformat(),
            'artifact_scan': artifact_scan,
            'progress_report': progress_report,
            'latest_files': latest_files,
            'key_metrics': key_metrics,
            'summary': self._create_summary(artifact_scan, latest_files)
        }

        # Save progress report for agents
        self._save_progress_update(sync_result)

        print(f"✅ Progress Sync Complete")
        print(f"   - Scanned {artifact_scan['total_files']} files")
        print(f"   - Latest: {len(latest_files)} recent artifacts")
        print(f"   - Metrics: {len(key_metrics)} extracted\n")

        return sync_result

    def _get_latest_files(self, artifact_scan: Dict) -> List[Dict]:
        """Extract most recent files from scan"""
        all_files = []

        for dir_data in artifact_scan['directories'].values():
            all_files.extend(dir_data.get('files', []))

        # Sort by modification time
        all_files.sort(key=lambda x: x['modified'], reverse=True)

        return all_files[:20]  # Top 20 most recent

    def _extract_key_metrics(self, latest_files: List[Dict]) -> Dict[str, Any]:
        """
        Extract key metrics from latest artifact files
        (JSON files, benchmark reports, etc.)
        """
        metrics = {
            'experiments_found': 0,
            'latest_experiment': None,
            'metrics': {}
        }

        for file_info in latest_files:
            if file_info['name'].endswith('.json'):
                try:
                    content = self.file_bridge.read_file('system', file_info['path'])
                    if content and len(content) < 100_000:  # Skip huge files
                        import json
                        data = json.loads(content)

                        # Look for common metric patterns
                        if 'ndcg' in str(data).lower() or 'compliance' in str(data).lower():
                            metrics['experiments_found'] += 1
                            if not metrics['latest_experiment']:
                                metrics['latest_experiment'] = {
                                    'file': file_info['path'],
                                    'modified': file_info['modified'],
                                    'preview': str(data)[:500]
                                }

                        # Extract specific metrics
                        for key in ['ndcg', 'compliance', 'latency', 'error_rate']:
                            if key in data:
                                metrics['metrics'][key] = data[key]

                except:
                    pass

        return metrics

    def _create_summary(self, artifact_scan: Dict, latest_files: List[Dict]) -> str:
        """Create human-readable summary"""
        lines = [
            "# Artifact Update Summary",
            f"**Scan Time**: {artifact_scan['scan_time']}",
            f"**Total Files**: {artifact_scan['total_files']}",
            f"**Total Size**: {artifact_scan['total_size'] / 1_000_000:.2f} MB",
            "\n## Recent Activity\n"
        ]

        for file_info in latest_files[:10]:
            lines.append(f"- `{file_info['name']}` - {file_info['modified']}")

        return "\n".join(lines)

    def _save_progress_update(self, sync_result: Dict):
        """Save progress update for agents to reference"""
        report_path = 'multi-agent/reports/progress_update.md'

        content = [
            sync_result['progress_report'],
            "\n---\n",
            sync_result['summary']
        ]

        self.file_bridge.write_file('system', report_path, "\n".join(content))

        # Also save full JSON
        import json
        json_path = 'multi-agent/reports/progress_update.json'
        self.file_bridge.write_file(
            'system',
            json_path,
            json.dumps(sync_result, indent=2)
        )


def create_agent_context_with_progress(project_root: Path) -> str:
    """
    Generate context string for agents that includes progress update
    Called at meeting start
    """
    hook = ProgressSyncHook(project_root)
    sync_result = hook.run_sync()

    context = f"""
## ARTIFACT PROGRESS UPDATE

{sync_result['summary']}

### Latest Files Detected:
{chr(10).join(f"- {f['name']} ({f['modified']})" for f in sync_result['latest_files'][:10])}

### Key Metrics Extracted:
- Experiments Found: {sync_result['key_metrics']['experiments_found']}
- Latest Experiment: {sync_result['key_metrics']['latest_experiment']['file'] if sync_result['key_metrics']['latest_experiment'] else 'None'}

**Full progress report saved to**: `multi-agent/reports/progress_update.md`

"""
    return context
