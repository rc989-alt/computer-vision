#!/usr/bin/env python3
"""
File Bridge / Data Gateway
Provides controlled file system access for agents with access policies
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


class FileAccessPolicy:
    """Defines what directories an agent can access"""

    def __init__(self, agent_id: str, accessible_dirs: List[str], read_only: bool = True):
        self.agent_id = agent_id
        self.accessible_dirs = [Path(d).resolve() for d in accessible_dirs]
        self.read_only = read_only

    def can_access(self, file_path: Path) -> bool:
        """Check if agent can access this file"""
        file_path = Path(file_path).resolve()
        for allowed_dir in self.accessible_dirs:
            try:
                file_path.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
        return False


class FileBridge:
    """
    Centralized file access gateway for agents
    Provides read/write/list with access control
    """

    def __init__(self, project_root: Path, access_policies: Dict[str, FileAccessPolicy]):
        self.project_root = Path(project_root).resolve()
        self.access_policies = access_policies
        self.access_log = []

    def read_file(self, agent_id: str, file_path: str) -> Optional[str]:
        """Read file with access control"""
        policy = self.access_policies.get(agent_id)
        if not policy:
            self._log_access(agent_id, file_path, "DENIED", "No policy")
            return None

        full_path = self.project_root / file_path
        if not policy.can_access(full_path):
            self._log_access(agent_id, file_path, "DENIED", "Not in accessible dirs")
            return None

        if not full_path.exists():
            self._log_access(agent_id, file_path, "NOT_FOUND")
            return None

        try:
            with open(full_path, 'r') as f:
                content = f.read()
            self._log_access(agent_id, file_path, "READ", f"{len(content)} bytes")
            return content
        except Exception as e:
            self._log_access(agent_id, file_path, "ERROR", str(e))
            return None

    def list_files(self, agent_id: str, directory: str, pattern: str = "*") -> List[Dict[str, Any]]:
        """List files in directory with metadata"""
        policy = self.access_policies.get(agent_id)
        if not policy:
            self._log_access(agent_id, directory, "DENIED", "No policy")
            return []

        full_path = self.project_root / directory
        if not policy.can_access(full_path):
            self._log_access(agent_id, directory, "DENIED", "Not in accessible dirs")
            return []

        if not full_path.exists():
            self._log_access(agent_id, directory, "NOT_FOUND")
            return []

        try:
            files = []
            for file_path in full_path.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        'path': str(file_path.relative_to(self.project_root)),
                        'name': file_path.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'hash': self._compute_hash(file_path) if stat.st_size < 10_000_000 else None
                    })

            files.sort(key=lambda x: x['modified'], reverse=True)
            self._log_access(agent_id, directory, "LIST", f"{len(files)} files")
            return files
        except Exception as e:
            self._log_access(agent_id, directory, "ERROR", str(e))
            return []

    def write_file(self, agent_id: str, file_path: str, content: str) -> bool:
        """Write file with access control"""
        policy = self.access_policies.get(agent_id)
        if not policy:
            self._log_access(agent_id, file_path, "DENIED", "No policy")
            return False

        if policy.read_only:
            self._log_access(agent_id, file_path, "DENIED", "Read-only policy")
            return False

        full_path = self.project_root / file_path
        if not policy.can_access(full_path):
            self._log_access(agent_id, file_path, "DENIED", "Not in accessible dirs")
            return False

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            self._log_access(agent_id, file_path, "WRITE", f"{len(content)} bytes")
            return True
        except Exception as e:
            self._log_access(agent_id, file_path, "ERROR", str(e))
            return False

    def get_file_diff(self, agent_id: str, directory: str, since_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get files modified since timestamp"""
        files = self.list_files(agent_id, directory, "**/*")

        if since_timestamp:
            since_dt = datetime.fromisoformat(since_timestamp)
            files = [f for f in files if datetime.fromisoformat(f['modified']) > since_dt]

        return files

    def scan_artifacts(self, agent_id: str, directories: List[str]) -> Dict[str, Any]:
        """Scan multiple directories for artifacts"""
        summary = {
            'scan_time': datetime.now().isoformat(),
            'agent_id': agent_id,
            'directories': {},
            'total_files': 0,
            'total_size': 0
        }

        for directory in directories:
            files = self.list_files(agent_id, directory, "**/*")
            summary['directories'][directory] = {
                'file_count': len(files),
                'total_size': sum(f['size'] for f in files),
                'latest_modified': files[0]['modified'] if files else None,
                'files': files[:10]  # Top 10 most recent
            }
            summary['total_files'] += len(files)
            summary['total_size'] += sum(f['size'] for f in files)

        return summary

    def _compute_hash(self, file_path: Path) -> str:
        """Compute file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None

    def _log_access(self, agent_id: str, path: str, action: str, details: str = ""):
        """Log file access for audit"""
        self.access_log.append({
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'path': path,
            'action': action,
            'details': details
        })

    def get_access_log(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get access log, optionally filtered by agent"""
        if agent_id:
            return [entry for entry in self.access_log if entry['agent_id'] == agent_id]
        return self.access_log


class ProgressTracker:
    """Tracks progress by monitoring file changes"""

    def __init__(self, file_bridge: FileBridge, tracked_dirs: List[str]):
        self.file_bridge = file_bridge
        self.tracked_dirs = tracked_dirs
        self.baseline = {}
        self.update_baseline()

    def update_baseline(self):
        """Update baseline state"""
        self.baseline = {
            'timestamp': datetime.now().isoformat(),
            'directories': {}
        }

        for directory in self.tracked_dirs:
            files = self.file_bridge.list_files('system', directory, "**/*")
            self.baseline['directories'][directory] = {
                'files': {f['path']: f['hash'] for f in files if f['hash']},
                'count': len(files)
            }

    def generate_progress_report(self, agent_id: str = 'system') -> str:
        """Generate progress report comparing current state to baseline"""
        report = [
            "# Progress Update Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Baseline**: {self.baseline['timestamp']}",
            "\n---\n"
        ]

        changes = {
            'new_files': [],
            'modified_files': [],
            'deleted_files': [],
            'unchanged_count': 0
        }

        for directory in self.tracked_dirs:
            current_files = self.file_bridge.list_files(agent_id, directory, "**/*")
            baseline_files = self.baseline['directories'].get(directory, {}).get('files', {})

            current_paths = {f['path']: f['hash'] for f in current_files if f['hash']}

            # Detect changes
            for path, current_hash in current_paths.items():
                if path not in baseline_files:
                    changes['new_files'].append(path)
                elif baseline_files[path] != current_hash:
                    changes['modified_files'].append(path)
                else:
                    changes['unchanged_count'] += 1

            for path in baseline_files:
                if path not in current_paths:
                    changes['deleted_files'].append(path)

        # Format report
        report.append(f"## Summary")
        report.append(f"- **New files**: {len(changes['new_files'])}")
        report.append(f"- **Modified files**: {len(changes['modified_files'])}")
        report.append(f"- **Deleted files**: {len(changes['deleted_files'])}")
        report.append(f"- **Unchanged files**: {changes['unchanged_count']}")

        if changes['new_files']:
            report.append(f"\n## New Files\n")
            for path in changes['new_files'][:20]:
                report.append(f"- `{path}`")

        if changes['modified_files']:
            report.append(f"\n## Modified Files\n")
            for path in changes['modified_files'][:20]:
                report.append(f"- `{path}`")

        if changes['deleted_files']:
            report.append(f"\n## Deleted Files\n")
            for path in changes['deleted_files'][:20]:
                report.append(f"- `{path}`")

        return "\n".join(report)


def create_default_policies(project_root: Path) -> Dict[str, FileAccessPolicy]:
    """Create default access policies for agents"""

    # Convert to absolute paths relative to project_root
    def make_absolute(dirs):
        return [str((Path(project_root) / d).resolve()) for d in dirs]

    # Shared read-only directories for all planning agents
    planning_dirs_relative = [
        'results',
        'logs',
        'benchmarks',
        'data',
        'research',
        'docs',
        'analysis'
    ]
    planning_dirs = make_absolute(planning_dirs_relative)

    # Technical Analysis has broader access
    tech_analysis_dirs = planning_dirs + make_absolute([
        'runs',
        'checkpoints',
        'cache'
    ])

    # Moderator can write reports
    moderator_dirs = planning_dirs + make_absolute([
        'multi-agent/reports',
        'multi-agent/decisions'
    ])

    policies = {
        'pre_architect': FileAccessPolicy('pre_architect', planning_dirs, read_only=True),
        'v1_production': FileAccessPolicy('v1_production', planning_dirs, read_only=True),
        'v2_scientific': FileAccessPolicy('v2_scientific', planning_dirs, read_only=True),
        'cotrr_team': FileAccessPolicy('cotrr_team', planning_dirs, read_only=True),
        'tech_analysis': FileAccessPolicy('tech_analysis', tech_analysis_dirs, read_only=True),
        'critic': FileAccessPolicy('critic', planning_dirs, read_only=True),
        'integrity_guardian': FileAccessPolicy('integrity_guardian', planning_dirs, read_only=True),
        'data_analyst': FileAccessPolicy('data_analyst', planning_dirs, read_only=True),
        'moderator': FileAccessPolicy('moderator', moderator_dirs, read_only=False),
        'system': FileAccessPolicy('system', ['.'], read_only=False),  # Full read/write access for internal operations
    }

    return policies
