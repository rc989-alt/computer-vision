#!/usr/bin/env python3
"""
I/O Utilities
Helper functions for file operations and data loading
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Path, indent: int = 2):
    """Save data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_prompt(prompt_dir: Path, filename: str) -> str:
    """Load prompt template from file"""
    filepath = prompt_dir / filename
    if not filepath.exists():
        return f"Prompt file not found: {filename}"

    with open(filepath, 'r') as f:
        return f.read()


def ensure_dir(dirpath: Path):
    """Ensure directory exists"""
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def read_data_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Read data file (auto-detect JSON/YAML)"""
    if not filepath.exists():
        return None

    if filepath.suffix == '.json':
        return load_json(filepath)
    elif filepath.suffix in ['.yaml', '.yml']:
        return load_yaml(filepath)
    else:
        with open(filepath, 'r') as f:
            return {'content': f.read()}


class ProjectContext:
    """Loads and manages project context data"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.context = {}

    def load_all(self):
        """Load all available context files"""
        if not self.data_dir.exists():
            return {}

        for filepath in self.data_dir.glob('*.json'):
            key = filepath.stem
            self.context[key] = load_json(filepath)

        for filepath in self.data_dir.glob('*.yaml'):
            key = filepath.stem
            self.context[key] = load_yaml(filepath)

        return self.context

    def get(self, key: str, default=None):
        """Get context by key"""
        return self.context.get(key, default)

    def summary(self) -> str:
        """Get summary of available context"""
        lines = ["# Available Context\n"]
        for key, data in self.context.items():
            if isinstance(data, dict):
                lines.append(f"- {key}: {len(data)} items")
            else:
                lines.append(f"- {key}: {type(data).__name__}")
        return "\n".join(lines)
