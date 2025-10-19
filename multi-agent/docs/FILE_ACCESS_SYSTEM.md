# File Access System

## Overview

The multi-agent system now has controlled file system access through a File Bridge architecture with access policies and automatic progress tracking.

## Components

### 1. File Bridge (`tools/file_bridge.py`)

Centralized file access gateway providing:
- **FileAccessPolicy**: Defines which directories each agent can access
- **FileBridge**: Core gateway with read/write/list operations + access control
- **ProgressTracker**: Monitors file changes and generates progress reports

### 2. Progress Sync Hook (`tools/progress_sync_hook.py`)

Automatically runs at meeting start to:
- Scan artifact directories for new/modified files
- Extract key metrics from JSON files (NDCG, compliance, latency, etc.)
- Generate progress update reports for agents
- Save progress context to `multi-agent/reports/progress_update.md`

### 3. Access Policies

**Read-Only Agents** (planning team):
- pre_architect, v1_production, v2_scientific, cotrr_team
- tech_analysis, critic, integrity_guardian, data_analyst
- Access: `./results`, `./logs`, `./benchmarks`, `./data`, `./research`, `./docs`, `./analysis`

**Write Access Agents**:
- **moderator**: Can write to `./multi-agent/reports` and `./multi-agent/decisions`
- **system**: Full read/write access for internal operations

**Tech Analysis** has broader access:
- All planning directories + `./runs`, `./checkpoints`, `./cache`

## Configuration

Enable file access in `configs/meeting.yaml`:

```yaml
file_access:
  enable_file_bridge: true
  tracked_directories:
    - results
    - logs
    - benchmarks
    - research/day3_results
    - research/day4_results
    - runs/report
  accessible_directories:
    planning_agents:
      - ./results
      - ./logs
      # ... etc
```

## Usage

### In Meeting Orchestrator

```python
from tools.file_bridge import FileBridge, create_default_policies
from tools.progress_sync_hook import create_agent_context_with_progress

# Initialize at meeting start
file_bridge = FileBridge(project_root, create_default_policies(project_root))

# Run progress sync
progress_context = create_agent_context_with_progress(project_root)

# Progress context is injected into agent prompts
```

### Agent File Access

Agents receive file information through:
1. **Automatic progress updates** at meeting start
2. **Context from ProjectContext** (existing system)
3. **Direct file bridge access** (for specialized agents)

## Testing

Run the test suite:

```bash
cd /Users/guyan/computer_vision/computer-vision
python3 multi-agent/test_file_access.py
```

Expected output:
- File Bridge: Operational
- Access Control: Working
- Progress Sync Hook: Ready
- 28+ artifact files detected (research results, benchmarks, logs)

## Security Features

1. **Access Control**: Each agent has defined accessible directories
2. **Read-Only Default**: Most agents cannot modify files
3. **Access Logging**: All file operations logged for audit
4. **Path Validation**: Prevents directory traversal attacks

## Generated Artifacts

When progress sync runs:
- `multi-agent/reports/progress_update.md` - Human-readable progress report
- `multi-agent/reports/progress_update.json` - Full structured data

Reports include:
- File change summary (new/modified/deleted files)
- Latest 20 artifacts with modification times
- Extracted metrics from experiment results
- Total file count and size statistics

## Benefits

1. **Agents aware of latest results**: No need to manually inform agents
2. **Evidence-based decisions**: Agents reference actual experiment outputs
3. **Automatic progress tracking**: File diff detection between meetings
4. **Audit trail**: Complete log of file access attempts
5. **Security**: Controlled access prevents accidental modifications
