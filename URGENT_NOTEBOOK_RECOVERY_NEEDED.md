# URGENT: Notebook Recovery Needed

**Date:** 2025-10-15
**Status:** ❌ NOTEBOOK CORRUPTED
**Affected File:** `research/colab/cvpr_autonomous_execution_cycle.ipynb`

---

## Problem

When fixing the indentation and attribute errors, I accidentally concatenated all lines in Cells 11 and 16 into single long strings **without preserving newline characters**. This makes the notebook unreadable and non-executable.

**Affected Cells:**
- **Cell 11**: Task execution loop with 3-agent approval (4,061 chars in single line)
- **Cell 16**: Phase 5.5 evidence verification (3,821 chars in single line)

---

## Recovery Options

### Option 1: Restore from Colab Revision History (RECOMMENDED)

If you have the notebook open in Google Colab:

1. **File → Revision history**
2. Look for revision **BEFORE** "2025-10-15 02:42" (current corrupted version)
3. Find a version from earlier today or yesterday
4. **Restore** that version
5. Then I'll reapply the fixes CAREFULLY

### Option 2: Restore from Google Drive Version History

1. Go to Google Drive
2. Right-click `cvpr_autonomous_execution_cycle.ipynb`
3. **Manage versions**
4. Download a previous version (before today)
5. Replace current file
6. I'll reapply fixes

### Option 3: Manual Reconstruction (LAST RESORT)

I can reconstruct Cells 11 and 16 from the documented fixes, but this is error-prone.

---

## What I'll Do Once You Restore

Once you have a clean version with properly formatted cells:

1. I'll apply the **Cell 11 fix** (indentation error) CAREFULLY:
   - Move `determine_task_status()` function outside loop
   - Use proper line-by-line edits (NOT concatenation)

2. I'll apply the **Cell 16 fix** (attribute error) CAREFULLY:
   - Change `tracker.tasks` → `tracker.task_results`
   - Change `task.status` → `task['status']`
   - Change `task.action` → `task['action']`
   - Use find/replace on PROPERLY FORMATTED lines

---

## What Went Wrong

My mistake was using this pattern:
```python
source = ''.join(cell['source'])  # Concatenates ALL lines
corrected_source = source.replace(...)  # Fix code
cell['source'] = corrected_source.split('\n')  # Split back
```

**Problem**: When the source was already concatenated (no `\n` inside), `split('\n')` created ONE element, not multiple lines.

**Correct approach** (what I should have done):
```python
# Work on source array directly
for i, line in enumerate(cell['source']):
    cell['source'][i] = line.replace('old', 'new')
# This preserves line structure!
```

---

## Immediate Action Required

**Please restore the notebook from revision history ASAP so I can reapply fixes correctly.**

In Colab: **File → Revision history** → Pick version from before today

Let me know when you've restored it, and I'll fix it properly this time.

---

## Lesson Learned

**NEVER use `.join()` then `.split()` on notebook cells!**
- Jupyter notebooks store each line as a separate array element
- Each line (except last) ends with `\n`
- Concatenating destroys this structure
- Always edit line-by-line or use proper JSON preservation

---

**Status:** ⏳ WAITING FOR USER TO RESTORE FROM REVISION HISTORY

CVPR 2025 Autonomous Execution Cycle

**Purpose:** Run autonomous Planning-Executive cycles for CVPR 2025 research

**Cycle Flow:**
1. 📥 **Read** `pending_actions.json` from Planning Team
2. 🤖 **Execute** tasks in priority order (HIGH → MEDIUM → LOW)
3. 📊 **Track** progress and results in real-time
4. 📤 **Report** results to `execution_progress_update.md`
5. 🔄 **Trigger** next Planning Team meeting with results
6. ⏸️ **Pause** for manual review before next cycle

**Timeline:** Week 1 (Oct 14-20) - Cross-architecture attention collapse validation

文本单元格 <Z1gxKDqjDAaf>
# %% [markdown]
---
## Setup: Mount Google Drive & Install Dependencies

代码单元格 <Y0IMX3ytDAaf>
# %% [code]
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import sys

# Set project paths
GDRIVE_ROOT = '/content/drive/MyDrive/cv_multimodal/project'
PROJECT_ROOT = f'{GDRIVE_ROOT}/computer-vision-clean'
MULTI_AGENT_ROOT = f'{PROJECT_ROOT}/multi-agent'

# Add to Python path
sys.path.insert(0, MULTI_AGENT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f"✅ Google Drive mounted")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"🤖 Multi-agent root: {MULTI_AGENT_ROOT}")

代码单元格 <J32UcLY0DAaf>
# %% [code]
import os
from pathlib import Path

print("="*80)
print("🔍 FINDING AND LOADING API KEYS")
print("="*80)

# Search for .env file in multiple locations
search_paths = [
    f'{GDRIVE_ROOT}/.env',
    f'{GDRIVE_ROOT}/computer-vision-clean/.env',
    '/content/drive/MyDrive/cv_multimodal/.env',
    '/content/drive/My Drive/cv_multimodal/project/.env',
    f'{PROJECT_ROOT}/.env'
]

env_file = None
for path in search_paths:
    if Path(path).exists():
        env_file = Path(path)
        print(f"\n✅ Found .env file: {path}")
        print(f"   Size: {env_file.stat().st_size} bytes\n")
        break
    else:
        print(f"   Checking: {path}... not found")

if not env_file:
    print("\n🔍 Searching entire cv_multimodal directory...")
    base = Path('/content/drive/MyDrive/cv_multimodal')
    if base.exists():
        all_env = list(base.rglob('*.env')) + list(base.rglob('.env*'))
        if all_env:
            print(f"\n✅ Found .env files:")
            for f in all_env:
                print(f"   📄 {f}")
            env_file = all_env[0]

    if not env_file:
        print("\n❌ No .env file found!")
        print("\n💡 Options:")
        print("   1. Upload .env to MyDrive/cv_multimodal/project/")
        print("   2. Use Colab Secrets (🔑 icon in left sidebar)")
        raise FileNotFoundError("No .env file found - check Google Drive or use Colab Secrets")

# Load all API keys
loaded_keys = []
with open(env_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Remove quotes
            os.environ[key] = value
            loaded_keys.append(key)

# Verify required keys for multi-agent system
required = {
    'ANTHROPIC_API_KEY': 'Claude Sonnet (Ops, Quality, Infrastructure)',
    'OPENAI_API_KEY': 'GPT-4 (Critical Evaluator)',
    'GOOGLE_API_KEY': 'Gemini (Research Advisor)'
}

print("Verifying required API keys:\n")
all_loaded = True

for key, usage in required.items():
    value = os.environ.get(key)
    if value and len(value) > 10:
        masked = f"{value[:8]}...{value[-4:]}"
        print(f"✅ {key}")
        print(f"   Value: {masked}")
        print(f"   Used by: {usage}\n")
    else:
        print(f"❌ {key} - NOT FOUND OR INVALID")
        print(f"   Needed by: {usage}\n")
        all_loaded = False

print("="*80)
if all_loaded:
    print("✅ ALL API KEYS LOADED SUCCESSFULLY")
    print(f"✅ Loaded {len(loaded_keys)} total keys from: {env_file}")
    print("✅ Ready to initialize agents")
else:
    print("❌ SOME API KEYS MISSING OR INVALID")
    print("⚠️ Agent execution will fail without all 3 keys")
print("="*80)

代码单元格 <dROKLbj1DAaf>
# %% [code]
# Install dependencies
!pip install -q anthropic openai google-generativeai python-dotenv pyyaml mlflow tiktoken
!pip install -q torch torchvision transformers open_clip_torch pillow matplotlib seaborn

print("✅ Dependencies installed")

文本单元格 <ZOndLm4fDAaf>
# %% [markdown]
---
## Phase 1: Read Pending Actions from Planning Team

代码单元格 <mBg8Ijx8DAag>
# %% [code]
import json
from datetime import datetime
from pathlib import Path

# Read pending actions from Planning Team
pending_actions_file = Path(MULTI_AGENT_ROOT) / 'reports/handoff/pending_actions.json'

if not pending_actions_file.exists():
    print(f"❌ No pending actions found at {pending_actions_file}")
    print("⚠️ Planning Team must generate pending_actions.json first")
    raise FileNotFoundError(f"Missing: {pending_actions_file}")

with open(pending_actions_file, 'r') as f:
    pending_actions = json.load(f)

print("="*80)
print("📥 PENDING ACTIONS FROM PLANNING TEAM")
print("="*80)
print(f"\n📋 Meeting ID: {pending_actions.get('meeting_id')}")
print(f"🗓️  Generated: {pending_actions.get('generated_at')}")
print(f"🎯 Context: {pending_actions.get('context')}")

decisions = pending_actions.get('decisions', [])
print(f"\n📊 Total tasks: {len(decisions)}")

# Group by priority
high_priority = [d for d in decisions if d.get('priority') == 'HIGH']
medium_priority = [d for d in decisions if d.get('priority') == 'MEDIUM']
low_priority = [d for d in decisions if d.get('priority') == 'LOW']

print(f"   ⭐ HIGH: {len(high_priority)}")
print(f"   🟠 MEDIUM: {len(medium_priority)}")
print(f"   🔵 LOW: {len(low_priority)}")

print("\n📋 Task List:")
for i, decision in enumerate(decisions, 1):
    priority = decision.get('priority', 'UNKNOWN')
    action = decision.get('action', 'No action specified')
    owner = decision.get('owner', 'unassigned')
    deadline = decision.get('deadline', 'No deadline')

    priority_icon = '⭐' if priority == 'HIGH' else '🟠' if priority == 'MEDIUM' else '🔵'
    print(f"\n{i}. {priority_icon} [{priority}] {action}")
    print(f"   👤 Owner: {owner}")
    print(f"   ⏰ Deadline: {deadline}")

print("\n" + "="*80)

文本单元格 <gpcBCXq4DAag>
# %% [markdown]
---
## Phase 2: Initialize Executive Team Agents

代码单元格 <YdhJTYuEDAag>
# %% [code]
# Import multi-agent system components
os.chdir(MULTI_AGENT_ROOT)

from agents.roles import Agent, AgentConfig, AgentTeam
from agents.router import AgentRouter, RoutingStrategy, Message
from tools.file_bridge import FileBridge, create_default_policies

print("✅ Multi-agent system imported")

# Initialize Executive Team (3 agents)
executive_team_agents = {}
prompt_dir = Path(MULTI_AGENT_ROOT) / 'agents/prompts/executive_team'

# Define Executive Team configuration
executive_config = {
    'ops_commander': {
        'name': 'Ops Commander',
        'model': 'claude-sonnet-4-20250514',
        'provider': 'anthropic',
        'role': 'Execute research experiments and deployments',
        'prompt_file': '02_ops_commander.md'
    },
    'quality_safety': {
        'name': 'Quality & Safety Officer',
        'model': 'claude-sonnet-4-20250514',
        'provider': 'anthropic',
        'role': 'Ensure code quality, safety, and reproducibility',
        'prompt_file': '01_quality_safety_officer.md'
    },
    'infrastructure': {
        'name': 'Infrastructure & Performance Monitor',
        'model': 'claude-sonnet-4-20250514',
        'provider': 'anthropic',
        'role': 'Monitor infrastructure and performance',
        'prompt_file': '03_infrastructure_performance_monitor.md'
    }
}

print("\n🤖 Initializing Executive Team:")
for agent_id, config in executive_config.items():
    agent_cfg = AgentConfig(
        name=config['name'],
        model=config['model'],
        provider=config['provider'],
        role=config['role'],
        prompt_file=config['prompt_file']
    )
    executive_team_agents[agent_id] = Agent(agent_cfg, prompt_dir)
    print(f"   ✅ {config['name']} ({config['model']})")

# Create agent team and router
executive_team = AgentTeam(executive_team_agents)
executive_router = AgentRouter(executive_team)

print("\n✅ Executive Team initialized (3 agents)")

文本单元格 <NbqWF2t2DAag>
# %% [markdown]
---
## Phase 3: Execute Tasks in Priority Order

代码单元格 <HdL7TSQkDAag>
# %% [code]
# Task execution tracker
class TaskExecutionTracker:
    def __init__(self):
        self.task_results = []
        self.start_time = datetime.now()
        self.current_task = None

    def start_task(self, task_id, action, priority):
        self.current_task = {
            'task_id': task_id,
            'action': action,
            'priority': priority,
            'status': 'in_progress',
            'start_time': datetime.now().isoformat(),
            'outputs': [],
            'errors': [],
            'agent_responses': {}
        }
        print(f"\n🚀 Starting Task {task_id}: {action}")
        print(f"   Priority: {priority}")
        print(f"   Started: {self.current_task['start_time']}")

    def log_agent_response(self, agent_name, response):
        if self.current_task:
            self.current_task['agent_responses'][agent_name] = response
            print(f"   ✅ {agent_name} responded ({len(response)} chars)")

    def log_output(self, output_type, content, file_path=None):
        if self.current_task:
            self.current_task['outputs'].append({
                'type': output_type,
                'content': content,
                'file_path': file_path,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   📄 Output: {output_type}" + (f" → {file_path}" if file_path else ""))

    def log_error(self, error_msg):
        if self.current_task:
            self.current_task['errors'].append({
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   ❌ Error: {error_msg}")

    def complete_task(self, status='completed'):
        if self.current_task:
            self.current_task['status'] = status
            self.current_task['end_time'] = datetime.now().isoformat()
            self.task_results.append(self.current_task)

            duration = (datetime.fromisoformat(self.current_task['end_time']) -
                       datetime.fromisoformat(self.current_task['start_time'])).total_seconds()

            print(f"   ✅ Task completed in {duration:.1f}s")
            print(f"   Status: {status}")
            print(f"   Outputs: {len(self.current_task['outputs'])}")
            print(f"   Errors: {len(self.current_task['errors'])}")

            self.current_task = None

    def get_summary(self):
        completed = len([t for t in self.task_results if t['status'] == 'completed'])
        failed = len([t for t in self.task_results if t['status'] == 'failed'])
        total_duration = (datetime.now() - self.start_time).total_seconds()

        return {
            'total_tasks': len(self.task_results),
            'completed': completed,
            'failed': failed,
            'total_duration_seconds': total_duration,
            'task_results': self.task_results
        }

tracker = TaskExecutionTracker()
print("✅ Task execution tracker initialized")

代码单元格 <TW2nPw6GDAah>
# %% [code]
# Execute tasks in priority order
print("="*80)
print("🚀 EXECUTIVE TEAM EXECUTION - WEEK 1 TASKS")
print("="*80)

# Sort decisions by priority (HIGH → MEDIUM → LOW)
priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
sorted_decisions = sorted(decisions, key=lambda x: priority_order.get(x.get('priority', 'LOW'), 3))

for task_id, decision in enumerate(sorted_decisions, 1):
    action = decision.get('action', 'No action specified')
    priority = decision.get('priority', 'MEDIUM')
    owner = decision.get('owner', 'ops_commander')
    rationale = decision.get('rationale', 'No rationale provided')
    acceptance_criteria = decision.get('acceptance_criteria', [])
    dependencies = decision.get('dependencies', [])

    # Start task tracking
    tracker.start_task(f"task_{task_id}", action, priority)

    # Check dependencies
    if dependencies:
        print(f"\n⚠️ Task has dependencies: {dependencies}")
        # In a full system, check if dependencies are completed

    # Create task context for agents
    task_context = f"""# EXECUTIVE TEAM TASK EXECUTION

## Task Details
**Task ID:** {task_id}
**Priority:** {priority}
**Action:** {action}
**Owner:** {owner}
**Deadline:** {decision.get('deadline', 'No deadline')}

## Rationale
{rationale}

## Acceptance Criteria
{''.join([f"- {criterion}\n" for criterion in acceptance_criteria])}

## Evidence Paths
{', '.join(decision.get('evidence_paths', []))}

## Your Role
You are the **{owner}** agent. Execute this task following these guidelines:

1. **Real Deployment Tools:** Use actual Python code, MLflow tracking, GPU resources
2. **Evidence-Based:** All claims must cite file paths and run_ids
3. **Reproducible:** Document all steps, save all outputs
4. **Report Progress:** Provide clear status updates

## Expected Output Format

Provide:
1. **Status:** COMPLETED / IN_PROGRESS / BLOCKED / FAILED
2. **Work Done:** What was accomplished
3. **Outputs Generated:** Files created, experiments run, metrics collected
4. **Evidence:** File paths, MLflow run_ids, line numbers
5. **Next Steps:** What remains (if not completed)
6. **Blockers:** Any issues encountered

Execute the task now.
"""

    # Route task to appropriate agent
    try:
        message = Message(
            sender="executive_coordinator",
            recipients=[owner] if owner in executive_team_agents else list(executive_team_agents.keys()),
            content=task_context,
            message_type="task"
        )

        # Get agent responses
        responses = executive_router.route_message(message)

        # Log all agent responses
        for agent_name, response in responses.items():
            tracker.log_agent_response(agent_name, response)

            # Display response preview
            print(f"\n📝 Response from {agent_name}:")
            print(f"{response[:500]}..." if len(response) > 500 else response)

        # Mark task as completed (in real system, parse response for actual status)
        tracker.complete_task('completed')

    except Exception as e:
        tracker.log_error(str(e))
        tracker.complete_task('failed')
        print(f"\n❌ Task failed: {e}")

    print("\n" + "-"*80)

print("\n" + "="*80)
print("✅ EXECUTIVE TEAM EXECUTION COMPLETE")
print("="*80)

文本单元格 <DUc7iU-kDAah>
# %% [markdown]
---
## Phase 4: Generate Progress Report for Planning Team

代码单元格 <Q1J9Y-StDAah>
# %% [code]
# Generate execution progress update
summary = tracker.get_summary()

progress_report = f"""# Executive Team Progress Update

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Mission:** CVPR 2025 Week 1 - Cross-architecture attention collapse validation
**Meeting ID:** {pending_actions.get('meeting_id', 'unknown')}

---

## 📊 Execution Summary

**Total Tasks:** {summary['total_tasks']}
**Completed:** {summary['completed']} ✅
**Failed:** {summary['failed']} ❌
**Execution Time:** {summary['total_duration_seconds']:.1f}s

---

## 📋 Task Results

"""

for i, task_result in enumerate(summary['task_results'], 1):
    status_icon = '✅' if task_result['status'] == 'completed' else '❌'
    progress_report += f"""
### Task {i}: {task_result['action']}

**Priority:** {task_result['priority']}
**Status:** {status_icon} {task_result['status'].upper()}
**Duration:** {(datetime.fromisoformat(task_result['end_time']) - datetime.fromisoformat(task_result['start_time'])).total_seconds():.1f}s

**Agent Responses:**
"""

    for agent_name, response in task_result['agent_responses'].items():
        progress_report += f"""
#### {agent_name}
```
{response[:1000]}{'...' if len(response) > 1000 else ''}
```
"""

    if task_result['outputs']:
        progress_report += "\n**Outputs:**\n"
        for output in task_result['outputs']:
            progress_report += f"- {output['type']}: {output.get('file_path', 'N/A')}\n"

    if task_result['errors']:
        progress_report += "\n**Errors:**\n"
        for error in task_result['errors']:
            progress_report += f"- {error['message']}\n"

    progress_report += "\n---\n"

progress_report += f"""
## 🎯 Week 1 Progress Toward GO/NO-GO Decision

**Target Date:** October 20, 2025

**Validation Goals:**
- [ ] Diagnostic tools work on ≥3 external models
- [ ] Statistical evidence collected (p<0.05 threshold)
- [ ] CLIP diagnostic completed
- [ ] ALIGN diagnostic attempted
- [ ] Results logged to MLflow

**Recommendation:** [TO BE FILLED BY EXECUTIVE TEAM]

---

## 📤 Handoff to Planning Team

**Status:** Executive Team execution cycle complete
**Next Action:** Planning Team review and next cycle planning
**Generated:** {datetime.now().isoformat()}

---

**Cycle Complete:** ✅
**Awaiting:** Manual review before next cycle
"""

# Save progress report
progress_file = Path(MULTI_AGENT_ROOT) / 'reports/handoff/execution_progress_update.md'
progress_file.parent.mkdir(parents=True, exist_ok=True)

with open(progress_file, 'w') as f:
    f.write(progress_report)

print("="*80)
print("📤 PROGRESS REPORT GENERATED")
print("="*80)
print(f"\n📄 Report saved to: {progress_file}")
print(f"\n{progress_report}")
print("\n" + "="*80)

文本单元格 <mae6QDUcDAah>
# %% [markdown]
---
## Phase 5: Auto-Sync to Google Drive

代码单元格 <9ZadMyBGDAah>
# %% [code]
# Generate execution progress update
from datetime import datetime
from pathlib import Path
import shutil

# Generate timestamp FIRST (will be reused for all files)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

summary = tracker.get_summary()

progress_report = f"""# Executive Team Progress Update

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Mission:** CVPR 2025 Week 1 - Cross-architecture attention collapse validation
**Meeting ID:** {pending_actions.get('meeting_id', 'unknown')}

---

## 📊 Execution Summary

**Total Tasks:** {summary['total_tasks']}
**Completed:** {summary['completed']} ✅
**Failed:** {summary['failed']} ❌
**Execution Time:** {summary['total_duration_seconds']:.1f}s

---

## 📋 Task Results

"""

for i, task_result in enumerate(summary['task_results'], 1):
    status_icon = '✅' if task_result['status'] == 'completed' else '❌'
    progress_report += f"""
### Task {i}: {task_result['action']}

**Priority:** {task_result['priority']}
**Status:** {status_icon} {task_result['status'].upper()}
**Duration:** {(datetime.fromisoformat(task_result['end_time']) - datetime.fromisoformat(task_result['start_time'])).total_seconds():.1f}s

**Agent Responses:**
"""

    for agent_name, response in task_result['agent_responses'].items():
        progress_report += f"""
#### {agent_name}
```
{response[:1000]}{'...' if len(response) > 1000 else ''}
```
"""

    if task_result['outputs']:
        progress_report += "\n**Outputs:**\n"
        for output in task_result['outputs']:
            progress_report += f"- {output['type']}: {output.get('file_path', 'N/A')}\n"

    if task_result['errors']:
        progress_report += "\n**Errors:**\n"
        for error in task_result['errors']:
            progress_report += f"- {error['message']}\n"

    progress_report += "\n---\n"

progress_report += f"""
## 🎯 Week 1 Progress Toward GO/NO-GO Decision

**Target Date:** October 20, 2025

**Validation Goals:**
- [ ] Diagnostic tools work on ≥3 external models
- [ ] Statistical evidence collected (p<0.05 threshold)
- [ ] CLIP diagnostic completed
- [ ] ALIGN diagnostic attempted
- [ ] Results logged to MLflow

**Recommendation:** [TO BE FILLED BY EXECUTIVE TEAM]

---

## 📤 Handoff to Planning Team

**Status:** Executive Team execution cycle complete
**Next Action:** Planning Team review and next cycle planning
**Generated:** {datetime.now().isoformat()}

---

**Cycle Complete:** ✅
**Awaiting:** Manual review before next cycle
"""

# ============================================================
# FIX: Save with TIMESTAMP in filename
# ============================================================

# Save timestamped progress report to handoff directory
progress_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/execution_progress_update_{timestamp}.md'
progress_file.parent.mkdir(parents=True, exist_ok=True)

with open(progress_file, 'w') as f:
    f.write(progress_report)

print("="*80)
print("📤 PROGRESS REPORT GENERATED")
print("="*80)
print(f"\n✅ Report saved to: {progress_file}")
print(f"   📄 Filename: execution_progress_update_{timestamp}.md")
print(f"   📊 Tasks: {summary['completed']}/{summary['total_tasks']} completed")
print(f"   ⏱️  Duration: {summary['total_duration_seconds']:.1f}s")

# Also save to summaries directory for backup
summary_file = Path(MULTI_AGENT_ROOT) / f'reports/execution/summaries/execution_summary_{timestamp}.md'
summary_file.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(progress_file, summary_file)

print(f"\n✅ Backup saved to: {summary_file}")

# Save task results as JSON for programmatic access
results_json = Path(MULTI_AGENT_ROOT) / f'reports/execution/results/execution_results_{timestamp}.json'
results_json.parent.mkdir(parents=True, exist_ok=True)

with open(results_json, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Results JSON saved: {results_json}")

print("\n" + "="*80)
print(f"✅ ALL FILES SAVED WITH TIMESTAMP: {timestamp}")
print("="*80)

代码单元格 <hf118SYtAncf>
# %% [code]
# ============================================================
# PHASE 5.5: EVIDENCE VERIFICATION (CRITICAL - DO NOT SKIP)
# ============================================================

print("="*80)
print("🔍 EVIDENCE VERIFICATION - CHECKING TASK COMPLETION CLAIMS")
print("="*80)
import mlflow
from pathlib import Path
import json

def verify_task_evidence(task_id, task_name, task_result):
    """Verify all evidence exists for a completed task"""
    print(f"\n🔍 Verifying Task {task_id}: {task_name}")

    missing_evidence = []

    # Check for MLflow run_id in agent responses
    has_mlflow = False
    for agent_name, response in task_result.get('agent_responses', {}).items():
        if 'run_id' in response.lower() or 'mlflow' in response.lower():
            # Try to extract run_id
            import re
            run_ids = re.findall(r'run_id[:\s]+([a-z0-9]+)', response, re.IGNORECASE)
            if run_ids:
                run_id = run_ids[0]
                try:
                    run = mlflow.get_run(run_id)
                    print(f"   ✅ MLflow Run: {run_id}")
                    has_mlflow = True
                except:
                    print(f"   ❌ MLflow Run INVALID: {run_id}")
                    missing_evidence.append(f"MLflow run {run_id} not found")

    if not has_mlflow:
        print(f"   ❌ No MLflow run_id found in responses")
        missing_evidence.append("No MLflow tracking")

    # Check for results files mentioned in outputs
    expected_dirs = [
        'runs/temporal_stability',
        'runs/intervention',
        'runs/ablation',
        'runs/coca_diagnostic',
        'analysis',
        'figures'
    ]

    files_found = 0
    for output in task_result.get('outputs', []):
        filepath = output.get('file_path')
        if filepath:
            path = Path(MULTI_AGENT_ROOT) / filepath
            if path.exists():
                size = path.stat().st_size
                print(f"   ✅ File: {filepath} ({size} bytes)")
                files_found += 1
            else:
                print(f"   ❌ File MISSING: {filepath}")
                missing_evidence.append(filepath)

    if files_found == 0:
        print(f"   ⚠️  No results files documented")
        missing_evidence.append("No results files verified")

    # Verdict
    if missing_evidence:
        print(f"\n   ❌ VERIFICATION FAILED: {len(missing_evidence)} issues")
        for issue in missing_evidence:
            print(f"      - {issue}")
        return False
    else:
        print(f"\n   ✅ VERIFICATION PASSED")
        return True

# Verify all completed tasks
print("\n" + "="*80)
print("📋 VERIFYING ALL COMPLETED TASKS")
print("="*80)

verification_failures = []

for i, task in enumerate(tracker.tasks, 1):
    if task.status == "completed":
        verified = verify_task_evidence(
            i,
            task.action,
            task
        )

        if not verified:
            verification_failures.append(i)
            # CRITICAL: Downgrade task status to FAILED
            task.status = "failed"
            print(f"   ⚠️  Task {i} downgraded to FAILED due to missing evidence")

# Final verdict
print("\n" + "="*80)
if verification_failures:
    print(f"❌ EVIDENCE VERIFICATION FAILED")
    print(f"   {len(verification_failures)} tasks lack required evidence")
    print(f"   Failed tasks: {verification_failures}")
    print(f"\n⚠️  CYCLE MARKED AS FAILED DUE TO INTEGRITY VIOLATIONS")
    print(f"\n🚨 ACTION REQUIRED:")
    print(f"   1. Review failed tasks in execution summary")
    print(f"   2. Do NOT proceed to Planning Team meeting")
    print(f"   3. Re-execute failed tasks with proper evidence logging")
else:
    print(f"✅ EVIDENCE VERIFICATION PASSED")
    completed_count = len([t for t in tracker.tasks if t.status == 'completed'])
    print(f"   All {completed_count} completed tasks have verified evidence")
print("="*80)

文本单元格 <iPOGGbSmDAah>
# %% [markdown]
---
## Phase 6: Trigger Next Planning Team Meeting (Manual Checkpoint)

代码单元格 <6PAzqVucy04g>
# %% [code]
# Create trigger for next Planning Team meeting
import json
from datetime import datetime
from pathlib import Path

print("="*80)
print("⏸️  MANUAL CHECKPOINT - READY FOR PLANNING TEAM REVIEW")
print("="*80)

# Get execution summary
summary = tracker.get_summary()

# Create next meeting trigger
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

next_meeting_trigger = {
    'trigger_type': 'executive_team_complete',
    'timestamp': datetime.now().isoformat(),
    'cycle_number': 1,  # Increment this for each cycle
    'executive_summary': {
        'total_tasks': summary['total_tasks'],
        'completed': summary['completed'],
        'failed': summary['failed'],
        'duration_seconds': summary['total_duration_seconds']
    },
    'next_meeting': {
        'team': 'planning',
        'purpose': 'Review Week 1 execution results and plan next cycle',
        'required_inputs': [
            f'execution_progress_update_{timestamp}.md',
            f'execution_results_{timestamp}.json'
        ],
        'agenda': [
            'Review task completion status',
            'Assess progress toward Week 1 GO/NO-GO criteria',
            'Identify blockers and risks',
            'Plan next cycle tasks (if needed)',
            'Generate new pending_actions.json'
        ]
    },
    'manual_checkpoint': True,
    'checkpoint_message': '⏸️ MANUAL REVIEW REQUIRED: Check execution results before starting next Planning Team meeting'
}

# Save trigger file with timestamp
trigger_file = Path(MULTI_AGENT_ROOT) / f'reports/handoff/next_meeting_trigger_{timestamp}.json'
trigger_file.write_text(json.dumps(next_meeting_trigger, indent=2), encoding='utf-8')

print(f"\n✅ Next meeting trigger saved:")
print(f"   📄 File: next_meeting_trigger_{timestamp}.json")
print(f"   📊 Tasks completed: {summary['completed']}/{summary['total_tasks']}")
print(f"   ⏱️  Duration: {summary['total_duration_seconds']:.1f}s ({summary['total_duration_seconds']/60:.1f} minutes)")

print(f"\n📋 Planning Team Agenda:")
for i, item in enumerate(next_meeting_trigger['next_meeting']['agenda'], 1):
    print(f"   {i}. {item}")

print(f"\n⏸️  MANUAL CHECKPOINT:")
print(f"   ⚠️  {next_meeting_trigger['checkpoint_message']}")
print(f"\n🔄 Next Steps:")
print(f"   1. Review execution results in reports/handoff/")
print(f"   2. If satisfied, run Planning Team notebook: planning_team_review_cycle.ipynb")
print(f"   3. Planning Team will read these results and plan next cycle")
print(f"   4. Execute next cycle with updated pending_actions.json")

print("\n" + "="*80)
print("✅ EXECUTIVE TEAM CYCLE COMPLETE")
print("="*80)
print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results saved to: reports/handoff/")
print(f"Ready for Planning Team review")
print("="*80)



文本单元格 <wMWFXl2oDAah>
# %% [markdown]
---
## Summary Dashboard

代码单元格 <9sLlOQTlDAai>
# %% [code]
import matplotlib.pyplot as plt
import seaborn as sns

# Create summary dashboard
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Task completion pie chart
completion_data = [summary['completed'], summary['failed']]
completion_labels = ['Completed', 'Failed']
colors = ['#4CAF50', '#F44336']

axes[0].pie(completion_data, labels=completion_labels, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[0].set_title('Task Completion Status')

# Priority distribution
priority_counts = {}
for task in summary['task_results']:
    priority = task.get('priority', 'UNKNOWN')
    priority_counts[priority] = priority_counts.get(priority, 0) + 1

axes[1].bar(priority_counts.keys(), priority_counts.values(), color=['#F44336', '#FF9800', '#2196F3'])
axes[1].set_title('Tasks by Priority')
axes[1].set_ylabel('Number of Tasks')

# Execution time
task_durations = []
task_names = []
for i, task in enumerate(summary['task_results'], 1):
    duration = (datetime.fromisoformat(task['end_time']) -
               datetime.fromisoformat(task['start_time'])).total_seconds()
    task_durations.append(duration)
    task_names.append(f"Task {i}")

axes[2].barh(task_names, task_durations, color='#4CAF50')
axes[2].set_title('Task Execution Time (seconds)')
axes[2].set_xlabel('Duration (s)')

plt.tight_layout()
plt.savefig(Path(MULTI_AGENT_ROOT) / f'reports/execution/results/execution_dashboard_{timestamp}.png',
            dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Dashboard saved to: reports/execution/results/execution_dashboard_{timestamp}.png")


