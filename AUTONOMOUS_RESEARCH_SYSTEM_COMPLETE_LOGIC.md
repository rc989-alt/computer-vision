# Autonomous Multi-Agent Research System
## Complete Architecture & Logic Documentation

**System Name:** CVPR 2025 Autonomous Research System
**Version:** 3.0 (Production)
**Date:** 2025-10-18
**Purpose:** Complete architectural documentation for the three-system autonomous research pipeline

---

## 🎯 EXECUTIVE SUMMARY

The **Autonomous Multi-Agent Research System** is a self-improving, reflection-guided platform that autonomously conducts scientific research through the coordination of three integrated subsystems:

### The Three Systems

1. **🧭 PLANNING SYSTEM** - Strategic task decomposition and resource allocation
2. **⚙️ EXECUTIVE SYSTEM** - Multi-agent execution with approval gates and verification
3. **🧠 REFLECTION SYSTEM** - Continuous learning and adaptive self-correction

### Core Innovation: Integrated Autonomy

```
┌─────────────────────────────────────────────────────────┐
│  PLANNING SYSTEM (Strategic Intelligence)               │
│  • Decomposes research goals → executable tasks         │
│  • Estimates resources and timelines                    │
│  • Learns from previous cycle outcomes                  │
└────────────────────┬────────────────────────────────────┘
                     │ pending_actions.json
                     ↓
┌─────────────────────────────────────────────────────────┐
│  EXECUTIVE SYSTEM (Tactical Execution)                  │
│  • 3-agent approval gates (Ops/Quality/Infrastructure)  │
│  • Sandboxed code execution with safety bounds          │
│  • Evidence-based verification (MLflow + artifacts)     │
└────────────────────┬────────────────────────────────────┘
                     ↕ (signals & patches - parallel)
┌─────────────────────────────────────────────────────────┐
│  REFLECTION SYSTEM (Adaptive Intelligence)              │
│  • Pattern mining and root cause analysis               │
│  • Confidence-weighted retry patches                    │
│  • Cross-cycle learning through memory                  │
└─────────────────────────────────────────────────────────┘
```

### Key Capabilities

- ✅ **Autonomous operation** - Runs complete research cycles without human intervention
- ✅ **Self-healing** - Automatically retries with intelligent, reflection-guided patches
- ✅ **Evidence-based** - All claims verified through MLflow runs and file artifacts
- ✅ **Progressive learning** - Gets smarter across cycles via semantic memory
- ✅ **Multi-agent consensus** - Triple-gate approval ensures quality and safety

---

## 📋 DOCUMENT STRUCTURE

This document is organized into four main parts:

- **PART I: PLANNING SYSTEM** - Strategic task management
- **PART II: EXECUTIVE SYSTEM** - Multi-agent execution pipeline (8 phases)
- **PART III: REFLECTION SYSTEM** - Learning and adaptation (R0-R6 phases)
- **PART IV: SYSTEM INTEGRATION** - How the three systems coordinate

Supporting sections include debugging guides, file structures, metrics, and operational procedures.

---

# PART I: PLANNING SYSTEM

The Planning System is the **strategic intelligence layer** that translates high-level research objectives into concrete, executable task specifications.

## 1.1 Overview

**Primary Function:** Strategic decomposition of research goals into prioritized, resource-aware tasks

**Core Responsibilities:**
- Analyze research milestones and current progress
- Decompose high-level objectives into executable units
- Estimate resource requirements (GPU hours, memory, storage)
- Assign task priorities based on critical path analysis
- Resolve inter-task dependencies
- Learn from previous execution cycle outcomes

**Key Output:** `pending_actions.json` - Validated task specifications with full context

---

## 1.2 Planning → Execution Cycle Model

The system operates on a **dual-cycle feedback loop**:

```
┌──────────────────────────────────────────────┐
│ PLANNING CYCLE N (Strategic)                 │
│  1. Review previous cycle results            │
│  2. Assess progress toward milestones        │
│  3. Identify blockers and opportunities      │
│  4. Decompose next phase objectives          │
│  5. Generate pending_actions.json            │
└───────────────┬──────────────────────────────┘
                │
                ↓
┌──────────────────────────────────────────────┐
│ EXECUTION CYCLE N (Tactical)                 │
│  1. Initialize executive agents              │
│  2. Execute tasks (approval → run → verify)  │
│  3. Collect evidence and metrics             │
│  4. Report results with provenance           │
│  5. Trigger next planning cycle              │
└───────────────┬──────────────────────────────┘
                │
                ↓ (feedback)
┌──────────────────────────────────────────────┐
│ PLANNING CYCLE N+1                           │
│  • Incorporate lessons from Cycle N          │
│  • Adjust resource estimates                 │
│  • Update task priorities                    │
│  • Apply semantic memory rules               │
└──────────────────────────────────────────────┘
```

**Key Principle:** Each execution cycle informs the next planning cycle, creating continuous improvement.

---

## 1.3 Task Decomposition Process

### Strategic Analysis Phase

**Input:** High-level research goal (e.g., "Validate attention collapse hypothesis across architectures")

**Process:**
1. **Milestone Mapping** - Break goal into measurable milestones
2. **Work Breakdown** - Decompose each milestone into atomic tasks
3. **Critical Path Analysis** - Identify which tasks block others
4. **Resource Profiling** - Estimate GPU, time, storage per task

**Example Decomposition:**

```
Research Goal: "CVPR 2025 - Validate attention collapse"
  ↓
Milestone 1: Cross-architecture baseline validation
  ↓
  Task 1.1: CLIP baseline extraction (HIGH)
    - Acceptance: MLflow run + attention maps
    - Resources: A100, 1.5h, 2GB storage
    - Dependencies: None

  Task 1.2: ALIGN diagnostic (HIGH)
    - Acceptance: Diagnostic metrics logged
    - Resources: A100, 2h, 1GB storage
    - Dependencies: None

  Task 1.3: Statistical comparison (MEDIUM)
    - Acceptance: P-value < 0.05
    - Resources: CPU only, 30min
    - Dependencies: Tasks 1.1, 1.2

  Task 1.4: Summary report (LOW)
    - Acceptance: Markdown report generated
    - Resources: CPU only, 15min
    - Dependencies: Task 1.3
```

---

## 1.4 Task Specification Schema

Each task in `pending_actions.json` follows this comprehensive schema:

```json
{
  "task_id": 1,
  "priority": "HIGH",
  "action": "Re-execute Task 1: CLIP Integration with REAL attention extraction and MLflow tracking",
  "owner": "ops_commander",
  "deadline": "2025-10-15T18:00:00Z",
  "rationale": "Required for Week 1 GO/NO-GO decision on attention collapse hypothesis",

  "acceptance_criteria": [
    "CLIP/OpenCLIP model loads successfully on A100 GPU",
    "Real attention extraction (not simulated placeholder data)",
    "MLflow run tracking with valid run_id",
    "Output artifact: baseline_attention.json with ≥100 samples",
    "Execution time < 2 hours"
  ],

  "dependencies": [],

  "estimated_resources": {
    "gpu_type": "A100",
    "gpu_hours": 1.5,
    "memory_gb": 32,
    "storage_gb": 2,
    "parallel_compatible": false
  },

  "context": {
    "previous_attempts": 0,
    "known_blockers": [],
    "reference_docs": ["docs/clip_setup.md"],
    "dataset_path": "/data/coco_val2017",
    "model_checkpoint": "openai/clip-vit-base-patch32"
  }
}
```

**Required Fields:**
- `task_id`, `priority`, `action`, `acceptance_criteria`

**Optional but Recommended:**
- `dependencies`, `estimated_resources`, `context`, `deadline`

---

## 1.5 Priority Assignment Logic

### Priority Levels

**HIGH** - Critical path, blocks other work
- Week 1 milestones
- Experimental validation tasks
- Tasks with external deadlines

**MEDIUM** - Important but not blocking
- Analysis and comparison tasks
- Secondary experiments
- Performance optimization

**LOW** - Nice-to-have, can be deferred
- Documentation
- Code cleanup
- Visualization improvements

### Priority Assignment Algorithm

```python
def assign_priority(task, context):
    priority_score = 0

    # Critical path analysis
    if blocks_other_tasks(task):
        priority_score += 30

    # Deadline urgency
    days_until_deadline = (task.deadline - now()).days
    if days_until_deadline <= 2:
        priority_score += 25

    # Milestone dependency
    if contributes_to_go_no_go_decision(task):
        priority_score += 20

    # External dependencies
    if has_external_dependencies(task):
        priority_score += 15

    # Previous failure recovery
    if is_retry_from_previous_cycle(task):
        priority_score += 10

    # Classify
    if priority_score >= 50:
        return "HIGH"
    elif priority_score >= 25:
        return "MEDIUM"
    else:
        return "LOW"
```

---

## 1.6 Resource Estimation

### GPU Hour Estimation

**Factors:**
- Model size (parameters)
- Dataset size (samples)
- Batch size
- Number of forward/backward passes

**Estimation Formula:**
```python
def estimate_gpu_hours(model_params, num_samples, batch_size):
    # Baseline: 1B params, 1000 samples, batch=32 → ~1 hour
    baseline_time = 1.0

    param_factor = model_params / 1e9
    sample_factor = num_samples / 1000
    batch_factor = 32 / batch_size

    estimated_hours = baseline_time * param_factor * sample_factor * batch_factor

    # Add 20% buffer for overhead
    return estimated_hours * 1.2
```

**Example:**
- CLIP ViT-B/32: 150M params, 500 samples, batch=32 → 0.15 * 0.5 * 1.0 * 1.2 = **0.09 hours (5.4 min)**
- CLIP ViT-L/14: 430M params, 500 samples, batch=16 → 0.43 * 0.5 * 2.0 * 1.2 = **0.52 hours (31 min)**

### Storage Estimation

**Attention Maps:** ~10 KB per sample
**Model Checkpoints:** 500 MB - 2 GB
**Logs:** 1-5 MB per run
**MLflow Metadata:** 50-100 MB per experiment

---

## 1.7 Dependency Resolution

### Dependency Types

1. **Data Dependency** - Task B needs artifacts from Task A
2. **Sequential Logic** - Task B conceptually follows Task A
3. **Resource Conflict** - Tasks A and B need exclusive access to same GPU

### Topological Sort Algorithm

```python
def resolve_dependencies(tasks):
    """Topological sort for dependency-respecting execution order"""

    # Build dependency graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for task in tasks:
        for dep in task.dependencies:
            graph[dep].append(task.task_id)
            in_degree[task.task_id] += 1

    # Kahn's algorithm
    queue = [t.task_id for t in tasks if in_degree[t.task_id] == 0]
    sorted_tasks = []

    while queue:
        current = queue.pop(0)
        sorted_tasks.append(current)

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(sorted_tasks) != len(tasks):
        raise ValueError("Circular dependency detected!")

    return sorted_tasks
```

**Example:**
```
Given: [Task 3 depends on [Task 1, Task 2]]
Output: [Task 1, Task 2, Task 3] (or [Task 2, Task 1, Task 3])
```

---

## 1.8 Adaptive Planning (Learning from Execution)

### Feedback Analysis

After each execution cycle, Planning System analyzes:

1. **Task Completion Metrics**
   - Which tasks completed successfully?
   - Actual vs estimated time/resources
   - Quality of generated evidence

2. **Failure Root Causes**
   - Which tasks failed permanently?
   - Common error patterns
   - Infrastructure issues

3. **Resource Utilization**
   - GPU efficiency
   - Memory peaks
   - Storage consumption

### Planning Adjustments

**Example Feedback Loop:**

```
Cycle 1 Execution Result:
  Task "CLIP baseline"
  - Estimated: 2h, 32GB RAM
  - Actual: 45 min, 28GB RAM
  - Status: Completed

Planning Team Learning:
  → Update estimation model: CLIP ViT-B/32 → 1h, 30GB
  → Next cycle: Reduce buffer, allocate saved time elsewhere
```

```
Cycle 1 Execution Result:
  Task "ALIGN diagnostic"
  - Status: Failed (model unavailable on HuggingFace)
  - Root cause: External dependency blocker

Planning Team Decision:
  → Document blocker in context
  → Add fallback task: "Use CoCa diagnostic instead"
  → Next cycle: Task 5_fallback with updated model path
```

---

## 1.9 Reflective Planning (Meta-Learning)

### Semantic Memory Integration

Planning Team reads from `/memory/semantic.yml` to apply learned patterns:

```yaml
# Example semantic rule
rules:
  - rule_id: "plan_001"
    pattern: "CLIP ViT-L/14 attention extraction on A100"
    observation: "Batch size > 16 causes OOM consistently"
    recommendation:
      action: "set_batch_size_max_12"
      parameters:
        batch_size: 12
    confidence: 0.95
    evidence_count: 5
```

**Application in Planning:**
```python
def generate_task_spec(goal, model, gpu):
    base_spec = create_base_spec(goal, model, gpu)

    # Check semantic memory for applicable rules
    applicable_rules = query_semantic_memory(
        pattern=f"{model} on {gpu}"
    )

    for rule in applicable_rules:
        if rule.confidence > 0.85:
            # Proactively apply learned optimization
            apply_rule_to_spec(base_spec, rule)
            log(f"Applied semantic rule {rule.rule_id}: {rule.recommendation}")

    return base_spec
```

**Result:** Planning becomes progressively smarter, avoiding known pitfalls.

---

## 1.10 Planning Horizon & Milestones

### Week 1 Example: Cross-Architecture Validation

**Research Objective:** GO/NO-GO decision on attention collapse hypothesis

**Planned Milestones:**
```
Week 1 (Oct 14-20): Cross-architecture validation
  ├─ Milestone 1: CLIP baseline complete
  │   └─ Tasks: 1 (HIGH), 4 (HIGH)
  ├─ Milestone 2: ALIGN/CoCa diagnostic
  │   └─ Task: 5 (HIGH)
  ├─ Milestone 3: Statistical validation
  │   └─ Tasks: 6 (MEDIUM), 7 (MEDIUM)
  └─ Milestone 4: Documentation
      └─ Task: 8 (LOW)

Success Criteria:
  - ≥ 3 external models tested
  - Statistical significance (p < 0.05)
  - All results logged to MLflow
  - Evidence artifacts verified
```

### Planning Cadence

- **Daily:** Micro-adjustments (re-prioritize based on blockers)
- **Weekly:** Cycle planning (generate new `pending_actions.json`)
- **Bi-weekly:** Milestone review (assess GO/NO-GO criteria)

---

## 1.11 Planning Quality Metrics

### Key Performance Indicators

**Task Completion Rate:**
```
Completion Rate = (Completed Tasks / Total Planned Tasks) * 100%
Target: ≥ 80% for HIGH priority tasks
```

**Estimation Accuracy:**
```
Time Accuracy = 1 - |Actual - Estimated| / Estimated
Target: ≥ 70% (within 30% of estimate)
```

**Priority Calibration:**
```
Did HIGH priority tasks truly block subsequent work?
Were MEDIUM tasks appropriately sequenced?
```

**Dependency Correctness:**
```
Were all dependencies properly specified?
Were there unplanned blockers?
```

### Continuous Improvement

Planning Team reviews these metrics after each cycle and adjusts:
- Estimation heuristics
- Priority assignment logic
- Dependency analysis depth
- Resource buffer percentages

---

## 1.12 Planning System Outputs

### Primary Output: pending_actions.json

```json
{
  "meeting_id": "cvpr_planning_cycle2_20251015_080000",
  "generated_at": "2025-10-15T08:00:00Z",
  "context": "Week 1 Cycle 2: Complete experimental tasks with real MLflow tracking",
  "decisions": [
    {
      "task_id": 1,
      "priority": "HIGH",
      "action": "Re-execute Task 1: CLIP Integration...",
      "owner": "ops_commander",
      "deadline": "2025-10-15T18:00:00Z",
      "rationale": "...",
      "acceptance_criteria": [...],
      "dependencies": [],
      "estimated_resources": {...}
    },
    // ... more tasks
  ]
}
```

### Supporting Outputs

- `planning_context.md` - Human-readable strategic rationale
- `resource_allocation.json` - Budget breakdown and constraints
- `dependency_graph.png` - Visual task dependencies

---

# PART II: EXECUTIVE SYSTEM

The Executive System is the **tactical execution layer** that transforms planning decisions into verified research outcomes through a multi-agent architecture.

## 2.1 Overview

**Primary Function:** Execute approved tasks with multi-agent consensus and evidence verification

**Core Architecture:**
- **3 Specialized Agents:** Ops Commander, Quality & Safety, Infrastructure Monitor
- **8-Phase Pipeline:** Intake → Init → Implementation → Approval → Execution → Verification → Finalization → Reporting
- **Multi-Level Retries:** Approval (4.5), Execution (5.5), Verification (6.5)

**Key Principle:** No task succeeds without verified evidence (MLflow runs + artifacts)

---

## 2.2 Agent Team Structure

### Agent 1: Ops Commander

**Model:** Claude Sonnet 4
**Primary Role:** Implementation and code generation

**Responsibilities:**
- Generate executable Python code from task specifications
- Structure job specifications (entry point, args, environment)
- Propose resource allocation (GPU, memory, batch size)
- Emit structured JSON contracts with confidence scores

**Output Format:**
```json
{
  "verdict": "APPROVE",
  "confidence": 0.92,
  "proposed_jobs": [{
    "id": "clip_baseline_001",
    "entry": "scripts/run_clip_baseline.py",
    "args": {
      "model_name": "ViT-B/32",
      "batch_size": 32,
      "num_samples": 500
    },
    "expected_artifacts": [
      "baseline_attention.json",
      "metrics.csv"
    ],
    "expected_duration": 300
  }],
  "reasoning": "Task is well-specified with clear acceptance criteria..."
}
```

### Agent 2: Quality & Safety Officer

**Model:** Claude Sonnet 4
**Primary Role:** Code review and compliance validation

**Responsibilities:**
- Review Ops Commander's proposed implementation
- Flag logic errors, edge cases, unsafe operations
- Verify compliance with acceptance criteria
- Assess reproducibility and scientific rigor

**Output Format:**
```json
{
  "verdict": "APPROVE",
  "confidence": 0.88,
  "checks": {
    "logic_sound": true,
    "safety_verified": true,
    "acceptance_criteria_met": true,
    "reproducible": true
  },
  "flags": {
    "warnings": ["Batch size may cause OOM on smaller GPUs"],
    "critical": []
  },
  "reasoning": "Implementation is sound. Minor warning about GPU compatibility."
}
```

### Agent 3: Infrastructure & Performance Monitor

**Model:** Claude Sonnet 4
**Primary Role:** Resource validation and performance bounds

**Responsibilities:**
- Verify environment dependencies (CUDA, PyTorch versions)
- Validate resource availability (GPU memory, disk space)
- Check MLflow tracking configuration
- Assess performance constraints (timeout, memory limits)

**Output Format:**
```json
{
  "verdict": "APPROVE",
  "confidence": 0.95,
  "checks": {
    "env_compatible": true,
    "resources_available": true,
    "mlflow_configured": true,
    "performance_bounded": true
  },
  "resource_validation": {
    "gpu_memory_required": "28GB",
    "gpu_memory_available": "40GB",
    "disk_space_required": "2GB",
    "disk_space_available": "50GB"
  },
  "reasoning": "All infrastructure checks passed. A100 GPU has sufficient resources."
}
```

---

## 2.3 The 8-Phase Execution Pipeline

### Phase 1: Planning Intake

**Cell:** 6
**Input:** `pending_actions.json` from Planning Team
**Output:** Validated task queue

**Process:**
1. Load `pending_actions.json`
2. Validate schema and required fields
3. Sort tasks by priority (HIGH → MEDIUM → LOW)
4. Initialize task tracking structure
5. Create reflection memory anchor

**Reflection Hook:** Signal subscription begins (R0)

---

### Phase 2: Agent Initialization

**Cell:** 8
**Input:** Task queue from Phase 1
**Output:** Initialized 3-agent team

**Process:**
1. Load agent configurations from `/agents/prompts/executive_team/`
2. Initialize API clients (Anthropic, OpenAI, Google)
3. Verify API keys and MLflow tracking URI
4. Perform agent readiness handshake
5. Create coordination bus for inter-agent messaging

**Key Feature:** All agents emit structured JSON (not free text)

---

### Phase 3: Task Implementation & Response Collection

**Cells:** 9-10
**Input:** Prioritized task from queue
**Output:** Multi-agent contracts (JSON)

**Process:**
For each task (in priority order):

1. **Dispatch** - Broadcast task to all 3 agents
2. **Ops Commander** - Generates implementation plan (code + job spec)
3. **Quality & Safety** - Reviews logic and flags issues
4. **Infrastructure** - Validates resources and environment
5. **Contract Consolidation** - Combine into structured response

**Important:** No code execution occurs in this phase—pure reasoning

**Contract Schema:**
```json
{
  "task_id": 1,
  "contracts": {
    "ops_commander": { "verdict": "APPROVE", ... },
    "quality_safety": { "verdict": "APPROVE", ... },
    "infrastructure": { "verdict": "APPROVE", ... }
  },
  "preliminary_status": null
}
```

**Reflection Signal:** Agents' reasoning and divergences logged to `reflection_buffer.jsonl`

---

### Phase 4: Approval Gate

**Cell:** 11
**Input:** Task contracts from Phase 3
**Output:** Preliminary approval status

**Decision Logic:**
```python
def determine_approval(ops, quality, infra):
    if (ops.verdict == "APPROVE" and
        quality.verdict == "APPROVE" and
        infra.verdict == "APPROVE" and
        len(ops.flags.critical) == 0 and
        len(quality.flags.critical) == 0 and
        len(infra.flags.critical) == 0):
        return "approved"
    else:
        return "rejected"
```

**Key Innovation:** Uses explicit JSON field parsing—no text pattern matching

**Outcomes:**
- `"approved"` → Proceed to Phase 5 (Execution)
- `"rejected"` → Route to Phase 4.5 (Approval Retry)

**Reflection Output:** Rejected tasks logged with reasons to `reflections.jsonl`

---

### Phase 4.5: Approval Retry

**Cell:** 11 (integrated)
**Trigger:** `preliminary_status == "rejected"`
**Input:** Reflection artifacts (`job_spec_patch.json`, `risk_scores.json`)

**Process:**

1. **Eligibility Check**
   - Reflection confidence ≥ 0.70?
   - Retry attempts < MAX_RETRIES (2)?

2. **Patch Application**
   - Load patch from Reflection System
   - Merge changes into job spec (sandboxed)
   - Validate safety (no permission expansion)

3. **Re-Submission**
   - Send patched contract back to 3 agents
   - Run approval gate again

4. **Outcome Handling**
   - Approved → Status = `"approved_retry"`, proceed to Phase 5
   - Rejected again → Status = `"failed_retry"`, skip to Phase 7

**Safety Guardrails:**
- No permission scope expansion
- No network/domain changes
- No write-path widening
- Must change job spec (no identical retries)

**Provenance:** Every retry logged with `attempt_id`, `patch_id`, checksums

---

### Phase 5: Execution (Code Executor)

**Cell:** 11.5 (Cell 13 in v2)
**Input:** Tasks with `preliminary_status == "approved"`
**Output:** Execution results + MLflow run_ids

**Process:**

For each approved task:

1. **Filter** - Skip if not approved
2. **Extract Code** - Parse Python blocks from Ops Commander response
3. **Sandbox Setup** - Create isolated subprocess
4. **Execute Code**
   - Apply timeout (300s default)
   - Stream stdout/stderr to logs
   - Capture MLflow run_id from output
5. **Result Capture**
   - Status: `SUCCEEDED`, `FAILED`, or `RETRYABLE_FAILURE`
   - Artifacts: List of generated files
   - Metrics: Execution time, GPU usage

**Safety Features:**
- No direct `exec()` in notebook kernel
- Sandboxed environment with resource quotas
- Output captured in `StringIO` (no kernel pollution)

**Example Execution:**
```python
def safe_execute_code(code, task_id):
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        exec_globals = {
            '__name__': '__main__',
            'MULTI_AGENT_ROOT': MULTI_AGENT_ROOT,
            'Path': Path
        }

        exec(code, exec_globals)

        return {
            'success': True,
            'stdout': stdout_capture.getvalue(),
            'run_id': extract_mlflow_run_id(stdout_capture.getvalue())
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'stderr': stderr_capture.getvalue()
        }
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
```

**Reflection Signal:** Live telemetry (logs, exit codes, MLflow events) streamed to Reflection System

---

### Phase 5.5: Execution Retry

**Cell:** 11.5 (integrated)
**Trigger:** Execution status ∈ {`FAILED`, `RETRYABLE_FAILURE`}
**Input:** Reflection patch (`job_spec_patch.json`)

**Process:**

1. **Collect Reflection**
   - Root cause diagnosis
   - Proposed fix (e.g., reduce batch size)
   - Confidence score

2. **Eligibility**
   - Confidence ≥ 0.70?
   - Attempts < MAX_EXEC_RETRIES (2)?

3. **Apply Patch**
   - Merge patch into job spec
   - Rebuild sandbox with corrected config

4. **Retry Execution**
   - Run with same safety constraints
   - Capture new telemetry

5. **Outcome**
   - Success → `RETRY_SUCCEEDED`, proceed to Phase 6
   - Failure → `RETRY_FAILED`, mark for Phase 7

**Regression Check:** Compare new vs old metrics—rollback if regression detected

---

### Phase 6: Evidence Verification

**Cell:** 17
**Input:** All executed tasks (original + retried)
**Output:** Verification verdicts (`pass` / `fail`)

**Task-Type Aware Logic:**

**Experimental Tasks** (keywords: execute, run, diagnostic, GPU, model):
```python
REQUIRES:
  - MLflow run_id (verified via mlflow.get_run())
  - Expected artifacts (file exists + size > 0)
```

**Documentation Tasks** (keywords: write, design, draft, doc):
```python
REQUIRES:
  - Output files (verified via Path.exists())
  - MLflow optional
```

**Verification Steps:**

1. **Classify Task Type**
   - Check action keywords
   - Determine evidence requirements

2. **Verify MLflow** (if required)
   - Query `mlflow.get_run(run_id)`
   - Check params, metrics, tags

3. **Verify Artifacts**
   - Check file existence
   - Validate size > 0
   - (Optional) Checksum validation

4. **Combine Results**
   - All required evidence present → `verification = "pass"`
   - Any missing → `verification = "fail"`

5. **Emit Reflection**
   - Log missing evidence patterns
   - Trigger Phase 6.5 if eligible

**Example:**
```python
def verify_task_evidence(task_id, task_action, task_result):
    requires_mlflow = should_require_mlflow(task_action)

    missing = []

    # Check MLflow
    if requires_mlflow:
        run_id = extract_run_id(task_result)
        if not run_id or not mlflow_run_exists(run_id):
            missing.append("MLflow run_id")

    # Check artifacts
    for artifact in task_result.expected_artifacts:
        if not Path(artifact).exists():
            missing.append(f"Artifact: {artifact}")

    if missing:
        return {"verification": "fail", "missing": missing}
    else:
        return {"verification": "pass"}
```

---

### Phase 6.5: Verification Retry

**Cell:** 17 (integrated)
**Trigger:** `verification == "fail"`
**Purpose:** Recover from artifact/logging issues without re-execution

**Process:**

1. **Diagnose**
   - What evidence is missing?
   - MLflow run vs artifacts?

2. **Remediate**
   - Re-query MLflow (may have been delayed)
   - Regenerate manifest files
   - Rebuild missing artifacts from cached logs

3. **Re-Verify**
   - Run Phase 6 checks again
   - If pass → `verification_passed_after_retry`
   - If fail → increment attempts

4. **Limit**
   - MAX_VERIF_RETRIES = 2
   - Beyond that → terminal failure

**Key Insight:** Often artifacts exist but weren't linked—lightweight recovery

---

### Phase 7: Final Status Determination

**Cell:** 17 (after verification)
**Input:** `preliminary_status` + `verification` + retry summaries
**Output:** Terminal `final_status`

**Status Rules (ordered priority):**

```python
if preliminary_status == "rejected":
    final_status = "failed"
    reason = "Did not pass 3-agent approval gate"

elif verification == "pass":
    final_status = "completed"
    reason = "Approved + evidence verified"

elif verification == "fail" and retry_exhausted:
    final_status = "failed_final"
    reason = "Evidence missing after max retries"

elif verification == "fail":
    final_status = "failed"
    reason = "Approved but execution failed (no evidence)"

else:
    final_status = "failed"
    reason = "Unresolved state"
```

**Provenance Record:**
```json
{
  "task_id": 1,
  "status": "completed",
  "status_reason": "Approved + evidence verified",
  "attempts_total": 1,
  "provenance": {
    "approval": {"attempts": 1, "approved_after_patch": false},
    "execution": {"attempts": 1},
    "verification": {"attempts": 1},
    "mlflow_run_ids": ["abc123"],
    "artifacts": ["baseline_attention.json"],
    "reflection_digest_id": "ref_20251018_001"
  }
}
```

---

### Phase 8: Reporting & Handoff

**Cells:** 14, 16, 19
**Input:** Finalized task results
**Output:** Progress reports + next cycle trigger

**Process:**

1. **Aggregate Results**
   - Compute KPIs (completion rate, retry success rate)
   - Join with MLflow metadata and artifacts

2. **Generate Reports**
   - `execution_summary_{timestamp}.md` (human-readable)
   - `execution_results_{timestamp}.json` (machine-readable)
   - `reflection_summary.md` (patterns and learnings)

3. **Archive Artifacts**
   - Bundle logs to tar.gz with checksums
   - Snapshot ledgers (immutable)
   - Record retention policy

4. **Trigger Next Planning Cycle**
   - Create `next_meeting_trigger_{timestamp}.json`
   - Include task outcomes and blockers
   - Notify Planning Team

**Timestamp Consistency:** All Phase 8 files use single timestamp for traceability

---

## 2.4 Execution Metrics

### Success Criteria

**For Individual Task:**
- ✅ All 3 agents approved (Phase 4 or 4.5)
- ✅ Code executed successfully (Phase 5 or 5.5)
- ✅ Evidence verified (Phase 6 or 6.5)
- ✅ No critical flags
- ✅ Within retry budget

**For Execution Cycle:**
- ✅ ≥70% HIGH priority tasks completed
- ✅ All completed tasks have verified evidence
- ✅ No integrity violations
- ✅ Progress report generated

---

# PART III: REFLECTION SYSTEM

The Reflection System is the **adaptive intelligence layer** that enables continuous learning and self-correction across execution cycles.

## 3.1 Overview

**Primary Function:** Observe, learn, and adapt in real-time

**Core Capabilities:**
- Pattern detection across failures
- Root cause analysis
- Patch generation for retries
- Memory management (episodic + semantic + procedural)

**Key Principle:** Every failure is a learning opportunity

---

## 3.2 Architecture

```
┌───────────────────────────────────────────────────┐
│           REFLECTION SERVICE                      │
│         (Non-Blocking Parallel Layer)             │
├───────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  Signal  │→ │  Pattern  │→ │  Patch &     │  │
│  │  Listen  │  │  Mining   │  │  Risk Score  │  │
│  │ (R0-R1)  │  │ (R2-R3)   │  │  (R4-R5)     │  │
│  └──────────┘  └───────────┘  └──────────────┘  │
│       ↓              ↓                ↓          │
│  ┌──────────────────────────────────────────┐   │
│  │      Memory Update & Rule Promotion      │   │
│  │            (R6: Learning)                │   │
│  └──────────────────────────────────────────┘   │
│                                                    │
└───────────────────────────────────────────────────┘
```

**Core Principles:**
1. **Non-Blocking** - Never delays execution
2. **Event-Driven** - Triggered by phase signals
3. **Confidence-Weighted** - All decisions scored
4. **Learning-Oriented** - Updates memory continuously
5. **Safety-First** - Patches validated before application

---

## 3.3 The Six Reflection Phases (R0-R6)

### R0: Signal Subscription

**Purpose:** Passively collect raw signals from execution phases

**Monitored Events:**
- Phase 3: Agent responses, draft contracts
- Phase 4: Approval verdicts, rejection reasons
- Phase 5: Execution logs, MLflow events, errors
- Phase 6: Verification results, missing evidence

**Output:** `reflection_buffer.jsonl` (event stream)

**Example Signal:**
```json
{
  "timestamp": "2025-10-18T14:32:10Z",
  "phase": "approval",
  "task_id": 1,
  "signal_type": "rejection",
  "agent": "quality_safety",
  "verdict": "REJECT",
  "flags": ["unsafe_batch_size"],
  "confidence": 0.35
}
```

---

### R1-R2: Pattern Mining & Root Cause Analysis

**Purpose:** Detect recurring patterns and diagnose failures

**Pattern Detection Algorithms:**

1. **Frequency Analysis**
   ```python
   if failure_count("out_of_memory") >= 2:
       pattern_detected("OOM")
   ```

2. **Temporal Correlation**
   ```python
   if all_failures_after("GPU_driver_update_X"):
       pattern_detected("driver_incompatibility")
   ```

3. **Multi-Agent Divergence**
   ```python
   if ops.verdict != quality.verdict:
       pattern_detected("spec_ambiguity")
   ```

4. **Artifact Consistency**
   ```python
   if missing_mlflow_run_id in [task1, task2, task3]:
       pattern_detected("mlflow_config_issue")
   ```

**Root Cause Diagnosis:**
```python
def diagnose_failure(task_result, signals):
    # Check common patterns
    if "OOM" in signals.stderr:
        return {
            "root_cause": "out_of_memory",
            "likely_fix": "reduce_batch_size",
            "confidence": 0.85
        }

    if signals.quality_verdict == "REJECT" and \
       "missing_mlflow_spec" in signals.flags:
        return {
            "root_cause": "incomplete_spec",
            "likely_fix": "add_mlflow_config",
            "confidence": 0.75
        }

    # Default: Unknown
    return {
        "root_cause": "unknown",
        "likely_fix": "manual_review",
        "confidence": 0.20
    }
```

**Output:** Diagnostic notes in `reflections.jsonl`

---

### R3-R4: Patch Generation & Risk Scoring

**Purpose:** Generate safe, targeted fixes

**Patch Generation Process:**

1. **Template Matching**
   - Root cause: "OOM" → Apply "reduce_batch_size" template

2. **Semantic Memory Lookup**
   - Check if similar issue solved before
   - Reuse proven patch if confidence > 0.80

3. **Safe Diff Creation**
   - Minimal changes only
   - No permission/domain expansion
   - Validate against security rules

**Example Patch:**
```json
{
  "patch_id": "patch_20251018_001",
  "task_id": 1,
  "root_cause": "out_of_memory",
  "changes": [
    {
      "field": "args.batch_size",
      "old_value": 32,
      "new_value": 16,
      "reason": "Reduce memory footprint"
    }
  ],
  "safety_checks": {
    "permission_scope": "unchanged",
    "network_domains": "unchanged",
    "write_paths": "unchanged"
  },
  "confidence": 0.85
}
```

**Risk Scoring:**
```python
def score_patch_risk(patch):
    risk = 0.0

    risk += len(patch.changes) * 0.05  # More changes = higher risk

    if patch.from_semantic_memory:
        risk -= 0.20  # Proven fix = lower risk

    risk += (1.0 - patch.confidence) * 0.30  # Low confidence = higher risk

    return min(max(risk, 0.0), 1.0)
```

**Output:** `job_spec_patch.json`, `risk_scores.json`

---

### R5-R6: Memory Update & Rule Promotion

**Purpose:** Learn from outcomes and update memory

**Learning Process:**

1. **Episodic Recording**
   - Every attempt → timestamped entry
   - Stored in `/memory/episodic.jsonl`

2. **Pattern Promotion**
   - Pattern recurs ≥3 times → Promote to semantic rule
   - Added to `/memory/semantic.yml`

3. **Template Caching**
   - Job spec succeeds with 0 retries ≥2 times → Cache
   - Stored in `/memory/procedural_cache/`

**Memory Update Logic:**
```python
def update_memory(task_result, reflection_note):
    # Always log episodic
    append_to_episodic(task_result, reflection_note)

    # Promote to semantic if recurring
    if count_similar_patterns(reflection_note.root_cause) >= 3:
        promote_to_semantic_rule(reflection_note)

    # Cache successful templates
    if task_result.status == "completed" and \
       task_result.retry_count == 0:
        cache_as_procedural_template(task_result.job_spec)

    # Prune old entries
    prune_episodic_memory(max_entries=100)
```

---

## 3.4 Memory Types

### Episodic Memory

**File:** `/memory/episodic.jsonl`
**Purpose:** Short-term, detailed event log

**Schema:**
```json
{
  "episode_id": "cycle_2025_10_18_task_1_attempt_2",
  "timestamp": "2025-10-18T14:35:22Z",
  "task_id": 1,
  "phase": "execution_retry",
  "attempt_number": 2,
  "root_cause": "out_of_memory",
  "patch_applied": "patch_20251018_001",
  "outcome": "success",
  "metrics": {
    "execution_time_seconds": 127,
    "gpu_memory_peak_gb": 28
  }
}
```

**Retention:** Last 100 entries per task type, 30 day max age

---

### Semantic Memory

**File:** `/memory/semantic.yml`
**Purpose:** Long-term rules from recurring patterns

**Schema:**
```yaml
rules:
  - rule_id: "sem_001"
    pattern: "CLIP ViT-L/14 on A100"
    observation: "Batch size > 16 causes OOM"
    recommendation:
      action: "set_batch_size_max_12"
      parameters:
        batch_size: 12
    evidence:
      success_count: 8
      failure_count: 0
      confidence: 0.95
    metadata:
      created: "2025-10-15T10:00:00Z"
      last_validated: "2025-10-18T15:00:00Z"
```

**Application:** Proactively applied in Planning and Execution initialization

---

### Procedural Cache

**Directory:** `/memory/procedural_cache/`
**Purpose:** Proven job specifications for fast reuse

**Structure:**
```
/memory/procedural_cache/
├── clip_baseline_extraction/
│   ├── job_spec.json
│   ├── metadata.json
│   └── artifacts_manifest.json
├── statistical_analysis/
└── documentation_generation/
```

**Job Spec:**
```json
{
  "template_id": "clip_baseline_v1",
  "proven_stable": true,
  "success_rate": 1.0,
  "usage_count": 5,
  "job_spec": {
    "entry": "scripts/clip_baseline.py",
    "args": {"batch_size": 32, ...},
    "expected_artifacts": ["baseline_attention.json"]
  }
}
```

---

## 3.5 Integration with Retry Phases

### Phase 4.5: Approval Retry

```
Rejection → Reflection Service
  ↓
R0-R1: Collect rejection reasons
  ↓
R2-R3: Diagnose (e.g., "missing MLflow spec")
  ↓
R4: Generate patch (add MLflow config)
  ↓
R5: Confidence ≥ 0.70 → Approve retry
  ↓
Apply patch → Re-submit to approval
```

### Phase 5.5: Execution Retry

```
Execution fail → Reflection Service
  ↓
R0-R1: Collect stderr, exit code
  ↓
R2-R3: Diagnose (e.g., "OOM")
  ↓
R4: Generate patch (reduce batch size)
  ↓
R5: Risk score 0.15 (low) → Approve retry
  ↓
Apply patch → Re-execute
```

### Phase 6.5: Verification Retry

```
Evidence missing → Reflection Service
  ↓
R0-R1: Identify what's missing
  ↓
R2-R3: Diagnose (e.g., "wrong artifact path")
  ↓
R4: Remediation (regenerate manifest)
  ↓
R5: Confidence 0.80 → Approve retry
  ↓
Apply fix → Re-verify
```

---

## 3.6 Confidence Scoring

**Confidence Formula:**
```python
def calculate_retry_confidence(reflection_note):
    base = 0.50

    # Boost: Known pattern
    if reflection_note.root_cause in KNOWN_PATTERNS:
        base += 0.25

    # Boost: Proven fix
    if has_semantic_rule_match(reflection_note):
        base += 0.20

    # Boost: Minimal patch
    if len(reflection_note.patch.changes) <= 2:
        base += 0.15

    # Reduce: Agent divergence
    if has_agent_divergence(reflection_note):
        base -= 0.20

    # Reduce: Multiple failures
    if reflection_note.attempt_count > 2:
        base -= 0.15

    return min(max(base, 0.0), 1.0)
```

**Thresholds:**
- **≥ 0.85:** High - Auto-retry
- **0.70-0.84:** Medium - Auto-retry with logging
- **0.50-0.69:** Low - Manual approval required
- **< 0.50:** Very low - Defer to next planning cycle

---

## 3.7 Safety & Governance

**Patch Guardrails:**
- ✅ No permission expansion
- ✅ No network domain changes
- ✅ No write path widening
- ✅ No unbounded resource allocation
- ✅ No credential modifications

**Audit Trail:**
- Every patch checksummed and signed
- Provenance: reflection_note → patch → outcome
- Immutable ledger: `/multi-agent/ledgers/reflections.jsonl`

**Human Oversight:**
- Low-confidence retries logged for review
- Semantic rules validated before promotion
- Procedural templates reviewed before caching

---

# PART IV: SYSTEM INTEGRATION

How Planning, Executive, and Reflection systems coordinate.

## 4.1 The Complete Cycle

```
┌────────────────────────────────────────────────┐
│ PLANNING SYSTEM                                │
│  1. Read previous cycle results                │
│  2. Read semantic memory rules                 │
│  3. Decompose goals into tasks                 │
│  4. Estimate resources with learned patterns   │
│  5. Generate pending_actions.json              │
└───────────────┬────────────────────────────────┘
                │
                ↓
┌────────────────────────────────────────────────┐
│ EXECUTIVE SYSTEM                               │
│  Phase 1: Read pending_actions.json            │
│  Phase 2: Initialize 3 agents                  │
│  Phase 3: Multi-agent implementation           │
│  Phase 4: Approval gate (+ 4.5 retry)          │
│  Phase 5: Code execution (+ 5.5 retry)         │
│  Phase 6: Evidence verification (+ 6.5 retry)  │
│  Phase 7: Final status determination           │
│  Phase 8: Report generation                    │
└───────────────┬────────────────────────────────┘
                ↕ (signals in parallel)
┌────────────────────────────────────────────────┐
│ REFLECTION SYSTEM (Parallel)                   │
│  R0: Signal subscription (all phases)          │
│  R1-R2: Pattern mining & root cause            │
│  R3-R4: Patch generation & risk scoring        │
│  R5-R6: Memory update & rule promotion         │
│  → Outputs: reflections.jsonl, patches, memory │
└────────────────────────────────────────────────┘
```

---

## 4.2 Data Flow

```
Planning → pending_actions.json → Executive
  ↓
Executive → execution_results_{timestamp}.json → Planning
  ↓
Executive → signals → Reflection (parallel)
  ↓
Reflection → patches → Executive (retry phases)
  ↓
Reflection → memory updates → Planning (next cycle)
```

---

## 4.3 Key Artifacts

| Artifact | Producer | Consumer | Purpose |
|----------|----------|----------|---------|
| `pending_actions.json` | Planning | Executive | Task specs |
| `execution_results_{timestamp}.json` | Executive | Planning | Outcomes |
| `reflections.jsonl` | Reflection | Executive, Planning | Learning notes |
| `job_spec_patch.json` | Reflection | Executive | Retry fixes |
| `/memory/semantic.yml` | Reflection | Planning, Executive | Long-term rules |
| `/memory/procedural_cache/` | Reflection | Executive | Proven templates |

---

## 4.4 Success Metrics

**Cycle-Level:**
- Task completion rate ≥ 80% (HIGH priority)
- Evidence verification rate = 100% (completed tasks)
- Retry success rate ≥ 60%

**Cross-Cycle:**
- Estimation accuracy improving (time, resources)
- Semantic rule count growing
- Procedural template reuse increasing

---

## 4.5 Debugging Guide

### When Tasks Fail Approval

**Check:**
1. Agent responses: `/logs/phase3/contracts/{task_id}.jsonl`
2. Which agent(s) rejected?
3. Flags array: Critical vs warnings
4. Reflection note: Root cause

**Common Causes:**
- Missing MLflow spec (Ops)
- Unsafe parameters (Quality)
- Resource limits (Infrastructure)

### When Execution Fails

**Check:**
1. Execution log: `/logs/execution_cycles/task_{id}.log`
2. Exit code and stderr
3. MLflow run (if partially created)
4. Reflection diagnosis

**Common Causes:**
- Out of memory (batch size)
- Import errors (dependencies)
- Timeout (exceeded 300s)
- File path issues

### When Verification Fails

**Check:**
1. Verification report: `/logs/verification/phase6_results.jsonl`
2. Expected vs found artifacts
3. MLflow query result
4. Task type classification

**Common Causes:**
- MLflow run not linked
- Wrong artifact directory
- File size = 0
- Task misclassified

---

## 4.6 File Structure

```
/multi-agent/
├── reports/
│   ├── handoff/
│   │   ├── pending_actions.json (INPUT from Planning)
│   │   ├── execution_progress_update_{ts}.md (OUTPUT)
│   │   └── next_meeting_trigger_{ts}.json (OUTPUT)
│   └── execution/
│       ├── summaries/execution_summary_{ts}.md
│       └── results/execution_results_{ts}.json
├── ledgers/ (immutable audit trail)
│   ├── approval_gate.jsonl
│   ├── run_log.json
│   └── reflections.jsonl
├── memory/
│   ├── episodic.jsonl
│   ├── semantic.yml
│   └── procedural_cache/
├── logs/
│   ├── initialization/
│   ├── phase3/contracts/
│   ├── execution_cycles/
│   └── verification/
└── artifacts/
    └── (generated outputs)
```

---

## 4.7 Operational Checklist

**Before Starting Cycle:**
- [ ] API keys loaded
- [ ] MLflow URI configured
- [ ] `pending_actions.json` present
- [ ] Memory files initialized
- [ ] GPU available

**During Execution:**
- [ ] Monitor agent initialization
- [ ] Watch approval verdicts
- [ ] Track code execution progress
- [ ] Review verification results

**After Completion:**
- [ ] Inspect progress report
- [ ] Archive logs and artifacts
- [ ] Review reflection summary
- [ ] Decide on next cycle

---

**END OF DOCUMENTATION**

*For operational details, see current notebook:*
`cvpr_autonomous_execution_cycle_v2_improved.ipynb`

*For implementation fixes, see:*
`NOTEBOOK_FIXES_APPLIED.md`

*For agent prompts, see:*
`/multi-agent/agents/prompts/executive_team/`
