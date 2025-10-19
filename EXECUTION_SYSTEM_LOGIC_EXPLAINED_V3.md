# Execution System Logic - Complete Explanation (v3)

**Date:** 2025-10-18
**System:** CVPR 2025 Autonomous Multi-Agent Research System

---

## 🎯 HIGH-LEVEL OVERVIEW

The execution system follows a **comprehensive 8-phase pipeline with integrated retry mechanisms**:

1. **Phase 1: PLANNING INTAKE** - Load pending actions from Planning Team
2. **Phase 2: INITIALIZATION** - Create 3-agent Executive Team
3. **Phase 3: IMPLEMENTATION** - Agents draft executable plans (no execution yet)
4. **Phase 4: APPROVAL** - 3-agent consensus gate using structured contracts
   - **Phase 4.5: APPROVAL RETRY** - Reflection-guided plan refinement
5. **Phase 5: EXECUTION** - Sandboxed code execution with telemetry
   - **Phase 5.5: EXECUTION RETRY** - Reflection-driven execution recovery
6. **Phase 6: VERIFICATION** - Evidence validation (MLflow + artifacts)
   - **Phase 6.5: VERIFICATION RETRY** - Evidence reconstruction
7. **Phase 7: FINALIZATION** - Terminal status assignment
8. **Phase 8: REPORTING** - KPI rollup, memory promotion, next-cycle handoff

**Parallel System:** Reflection Service monitors all phases, emitting patches and confidence scores for automated self-repair.

**Key Insight:** The system uses **structured JSON contracts** (not text parsing), **sandboxed execution** (not kernel exec), and **reflection-guided retries** (not blind retries) at three critical points: approval, execution, and verification.

---

## 📊 COMPLETE SYSTEM FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Read Planning Team's Decisions                         │
│ Cell 6: Load pending_actions.json                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Initialize Executive Team (3 Agents)                   │
│ Cell 8: Create Ops Commander, Quality & Safety, Infrastructure │
│                                                                 │
│  Step 1: Load core agent modules                                │
│  ├─→ Import multi-agent orchestration system                    │
│  ├─→ Load model configs, environment variables (.env)           │
│  └─→ Mount MLflow + Drive paths                                 │
│                                                                 │
│  Step 2: Create Agents                                          │
│  ├─→ 🧭 Ops Commander – implementation & code generation        │
│  ├─→ 🧩 Quality & Safety – logic review & compliance check      │
│  └─→ ⚙️ Infrastructure – runtime & resource validation          │
│                                                                 │
│  Step 3: Assign Structured Output Contracts                     │
│  ├─→ Each agent emits JSON schema {verdict, flags, checks…}     │
│  └─→ Ensures downstream gating uses typed fields, not text      │
│                                                                 │
│  OUTPUT: executive_team_manifest.json, agent_status_report.md   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
──────────────────────────────────────────────────────────────────
  Parallel System: Reflection Service (runs during Phases 3–6)
  Emits → reflections.jsonl | job_spec_patch.json | risk_scores.json
  Consumed by → Phase 4.5 (Approval Retry), 5.5 (Exec Retry), 6.5 (Verify Retry)
──────────────────────────────────────────────────────────────────
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Task Implementation & Agent Response Collection        │
│ (No execution – agents produce drafts + structured contracts)   │
│                                                                 │
│ FOR EACH TASK (HIGH → MEDIUM → LOW priority):                   │
│                                                                 │
│   Step 1: Dispatch task to 3 agents                             │
│   ├─→ Ops Commander – "Propose implementation (code as text)"   │
│   ├─→ Quality & Safety – "Assess logic & risks"                │
│   └─→ Infrastructure – "Assess env/resources & performance"     │
│                                                                 │
│   Step 2: Agents respond (no execution)                         │
│   ├─→ Ops Commander: draft Python/code + rationale              │
│   ├─→ Quality & Safety: review notes & risk flags               │
│   └─→ Infrastructure: resource/perf analysis                    │
│       (All 3 emit **structured JSON contracts** + commentary)   │
│       Contract schema (example):                                │
│       { verdict, flags:{critical[], warnings[]},                │
│         checks:{requirements_passed, mlflow_required},          │
│         proposed_jobs:[{id, entry, args, expected_artifacts}] } │
│                                                                 │
│ END FOR EACH TASK                                               │
│                                                                 │
│ OUTPUT → task_drafts.json (multi-agent contracts)               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: APPROVAL (Cell 11)                                     │
│ 3-Agent Approval Gate – Contract-Driven Decision                 │
│                                                                 │
│ FOR EACH task_draft:                                            │
│                                                                 │
│   Step 1: Gate on structured verdicts                           │
│   ├─→ Ops Commander verdict == "APPROVE"?                       │
│   ├─→ Quality & Safety verdict == "APPROVE"?                    │
│   └─→ Infrastructure verdict == "APPROVE"?                      │
│       Decision uses **explicit JSON fields**, not text search   │
│                                                                 │
│   Step 2: Apply strict rule                                     │
│   ├─→ IF (all three APPROVE) AND (no critical flags) →          │
│   │      preliminary_status = "approved"                        │
│   └─→ ELSE → preliminary_status = "rejected"                    │
│                                                                 │
│   Step 3: Persist decision                                      │
│   └─→ Store {task_id, preliminary_status, contracts, reasons}   │
│       (If rejected → send to Phase 4.5 Approval Retry)         │
│                                                                 │
│ END FOR EACH                                                    │
│                                                                 │
│ OUTPUT → task_results = [                                       │
│    {task_id: 1, preliminary_status: "approved", approved_jobs…}, │
│    {task_id: 2, preliminary_status: "rejected", reasons…}       │
│ ]                                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4.5: Approval Retry Mechanism                             │
│ (Triggered when any agent rejects or flags critical issues)      │
│                                                                 │
│ FOR EACH task_result WHERE preliminary_status == "rejected":     │
│                                                                 │
│   Step 1: Collect Reflection Feedback                           │
│   ├─→ Pull from Reflection Service:                              │
│   │    • reflections.jsonl (notes & root-cause)                  │
│   │    • job_spec_patch.json (safe diffs for plan/args/env)      │
│   │    • risk_scores.json (confidence, severity, cost)           │
│   └─→ Parse: failure_reason, proposed_fix, confidence_score      │
│                                                                 │
│   Step 2: Retry Eligibility Gate                                 │
│   ├─→ IF confidence_score ≥ τ (default 0.70) → proceed           │
│   ├─→ ELSE → mark non-retriable; log reason                     │
│   └─→ Enforce guardrails (no permission scope expansion)        │
│                                                                 │
│   Step 3: Apply Patch (Sandboxed)                                │
│   ├─→ Merge job_spec_patch.json into task draft                 │
│   ├─→ Update contract metadata: retry_attempt += 1              │
│   └─→ Generate provenance IDs (attempt_id, patch_id)            │
│                                                                 │
│   Step 4: Re-submit to Approval Gate                             │
│   ├─→ Send revised draft back to the 3 agents                   │
│   ├─→ Run Phase 4 checks (explicit verdict fields, flags)        │
│   └─→ Persist new verdicts under attempt_id + timestamp          │
│                                                                 │
│   Step 5: Limit Enforcement                                      │
│   ├─→ IF retry_attempt > MAX_RETRIES (default = 2) → abort       │
│   └─→ Record terminal reason: "Exceeded retry limit"             │
│                                                                 │
│ END FOR EACH                                                     │
│                                                                 │
│ OUTPUT → updated_task_results with retry outcomes               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: EXECUTION (Cell 11.5)                                 │
│ Automatic Code Executor – Run Approved Plans                    │
│                                                                 │
│ FOR EACH task_result IN task_results:                          │
│                                                                 │
│   Step 1: Check Preliminary Status                              │
│   ├─→ If preliminary_status != "approved" → Skip (⏭️)          │
│   └─→ Else → Continue to execution                              │
│                                                                 │
│   Step 2: Retrieve Job Specification                            │
│   ├─→ Extract from Ops Commander's structured JSON              │
│   ├─→ job_spec = {entry, args, expected_artifacts, env_keys[]}  │
│   └─→ Verify all required artifacts & MLflow keys declared      │
│                                                                 │
│   Step 3: Safe Job Execution (Sandboxed)                        │
│   ├─→ Spawn isolated subprocess/container                       │
│   ├─→ Apply timeouts, CPU/GPU/memory quotas                     │
│   ├─→ Stream stdout/stderr → /logs/execution_cycles/...          │
│   ├─→ Capture telemetry: MLFLOW_RUN_ID, ARTIFACT, METRIC        │
│   ├─→ On crash → tag as RETRYABLE_FAILURE or ERROR              │
│   └─→ No direct `exec()` in kernel (security isolation)         │
│                                                                 │
│   Step 4: Record Execution Results                              │
│   ├─→ execution.status = SUCCEEDED | FAILED | RETRYABLE_FAILURE │
│   ├─→ Persist run_id, artifacts[], metrics{}                    │
│   └─→ Update task_result with telemetry + log path              │
│                                                                 │
│   Step 5: Post-Execution Routing                                │
│   ├─→ IF execution.status == SUCCEEDED →                        │
│   │     • Mark executed_ready_for_verification = True           │
│   │     • Enqueue to Phase 6: Verification                      │
│   └─→ IF execution.status ∈ {FAILED, RETRYABLE_FAILURE} →       │
│         • Route to Phase 5.5: Reflection-Guided Retry           │
│         • Increment attempt counter (attempt_id += 1)           │
│                                                                 │
│ END FOR EACH                                                   │
│                                                                 │
│ OUTPUT: Successful jobs → Phase 6 | Failed jobs → Phase 5.5    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5.5: Reflection-Guided Retry (During Execution)           │
│ (Triggered when Phase 5 status ∈ {FAILED, RETRYABLE_FAILURE})    │
│                                                                 │
│ FOR EACH failed_attempt FROM Phase 5:                           │
│                                                                 │
│   Step 1: Collect Reflection Outputs                            │
│   ├─→ Pull artifacts from Reflection Service:                    │
│   │    • reflections.jsonl  (root cause, notes)                 │
│   │    • job_spec_patch.json (safe diffs: args/code/env)        │
│   │    • risk_scores.json   (confidence, severity, cost)        │
│   └─→ Parse: failure_reason, proposed_fix, confidence_score      │
│                                                                 │
│   Step 2: Eligibility & Guardrails                               │
│   ├─→ Require confidence_score ≥ τ (default 0.70)               │
│   ├─→ Enforce safety: no permission scope expansion             │
│   └─→ If not eligible → skip retry; mark for Phase 7            │
│                                                                 │
│   Step 3: Apply Patch (Versioned + Sandboxed)                    │
│   ├─→ Merge job_spec_patch.json → new job_spec_v{n+1}           │
│   ├─→ Generate provenance IDs: attempt_id, patch_id, checksum   │
│   └─→ Diff-scan & sign patch; persist under /patches/           │
│                                                                 │
│   Step 4: Retry Execution (Isolated)                             │
│   ├─→ Run in the same sandbox constraints (timeouts/quotas)     │
│   ├─→ Stream logs; capture MLflow run_id, metrics, artifacts    │
│   ├─→ On success → tag as RETRY_SUCCEEDED                       │
│   └─→ On failure  → tag as RETRY_FAILED                         │
│                                                                 │
│   Step 5: Loop Control                                           │
│   ├─→ If RETRY_SUCCEEDED → enqueue to Phase 6 (Verification)    │
│   ├─→ Else if attempts ≥ MAX_EXEC_RETRIES (default = 2)         │
│   │   → stop; mark "failed_final"; escalate in Phase 7          │
│   └─→ Else → request fresh reflection; iterate again            │
│                                                                 │
│ END FOR EACH                                                     │
│                                                                 │
│ OUTPUT → retry_results with RETRY_SUCCEEDED or failed_final     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6: VERIFICATION (Cell 17)                                │
│ Evidence Check – Verify Real Results Exist                      │
│                                                                 │
│ FOR EACH task WITH preliminary_status == "approved":            │
│                                                                 │
│   Step 1: Determine Task Type                                   │
│   ├─→ EXPERIMENTAL? (execute, run, diagnostic, gpu, model) →   │
│   │    Requires MLflow run_id + declared result artifacts       │
│   └─→ DOCUMENTATION? (write, design, draft, doc) →              │
│        Requires output files only (MLflow optional)             │
│                                                                 │
│   Step 2: Verify MLflow Evidence (if required)                  │
│   ├─→ Retrieve run_id from structured telemetry                 │
│   ├─→ Query MLflow: mlflow.get_run(run_id)                      │
│   ├─→ If run exists → ✅ Evidence confirmed                    │
│   └─→ If missing or invalid → ❌ Evidence missing               │
│                                                                 │
│   Step 3: Verify Artifact Evidence                              │
│   ├─→ Load expected_artifacts[] from task_result schema         │
│   ├─→ For each path: Path.exists() and size > 0 ?              │
│   ├─→ If all exist → ✅ Files verified                         │
│   └─→ If any missing → ❌ Incomplete evidence                   │
│                                                                 │
│   Step 4: Combine Results → Verification Verdict                │
│   ├─→ If all required evidence present → verification = "pass"  │
│   └─→ Else → verification = "fail"                              │
│                                                                 │
│   Step 5: Routing                                               │
│   ├─→ If verification == "pass" → continue to Phase 7          │
│   └─→ If verification == "fail" → send to Phase 6.5 (Retry)    │
│                                                                 │
│ END FOR EACH                                                   │
│                                                                 │
│ OUTPUT: verification_results with pass/fail per task            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6.5: Verification Retry Mechanism                         │
│ (Triggered when verification == "fail")                         │
│                                                                 │
│ FOR EACH task FAILED VERIFICATION:                              │
│                                                                 │
│  Step 1: Collect Reflection Outputs                             │
│  ├─→ Pull from Reflection Service:                              │
│  │   • reflections.jsonl (missing artifacts, failure notes)     │
│  │   • job_spec_patch.json (regenerate logic / fix targets)     │
│  │   • risk_scores.json (confidence, severity, cost)            │
│  └─→ Parse root cause and confidence score                      │
│                                                                 │
│  Step 2: Eligibility Check                                      │
│  ├─→ If confidence ≥ τ (default 0.7) → continue                │
│  └─→ Else → mark non-retriable; log reason; Phase 7             │
│                                                                 │
│  Step 3: Rebuild Artifacts / Re-Query Evidence (Sandboxed)      │
│  ├─→ Apply patch to job_spec or re-run MLflow query            │
│  ├─→ Regenerate missing files from cached inputs or logs        │
│  └─→ Capture new run_id and artifacts                           │
│                                                                 │
│  Step 4: Verification Re-Check                                  │
│  ├─→ Repeat Phase 6 steps (MLflow + artifact validation)       │
│  └─→ If all pass → mark "verification_passed_after_retry"       │
│                                                                 │
│  Step 5: Limit and Escalation                                   │
│  ├─→ If retry_attempt > MAX_VERIF_RETRIES (default = 2) → abort │
│  └─→ Store terminal reason = "Verification failed after retries"│
│                                                                 │
│ END FOR EACH                                                    │
│                                                                 │
│ OUTPUT → verification_retry_results                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 7: FINAL STATUS DETERMINATION (Cell 17)                   │
│ Retroactive Status Assignment Based on Verification & Retries   │
│                                                                 │
│ FOR EACH task_result:                                           │
│                                                                 │
│   Step 1: Normalize Upstream Signals                            │
│   ├─→ Read: preliminary_status, verification (pass|fail|na),    │
│   │     retry_summaries{approval, exec, verify}, attempts.count │
│   └─→ Ingest last reflection note (if present) for context      │
│                                                                 │
│   Step 2: Terminal Status Rules (ordered)                       │
│   ├─→ IF preliminary_status == "rejected"                       │
│   │     → final_status = "failed"                               │
│   │     → reason = "Did not pass 3-agent approval gate"         │
│   │                                                           │
│   ├─→ ELIF verification == "pass"                               │
│   │     → final_status = "completed"                            │
│   │     → reason = "Approved + evidence verified"              │
│   │                                                           │
│   ├─→ ELIF verification == "fail" AND retry_exhausted == true   │
│   │     → final_status = "failed_final"                         │
│   │     → reason = "Evidence missing after max retries"         │
│   │                                                           │
│   └─→ ELSE                                                      │
│         final_status = "failed"                                 │
│         reason = "Approved but no evidence (execution failed)"  │
│                                                                 │
│   Step 3: Persist Canonical Close-Out Record                    │
│   ├─→ task_result['status'] = final_status                      │
│   ├─→ task_result['status_reason'] = reason                     │
│   ├─→ task_result['attempts_total'] = attempts.count            │
│   └─→ task_result['provenance'] = {approval, execution,         │
│         verification, mlflow_run_ids, artifacts, ledger_paths}  │
│                                                                 │
│ END FOR EACH                                                    │
│                                                                 │
│ OUTPUT: Finalized task_results with terminal statuses           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 8: REPORTING & HANDOFF                                    │
│ Cells 14, 16, 19 – Generate Reports and Trigger Next Cycle      │
│                                                                 │
│ Step 1: Aggregate Results                                        │
│ ├─→ Collect finalized task_results from Phase 7                 │
│ ├─→ Join with run ledger, MLflow metadata, artifacts index       │
│ └─→ Compute KPIs: completed%, retry_success%, evidence_rate     │
│                                                                  │
│ Step 2: Generate Reports                                         │
│ ├─→ Write machine report: /reports/summary.json                 │
│ ├─→ Write human report: /reports/report.md (PDF)                │
│ ├─→ Include links: MLflow runs, artifact manifests, logs        │
│ └─→ Attach reflection_summary.md                                │
│                                                                  │
│ Step 3: Memory Promotion (Learning)                              │
│ ├─→ From reflections.jsonl → update /memory/semantic.yml        │
│ ├─→ Promote stable job_specs → /memory/procedural_cache/        │
│ └─→ Prune /memory/episodic.jsonl to top-K recent per task       │
│                                                                  │
│ Step 4: Trigger Next Planning Cycle                              │
│ ├─→ Compose /reports/handoff/pending_actions.json (vNext)       │
│ │    • include unresolved/failed_final tasks with reasons        │
│ │    • include pre-patch hints from semantic.yml                 │
│ └─→ Notify Planning Team: "Execution cycle complete"             │
│                                                                  │
│ OUTPUT: summary.json, report.md, reflection_summary.md,          │
│         updated memory, handoff: pending_actions.json            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 DETAILED BREAKDOWN OF EACH PHASE

### **PHASE 1: READ PLANNING TEAM'S DECISIONS** (Cell 6)

**Purpose:** Load the latest planning directives from Planning Team and prepare the task queue.

#### **1.1: File Loading**

```python
from pathlib import Path
import json

PLANNING_PATH = Path("/multi-agent/reports/handoff/pending_actions.json")

with open(PLANNING_PATH, "r") as f:
    pending_actions = json.load(f)

print(f"✅ Loaded {len(pending_actions)} actions from Planning Team.")
```

Example structure:
```json
[
  {
    "task_id": 1,
    "priority": "HIGH",
    "action": "Re-execute Task 1 – CLIP Integration with real attention extraction",
    "acceptance_criteria": [
      "CLIP/OpenCLIP must load on A100 GPU",
      "Real attention extraction (not simulated)",
      "MLflow run tracking with run_id",
      "Output: baseline_attention.json"
    ]
  }
]
```

#### **1.2: Schema Validation**

```python
import jsonschema

schema = {
  "type": "array",
  "items": {
    "type": "object",
    "required": ["task_id", "priority", "action", "acceptance_criteria"]
  }
}

jsonschema.validate(instance=pending_actions, schema=schema)
print("✅ Planning data validated against schema v1.2")
```

#### **1.3: Reflection System Intake**

```python
reflection_memory = {
  "episode_id": "cycle_2025_10_18",
  "planning_snapshot": {
    "source": str(PLANNING_PATH),
    "tasks": len(task_queue)
  }
}

with open("/multi-agent/memory/episodic.jsonl", "a") as f:
    f.write(json.dumps(reflection_memory) + "\n")
```

---

### **PHASE 2: INITIALIZE EXECUTIVE TEAM** (Cell 8)

**Purpose:** Instantiate the 3 core agents with structured output contracts.

**Process Overview:**
- **Ops Commander** handles implementation and code generation
- **Quality & Safety** performs compliance verification
- **Infrastructure Monitor** checks environmental readiness

#### **2.1: Agent Definition and Configuration**

Each agent initialized with:
- Role name and responsibilities
- API endpoint or callable interface
- Memory mode (short-term or long-term)
- Execution permissions

#### **2.2: Structured Output Contracts**

```json
{
  "verdict": "APPROVE" | "REJECT" | "CONDITIONAL",
  "flags": {
    "critical": [],
    "warnings": []
  },
  "checks": {
    "requirements_passed": true,
    "mlflow_required": true
  },
  "proposed_jobs": [
    {
      "id": "clip_baseline_001",
      "entry": "run_clip_baseline.py",
      "args": {"batch_size": 16, "device": "cuda"},
      "expected_artifacts": ["baseline_attention.json", "setup.md"]
    }
  ]
}
```

**Key Innovation:** No text parsing! Approval gate reads `verdict` field directly.

---

### **PHASE 3: TASK IMPLEMENTATION & AGENT RESPONSE COLLECTION**

**Purpose:** Agents draft executable plans (no execution yet).

#### **3.1: Typical Flow**

1. High-priority task "Run real attention extraction" sent to all 3 agents
2. Ops Commander produces Python block with MLflow tracking
3. Quality & Safety inspects logic, suggests batch size reduction
4. Infrastructure validates CUDA 12 and Torch 2.1 compatibility
5. All responses converted to structured JSON contracts

**Output:** `task_drafts.json` with full multi-agent contracts.

---

### **PHASE 4: APPROVAL GATE** (Cell 11)

**Purpose:** Multi-agent consensus checkpoint using structured verdicts.

#### **4.1: Approval Logic (Contract-Based)**

```python
def determine_approval(ops, quality, infra):
    """
    Use explicit JSON fields, not text matching.
    """
    all_approve = (
        ops["verdict"] == "APPROVE" and
        quality["verdict"] == "APPROVE" and
        infra["verdict"] == "APPROVE"
    )

    no_critical_flags = (
        len(ops["flags"]["critical"]) == 0 and
        len(quality["flags"]["critical"]) == 0 and
        len(infra["flags"]["critical"]) == 0
    )

    if all_approve and no_critical_flags:
        return "approved"
    else:
        return "rejected"
```

**Decision logged to:** `/multi-agent/ledgers/approval_gate.jsonl`

```json
{
  "task_id": 1,
  "preliminary_status": "approved",
  "timestamp": "2025-10-18T16:04:21Z",
  "verdicts": {...},
  "critical_flags": [],
  "approved_jobs": ["clip_integration_baseline"]
}
```

---

### **PHASE 4.5: APPROVAL RETRY MECHANISM**

**Purpose:** Self-correction for rejected proposals using reflection feedback.

#### **4.5.1: Retry Eligibility Gate**

```python
def should_retry_approval(task_result, reflection_feedback):
    """
    Only retry if reflection confidence ≥ threshold.
    """
    confidence = reflection_feedback["risk_scores"]["confidence"]
    attempt_count = task_result.get("retry_attempt", 0)

    if confidence < 0.70:  # τ threshold
        return False, "Low confidence patch"

    if attempt_count >= 2:  # MAX_RETRIES
        return False, "Exceeded retry limit"

    return True, "Eligible for retry"
```

#### **4.5.2: Patch Application**

```python
def apply_approval_patch(task_draft, patch):
    """
    Merge reflection patch into job spec.
    """
    # Example: patch reduces batch size from 32 → 16
    for job in task_draft["proposed_jobs"]:
        if "batch_size" in patch["args"]:
            job["args"]["batch_size"] = patch["args"]["batch_size"]

    task_draft["retry_attempt"] = task_draft.get("retry_attempt", 0) + 1
    task_draft["patch_id"] = patch["patch_id"]

    return task_draft
```

**Output:** Updated `task_results` with retry outcomes (`"approved_retry"` or `"failed_retry"`).

---

### **PHASE 5: EXECUTION** (Cell 11.5)

**Purpose:** Sandboxed execution of approved jobs with telemetry.

#### **5.1: Execution Filter**

```python
for task_result in task_results:
    if task_result['preliminary_status'] not in ['approved', 'approved_retry']:
        print(f"⏭️  Task {task_id}: Skipping (not approved)")
        continue

    # Task is approved → extract and execute
    execute_task(task_result)
```

#### **5.2: Sandboxed Execution**

```python
def execute_task_safely(job_spec):
    """
    Run in isolated subprocess with resource limits.
    NO kernel exec() - use subprocess isolation.
    """
    import subprocess
    import resource

    # Build command
    cmd = [
        "python", job_spec["entry"],
        "--batch_size", str(job_spec["args"]["batch_size"]),
        "--device", job_spec["args"]["device"]
    ]

    # Resource limits
    def set_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour
        resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))  # 16GB

    # Execute in subprocess
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=7200,  # 2 hours
            preexec_fn=set_limits,
            text=True
        )

        # Extract run_id from output
        run_id_match = re.search(r'MLflow Run ID:\s*([a-f0-9]+)', result.stdout)
        run_id = run_id_match.group(1) if run_id_match else None

        return {
            "status": "SUCCEEDED" if result.returncode == 0 else "FAILED",
            "run_id": run_id,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {"status": "RETRYABLE_FAILURE", "reason": "Timeout"}

    except Exception as e:
        return {"status": "FAILED", "reason": str(e)}
```

**What actually happens:**
1. Code runs in isolated subprocess (not kernel)
2. Real MLflow run created: `run_id = "abc123def456"`
3. Real files saved: `baseline_attention.json`
4. Telemetry logged to `/logs/execution_cycles/task_001.log`

---

### **PHASE 5.5: REFLECTION-GUIDED RETRY (During Execution)**

**Purpose:** Intelligent retry for failed executions using reflection patches.

#### **5.5.1: Patch Application Example**

```python
def apply_execution_patch(job_spec, reflection_patch):
    """
    Apply reflection-suggested fixes.
    """
    # Example: Reduce batch size to avoid OOM
    if "batch_size" in reflection_patch["args"]:
        job_spec["args"]["batch_size"] = reflection_patch["args"]["batch_size"]

    # Example: Add random seed for reproducibility
    if "seed" in reflection_patch["args"]:
        job_spec["args"]["seed"] = reflection_patch["args"]["seed"]

    # Example: Fix file path
    if "output_path" in reflection_patch["args"]:
        job_spec["args"]["output_path"] = reflection_patch["args"]["output_path"]

    return job_spec
```

#### **5.5.2: Retry Decision**

```python
def decide_execution_retry(execution_result, reflection_feedback):
    """
    Decide if retry is safe and likely to succeed.
    """
    confidence = reflection_feedback["risk_scores"]["confidence"]
    attempt = execution_result.get("attempt", 0)

    if confidence < 0.70:
        return "ESCALATE", "Low confidence in patch"

    if attempt >= 2:
        return "ABORT", "Max retries exceeded"

    if execution_result["status"] == "RETRYABLE_FAILURE":
        return "RETRY", f"Applying patch: {reflection_feedback['patch_summary']}"

    return "ABORT", "Non-retriable failure"
```

---

### **PHASE 6: VERIFICATION** (Cell 17)

**Purpose:** Validate that approved tasks have real, verifiable evidence.

#### **6.1: Task Type Determination**

```python
def should_require_mlflow(task_action):
    """
    Determine evidence requirements based on task type.
    """
    action_lower = task_action.lower()

    # Experimental tasks require MLflow
    experimental_keywords = ['execute', 'run', 'diagnostic', 'experiment', 'gpu', 'model']
    for keyword in experimental_keywords:
        if keyword in action_lower:
            return True  # → Requires MLflow run_id

    # Documentation tasks don't require MLflow
    doc_keywords = ['write', 'design', 'draft', 'document', 'paper']
    for keyword in doc_keywords:
        if keyword in action_lower:
            return False  # → Requires code/doc files only

    return True  # Default: require MLflow
```

#### **6.2: MLflow Verification**

```python
def verify_mlflow_evidence(task_result):
    """
    Check if MLflow run actually exists.
    """
    import mlflow

    # Extract run_id from execution results
    run_id = task_result.get("execution", {}).get("run_id")

    if not run_id:
        return {'found': False, 'reason': 'No run_id in execution result'}

    # Query MLflow API
    try:
        run = mlflow.get_run(run_id)  # ← Queries real MLflow database
        return {
            'found': True,
            'run_id': run_id,
            'experiment': run.info.experiment_id,
            'status': run.info.status
        }
    except Exception as e:
        return {
            'found': False,
            'run_id': run_id,
            'reason': f'Run not found in MLflow: {e}'
        }
```

#### **6.3: File Verification**

```python
def verify_file_evidence(task_result):
    """
    Check if result files actually exist on disk.
    """
    from pathlib import Path

    missing_files = []
    found_files = []

    # Check expected artifacts
    for artifact_path in task_result.get("expected_artifacts", []):
        full_path = Path(artifact_path)

        if full_path.exists() and full_path.stat().st_size > 0:
            found_files.append({
                'path': str(artifact_path),
                'size': full_path.stat().st_size
            })
        else:
            missing_files.append(str(artifact_path))

    return {
        'found_files': found_files,
        'missing_files': missing_files,
        'all_present': len(missing_files) == 0
    }
```

#### **6.4: Verification Verdict**

```python
def verify_task_evidence(task_id, task_result):
    """
    Complete verification for one task.
    """
    print(f"\n🔍 Verifying Task {task_id}")

    # Determine requirements
    requires_mlflow = should_require_mlflow(task_result['action'])
    print(f"   📊 Task Type: {'EXPERIMENTAL' if requires_mlflow else 'DOCUMENTATION'}")

    # Check MLflow
    mlflow_result = verify_mlflow_evidence(task_result)
    if requires_mlflow and not mlflow_result['found']:
        print(f"   ❌ MLflow: {mlflow_result['reason']}")
        return False

    # Check files
    file_result = verify_file_evidence(task_result)
    if not file_result['all_present']:
        for missing in file_result['missing_files']:
            print(f"   ❌ File MISSING: {missing}")
        return False

    print(f"\n   ✅ VERIFICATION PASSED")
    return True
```

---

### **PHASE 6.5: VERIFICATION RETRY MECHANISM**

**Purpose:** Reconstruct missing evidence without full re-execution.

#### **6.5.1: Evidence Remediation**

```python
def remediate_evidence(task_result, reflection_patch):
    """
    Lightweight evidence reconstruction.
    """
    # Re-query MLflow (maybe run was logged with delay)
    if "mlflow_requery" in reflection_patch["actions"]:
        time.sleep(5)  # Wait for async logging
        mlflow_result = verify_mlflow_evidence(task_result)
        if mlflow_result['found']:
            return {"status": "remediated", "method": "mlflow_requery"}

    # Regenerate artifacts from cached logs
    if "regenerate_artifacts" in reflection_patch["actions"]:
        # Example: Re-export from cached tensors
        cached_data = load_from_cache(task_result["run_id"])
        export_artifacts(cached_data, task_result["expected_artifacts"])
        return {"status": "remediated", "method": "artifact_regeneration"}

    return {"status": "failed", "reason": "No remediation available"}
```

---

### **PHASE 7: FINAL STATUS DETERMINATION**

**Purpose:** Convert all signals into terminal status.

#### **7.1: Status Logic**

```python
def determine_final_status(task_result, verification_passed):
    """
    Determine final status based on approval + verification.
    """
    preliminary = task_result['preliminary_status']

    if preliminary in ['rejected', 'failed_retry']:
        return {
            'status': 'failed',
            'reason': 'Did not pass 3-agent approval gate'
        }

    elif preliminary in ['approved', 'approved_retry']:
        if verification_passed:
            return {
                'status': 'completed',
                'reason': 'Approved + evidence verified'
            }
        else:
            return {
                'status': 'failed',
                'reason': 'Approved but no evidence (execution failed)'
            }
```

#### **7.2: Truth Table**

| Preliminary Status | Verification | Final Status | Reason |
|-------------------|--------------|--------------|--------|
| rejected | N/A | **failed** | Didn't pass approval |
| approved | True | **completed** ✅ | Approved + evidence exists |
| approved | False | **failed** ❌ | Approved but execution failed |
| approved_retry | True | **completed** ✅ | Approved after retry + evidence |

---

### **PHASE 8: REPORTING & HANDOFF**

**Purpose:** Generate reports, promote learning, trigger next cycle.

#### **8.1: KPI Rollup**

```python
def compute_kpis(task_results):
    """
    Aggregate execution metrics.
    """
    total = len(task_results)
    completed = sum(1 for t in task_results if t['status'] == 'completed')

    retry_succeeded = sum(
        1 for t in task_results
        if 'approved_retry' in t.get('preliminary_status', '') and t['status'] == 'completed'
    )

    return {
        "total_tasks": total,
        "completed": completed,
        "completion_rate": completed / total if total > 0 else 0,
        "retry_success_rate": retry_succeeded / total if total > 0 else 0
    }
```

#### **8.2: Memory Promotion**

```python
def promote_to_semantic_memory(reflections):
    """
    Convert repeated patterns into long-term rules.
    """
    pattern_counts = {}

    for reflection in reflections:
        root_cause = reflection["root_cause"]
        pattern_counts[root_cause] = pattern_counts.get(root_cause, 0) + 1

    # Promote patterns seen ≥3 times
    semantic_rules = []
    for pattern, count in pattern_counts.items():
        if count >= 3:
            semantic_rules.append({
                "pattern": pattern,
                "fix": reflections[0]["proposed_fix"],  # First successful fix
                "confidence": 0.85,
                "occurrences": count
            })

    # Write to semantic.yml
    with open("/memory/semantic.yml", "w") as f:
        yaml.dump(semantic_rules, f)
```

---

## 🔄 PARALLEL REFLECTION SERVICE

### **Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                      REFLECTION SERVICE                          │
│    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│    │  Error Miner │   │   Evaluator  │   │ Patch Builder│       │
│    └──────────────┘   └──────────────┘   └──────────────┘       │
│             │                │                   │              │
│             └────────────────┼───────────────────┘              │
│                              │                                  │
│       ┌──────────────────────▼───────────────────────┐          │
│       │      Policy & Memory Manager                 │          │
│       │  • Decide: retry / escalate / abort          │          │
│       │  • Maintain episodic, semantic, procedural   │          │
│       └──────────────────────┬────────────────────────┘         │
│                              │                                  │
│        ┌─────────────────────▼───────────────────────┐          │
│        │ Emits: job_spec_patch.json, risk_scores.json│          │
│        │        reflections.jsonl                     │          │
│        └────────────────────┬────────────────────────┘          │
│                             │                                   │
│  ↳ Feeds retry logic at Phases 4.5 / 5.5 / 6.5                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Reflection Phases (R0–R6)**

| Phase | Name | Function |
|-------|------|----------|
| **R0** | Signal Subscription | Non-blocking listener on all execution logs |
| **R1** | Data Ingestion | Normalize telemetry into standard schema |
| **R2** | Error Mining | Detect patterns (OOM, timeout, schema mismatch) |
| **R3** | Root Cause Eval | Cluster failures, compute confidence/severity |
| **R4** | Reflection Note | Generate structured diagnostic |
| **R5** | Patch Generation | Produce safe diffs for job_spec/code/env |
| **R6** | Risk Scoring | Compute retry priority = f(confidence, severity, cost) |

### **Memory Subsystem**

#### **Episodic Memory (Short-Term)**
- **File:** `/memory/episodic.jsonl`
- **TTL:** Current cycle only
- **Purpose:** Immediate retry guidance
- **Example:**
```json
{
  "task_id": 1,
  "attempt": 2,
  "root_cause": "OOM during attention extraction",
  "proposed_fix": "Reduce batch_size to 8",
  "confidence": 0.87
}
```

#### **Semantic Memory (Long-Term)**
- **File:** `/memory/semantic.yml`
- **TTL:** Persistent across cycles
- **Purpose:** Generalized rules
- **Example:**
```yaml
rules:
  - pattern: "OOM on CLIP-eval"
    fix: "Set batch_size=8, gradient_checkpointing=True"
    confidence: 0.92
    occurrences: 5
```

#### **Procedural Cache (Stable Templates)**
- **Directory:** `/memory/procedural_cache/`
- **Purpose:** Fast-start templates
- **Promotion:** After ≥3 successful runs
- **Example:** `clip_baseline_template_v2.json`

---

## 🎯 KEY DECISION POINTS

### **Decision Point 1: Phase 4 Approval Gate**

```
IF all 3 agents APPROVE AND no critical flags:
    preliminary_status = "approved"
    → GO TO Phase 5 (execute)
ELSE:
    preliminary_status = "rejected"
    → GO TO Phase 4.5 (approval retry)
```

### **Decision Point 2: Phase 4.5 Retry Eligibility**

```
IF confidence ≥ 0.70 AND attempts < 2:
    → Apply patch, resubmit to approval
ELSE:
    → Mark "failed_retry", escalate to Phase 7
```

### **Decision Point 3: Phase 5 Execution Filter**

```
IF preliminary_status ∈ ["approved", "approved_retry"]:
    → Execute in sandbox
ELSE:
    → Skip execution
```

### **Decision Point 4: Phase 5.5 Execution Retry**

```
IF status ∈ ["FAILED", "RETRYABLE_FAILURE"] AND confidence ≥ 0.70 AND attempts < 2:
    → Apply patch, retry execution
ELSE:
    → Mark "failed_final", proceed to Phase 7
```

### **Decision Point 5: Phase 6 Verification**

```
IF task_type == EXPERIMENTAL:
    IF MLflow run exists AND all artifacts exist:
        verification = "pass"
    ELSE:
        verification = "fail" → Phase 6.5
ELIF task_type == DOCUMENTATION:
    IF all artifacts exist:
        verification = "pass"
    ELSE:
        verification = "fail" → Phase 6.5
```

### **Decision Point 6: Phase 7 Final Status**

```
IF preliminary_status == "rejected":
    final_status = "failed"

ELIF verification == "pass":
    final_status = "completed" ✅

ELIF verification == "fail" AND retry_exhausted:
    final_status = "failed_final" ❌

ELSE:
    final_status = "failed" ❌
```

---

## 📊 COMPLETE OUTCOMES TABLE

| Approval | Execution | Verification | Retry Used | Final Status | Example Scenario |
|----------|-----------|--------------|------------|--------------|------------------|
| rejected | N/A | N/A | No | **failed** | Quality flagged critical issue |
| rejected | N/A | N/A | Yes (4.5) → approved | **completed** | Batch size reduced, then passed |
| approved | SUCCEEDED | pass | No | **completed** ✅ | Perfect execution |
| approved | SUCCEEDED | pass | Yes (5.5) | **completed** ✅ | OOM fixed by retry |
| approved | SUCCEEDED | fail | No | **failed** ❌ | MLflow logging failed |
| approved | SUCCEEDED | fail | Yes (6.5) → pass | **completed** ✅ | Evidence reconstructed |
| approved | FAILED | N/A | No | **failed** ❌ | Code crash, no retry |
| approved | FAILED | N/A | Yes (5.5) → SUCCEEDED | **completed** ✅ | Import error fixed |
| approved | FAILED | N/A | Yes (5.5) → FAILED | **failed_final** ❌ | Non-retriable error |

---

## 🔧 SYSTEM IMPROVEMENTS (V2 → V3)

### **What Changed:**

1. **Structured Contracts (No Text Parsing)**
   - **V2:** Approval gate searched for "✅ PASS" or "❌ FAIL" in agent text
   - **V3:** Agents emit JSON with explicit `verdict` field
   - **Impact:** Eliminates ambiguity and parsing errors

2. **Sandboxed Execution (No Kernel Exec)**
   - **V2:** Used `exec(code)` in notebook kernel
   - **V3:** Spawns isolated subprocess with resource limits
   - **Impact:** Security isolation, crash protection, rollback support

3. **Three Retry Points**
   - **V2:** Single retry after execution failure
   - **V3:** Retries at approval (4.5), execution (5.5), verification (6.5)
   - **Impact:** 3× higher recovery rate

4. **Reflection-Guided Patches**
   - **V2:** Blind retries with same parameters
   - **V3:** Reflection Service analyzes failures, proposes targeted fixes
   - **Impact:** Retries change job_spec based on root cause

5. **Memory System**
   - **V2:** No learning across cycles
   - **V3:** Episodic → Semantic → Procedural memory promotion
   - **Impact:** Self-improving system, pre-patches future tasks

6. **Evidence-Based Verification**
   - **V2:** Trusted agent claims of completion
   - **V3:** Queries real MLflow API, checks file existence
   - **Impact:** Prevents false positives

---

## 🎯 BOTTOM LINE

The V3 execution system is a **self-healing, evidence-based autonomous pipeline** that:

1. **Plans with structure** - JSON contracts, not text patterns
2. **Executes with safety** - Sandboxed processes, not kernel exec
3. **Verifies with proof** - Real MLflow queries, not agent claims
4. **Retries with intelligence** - Reflection-guided patches, not blind retries
5. **Learns from experience** - Memory promotion across cycles

**Retry Philosophy:**
- **Phase 4.5:** Fix the plan (reduce batch size, add safety checks)
- **Phase 5.5:** Fix the execution (adjust env, rebind GPU, add seed)
- **Phase 6.5:** Fix the evidence (re-query MLflow, regenerate artifacts)

**Confidence Threshold (τ):** 0.70 - Only retry when reflection confidence ≥ 70%

**Max Retries:** 2 per phase - Prevents infinite loops, escalates persistent failures

**Output:** Every task ends with a **terminal status** (`completed`, `failed`, `failed_final`) backed by **verifiable evidence** and **complete provenance** (attempt logs, patch IDs, MLflow run IDs, artifact checksums).

---

**Generated:** 2025-10-18
