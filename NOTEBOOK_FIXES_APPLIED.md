# Execution System Logic - Complete Explanation (v2)

**Date:** 2025-10-15 **System:** CVPR 2025 Autonomous Multi-Agent
Research System

------------------------------------------------------------------------

## 🎯 HIGH-LEVEL OVERVIEW

The execution system follows a **clear 4-phase pipeline**:

1.  **Phase 3: APPROVAL** - 3 agents review task and approve/reject
    implementation plan
2.  **Phase 4: EXECUTION** - Approved tasks have their code extracted
    and run
3.  **Phase 5: VERIFICATION** - Real evidence is checked (MLflow runs,
    files)
4.  **Phase 6: FINAL STATUS** - Task status confirmed based on
    verification results

**Key Insight:** Status determination happens AFTER verification, not
before. The approval gate (Phase 3) gives a preliminary status, but
Phase 5 verification is what makes it final.

------------------------------------------------------------------------

## 📊 COMPLETE SYSTEM FLOW

```         
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Read Planning Team's Decisions                         │
│ Cell 6: Load pending_actions.json                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Initialize Executive Team (3 Agents)                   │
│ Cell 8: Create Ops Commander, Quality & Safety, Infrastructure  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
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
│  Step 4: Environment & Dependency Check                         │
│  ├─→ Verify all required APIs and credentials loaded            │
│  ├─→ Confirm MLflow tracking URI reachable                      │
│  └─→ Confirm storage & logging directories exist                │
│                                                                 │
│  Step 5: READY Verification                                     │
│  ├─→ All 3 agents must return status = “READY”                  │
│  └─→ If any agent fails, halt initialization                    │
│                                                                 │
│  OUTPUT: Initialization log written to                          │
│          /multi-agent/logs/system_init/exec_team_init.log       │
│                                                                 │
│  → Continue to Phase 3: Approval Gate once all agents READY ✅   │
└─────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────
  Parallel System: Reflection Service (runs during Phases 3–6)
  Emits → reflections.jsonl | job_spec_patch.json | risk_scores.json
  Consumed by → Phase 3.5 (Approval Retry), 4.5 (Exec Retry), 6.5 (Verification Retry)
──────────────────────────────────────────────────────────────────

                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Task Implementation & Agent Response Collection        │
│ (No execution – agents produce drafts + structured contracts)   │
└─────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────
  Parallel Reflection Listening Layer (R0: Signal Subscription)
  • Subscribes to live agent responses and inter-agent discrepancies
  • Captures early reasoning traces, draft risk flags, and schema quality
  • Streams to Reflection Service (Phases R1–R2) for pre-approval diagnostics
──────────────────────────────────────────────────────────────────
                            ↓
│ FOR EACH TASK (HIGH → MEDIUM → LOW priority):                   │
│                                                                 │
│   Step 1: Dispatch task to 3 agents                             │
│   ├─→ Ops Commander – “Propose implementation (code as text)”   │
│   ├─→ Quality & Safety – “Assess logic & risks”                │
│   └─→ Infrastructure – “Assess env/resources & performance”     │
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
│   Step 3: Emit Reflection Snapshot (Async Event)                │
│   ├─→ Publish mini-event to reflection_stream:                  │
│       {task_id, phase:"implementation", risk_flags, divergence} │
│   ├─→ Enables early pattern detection (R2: Error & Signal Mining) │
│   └─→ Logged to reflections.jsonl with timestamp & checksum    │
│                                                                 │
│ END FOR EACH TASK                                               │
│                                                                 │
│ OUTPUT → task_drafts = [                                        │
│    {task
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: APPROVAL (Cell 11)                                     │
│ 3-Agent Approval Gate – Contract-Driven Decision                 │
└─────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────
  Parallel Reflection Listening Layer (R0: Signal Subscription)
  • Subscribes to real-time Approval logs (verdicts, flags, rationale)
  • Streams non-blocking events → Reflection Service (Phases 3–6)
  • Used to trigger early diagnostic signals and confidence updates
──────────────────────────────────────────────────────────────────
                            
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
│   └─→ Store {task_id, preliminary_status, contracts, reasons, approved_jobs[]} │
│       (No execution here — handoff to Phase 5: Execution)       │
│       (If rejected → send to Phase 4.5 Approval Retry)         │
│                                                                 │
│   Step 4: Emit reflection signal snapshot                       │
│   ├─→ Publish minimal event → reflection_stream:                │
│       {task_id, phase: "approval", verdicts, flags, status}     │
│   ├─→ Enables reflection R1–R2 modules to start root-cause mining │
│   └─→ Logged to reflections.jsonl with timestamp & checksum     │
│                                                                 │
│ END FOR EACH                                                    │
│                                                                 │
│ OUTPUT → task_results = [                                       │
│    {task_id: 1, preliminary_status: "approved", approved_jobs…}, │
│    {task_id: 2, preliminary_status: "rejected", reasons…},      │
│    …                                                            │
│ ]
└─────────────────────────────────────────────────────────────────┘
                
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4.5: Approval Retry Mechanism                             │
│ (Triggered when any agent rejects or flags critical issues)      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
│ FOR EACH task_result WHERE preliminary_status == “rejected”:     │
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
│   ├─→ ELSE → mark non-retriable; log reason; continue to Phase 7 │
│   └─→ Enforce guardrails (no permission scope expansion, etc.)   │
│                                                                 │
│   Step 3: Apply Patch (Sandboxed)                                │
│   ├─→ Merge job_spec_patch.json into task draft (safe diff-scan) │
│   ├─→ Update contract metadata: retry_attempt += 1               │
│   └─→ Generate provenance IDs (attempt_id, patch_id, checksum)   │
│                                                                 │
│   Step 4: Re-submit to Approval Gate                             │
│   ├─→ Send revised draft back to the 3 agents (Ops/Quality/Infra)│
│   ├─→ Run Phase 4 checks (explicit verdict fields, flags)        │
│   └─→ Persist new verdicts under attempt_id + timestamp          │
│                                                                 │
│   Step 5: Limit Enforcement                                      │
│   ├─→ IF retry_attempt > MAX_RETRIES (default = 2) → abort       │
│   └─→ Record terminal reason: “Exceeded retry limit”             │
│                                                                 │
│   Step 6: Artifact Emission & Traceability (Reflection Output)   │
│   ├─→ Emit to /ledgers/ and /memory/:                            │
│   │    • /ledgers/reflections.jsonl (append attempt record)      │
│   │    • /patches/job_spec_patch.json (applied patch snapshot)   │
│   │    • /ledgers/risk_scores.json (decision features)           │
│   │    • /ledgers/contract_delta.json (pre→post gate diff)       │
│   │    • /ledgers/patch_applied.diff (unified diff)              │
│   ├─→ Link MLflow evidence if any pre-check runs were simulated: │
│   │    • mlflow.link: {task_id, attempt_id, related_run_ids[]}   │
│   └─→ Update memory for learning:                                │
│        • /memory/episodic.jsonl  (attempt-level reflection)      │
│        • /memory/semantic.yml     (if pattern recurs → rule)     │
│        • /memory/procedural_cache/ (if patch stabilizes)         │
│                                                                 │
│ END FOR EACH                                                     │
│                                                                 │
│ OUTPUT → updated_task_results = [                                │
│   {task_id: 1, retry_attempts: 1, status: “approved_after_patch”},│
│   {task_id: 2, retry_attempts: 2, status: “failed_final”},       │
│   …                                                              │
│ ]                                                                │
│                                                                 │
│ NOTE: All emitted artifacts are batch-flushed and versioned;     │
│ checksums enable end-to-end provenance from reflection → gate.   │
└─────────────────────────────────────────────────────────────────┘

                            ↓
                            
                            
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: EXECUTION (Cell 11.5)                                 │
│ Automatic Code Executor – Run Approved Plans                    │
└─────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────
  Parallel Reflection Listening Layer (R0: Signal Subscription)
  • Subscribes to runtime logs (stdout/stderr), exit codes, MLflow
  • Streams non-blocking telemetry → Reflection Service (R1–R6)
  • Enables rapid root-cause mining before 5.5 retry
──────────────────────────────────────────────────────────────────
                            ↓
│  FOR EACH task_result IN task_results:                          │
│                                                                 │
│   Step 1: Check Preliminary Status                              │
│   ├─→ If preliminary_status != "approved" → Skip (⏭️)          │
│   └─→ Else → Continue to execution                              │
│                                                                 │
│   Step 2: Retrieve Job Specification                            │
│   ├─→ Extract from Ops Commander’s structured JSON              │
│   ├─→ job_spec = {entry, args, expected_artifacts, env_keys[]}  │
│   └─→ Verify all required artifacts & MLflow keys declared      │
│                                                                 │
│   Step 3: Safe Job Execution (Sandboxed)                        │
│   ├─→ Spawn isolated subprocess/container                       │
│   ├─→ Apply timeouts, CPU/GPU/memory quotas                     │
│   ├─→ Stream stdout/stderr → /logs/execution_cycles/...          │
│   ├─→ Capture telemetry lines:                                  │
│   │     MLFLOW_RUN_ID=..., ARTIFACT=..., METRIC=...             │
│   ├─→ On crash → tag as RETRYABLE_FAILURE or ERROR              │
│   └─→ No direct `exec()` in kernel (security isolation)         │
│                                                                 │
│   Step 4: Record Execution Results                              │
│   ├─→ execution.status = SUCCEEDED | FAILED | RETRYABLE_FAILURE │
│   ├─→ Persist run_id, artifacts[], metrics{}                    │
│   ├─→ Update task_result with telemetry + log path              │
│   └─→ Append ledger: /multi-agent/ledgers/run_log.json          │
│                                                                 │
│   Step 5: Post-Execution Routing                                │
│   ├─→ IF execution.status == SUCCEEDED →                        │
│   │     • Emit success snapshot → reflection_stream (non-block) │
│   │     • Mark executed_ready_for_verification = True           │
│   │     • Enqueue to Phase 6: Verification                      │
│   │                                                             │
│   └─→ IF execution.status ∈ {FAILED, RETRYABLE_FAILURE} →       │
│         • Emit failure snapshot → reflections.jsonl             │
│         • Include: exit_code, stderr_tail, run_id?, artifacts   │
│         • Route to Phase 5.5: Reflection-Guided Retry           │
│         • Increment attempt counter (attempt_id += 1)           │
│                                                                 │
│  END FOR EACH                                                   │
│                                                                 │
│  OUTPUT:                                                        │
│   • Successful jobs → queued for Phase 6 (Verification)         │
│   • Unsuccessful jobs → routed to Phase 5.5 (Retry) with        │
│     complete telemetry for reflection (patch + risk scoring)    │
└─────────────────────────────────────────────────────────────────┘

                            ↓
                            
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5.5: Reflection-Guided Retry (During Execution)           │
│ (Triggered when Phase 5 status ∈ {FAILED, RETRYABLE_FAILURE})    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
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
│   ├─→ Enforce safety: no permission scope expansion,            │
│   │   no network/domain list changes, no write-path widening    │
│   └─→ If not eligible → skip retry; mark for Phase 7 reporting   │
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
│   Step 5: Regression & Rollback Check                            │
│   ├─→ Compare key metrics/artifacts vs previous attempt         │
│   ├─→ If regression detected (lower quality / missing files)    │
│   │   → rollback to last best artifacts; keep retry outcome log │
│   └─→ Store regression_reason in run ledger                     │
│                                                                 │
│   Step 6: Loop Control                                           │
│   ├─→ If RETRY_SUCCEEDED → enqueue to Phase 6 (Verification)    │
│   ├─→ Else if attempts ≥ MAX_EXEC_RETRIES (default = 2)         │
│   │   → stop; mark “failed_final”; escalate in Phase 7          │
│   └─→ Else → request fresh reflection; iterate again            │
│                                                                 │
│   Step 7: Artifact Emission & Memory Update                      │
│   ├─→ Emit:                                                     │
│   │    • /ledgers/reflections.jsonl (attempt snapshot)          │
│   │    • /ledgers/risk_scores.json (decision features)          │
│   │    • /patches/job_spec_patch.json (applied patch copy)      │
│   │    • /ledgers/run_log.json (append retry run)               │
│   ├─→ Link MLflow: {task_id, attempt_id, run_id(s)}             │
│   └─→ Update memory:                                            │
│        • /memory/episodic.jsonl (attempt-level record)          │
│        • /memory/semantic.yml (add rule if pattern recurs)      │
│        • /memory/procedural_cache/ (store stable template)      │
│                                                                 │
│ END FOR EACH                                                     │
│                                                                 │
│ OUTPUT → retry_results = [                                      │
│   {task_id: T1, attempt: 2, status: "RETRY_SUCCEEDED", run_id…},│
│   {task_id: T2, attempt: 3, status: "failed_final", reasons…},  │
│   …                                                             │
│ ]                                                                │
│ NOTE: All retries remain sandboxed; every iteration must change │
│       job_spec (no identical reruns). Confidence τ & max retries│
│       are configurable per task class/priority.                 │
└─────────────────────────────────────────────────────────────────┘
                            
                            
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6: VERIFICATION (Cell 17)                                │
│ Evidence Check – Verify Real Results Exist                      │
└─────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────
  Parallel Reflection Listening Layer (R0: Signal Subscription)
  • Subscribes to MLflow telemetry & artifact validation events
  • Streams non-blocking verification logs → Reflection Service (R1–R6)
  • Enables early detection of missing runs or incomplete outputs
──────────────────────────────────────────────────────────────────
                            ↓
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
│   Step 5: Emit Reflection Snapshot                              │
│   ├─→ Publish {task_id, phase:"verification", status, missing[]}
│   │     → reflection_stream for post-run analysis               │
│   └─→ Logged to reflections.jsonl with timestamp & checksum     │
│                                                                 │
│   Step 6: Routing                                               │
│   ├─→ If verification == "pass" → continue to Phase 7          │
│   └─→ If verification == "fail" → send to Phase 6.5 (Retry)    │
│                                                                 │
│ END FOR EACH                                                   │
│                                                                 │
│ OUTPUT: Structured verification report per approved task        │
│          {task_id, verification:"pass"|"fail", evidence:{...}} │
└─────────────────────────────────────────────────────────────────┘

                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6.5: Verification Retry Mechanism                         │
│ (Triggered when verification == "fail")                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
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
│  Step 6: Artifact Emission & Memory Update                      │
│  ├─→ Emit to ledgers/:                                          │
│  │   • reflections.jsonl      (verification retry record)       │
│  │   • risk_scores.json        (updated confidence)             │
│  │   • verification_log.json   (full validation trace)          │
│  └─→ Update /memory/:                                           │
│      • episodic.jsonl (attempt-level)                           │
│      • semantic.yml (new pattern rule)                          │
│      • procedural_cache/ (stable check templates)               │
│                                                                 │
│ END FOR EACH                                                    │
│                                                                 │
│ OUTPUT → verification_retry_results = [                         │
│   {task_id:T1, attempt:2, status:"verification_passed_after_retry"},│
│   {task_id:T2, attempt:3, status:"failed_final"} … ]            │
│ NOTE: All verification retries remain sandboxed; evidence must │
│                                                                │
│                                                                 │
│                                                                 │
│                                                                 │
│                                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 7: FINAL STATUS DETERMINATION (Cell 17)                   │
│ Retroactive Status Assignment Based on Verification & Retries   │
└─────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────
  Parallel Reflection Snapshot (non-blocking)
  • Publish terminal event → reflection_stream:
    {task_id, phase:"finalize", final_status, reasons[], attempts}
──────────────────────────────────────────────────────────────────
                            ↓
│ FOR EACH task_result:                                           │
│                                                                 │
│   Step 1: Normalize Upstream Signals                            │
│   ├─→ Read: preliminary_status, verification (pass|fail|na),    │
│   │     retry_summaries{approval, exec, verify}, attempts.count │
│   ├─→ Derive flags:                                            │
│   │     approved_after_patch?, retry_exhausted?, evidence_ready?│
│   └─→ Ingest last reflection note (if present) for context      │
│                                                                 │
│   Step 2: Terminal Status Rules (ordered)                       │
│   ├─→ IF preliminary_status == "rejected"                       │
│   │     → final_status = "failed"                               │
│   │     → reason = "Did not pass 3-agent approval gate"         │
│   │                                                           │
│   ├─→ ELIF verification == "pass"                               │
│   │     → final_status = "completed"                            │
│   │     → reason = (approved_after_patch?                       │
│   │        "Approved after retry + evidence verified" :         │
│   │        "Approved + evidence verified")                      │
│   │                                                           │
│   ├─→ ELIF verification == "fail" AND retry_exhausted == true   │
│   │     → final_status = "failed_final"                         │
│   │     → reason = "Evidence missing after max retries"         │
│   │                                                           │
│   ├─→ ELIF verification == "fail"                               │
│   │     → final_status = "failed"                               │
│   │     → reason = "Approved but no evidence (execution failed)"│
│   │                                                           │
│   └─→ ELSE (fallback)                                           │
│         final_status = "failed"                                 │
│         reason = "Unresolved state; see logs"                   │
│                                                                 │
│   Step 3: Persist Canonical Close-Out Record                    │
│   ├─→ task_result['status'] = final_status                      │
│   ├─→ task_result['status_reason'] = reason                     │
│   ├─→ task_result['attempts_total'] = attempts.count            │
│   ├─→ task_result['provenance'] = {                             │
│   │      approval:{approved_after_patch, attempts:retry_summaries.approval},│
│   │      execution:{attempts:retry_summaries.exec},             │
│   │      verification:{attempts:retry_summaries.verify},        │
│   │      mlflow_run_ids: [...], artifacts: [...],              │
│   │      reflection_digest_id, ledger_paths:[...]}              │
│   └─→ Append JSON line to /multi-agent/logs/final_status/phase7.jsonl │
│                                                                 │
│   Step 4: Emit Planning/Reporting Event                         │
│   ├─→ Publish compact summary → /reports/final_events.jsonl     │
│   │    {task_id, final_status, key_metrics, missing_gaps[]}     │
│   └─→ Used by Phase 8 and Planning Phase 1 (next cycle)         │
│                                                                 │
│   Step 5: Memory Hooks (non-blocking)                           │
│   ├─→ If final_status in {"failed_final","failed"} and          │
│   │    repeated root cause → update /memory/semantic.yml rule   │
│   └─→ If "completed" with strong uplift → cache template in     │
│        /memory/procedural_cache/                                │
│                                                                 │
│ END FOR EACH                                                    │
│                                                                 │
│ OUTPUT: Finalized task_results (terminal, auditable)            │
│  task_results = [                                               │
│    {task_id: 1, status:"completed", status_reason:"Approved + evidence verified"}, │
│    {task_id: 2, status:"failed_final", status_reason:"Evidence missing after max retries"}, │
│    {task_id: 3, status:"failed", status_reason:"Did not pass 3-agent approval gate"} │
│  ]                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Phase 8: REPORTING & HANDOFF                                    │
│ Cells 14, 16, 19 – Generate Reports and Trigger Next Cycle      │
└─────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────
  Parallel Reflection Snapshot (non-blocking)
  • Roll up reflections → reflection_summary.md
  • Update memory: semantic.yml (rules), procedural_cache/ (templates)
──────────────────────────────────────────────────────────────────
                            ↓
│ Step 1: Aggregate Results                                        │
│ ├─→ Collect finalized task_results from Phase 7 (canonical JSON) │
│ ├─→ Join with run ledger, MLflow metadata, artifacts index       │
│ └─→ Compute KPIs: completed%, retry_success%, evidence_rate,     │
│      avg_attempts_to_success, top_root_causes                    │
│                                                                  │
│ Step 2: Generate Reports                                         │
│ ├─→ Write machine report: /reports/execution_summary/summary.json│
│ ├─→ Write human report: /reports/execution_summary/report.md(PDF)│
│ ├─→ Include links: MLflow runs, artifact manifests, log bundles  │
│ └─→ Attach reflection_summary.md (patterns, fixes, confidence)   │
│                                                                  │
│ Step 3: Archive Logs & Artifacts                                 │
│ ├─→ Bundle /logs/execution_cycles/cycle_N → tar.gz + checksum    │
│ ├─→ Snapshot /multi-agent/ledgers/*.jsonl (immutable)            │
│ ├─→ Persist artifact manifest with sizes and SHA256              │
│ └─→ Record retention window & access policy                      │
│                                                                  │
│ Step 4: Memory Promotion (Learning)                              │
│ ├─→ From reflections.jsonl → update /memory/semantic.yml (rules) │
│ ├─→ Promote stable job_specs → /memory/procedural_cache/         │
│ └─→ Prune /memory/episodic.jsonl to top-K recent per task        │
│                                                                  │
│ Step 5: Governance & Redaction Checks                            │
│ ├─→ PII/secret scan on reports & logs; redact or block as needed │
│ ├─→ Verify schema versions & provenance IDs across outputs       │
│ └─→ Sign summary.json + report.md with release tag (vX.Y.cycleN) │
│                                                                  │
│ Step 6: Trigger Next Planning Cycle                              │
│ ├─→ Compose /reports/handoff/pending_actions.json (vNext)        │
│ │    • include unresolved/failed_final tasks with reasons        │
│ │    • include pre-patch hints from semantic.yml                 │
│ ├─→ Write handoff manifest: {version, checksums, URIs, counts}   │
│ └─→ Notify Planning Team: “Execution cycle complete”             │
│                                                                  │
│ OUTPUT:                                                          │
│  • summary.json, report.md/PDF, reflection_summary.md            │
│  • updated memory: semantic.yml, procedural_cache/, episodic.jsonl│
│  • archived logs/artifacts with checksums                        │
│  • handoff: pending_actions.json (for Phase 1: Planning)         │
└─────────────────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

## 🔍 DETAILED BREAKDOWN OF EACH PHASE

------------------------------------------------------------------------

### **PHASE 1: READ PLANNING TEAM’S DECISIONS** (Cell 6)

**Purpose:**\
Load the latest planning directives (`pending_actions.json`) from the
Planning Team\
and prepare the Execution Team’s task queue for initialization.

**Key Insight:**\
This is the **data-ingestion and alignment stage** — ensuring that every
action item from planning\
is parsed, validated, and transformed into an executable format for the
agents.

**Parallel Reflection Hook:**\
The **Reflection Service** begins passive listening (R0 Signal
Subscription).\
It records high-level intent summaries and pre-allocates memory context
for each task\
to support downstream self-feedback (used in Phases 3–6).

------------------------------------------------------------------------

#### **1.1: File Loading**

``` python
from pathlib import Path
import json

PLANNING_PATH = Path("/multi-agent/reports/handoff/pending_actions.json")

with open(PLANNING_PATH, "r") as f:
    pending_actions = json.load(f)
print(f"✅ Loaded {len(pending_actions)} actions from Planning Team.")
```

Example structure of pending_actions.json:

```         
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
  },
  {
    "task_id": 2,
    "priority": "MEDIUM",
    "action": "Regenerate attention maps for V2 baseline",
    "acceptance_criteria": ["Generate maps for 500 samples"]
  }
]
```

|                                                                                                                                               |
|:-----------------------------------------------------------------------|
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

Validation rules: |
| Every task has a unique task_id.                                                                                                              |
| Priority ∈ {HIGH, MEDIUM, LOW}.                                                                                                               |
| Each task includes acceptance criteria.                                                                                                       |
| Optional metadata (e.g., deadline, owner, dataset path) are retained. 

#### **1.3: Task Queue Initialization** 
```python
task_queue = [
  {
    "task_id": t["task_id"],
    "priority": t["priority"],
    "action": t["action"],
    "acceptance_criteria": t["acceptance_criteria"],
    "status": "pending",
    "preliminary_status": None
  }
  for t in pending_actions
]
```
This normalized queue will be used by Phase 2 (Initialize Executive Team).

#### **1.4: Reflection System Intake** 
```python
reflection_memory = {
  "episode_id": "cycle_2025_10_18",
  "planning_snapshot": {
    "source": str(PLANNING_PATH),
    "tasks": len(task_queue)
  },
  "signals": []
}

with open("/multi-agent/memory/episodic.jsonl", "a") as f:
    f.write(json.dumps(reflection_memory) + "\n")
```
This entry lets the Reflection Service correlate future self-feedback
with the original Planning Team intent — forming the “memory anchor”
for the cycle.

#### **1.5: Phase 1 Output** 
✅ Outputs generated

task_queue.json → normalized queue for execution initialization

Reflection memory anchor in /memory/episodic.jsonl

Schema validated plan snapshot (immutable copy in /ledgers/planning_snapshot_v1.json)
Phase 1 Summary:
----------------
Loaded 2 tasks from Planning Team
Validated ✅ schema v1.2
Prepared task_queue.json for Phase 2
Reflection Service anchored planning context

### **PHASE 2: INITIALIZE EXECUTIVE TEAM** (Cell 8)

**Purpose:**
This phase brings the Execution Team to life by instantiating its three core agents — **Ops Commander**, **Quality & Safety**, and **Infrastructure Monitor**. Together, they form the operational backbone of the autonomous system.
Their creation ensures every incoming task from the Planning Team has a designated executor, reviewer, and validator — each operating within a defined logic boundary and communication protocol.

---

**Process Overview:**
When Phase 2 starts, the system reads the validated `task_queue.json` produced by Phase 1 and spins up the three agents as independent subprocesses or callable API modules. Each agent registers its identity, role, and memory scope under the current cycle ID (e.g., `cycle_2025_10_18`).

The **Ops Commander** handles logic construction and code generation, turning natural-language task descriptions into runnable implementations.
The **Quality & Safety** agent performs real-time static analysis and logical verification, ensuring the generated approach meets compliance and safety constraints.
The **Infrastructure Monitor** checks environmental readiness—verifying hardware, CUDA drivers, dependencies, and performance budgets before any execution can occur.

---

**Internal Initialization Steps:**

1. **Agent Definition and Configuration:**
   Each agent is initialized with its operational schema, which defines:

   * Role name and responsibilities
   * API endpoint or callable interface
   * Memory mode (short-term or long-term)
   * Execution permissions (read-only, simulation, or full-run)

2. **Capability Verification:**
   Each agent conducts a self-diagnostic on startup. The Ops Commander checks model readiness (e.g., Python, PyTorch availability). The Quality agent loads its compliance ruleset. Infrastructure scans GPU availability, disk I/O, and environment variables. These diagnostics are stored under `/multi-agent/logs/initialization/agent_status.jsonl`.

3. **Handshake Protocol:**
   Once initialized, all three agents perform a handshake routine—confirming inter-agent communication channels via the shared `coordination_bus` object. This ensures task payloads can circulate seamlessly across the agents in subsequent phases.

---

**Reflection System Involvement:**
During initialization, the **Reflection Service** creates a **context registry** that associates each agent with its behavioral memory:

* **Episodic Memory** records the current state of agent creation and health checks.
* **Semantic Memory** links initialization anomalies to persistent lessons — for instance, “Infra agent failed GPU detection once due to driver version mismatch; auto-check before execution next time.”
  These entries populate `/memory/episodic.jsonl` and `/memory/semantic.yml`, enabling meta-learning across cycles.

---

**Expected Outputs:**
At the end of Phase 2, the system produces:

* `executive_team_manifest.json` – a registry describing all active agents, their roles, and readiness.
* `agent_status_report.md` – human-readable diagnostic summary with version info, GPU utilization, and memory footprints.
* `reflection_context_init.json` – snapshot of agent creation for feedback alignment.

---

**Narrative Summary:**
Phase 2 is the point where structure becomes intelligence. The system transforms raw planning instructions into coordinated agent entities ready to act. This modular initialization design not only guarantees reproducibility and accountability but also establishes a living memory architecture — allowing the Execution Team to improve autonomously with every new iteration.

---

Next step: **Phase 3 – Task Implementation & Agent Response Collection**, where the newly initialized agents begin interpreting and drafting executable plans.

### **PHASE 3: TASK IMPLEMENTATION & AGENT RESPONSE COLLECTION** (Cells 9–10)

**Purpose:**
In this phase, the Execution Team transitions from passive readiness to active reasoning.
Each task from the queue is dispatched to the three core agents — **Ops Commander**, **Quality & Safety**, and **Infrastructure Monitor** — who independently propose, evaluate, and validate possible implementation plans.

This stage is purely **cognitive**: no execution occurs here. Instead, agents produce structured JSON contracts that define *how* they intend to execute, *why* their plan should work, and *what* resources it will require.

---

**Process Overview:**
Each task is broadcast to all three agents with full context: the task action, its priority, acceptance criteria, and any relevant logs or artifacts from previous runs.

* The **Ops Commander** generates the primary implementation plan in Python (code as text).
* The **Quality & Safety** agent performs a reasoning-level evaluation — flagging logical errors, compliance issues, or unsafe parameters.
* The **Infrastructure Monitor** checks the proposed plan’s resource requirements, environment dependencies, and potential conflicts with current hardware allocation.

Together, their structured outputs are consolidated into a **multi-agent contract document** — the atomic unit of agreement for the approval gate that follows.

---

**Reflection Layer Involvement:**
During this phase, the **Reflection Service** begins listening through its **Parallel Signal Subscription (R0–R1)**.
It collects metadata on:

* How many proposed implementations differ from historical baselines
* Which agents show recurring disagreements or flag conflicts
* The semantic patterns in their reasoning chains (e.g., frequent “GPU mismatch” warnings)

These observations are appended to `/memory/episodic.jsonl` and become learning signals for later retry phases (4.5, 5.5).

---

**Typical Flow Narrative:**

1. A high-priority task, e.g., “Run real attention extraction for CLIP baseline,” is sent to all three agents simultaneously.
2. Ops Commander produces a Python block with MLflow tracking and GPU allocation.
3. Quality & Safety inspects the logic, finds that the batch size may exceed A100 memory limits, and suggests a reduction.
4. Infrastructure validates that CUDA 12 and Torch 2.1 are compatible and confirms low memory usage risk.
5. All responses are converted into structured JSON with verdicts and metadata.

At this point, no execution occurs. The system simply records structured decisions — ready for evaluation by the **Approval Gate** in Phase 4.

---

**Output Artifacts:**

* `task_drafts.json` – full multi-agent contracts for all active tasks
* `/logs/phase3/contracts/*.jsonl` – one file per task
* Reflection notes streamed to `/memory/reflection_buffer.jsonl`

---

**Key Takeaway:**
Phase 3 turns abstract goals into executable intent. It represents the **collective deliberation** of the Execution Team — an interpretable blueprint of what will be executed, under what constraints, and with what confidence. This interpretability is critical for autonomous safety and future reflection alignment.

### **PHASE 4: APPROVAL GATE** (Cell 11)

**Purpose:**
This phase acts as the **multi-agent consensus checkpoint**.
It determines whether a proposed task implementation is reliable enough to move forward to real execution.
The system applies a strict contract-based decision model: every agent (Ops Commander, Quality & Safety, Infrastructure Monitor) must issue an explicit **verdict = “APPROVE”** within its structured JSON response for the task to advance.

This gate ensures that execution never begins from ambiguous or unsafe intent.
It is the **autonomous equivalent of a human design review** — logic is verified, compliance is validated, and infrastructure readiness is confirmed.

---

**Process Overview:**
When Phase 4 begins, it retrieves all `task_drafts.json` objects produced in Phase 3 and parses their structured fields.
For each task, the approval algorithm evaluates three independent signals:

1. **Ops Commander verdict:** “Does the implementation plan compile and logically fulfill the task?”
2. **Quality & Safety verdict:** “Is the plan compliant with all safety and policy constraints?”
3. **Infrastructure verdict:** “Can this plan safely run on the target system without exceeding resource or version limits?”

Only if all three return `"APPROVE"` and no critical flags exist will the task be marked `"approved"` for execution. Otherwise, it’s routed to **Phase 4.5 (Approval Retry Mechanism)**.

---

**Decision Logic Flow:**
Each agent’s JSON contract contains explicit fields that the Approval Gate parses directly:

```json
{
  "ops": { "verdict": "APPROVE", "flags": [], "confidence": 0.94 },
  "quality": { "verdict": "APPROVE", "flags": [], "risk_score": 0.07 },
  "infra": { "verdict": "APPROVE", "env_pass": true, "compatibility": ["CUDA12", "torch2.1"] }
}
```

The evaluation logic is purely structured — **no text parsing or keyword matching.**
This design eliminates ambiguity (e.g., “✅ PASS” or “Looks good” no longer need to be interpreted heuristically).

The gate applies a deterministic rule:

```python
if all(v["verdict"] == "APPROVE" for v in [ops, quality, infra]) and no_critical_flags:
    preliminary_status = "approved"
else:
    preliminary_status = "rejected"
```

The decision is then logged into the run ledger:
`/multi-agent/ledgers/approval_gate.jsonl`
with the following structure:

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

**Reflection Layer Involvement:**
The **Reflection Service** operates in parallel during this phase, passively consuming all approval logs.
It identifies **patterns of disagreement** between agents or repeated rejection causes (e.g., “low confidence from Quality agent” or “missing MLflow spec in Ops response”).
Each rejection triggers a reflection record written to:
`/memory/reflections.jsonl` → used later by **Phase 4.5 (Approval Retry)** to generate targeted fix patches.

The reflection agent assigns a **retry confidence score (0–1)** indicating whether an automatic retry should be attempted, or if human intervention may be required.

---

**Outputs:**

* `task_results.json` → list of all tasks with preliminary approval states.
* `approval_gate.log` → detailed log of verdicts, reasons, and rejection causes.
* `reflections.jsonl` → feedback summary for rejected tasks (Phase 4.5 input).

---

**Summary:**
Phase 4 is the **logical filter of the autonomous pipeline** — ensuring that every executable plan is fully aligned across reasoning, safety, and infrastructure perspectives.
It transforms multiple streams of judgment into a single coherent decision, producing an audit trail that’s both machine-verifiable and human-readable.

### **PHASE 4.5: APPROVAL RETRY MECHANISM**

**Purpose:**
Phase 4.5 is the system’s **self-correction checkpoint** — it activates automatically when a task fails the Approval Gate.
Rather than discarding rejected proposals, this phase re-engages the agents and the Reflection Service to refine, patch, and re-evaluate them.
It embodies the *self-healing* logic of the pipeline: reflection-guided recovery without redundant computation or unsafe retries.

---

**Process Overview:**
When Phase 4 outputs any task with `preliminary_status = "rejected"`, the retry controller inspects the reflection artifacts:

* `reflections.jsonl` – failure summaries and detected patterns
* `job_spec_patch.json` – auto-generated diffs for code/spec corrections
* `risk_scores.json` – confidence and safety bounds

The retry controller filters these cases using two thresholds:

* **Max retries:** 2 (default, configurable)
* **Reflection confidence threshold (τ):** 0.65
  Only tasks whose reflection confidence ≥ τ proceed to a retry; others are deferred to the next planning cycle for manual review.

---

**Detailed Flow:**

1. **Trigger & Intake** – the controller loads all rejected task contracts and links them with the corresponding reflection entries.
2. **Patch Application** – it merges proposed fixes into the Ops Commander’s job spec (e.g., reduce batch size, change seed, rebind GPU).
3. **Re-Submission** – the patched contract is re-evaluated by all three agents through a condensed Approval Gate run.
4. **Outcome Handling** –

   * If all agents now approve → status changes to `"approved_retry"`, forwarded to Phase 5.
   * If still rejected → mark `"failed_retry"` and store reflection summary for analysis.
5. **Logging & Metrics** – every retry is logged with reason, iteration count, and improvement delta in `/multi-agent/ledgers/retry_attempts.jsonl`.

---

**Reflection Integration:**
The Reflection Service is deeply embedded here. It emits live artifacts during retry execution:

* **`reflections.jsonl`** → contextual reasoning on what changed and why.
* **`job_spec_patch.json`** → safe patch set aligned with previous errors.
* **`risk_scores.json`** → weighted evaluation of improvement probability.

This feedback loop lets the Execution Team perform *targeted, cost-bounded self-repair*, rather than re-running the entire approval process.

---

**Outputs:**

* Updated `task_results_retry.json` listing each retried task, patch applied, and new verdict.
* Reflection logs appended to `/memory/episodic.jsonl` and `/memory/semantic.yml` for pattern learning.
* Aggregated retry report: `/reports/retry_phase4_5_summary.md` documenting success rate and risk distribution.

---

**Narrative Summary:**
Phase 4.5 closes the loop between decision and adaptation.
It operationalizes *self-reflection as a control mechanism*: rejected ideas are not failures but opportunities for refinement, guided by measurable confidence and safety thresholds.
This mechanism drastically improves task throughput and robustness by ensuring that the Execution Team learns from its own rejections — transforming static validation into continuous self-improvement.

---

### **PHASE 4.5: APPROVAL RETRY MECHANISM**

**Purpose:**
Phase 4.5 is the system’s **self-correction checkpoint** — it activates automatically when a task fails the Approval Gate.
Rather than discarding rejected proposals, this phase re-engages the agents and the Reflection Service to refine, patch, and re-evaluate them.
It embodies the *self-healing* logic of the pipeline: reflection-guided recovery without redundant computation or unsafe retries.

---

**Process Overview:**
When Phase 4 outputs any task with `preliminary_status = "rejected"`, the retry controller inspects the reflection artifacts:

* `reflections.jsonl` – failure summaries and detected patterns
* `job_spec_patch.json` – auto-generated diffs for code/spec corrections
* `risk_scores.json` – confidence and safety bounds

The retry controller filters these cases using two thresholds:

* **Max retries:** 2 (default, configurable)
* **Reflection confidence threshold (τ):** 0.65
  Only tasks whose reflection confidence ≥ τ proceed to a retry; others are deferred to the next planning cycle for manual review.

---

**Detailed Flow:**

1. **Trigger & Intake** – the controller loads all rejected task contracts and links them with the corresponding reflection entries.
2. **Patch Application** – it merges proposed fixes into the Ops Commander’s job spec (e.g., reduce batch size, change seed, rebind GPU).
3. **Re-Submission** – the patched contract is re-evaluated by all three agents through a condensed Approval Gate run.
4. **Outcome Handling** –

   * If all agents now approve → status changes to `"approved_retry"`, forwarded to Phase 5.
   * If still rejected → mark `"failed_retry"` and store reflection summary for analysis.
5. **Logging & Metrics** – every retry is logged with reason, iteration count, and improvement delta in `/multi-agent/ledgers/retry_attempts.jsonl`.

---

**Reflection Integration:**
The Reflection Service is deeply embedded here. It emits live artifacts during retry execution:

* **`reflections.jsonl`** → contextual reasoning on what changed and why.
* **`job_spec_patch.json`** → safe patch set aligned with previous errors.
* **`risk_scores.json`** → weighted evaluation of improvement probability.

This feedback loop lets the Execution Team perform *targeted, cost-bounded self-repair*, rather than re-running the entire approval process.

---

**Outputs:**

* Updated `task_results_retry.json` listing each retried task, patch applied, and new verdict.
* Reflection logs appended to `/memory/episodic.jsonl` and `/memory/semantic.yml` for pattern learning.
* Aggregated retry report: `/reports/retry_phase4_5_summary.md` documenting success rate and risk distribution.

---

**Narrative Summary:**
Phase 4.5 closes the loop between decision and adaptation.
It operationalizes *self-reflection as a control mechanism*: rejected ideas are not failures but opportunities for refinement, guided by measurable confidence and safety thresholds.
This mechanism drastically improves task throughput and robustness by ensuring that the Execution Team learns from its own rejections — transforming static validation into continuous self-improvement.

### **PHASE 5: EXECUTION** (Cell 11.5)

**Purpose:**
This phase is where planning becomes action — the Execution Team finally runs the approved jobs in a safe, sandboxed environment.
Each task that passed Phases 3–4 (or Phase 4.5 retry) is executed according to its structured job specification, generating real evidence such as MLflow runs, log files, and artifacts.
The phase’s design prioritizes **safety, traceability, and rollback readiness**, ensuring that no unverified or unsafe code reaches runtime.

---

**Process Overview:**
Execution follows a deterministic flow:

1. **Filtering** – Only tasks with `preliminary_status == "approved"` or `"approved_retry"` enter this phase.
2. **Job Spec Extraction** – The system reads the validated job specification object from the Ops Commander’s contract.
3. **Sandbox Launch** – Each job runs in an isolated subprocess or container to prevent dependency or memory contamination.
4. **Execution Monitoring** – The system continuously streams logs, captures telemetry, and tracks anomalies (OOM, timeout, import failure).
5. **Evidence Recording** – Upon completion, the job’s outputs (MLflow run IDs, generated files, metrics) are logged and stored as traceable artifacts.

Every run is trace-linked to its task ID and reflection episode, creating a lineage between *plan, execution, and evidence*.

---

**Reflection Layer Involvement:**
During this phase, the **Reflection Service** actively monitors execution through its **Signal Subscription (R0–R2)** channels.
It mines telemetry for early failure signals — such as unstable GPU memory, missing files, or repeated runtime exceptions.
When anomalies are detected, the Reflection Service emits a **risk signal** to `/memory/reflection_buffer.jsonl` and prepares a **patch suggestion** for Phase 5.5 (Reflection-Guided Retry).
This mechanism enables *proactive self-repair*: the system doesn’t simply fail; it learns from the failure while it happens.

---

**Typical Flow Example:**

1. A validated CLIP baseline task is launched.
2. Execution starts inside the sandbox, loading the ViT-B/32 model and tracking metrics in MLflow.
3. Logs stream continuously to `/logs/execution_cycles/clip_run_001.log`.
4. If execution succeeds, the system registers:

   * `status = "SUCCEEDED"`
   * `mlflow_run_id = "abc123def456"`
   * `artifacts = ["baseline_attention.json", "setup.md"]`
5. If execution fails or crashes (e.g., out-of-memory), the Reflection Service flags it for retry with confidence and patch proposals.

---

**Outputs:**

* `execution_results.json` – detailed per-task results with success/failure metadata
* `/logs/execution_cycles/*.log` – stdout/stderr traces
* `/multi-agent/artifacts/*` – generated files for verification
* Reflection signals emitted to `/memory/reflection_buffer.jsonl`

---

**Narrative Summary:**
Phase 5 is the **operational heartbeat** of the autonomous system.
It transforms verified logic into real, empirical evidence — while maintaining safety constraints and recoverability.
Through the integrated reflection hooks, this stage is not just about running code but about *observing and learning from the act of execution itself*.

---

Next: **Phase 5.5 – Reflection-Guided Retry**, where failed or unstable executions are selectively retried using self-generated patches and confidence-weighted repair logic.

### **PHASE 5.5: REFLECTION-GUIDED RETRY (During Execution)**

**Purpose:**
Phase 5.5 activates when an execution fails, underperforms, or exhibits anomalies during Phase 5.
Rather than halting the workflow, the system performs an *intelligent retry* guided by reflection feedback.
This phase blends execution telemetry with learned self-correction patterns, enabling the agent system to recover automatically from transient or predictable failures.

---

**Process Overview:**
When a task’s execution result is tagged as `"FAILED"` or `"RETRYABLE_FAILURE"`,
the controller fetches the latest artifacts from the Reflection Service:

* **`reflections.jsonl`** – diagnostic reasoning about why execution failed.
* **`job_spec_patch.json`** – safe patch proposals derived from previous cycle experience.
* **`risk_scores.json`** – confidence, severity, and cost trade-offs for retry.

The retry decision is based on a **confidence threshold** (`τ_retry`, typically 0.7) and a **retry limit** (max 2 per task).
If both criteria are satisfied, the retry pipeline reconfigures the sandbox, applies patches, and reruns the task safely.

---

**Detailed Steps:**

1. **Trigger & Intake**

   * Load the failed task’s telemetry from `/logs/execution_cycles/*.log`
   * Ingest the corresponding reflection metadata for that task ID.

2. **Patch Application**

   * Integrate fixes from `job_spec_patch.json` (e.g., reduce batch size, add seed, fix file path).
   * Rebuild the sandbox environment with corrected configurations.

3. **Selective Retry Execution**

   * Launch a new run in isolation (`retry_mode = True`).
   * Apply stricter monitoring thresholds (e.g., shorter timeout, capped GPU memory).
   * Stream live telemetry to `/logs/retries/phase5_5_taskX.log`.

4. **Outcome Evaluation**

   * If the retry succeeds → mark `"recovered_success"`.
   * If it fails again → log as `"failed_retry"`, and Reflection Service updates the memory rule set for next cycle learning.

5. **Post-Retry Reflection Update**

   * The Reflection Service logs success/failure deltas.
   * Updates episodic memory (`/memory/episodic.jsonl`) and semantic memory (`/memory/semantic.yml`) with learned correction rules.

---

**Reflection System Role:**
The Reflection Service not only proposes patches but also *learns from the retry itself*.
For instance:

> “Reducing batch size from 32 → 16 eliminated OOM; this fix pattern increases reliability by 40% under similar GPU conditions.”

Such observations become **semantic rules** for future initialization phases, leading to self-stabilizing model behavior over time.

---

**Outputs:**

* `execution_retry_results.json` – detailed retry outcomes per task
* `reflection_summary.md` – what worked, what didn’t, and confidence evolution
* Updated entries in `/memory/episodic.jsonl` (event record) and `/memory/semantic.yml` (long-term rule)

---

**Narrative Summary:**
Phase 5.5 represents the **adaptive intelligence** of the Execution Team — a bridge between failure and resilience.
Instead of viewing errors as endpoints, the system treats them as feedback signals, applying learned corrections in real-time.
This ensures the autonomous pipeline continuously improves, achieving higher reliability without human intervention.

---

Next: **Phase 6 – Verification**, where both original and retried executions are validated for real evidence and artifact integrity.


### **PHASE 6: VERIFICATION** (Cell 17)

**Purpose:**
After execution (and any retries), this phase validates that each approved task has produced *real, verifiable evidence*—not merely logs or claimed outputs.
Verification acts as the **empirical safeguard** of the system: it ensures that every claimed success corresponds to an actual MLflow record, tangible artifacts, and measurable outcomes.
This phase closes the evidence loop before finalizing any task status.

---

**Process Overview:**
Verification operates in three layers:

1. **Task Classification** – determines what kind of evidence each task requires (e.g., MLflow runs for experiments, files for documentation).
2. **Evidence Validation** – cross-checks recorded MLflow run IDs, output artifacts, and metrics against the expected schema.
3. **Verdict Assignment** – assigns a binary `verification = "pass"` or `"fail"`, recording reasoning and missing evidence details.

Both original and reflection-guided retries from Phase 5.5 are included in this process, giving retried runs an equal opportunity to pass verification.

---

**Detailed Flow Narrative:**

* For experimental or execution-based tasks (like “Run baseline CLIP attention”), the system confirms that:

  * A real MLflow run exists and is accessible via API query.
  * Required artifacts (e.g., `baseline_attention.json`, `setup.md`) exist on disk and match expected size thresholds.
* For documentation-type tasks (e.g., “Draft validation summary”), MLflow is optional; verification focuses on output file existence and content hash validation.
* Failed checks are annotated with structured error codes such as `EVIDENCE_MISSING`, `RUN_NOT_FOUND`, or `ARTIFACT_CORRUPTED`.
  All results are written to `/multi-agent/logs/verification/phase6_results.jsonl` for downstream aggregation.

---

**Reflection System Role:**
The Reflection Service simultaneously listens for evidence anomalies through its **R0–R3 signal stream**.
When evidence is missing or incomplete, it emits feedback into `/memory/reflection_buffer.jsonl` and proposes targeted remediation instructions.
This triggers **Phase 6.5 (Verification Retry)** if the reflection confidence exceeds the minimum threshold.
Reflection thereby allows the system to attempt self-correction—rebuilding artifacts, re-querying MLflow, or reconstructing logs—before declaring a final failure.

---

**Outputs:**

* `verification_results.json` → list of all verified tasks and their evidence verdicts.
* `/logs/verification/` → audit trail of MLflow and file checks.
* Reflection updates streamed to `/memory/episodic.jsonl` and `/memory/semantic.yml` for learning across cycles.

---

**Narrative Summary:**
Phase 6 converts raw execution outcomes into verifiable truth.
It ensures that the pipeline’s claims are grounded in measurable results and creates a forensic record of every validation check.
This design makes the autonomous system not just self-running but **self-auditing**, enforcing scientific rigor before any success is finalized.

---

Next: **Phase 6.5 – Verification Retry Mechanism**, where missing or incomplete evidence triggers targeted self-repair actions guided by reflection confidence and previously learned correction patterns.

### **PHASE 6.5: VERIFICATION RETRY MECHANISM**

**Purpose:**
When Verification (Phase 6) fails due to missing or inconsistent evidence, Phase 6.5 attempts a **targeted, low-cost recovery**. Instead of declaring failure immediately, the system leverages reflection-guided fixes to **reconstruct evidence**, **re-run lightweight checks**, or **re-query sources** (e.g., MLflow) so that truly successful executions aren’t lost to incidental logging or artifact issues.

---

**Process Overview:**
Phase 6.5 is a short, bounded loop that operates only on tasks with `verification == "fail"` and where the Reflection Service indicates that a retry is **safe and likely to succeed** (confidence ≥ τ). It prioritizes **artifact regeneration and evidence reconciliation** over full re-execution:

1. **Intake & Diagnosis** – Load the task’s verification report (what’s missing or invalid) and the corresponding reflection note (why it likely failed verification).
2. **Patch & Plan** – Apply a minimal plan from `job_spec_patch.json` (e.g., re-export artifacts; fix output path; re-attach MLflow tags) without changing the experiment’s scientific content.
3. **Selective Remediation** – Re-query MLflow; regenerate manifest files; re-upload or re-hash artifacts; rebuild missing files from cached intermediates/logs.
4. **Re-Check** – Re-run the Verification checks from Phase 6 (MLflow + file integrity).
5. **Stop Conditions** – If evidence passes → proceed; if not and attempts < max, iterate once more; otherwise, stop and record a terminal verification failure.

---

**Detailed Steps:**

* **Step A – Evidence Delta Analysis:**
  Compare the **expected_artifacts** from the job spec against the **artifact manifest** found in the execution ledger. Identify minimal deltas (e.g., “file exists but size=0”, “hash mismatch”, “MLflow run present but missing tag”).

* **Step B – Minimal Patch Application:**
  Use reflection suggestions to update non-invasive parts of the job spec or metadata (e.g., add `mlflow.log_artifact(...)`, re-point output directory, restore `run_id` into the summary). Patches must pass safety guards (no permission escalation, no unbounded re-runs).

* **Step C – Remediation Actions (lightweight):**

  * **MLflow:** Attempt `mlflow.get_run(run_id)` again; if found, backfill missing params/tags.
  * **Artifacts:** Re-materialize derived files from cached tensors/logs; re-hash and write a fresh manifest.
  * **Index/Manifests:** Recreate `artifacts_manifest.json` and link to MLflow.

* **Step D – Verification Re-Check:**
  Execute the full Phase 6 checks. If **pass**, tag as `verification_passed_after_retry`; else, increment attempt counter and consider another bounded retry.

* **Step E – Provenance & Audit:**
  Append an immutable entry to `/multi-agent/logs/verification/phase6_5_attempts.jsonl` detailing: attempt number, patch IDs, deltas fixed, and the new verification verdict.

---

**Reflection System Role:**
The Reflection Service provides **diagnostic specificity** (e.g., “artifact path mismatch” vs “missing MLflow tag”), plus a **retry confidence** derived from history and current signals. It also learns from outcomes:

* Successful remediations become **semantic rules** (e.g., “always emit `artifacts_manifest.json` alongside model outputs”).
* Frequent failure motifs are promoted to **procedural templates** that pre-patch specs in future cycles (Phase 2/3).

---

**Outputs:**

* `verification_retry_results.json` – per-task status after remediation (`verification_passed_after_retry` or `failed_final`).
* Updated ledgers: `/logs/verification/phase6_5_attempts.jsonl`, refreshed `artifacts_manifest.json`, and MLflow tag corrections if applicable.
* Memory updates: episodic record of the attempt; semantic rule addition when a remediation pattern proves repeatedly effective.

---

**Narrative Summary:**
Phase 6.5 prevents “paper failures” by repairing **evidence plumbing** rather than recomputing experiments. It’s **fast, conservative, and audit-friendly**: only minimal, safe changes are permitted; every action is versioned; and outcomes feed back into reflection memory so the system gets **better at avoiding verification pitfalls** over time.


### **PHASE 7: FINAL STATUS DETERMINATION** (Cell 17)

**Purpose:**
Convert all upstream signals (approval verdicts, execution outcomes, verification results, and any retry summaries) into a **single, auditable terminal status** per task. Phase 7 is the bookkeeping and governance checkpoint: it closes the loop with clear success/failure semantics, provenance, and reasons that can be consumed by reporting and the next planning cycle.

**Process Overview:**
For each task, Phase 7 reads the **canonical records** produced earlier: the approval decision from Phase 4 (or 4.5), the execution/telemetry bundle from Phase 5 (and 5.5), and the verification verdict from Phase 6 (and 6.5). It then applies an ordered set of rules to produce one of a small set of terminal outcomes (e.g., `completed`, `failed`, `failed_final`). The rules are deterministic and preference is given to **verification truth**: if verification passed, the task is completed; if verification failed after exhausting bounded retries, the task is marked failed_final; if the task never passed the approval gate, it is failed with an explicit “approval gate” reason.

**What’s Decided:**

* **Status:** `completed` | `failed` | `failed_final`
* **Reason:** concise, human-readable rationale (e.g., “Approved + evidence verified”, “Did not pass 3-agent approval gate”, “Evidence missing after max retries”).
* **Provenance:** compact bundle of identifiers and links (number of attempts across 4.5/5.5/6.5, MLflow run IDs, artifact manifest references, ledger paths).
* **Metrics:** attempt counts, whether success came **after retry**, and any regression/rollback notes from 5.5.

**Reflection Layer Involvement:**
Phase 7 emits a **terminal snapshot** to the Reflection Service. This is non-blocking and serves two purposes: (1) it records the final causal chain for learning (what sequence of fixes led to success or failure), and (2) it allows reflection memory to promote recurring patterns into **semantic rules** and **procedural templates** (e.g., if multiple tasks only succeed after adding `artifacts_manifest.json`, that rule is promoted for pre-patching in future cycles).

**Outputs:**

* A single consolidated **final status record** per task (JSON line), suitable for downstream analytics and dashboards.
* Updated memory hooks when a pattern merits promotion (e.g., failed_final due to repeatable evidence issue).
* A compact **reporting event** that Phase 8 consumes to assemble cycle summaries, KPI rollups, and the next-cycle handoff.

**Narrative Summary:**
Phase 7 is the “closing argument” of the execution cycle. It enforces consistent, explainable outcomes and ties each decision back to concrete evidence. By standardizing terminal states and reasons, it keeps the system transparent and audit-ready, while feeding just enough structured insight back into reflection and planning to **continuously improve** the next iteration.

### **PHASE 8: REPORTING & HANDOFF** (Cells 14, 16, 19)

**Purpose:**
Turn the raw trail of evidence and decisions into ** clean, auditable artifacts** for stakeholders and into **actionable inputs** for the next Planning cycle. Phase 8 is where the system consolidates results, rolls up KPIs, promotes durable learnings into memory, and publishes a versioned, reproducible handoff (`pending_actions.json`) to restart the loop.

---

**Process Overview:**
Phase 8 ingests the **canonical final status** records from Phase 7, joins them with execution ledgers, verification logs, and MLflow metadata, and emits two synchronized outputs:

1. a **human report** with narrative insights, and 2) a **machine bundle** (JSON manifests + checksums) suitable for programmatic consumption and long-term archiving.

Typical steps:

1. **Aggregation & KPI Rollup** – Merge `task_results` with run ledgers and artifact indices; compute KPIs: completion rate, retry success rate (4.5/5.5/6.5), evidence coverage %, average attempts to success, top failure root causes, time-to-evidence.
2. **Report Generation** – Produce `summary.json` (machine-readable) and `report.md`/PDF (human). Include links to MLflow runs, artifact manifests, and log bundles. Highlight “wins after retry” and “failed_final with causes.”
3. **Governance & Provenance** – Redact secrets/PII, attach schema versions and SHA256 checksums, sign the bundle with a cycle tag (e.g., `v3.2_cycle_2025-10-18`).
4. **Memory Promotion** – Fold recurring, validated fixes from reflection into `semantic.yml` (rules) and copy stable job specs into `procedural_cache/` (templates). Prune `episodic.jsonl` to top-K per task to prevent drift.
5. **Handoff Construction** – Compose the **next-cycle** `pending_actions.json` by:

   * Carrying forward `failed_final` and unresolved items with structured reasons and suggested pre-patches.
   * Adding new opportunities derived from reflection patterns (e.g., “standardize artifacts_manifest for experiments”).
   * Embedding acceptance criteria and guardrails learned this cycle.
6. **Notify Planning** – Write a compact manifest with counts, URIs, and checksums; emit a completion event to the Planning Team: “Execution cycle complete—handoff ready.”

---

**Reflection Layer Involvement:**
Reflection runs **in parallel** here to produce `reflection_summary.md`: a concise roll-up of pattern discoveries (effective patches, frequent misconfigs, high-leverage rules). Phase 8 consumes that summary to:

* Promote rules into `semantic.yml` (e.g., “add MLflow tags X,Y on every run”),
* Update `procedural_cache/` with hardened, reusable job_spec templates, and
* Seed **pre-patch hints** that Planning Phase 2 will load before the next agent initialization.

---

**Outputs:**

* `/reports/execution_summary/summary.json` – KPIs, per-task terminal states, links to evidence.
* `/reports/execution_summary/report.md` (and PDF) – human narrative with charts/tables and MLflow links.
* `/multi-agent/archives/cycle_<id>.tar.gz` – logs/artifacts snapshot with checksums and retention notes.
* `/memory/semantic.yml` (updated), `/memory/procedural_cache/` (new/updated templates), pruned `/memory/episodic.jsonl`.
* `/reports/handoff/pending_actions.json` – **versioned** next-cycle plan (with pre-patches and acceptance criteria).
* `/reports/handoff/manifest.json` – bundle metadata: version tag, schema versions, counts, and SHA256 hashes.

---

**Narrative Summary:**
Phase 8 closes the loop with **evidence, learning, and continuity**. It turns a noisy stream of executions and retries into a signed, reproducible record—and it translates those learnings into concrete improvements for the next cycle. By the time the Planning Team picks up `pending_actions.json`, the system has already encoded what worked, what didn’t, and what to fix preemptively—ensuring the next iteration starts smarter than the last.

### **REFLECTION SYSTEM: PARALLEL SELF-FEEDBACK & MEMORY ENGINE**

**Purpose:**
The Reflection System is the *autonomous introspection layer* that runs in parallel with the execution workflow. Its mission is to **observe, analyze, and learn** from every decision and outcome — without interrupting the main process. It transforms logs, errors, and metrics into structured insights, producing actionable artifacts (`job_spec_patch.json`, `risk_scores.json`, `reflections.jsonl`) and continuously improving the system’s decision quality across cycles.

---

### **Operational Overview**

The Reflection System listens non-blockingly to all major execution phases (3–6), operating asynchronously through a structured feedback pipeline:

1. **Parallel Observation:** Subscribes to logs, approval verdicts, and telemetry as tasks are processed.
2. **Analysis & Diagnosis:** Detects anomalies, infers causes, estimates confidence, and classifies risk.
3. **Patch Generation:** Synthesizes lightweight, safe corrective actions that can be applied in retry phases.
4. **Policy Suggestion:** Decides whether to RETRY, ESCALATE, or ABORT based on impact vs. confidence.
5. **Memory Update:** Encodes both short-term (episodic) and long-term (semantic) knowledge for continual adaptation.

---

### **Architecture: Modules & Flow**

#### **R0–R1: Parallel Listening & Data Ingestion**

* Constantly monitors logs, stdout/stderr, MLflow telemetry, and approval records.
* Normalizes all data into a standard structure: `{task_id, phase, timestamp, signal_type, content}`.
* Operates asynchronously — does **not** block or slow execution flow.

#### **R2–R3: Error Mining & Root-Cause Evaluation**

* Detects patterns like: timeout, OOM, missing artifacts, schema mismatch, or repeated rejection patterns.
* Performs clustering to identify underlying causes (e.g., resource misallocation vs. code logic flaw).
* Computes three metrics per issue:

  * **Confidence:** probability that diagnosis is correct
  * **Severity:** operational impact (low → critical)
  * **Safety Impact:** potential to corrupt data or regress results

#### **R4–R6: Reflection Note, Patch, and Policy Formation**

* **R4 – Reflection Note Synthesis:** Generates structured explanation:
  “*Task 12 failed due to MLflow artifact path mismatch. Proposed fix: rebind artifact path using env var.*”
* **R5 – Patch Generation:** Converts insight into small, safe diffs for the job spec or environment (e.g., adjusting batch size, logging missing key).
* **R6 – Risk Scoring:** Combines severity, confidence, and cost to compute retry priority:
  `retry_priority = f(confidence, severity, recovery_cost)`.

#### **R7–R8: Artifact Emission & Retry Integration**

* Emits reflection outputs consumed by retry phases (4.5, 5.5, 6.5):

  * `reflections.jsonl` – detailed diagnostics per task
  * `job_spec_patch.json` – concrete diffs for fix proposals
  * `risk_scores.json` – retry priority and expected safety margins
* These artifacts are automatically read by the Executive Team when retry conditions are triggered.

---

### **Memory Subsystem**

#### **Episodic Memory (Short-Term)**

* Stores reflection data for the current cycle only.
* Tracks each task’s failures, retries, and success outcomes.
* File: `/memory/episodic.jsonl`
* Used for immediate reflection-driven retries (e.g., 4.5, 5.5, 6.5).

#### **Semantic Memory (Long-Term)**

* Aggregates recurring reflection patterns across cycles.
* Converts repeated insights into generalizable rules:

  > “Reduce batch size to 8 when OOM on CLIP-eval.”
* File: `/memory/semantic.yml`
* Loaded during Phase 2 initialization for **pre-patching** future job specs.

#### **Procedural Cache (Stable Templates)**

* Stores full job_spec variants that have repeatedly succeeded.
* Serves as *fast-start templates* for the next cycle.
* Directory: `/memory/procedural_cache/`
* Promoted when a reflection pattern proves robust over ≥3 cycles.

---

### **Lifecycle & Interaction with Execution Phases**

| Execution Phase                    | Reflection Role                                                | Outputs Consumed / Emitted          |
| ---------------------------------- | -------------------------------------------------------------- | ----------------------------------- |
| **3 – Task Implementation**        | Capture agent drafts & rejections for structural/risk patterns | reflections.jsonl                   |
| **4 – Approval Gate**              | Monitor verdicts; detect reasoning mismatches or logic flaws   | job_spec_patch.json (pre-patch)     |
| **4.5 – Approval Retry**           | Provide patch suggestions for rejected tasks                   | reflections.jsonl, risk_scores.json |
| **5 – Execution**                  | Real-time telemetry monitoring; detect execution anomalies     | reflections.jsonl                   |
| **5.5 – Execution Retry**          | Apply reflection-driven code/env patches                       | job_spec_patch.json                 |
| **6 – Verification**               | Observe evidence failures (e.g., missing artifacts)            | risk_scores.json                    |
| **6.5 – Verification Retry**       | Propose lightweight artifact or MLflow remediations            | reflections.jsonl                   |
| **7–8 – Finalization & Reporting** | Consolidate lessons and update memory                          | reflection_summary.md, semantic.yml |

---

### **Outputs Summary**

* **reflections.jsonl** – All reflection events with phase, cause, and context.
* **job_spec_patch.json** – Auto-generated microfixes.
* **risk_scores.json** – Retry confidence and safety evaluation.
* **reflection_summary.md** – Human-readable summary for reporting.
* **/memory/episodic.jsonl** – Cycle-level reflection history.
* **/memory/semantic.yml** – Long-term reflection rules.
* **/memory/procedural_cache/** – Stable reusable job templates.

---

### **Narrative Summary**

The Reflection System functions as a **cognitive layer** parallel to execution — silently learning while the agents work. It transforms runtime telemetry and decision trails into **structured knowledge** that continuously closes the feedback loop.
Where the Executive Team *acts*, the Reflection System *learns*; where execution retries *fix*, reflection *teaches*.

By integrating both short-term adaptive retries and long-term knowledge accumulation, it ensures that every failure leaves a trace — and every trace sharpens the next cycle’s intelligence.



┌────────────────────────────────────────────────────────────────┐
│                         PLANNING TEAM                          │
│                                                                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ Strategic    │   │ Empirical    │   │ Critical     │        │
│  │  Leader      │   │ Validation   │   │ Evaluator    │        │
│  │  (Claude)    │   │   Lead       │   │  (GPT-4)     │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│           │                 │                 │                │
│           └─────────────────┼─────────────────┘                │
│                             │                                  │
│                   ┌─────────▼─────────┐                        │
│                   │ Gemini Research   │                        │
│                   │     Advisor       │                        │
│                   └─────────┬─────────┘                        │
│                             │                                  │
│                   ┌─────────▼─────────┐                        │
│                   │ pending_actions   │                        │
│                   │     .json         │                        │
│                   └─────────┬─────────┘                        │
└─────────────────────────────┼──────────────────────────────────┘
│
│ Handoff → Execution Phase 1
│
┌─────────────────────────────▼──────────────────────────────────┐
│                         EXECUTIVE TEAM                         │
│                                                                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ Ops          │   │ Quality &    │   │ Infrastructure│       │
│  │ Commander    │   │  Safety      │   │  Monitor      │       │
│  │  (Claude)    │   │  (Claude)    │   │  (Claude)     │       │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                   │                   │              │
│         │   ┌───────────────▼──────────────┐     │             │
│         └──►│  3-Agent Approval Gate       │◄────┘             │
│             │ (Phase 3 – All must approve) │                   │
│             └───────────────┬──────────────┘                   │
│                             │                                  │
│                     ┌───────▼────────┐                         │
│                     │  Execution     │                         │
│                     │  & Evidence    │                         │
│                     │  Verification  │                         │
│                     │ (Phases 4–6.5) │                         │
│                     └───────┬────────┘                         │
│                             │                                  │
│                     ┌───────▼────────┐                         │
│                     │ execution_     │                         │
│                     │ progress.md    │                         │
│                     └───────┬────────┘                         │
└─────────────────────────────┼──────────────────────────────────┘
│
│ Handoff Back → Planning Phase 1
│
▼
Next Planning Cycle (Continuous Learning Loop)


══════════════════ PARALLEL REFLECTION SERVICE ══════════════════
│
│  • Continuously monitors all Execution Phases (3 → 6.5)
│  • Provides self-feedback artifacts for retry and planning
│  • Memory System retains episodic, semantic, and procedural data
│
│ ┌───────────────────────────────────────────────────────────────────┐
│ │                      REFLECTION SERVICE                           │
│ │    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│ │    │  Error Miner │   │   Evaluator  │   │ Patch Builder│         │
│ │    └──────────────┘   └──────────────┘   └──────────────┘         │
│ │             │                │                   │                │
│ │             └────────────────┼───────────────────┘                │
│ │                              │                                    │
│ │       ┌──────────────────────▼───────────────────────┐            │
│ │       │           Policy & Memory Manager            │            │
│ │       │.     • Decide: retry / escalate / abort      │            │
│ │       │• Maintain episodic.jsonl, semantic.yml, procedural_cache/ │
│ │       └──────────────────────┬────────────────────────┘           │
│ │                              │                                    │
│ │        ┌─────────────────────▼───────────────────────┐            │
│ │        │ Emits: job_spec_patch.json, risk_scores.json,            │
│ │        │ reflections.jsonl, reflection_summary.md                 │
│ │        └────────────────────┬────────────────────────┘            │
│ │                             │                                     │
│ │  ↳ Feeds retry logic at Execution Phases 3.5 / 4.5 / 6.5          │
│ │  ↳ Updates planning memory before next cycle (Phase 2)            │
│ └───────────────────────────────────────────────────────────────────┘
│
═════════════════════════════════════════════════════════════════




Execution Team Work Flow (v3.5 – with Dual Retry Points)

┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Read Planning Team’s Decisions                         │
│ Cell 6: Load pending_actions.json                               │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Initialize Executive Team (3 Agents)                   │
│ Cell 8: Create Ops Commander, Quality & Safety, Infrastructure  │
└─────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────
Parallel System: Reflection Service (monitors Phases 3–6)
Outputs: reflections.jsonl, job_spec_patch.json, risk_scores.json
──────────────────────────────────────────────────────────────────
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Task Implementation & Agent Response Collection        │
│ Agents produce draft plans, code, or analyses                   │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Approval                                               │
│ Cell 11: 3-Agent Approval Gate – Plan Review                    │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4.5: Approval Retry Mechanism                             │
│ Triggered if any agent rejects or fails the plan                │
│ • Apply reflection patch / fix proposal                         │
│ • Re-submit to Approval Gate (Phase 4)                          │
│ • Max retries = 2 | Skip if reflection confidence < τ           │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: Execution                                              │
│ Cell 11.5: Automatic Code Executor – Run Approved Plans         │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5.5: Reflection-Guided Retry (During Execution)           │
│ • Consumes reflection outputs (patch + score)                   │
│ • Runs safely within sandbox, rollback if regression detected   │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6: Verification                                           │
│ Cell 17: Evidence Check – Verify Real Results Exist             │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6.5: Verification Retry Mechanism                         │
│ • Triggered when evidence incomplete or corrupted               │
│ • Re-run diagnostic jobs, regenerate artifacts, sync MLflow     │
│ • Escalate to Planning Team if repeated failures                │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 7: Final Status Determination                             │
│ Assign outcome (completed / failed) based on verification pass  │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 8: Reporting & Handoff (Next Cycle Initialization)        │
│ Generate reports → seed next pending_actions.json               │
└─────────────────────────────────────────────────────────────────┘


# Parallel Non-Blocking System
┌─────────────────────────────────────────────────────────────────┐
│ Reflection Service (Parallel)                                   │
│ • Observes Phases 3–5 in real time (logs, metrics, MLflow)      │
│ • Produces: reflections.jsonl, job_spec_patch.json, risk_scores │
│ • Updates: episodic.jsonl → semantic.yml → procedural_cache/    │
│ • Feeds Retry Phases 3.5, 4.5, and 5.5                          │
└─────────────────────────────────────────────────────────────────┘



Reflection & memory system 
┌─────────────────────────────────────────────────────────────────┐
│ AUTONOMOUS COORDINATOR                                          │
│ Oversees synchronization and message passing between            │
│ Reflection Service, Planning Team, and Executive Team.          │
│ Ensures parallel read-only reflection, time-budget enforcement, │
│ and safe routing of patches to retry phases.                    │
└─────────────────────────────────────────────────────────────────┘
│
↔ Reflection runs concurrently with Execution (Phases 3–6)
<< read-only telemetry  |  write-back: reflection artifacts >>
┌────────────────────────────────────────────────────────────────┐
│ REFLECTION SERVICE (Parallel Process)                          │
│  • Error Miner – extract failure signals from logs/metrics     │
│  • Evaluator – identify root cause, confidence, risk level     │
│  • Patch Generator – propose job_spec/code/env modifications   │
│  • Policy Manager – decide retry / escalate / abort            │
│  • Memory System – store episodic, semantic & procedural data  │
└──────────┬─────────────────────────────────┬───────────────────┘
           │                                 │
      Input Sources                  Output Artifacts
           │                                 │
 ┌─────────▼───────────┐    ┌────────────────▼─────────────────────┐
 │ From Planning Team  │    │ reflections.jsonl (per attempt)      │
 │  • decisions.json   │    │ job_spec_patch.json (safe diffs)     │
 │  • actions.json     │    │ risk_scores.json (confidence log)    │
 │ From Execution Team │    │ memory/{episodic.jsonl, semantic.yml,│
 │  • logs/, metrics/, │    │          procedural_cache/}          │
 │    MLflow runs      │    │ reflection_summary.md (aggregated)   │
 │                     │    └─────────────────┬────────────────────┘
 └─────────┬───────────┘                      │
           │                                  │
 ┌─────────▼────────────────┐      ┌──────────▼─────────────────┐
 │ EXECUTION TEAM (3.5–6.5) │ <────│ Retry Agents (3.5/4.5/6.5) │
 │ (reads patches & scores) │      │ apply selective retry      │
 └──────────────────────────┘      └────────────────────────────┘
Reflection System Workflow (v3 — Parallel + Memory Integrated)
Runs continuously beside Execution Phases 3–6, asynchronously generating self-feedback artifacts.

Parallel Listening Layer
┌─────────────────────────────────────────────────────────────────┐
│ R0: Signal Subscription (Non-Blocking Listener)                 │
│ Watches: Approval logs, Execution telemetry, Verification output│
│ Streams data without pausing or modifying main execution.       │
└─────────────────────────────────────────────────────────────────┘
↓

Processing Pipeline
┌─────────────────────────────────────────────────────────────────┐
│ R1: Data Ingestion & Validation                                 │
│ Normalize → {task_id, attempt_id, phase, timestamp}.            │
│ Validate schema & compute checksums to prevent data drift.      │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ R2: Error & Signal Mining                                       │
│ Detect anomalies (OOM, timeout, schema mismatch, low metric).   │
│ Record both negative and successful runs for balance.           │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ R3: Root-Cause Evaluation                                       │
│ Cluster failure signals → infer causes; compute severity/risk.  │
│ confidence = α·model + β·human + γ·history weighting.           │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────────┐
│ R4: Reflection Note Synthesis                                       │
│ Create structured reflection_notes: what failed, why, fix, conf.    │
│ Append to reflections.jsonl with reflection_type=[plan|exec|verify].│
└─────────────────────────────────────────────────────────────────────┘
↓

Patch & Policy Layer
┌─────────────────────────────────────────────────────────────────┐
│ R5: Patch Generation                                            │
│ Produce safe job_spec/code/env diffs (sandbox-scanned).         │
│ Example: adjust batch size, add seed, rebind GPU, enforce limit.│
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ R6: Risk & Confidence Scoring                                   │
│ Estimate improvement & safety margin.                           │
│ retry_priority = f(confidence, severity, resource_cost).        │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ R7: Policy Suggestion                                           │
│ Recommend RETRY / ESCALATE / ABORT.                             │
│ Respect retry limits and confidence threshold τ.                │
└─────────────────────────────────────────────────────────────────┘

↓

Output Handoff
┌───────────────────────────────────────────────────────────────────┐
│ R8: Artifact Emission                                             │
│ Emit artifacts every N seconds or M attempts (batching).          │
│ Feeds Execution Phases 3.5 / 4.5 / 6.5 for reflection retries.    │
│ Outputs: reflections.jsonl, job_spec_patch.json, risk_scores.json │
└───────────────────────────────────────────────────────────────────┘
↓

Memory System (Adaptive + Decay)
┌─────────────────────────────────────────────────────────────────┐
│ R9: Episodic Memory (Short-Term)                                │
│ Logs attempt-level reflections (TTL = current cycle).           │
│ Auto-prune oldest K entries per task to prevent bloat.          │
│ File: /memory/episodic.jsonl                                    │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ R10: Semantic Memory (Long-Term)                                │
│ Aggregate recurring patterns into general heuristics.           │
│ Adds decay weighting: older rules down-ranked over time.        │
│ File: /memory/semantic.yml                                      │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ R11: Procedural Cache (Stable Templates)                        │
│ Stores proven job_spec variants. Linked to next pending_actions.│
│ Folder: /memory/procedural_cache/                               │
└─────────────────────────────────────────────────────────────────┘
↓

Reporting & Loop Closure
┌──────────────────────────────────────────────────────────────────┐
│ R12: Reflection Roll-up                                          │
│ Aggregate metrics: retry success %, avg confidence, evidence gain│
│ Output → reflection_summary.md for Planning Team.                │
│ Feeds Execution retry (3.5/4.5/6.5) and Planning pre-patching.   │
└──────────────────────────────────────────────────────────────────┘


Planning Team Work Flow (v2)

┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Read Executive Team’s Execution Results                │
│ Cell 6: Load execution_progress_update_*.md + results.json      │
│ Validate schema & integrity; register run summaries in memory.  │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Initialize Planning Team (4 Agents)                    │
│ Cell 8: Create Strategic Leader, Empirical Lead,                │
│         Critical Evaluator, Gemini Research Advisor             │
│ Sync previous reflection_summary.md + semantic.yml              │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Review Meeting                                         │
│ Cell 10: Analyze results, assess GO / PAUSE / PIVOT decisions   │
│ Reference reflection patterns and retry metrics.                │
│ Produce meeting log + decision_report.md                        │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Next Cycle Generation                                  │
│ Cell 12: Draft pending_actions.json for next execution cycle    │
│ Integrate reflection rules (semantic.yml) → pre-patched configs │
│ Assign task priorities & acceptance criteria                    │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: Save & Upload                                          │
│ Cell 13: Save new pending_actions.json to Google Drive          │
│ Validate JSON schema, then sync to repository trigger directory │
│ (Automatically triggers new Execution Cycle)                    │
└─────────────────────────────────────────────────────────────────┘




