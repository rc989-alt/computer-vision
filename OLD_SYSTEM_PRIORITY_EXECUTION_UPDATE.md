---
title: "Complete System Flow"
output:
  pdf_document:
    toc: false
    latex_engine: xelatex
    keep_tex: true
mainfont: "Courier New"
monofont: "Courier New"
---






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
│  └─→ ⚙️ Infrastructure – runtime & resource validation           │
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
│ (Triggered when any agent rejects or flags critical issues)     │
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


