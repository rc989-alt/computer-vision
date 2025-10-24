# Benchmark Task Suite (v1)

This suite defines 15 lightweight research execution tasks with clear goals, inputs, and acceptance criteria. Use these to evaluate Baseline A (Single Agent), Baseline B (Reflection without governance), and Ours (Full System).

Notes
- Each task should run via `prepare_execution_cycle()` + `run_execution_cycle()` with IDs selected below.
- Acceptance criteria intentionally reference MLflow run IDs and artifacts to enable verifiable scoring.

---

## Task T01 — CLIP Attention Extraction
- Goal: Export CLIP cross‑attention maps for a given image/text pair.
- Inputs: `TEST_IMAGE_PATH`, prompt="a dog on a beach".
- Acceptance: MLflow run logged with artifact `attentions/*.npy` and a preview `attn_grid.png`.
- Notes: Use built‑in attention patch; no code edits allowed.

## Task T02 — Evidence Regeneration (Minimal Script)
- Goal: When evidence is missing, generate a minimal MLflow script to re‑produce a run_id.
- Inputs: Broken prior run context (no run_id).
- Acceptance: New valid `run_id` produced and verified in Phase 6; logs include template script path.

## Task T03 — Import Error Recovery
- Goal: Detect and fix a missing import during execution.
- Inputs: Script with `ModuleNotFoundError`.
- Acceptance: Patch applied at Phase 5.5; subsequent Phase 5 succeeds; provenance recorded in ledgers.

## Task T04 — Data Path Validation
- Goal: Validate dataset path and provide fallback synthetic data.
- Inputs: Invalid `TEST_DATA_DIR`.
- Acceptance: Deterministic fallback generated in Phase 0; execution continues; telemetry logs fallback flag.

## Task T05 — Results Report Persistence
- Goal: Persist a structured results JSON and a short Markdown report.
- Inputs: Task metadata (id, owner, acceptance).
- Acceptance: Files in `reports/execution_summary/` and matching MLflow tags.

## Task T06 — Approval Contract Enforcement
- Goal: Reject a response that violates schema (missing required fields), then re‑submit.
- Inputs: Contract‑breaking agent output.
- Acceptance: Rejection at Phase 4; targeted fix at 4.5 passes with confidence ≥ tau.

## Task T07 — Risk‑Scoped Patch Application
- Goal: Apply a small parameter tweak with confidence ≥ tau.
- Inputs: Reflection suggestion with risk score.
- Acceptance: Patch applied; retry count increments; risk/confidence logged in `ledgers/`.

## Task T08 — Procedural Template Reuse
- Goal: Reuse a template from `memory/procedural_cache/`.
- Inputs: Repeated task description with known pattern.
- Acceptance: Template path recorded; reduced retries compared to first occurrence.

## Task T09 — API Evidence Check
- Goal: Verify an MLflow run via API with normalized run_id.
- Inputs: Candidate run_id.
- Acceptance: Phase 6 passes; mismatch triggers 6.5 regeneration.

## Task T10 — Timeout Handling
- Goal: Enforce a max runtime and safe abort with artifacts preserved.
- Inputs: Long‑running script.
- Acceptance: Abort recorded; partial logs saved; safe rollback to Phase 7 with clear status.

## Task T11 — Multi‑Agent Disagreement Resolution
- Goal: Resolve disagreement between Ops and Quality via structured confidence and rationale.
- Inputs: Conflicting proposals.
- Acceptance: Gate computes decision; rationale persisted; confidence reported per agent.

## Task T12 — Environment Pinning
- Goal: Ensure environment variables and versions are recorded for reproducibility.
- Inputs: Standard task.
- Acceptance: `environment` block (versions, env) logged to MLflow tags and report JSON.

## Task T13 — Minimal Visualization Export
- Goal: Export a small PNG summary of results.
- Inputs: Model outputs.
- Acceptance: `summary.png` present under MLflow artifacts; linked in Markdown.

## Task T14 — Retry Budget Exhaustion
- Goal: Respect retry caps and return audited failure.
- Inputs: Hard failure case.
- Acceptance: Capped retries reached; final status exported with reasons and timestamps.

## Task T15 — Owner/Deadline Compliance Check
- Goal: Ensure tasks include Owner and Deadline, validating acceptance criteria binding.
- Inputs: Handoff `pending_actions.json` with fields.
- Acceptance: Phase 3/4 enforce presence; missing fields trigger fix at 4.5 and succeed.

---

## Running the Suite

- Prepare: `ids = prepare_execution_cycle()`
- Filter desired IDs: `selected = ["T01","T02",...,"T10"]`
- Execute: `run_execution_cycle(selected)`
- Export: `export_task_contexts('benchmark_cycle')`

Scoring
- Success: Phase 7 final_status == success
- False completion: Claimed success but Phase 6 verification fails
- Verify pass rate: Fraction of tasks with Phase 6 passing
- Avg retries: Mean of retry counters across 4.5/5.5/6.5
- Human time: Sum of manual interventions (if any), from tracker logs

Artifacts
- Summary CSV recommended at `reports/experiment_summary.csv`
- Plots recommended at `reports/benchmark_bars.png`

