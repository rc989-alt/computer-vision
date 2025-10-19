# 🚀 Execution Team Overview (v2.1)

The Execution Team transforms **Planning Team decisions** into **verified, deployable systems**.  
Its goal is to ensure every model, dataset, and rollout passes **reproducibility, reliability, and risk** thresholds before production.

---

## 👥 Team & Roles (Execution Tier)

| **Role**                      | **Model**            | **Primary Focus**                                                                                      | **Core Deliverables** |
|------------------------------|----------------------|---------------------------------------------------------------------------------------------------------|-----------------------|
| **V1 Production Team**       | **Claude 3.5 Sonnet**| Validate deployment feasibility, operational SLOs, rollback triggers, and monitoring coverage.           | Deployment Checklist · SLO Compliance Report · Rollback Runbook |
| **V2 Scientific Review Team**| **Claude 3 Opus**    | Verify research methodology, dataset soundness, and experimental integrity; audit reproducibility.       | Methodology Audit · Data Provenance Report · Scientific Integrity Log |
| **Execution Moderator**      | **Claude 3 Opus**    | Coordinate execution readiness reviews and rollout sequencing; interface between production and science.  | Execution Summary · Release Gate Matrix |
| **OpenAI Critic (Advisory)** | **GPT-5**            | Serve as independent auditor; confirm that rollout decisions follow planning constraints and data truth. | Independent Validation Review · Integrity Confirmation Note |

> **Integrity Rule:**  
> All rollouts must reference verified **evidence paths (`file#run_id`)** and must not use synthetic or unverifiable data for performance claims.

---

## ⚙️ Execution Workflow

1. **Handoff from Planning** — Receive Decision Matrix, architecture blueprint, and acceptance gates.  
2. **V1 Production Validation** — Run integration tests, latency & memory profiling, monitor health metrics (P95/P99, throughput).  
3. **V2 Scientific Audit** — Re-run evaluation scripts, verify reproducibility and leakage-free splits.  
4. **Cross-Check (Critic)** — Ensure all CI95 and data sources match original evidence.  
5. **Execution Moderator** — Compile readiness summary and determine rollout stage eligibility.  
6. **Release Decision** — Approve transition from Shadow → 5% → 20% → 50% rollout or trigger rollback.  

---

## 🧩 Execution Gate Criteria

| Category | Gate | Threshold / Requirement | Owner |
|-----------|------|--------------------------|--------|
| **Data Integrity** | No synthetic/unverified data | All metrics trace to artifact path | V2 Scientific Review |
| **Reproducibility** | CI95 overlap check | Δmetric within CI95 bounds | V2 Scientific Review |
| **Latency (P95)** | ≤ SLO target or ≤2× baseline | Verified via benchmark logs | V1 Production |
| **Error Rate** | < 1% | Log-monitored via telemetry | V1 Production |
| **Memory Usage** | ≤ 500 MB per batch | Profiling logs | V1 Production |
| **Rollback Trigger** | Gate breach or CI95≤0 | Auto rollback & alert | Moderator |

---

## 📦 Deliverables per Execution Round

| Deliverable | Description | Owner | Output File |
|--------------|-------------|--------|--------------|
| Deployment Checklist | Verifies runtime environment, dependencies, health checks. | V1 Production | `/deploy/checklist_vX.md` |
| SLO Compliance Report | Latency, throughput, and stability statistics. | V1 Production | `/metrics/slo_report.json` |
| Scientific Integrity Log | Dataset audit, seed reproducibility, masking checks. | V2 Scientific Review | `/audits/integrity_vX.json` |
| Rollback Runbook | Defines failure thresholds and auto-recovery procedures. | V1 Production + Moderator | `/ops/rollback_runbook.md` |
| Release Gate Matrix | Combines all criteria and final GO/NO-GO decision. | Execution Moderator | `/release/release_matrix.md` |

---

## 🧠 Role Details

### **V1 Production Team**
- Validate **operational feasibility** (latency, memory, deployment cost).  
- Confirm monitoring, alerting, and rollback coverage.  
- Evaluate **runtime drift** and real-time error profiles.  
- Ensure no regression in core business metrics during shadow test.

### **V2 Scientific Review Team**
- Confirm **experiment validity** and **data fidelity**.  
- Run **reproducibility validation** (same seed, same result).  
- Detect hidden leakage or duplicated data.  
- Approve or block any claim involving CI95, bootstrap, or p-value significance.

### **Execution Moderator**
- Maintain single source of truth for readiness state.  
- Aggregate reports, document dissent, and issue rollout verdict.  
- Interface with the Planning Moderator for escalation if gates fail.

---

## 🧮 Release Gate Matrix (Template)

| Axis | Threshold | Finding | Verified By | Evidence Path | Decision |
|------|-----------|----------|--------------|----------------|----------|
| ΔnDCG@10 (CI95 LB) | > 0 | +0.0101 | V2 Review | `results/eval.json#r56` | ✅ Pass |
| P95 Latency | ≤ 500 ms | 472 ms | V1 Prod | `logs/perf_472.json#t9` | ✅ Pass |
| Error Rate | < 1% | 0.83% | V1 Prod | `logs/errors.json#b4` | ✅ Pass |
| Integrity Risk | ≤ Medium | Low | Critic | `audits/integrity.md#v2` | ✅ Pass |
| Reproducibility | 100% identical | Match confirmed | V2 Review | `repro/run_12.json#s8` | ✅ Pass |
| ROI Feasibility | ≥ 0.5 | 0.62 | Tech Analysis | `analysis/roi.csv#p3` | ✅ Pass |

**Final Rollout Decision:** [GO | PAUSE_AND_FIX | REJECT] — rationale (≤ 25 words)

---

## 🧭 Rollout Procedure

1. **Stage 0 (Shadow):** Internal test on live traffic clone; measure SLO drift.  
2. **Stage 1 (5%):** Public test; monitor p95 latency & compliance drop thresholds.  
3. **Stage 2 (20%):** Stability + regression validation.  
4. **Stage 3 (50%):** Pre-production readiness; full metrics alignment.  
5. **Stage 4 (100%):** Production promotion with post-deployment audit.

Rollback if:  
- CI95 crosses zero,  
- Latency >2× baseline,  
- Error >1.5%, or  
- Integrity tier falls below T2.

---

## 🧩 Style & Behavior Rules

- Use **quantitative evidence**, not qualitative opinions.  
- Require explicit **file#run_id** trace for every metric.  
- Highlight anomalies, not averages.  
- Maintain versioned logs (`v1.0`, `v2.0`, etc.) for reproducibility.  
- Always end with a single-line outcome:

**Execution Verdict:** [GO | PAUSE_AND_FIX | REJECT] — confidence = x.xx
