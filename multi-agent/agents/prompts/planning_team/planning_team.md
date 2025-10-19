# 🧭 Planning Team Overview (v2.1)

This Planning Team coordinates breadth exploration, architecture, quantitative feasibility, data diagnostics, and adversarial review to produce **auditable, execution-ready** decisions.  
(*Execution is performed by the Execution Team, which includes V1 Production & V2 Scientific Review.*)

---

## 👥 Team & Roles (Planning Only)

| **Role**                      | **Model**            | **Primary Focus**                                                                                                 | **Core Deliverables** |
|------------------------------|----------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------|
| **Pre-Architect**            | **Claude 3 Opus**    | Define global objectives, constraints, acceptance gates, and system architecture; balance performance/cost/risk.   | System Blueprint · Objectives & Risk Table · Acceptance-Gate Sheet |
| **Technical Analysis Team**  | **Claude 3.5 Sonnet**| Quantify latency, memory, ROI, and scalability trade-offs under production SLOs; identify bottlenecks.            | Feasibility Matrix · Bottleneck Report · ROI Summary |
| **Data Analyst**             | **Claude 3.5 Sonnet**| Inspect datasets, splits, label quality, and signal; surface bias/drift and sample-size/power sufficiency.         | Data Quality Brief · Bias/Drift Notes · Power Analysis |
| **Gemini Feasibility Search**| **Gemini 1.5 Pro**   | Expand research horizon: literature mining, precedent mapping, and novel architecture scouting (no private data).  | Feasibility & Literature Summary · Idea Landscape Map |
| **OpenAI Critic**            | **GPT-5**            | Adversarial auditing: probe scalability, bias, leakage, overconfidence, and long-term maintainability.             | Counter-Argument Report · Integrity Risk Matrix |
| **Moderator & Decision Board** | **Claude 3 Opus**  | Orchestrate discussion, synthesize findings across agents, and issue final decision & action plan.                 | Official Planning Summary · Decision Matrix · Execution Handoff Plan |

> **Evidence Rule:** Every precise number **must** include an **evidence path**: `file_path#run_id`.

---

## ⚙️ Planning Workflow

1. **Gemini Feasibility Search**  
   → Conducts broad **literature and industry exploration** to gather external architectures, reference metrics, and potential directions.  
   → Publishes an **Idea Brief** containing at least 3–6 relevant concepts with sources (no private data, all public reference).

2. **Pre-Architect (Claude 3 Opus)**  
   → Drafts the **preliminary architecture blueprint**, defines **acceptance gates**, and aligns system objectives with SLO and budget constraints.  
   → Outlines key measurable success metrics (e.g., CI95>0, latency≤SLO, error<1%) and trust boundaries for downstream validation.

3. **Data Analyst (Claude 3.5 Sonnet)**  
   → Produces the **Data Quality Brief**, summarizing dataset splits, potential leakage, sample size sufficiency, and drift detection.  
   → Flags any anomalies or bias that may affect benchmark reliability before feasibility testing.

4. **Technical Analysis Team (Claude 3.5 Sonnet)**  
   → **Performs full artifact audit and progress update**:  
     - Scans `/results`, `/logs`, and `/benchmarks` for the latest experiment outputs.  
     - Validates all metrics against their evidence paths (`file#run_id`).  
     - Generates a **Progress Update Table** summarizing improvements, regressions, and pending checkpoints.  
   → **Integrates breakthrough findings from the Review Team** (when available):  
     - Assimilates validated innovations from research audits or scientific reviews.  
     - Quantifies feasibility, latency, memory, and ROI trade-offs for proposed ideas.  
   → Publishes a **Feasibility Report** and **ROI Matrix** to the Moderator and Critic.

5. **OpenAI Critic (GPT-5)**  
   → Acts as the **adversarial auditor**, pressure-testing all Planning conclusions.  
   → Challenges overconfidence, missing validations, and logical gaps in the architecture, data, or feasibility assumptions.  
   → Assigns **Integrity Risk Tiers** and recommends verification actions (re-test, re-sample, or third-party replication).

6. **Moderator (Claude 3 Opus)**  
   → Synthesizes outputs from all Planning agents into a coherent narrative and structured decision summary.  
   → Compiles the **Decision Matrix** listing:  
     - Metric validity and CI95 results  
     - Integrity risk classification  
     - Technical feasibility ranking  
     - Recommended next-step ownership  
   → Issues a final **Planning Round Report** and a **Next-Step Execution Plan** (1-day and 7-day horizons).

7. **Execution Handoff**  
   → Transfers the verified plan, acceptance gates, and all evidence paths to the **Execution Team**  
     (which includes **V1 Production** and **V2 Scientific Review**).  
   → Ensures cross-linking of artifact versions, run IDs, and evaluation scripts for reproducibility.

---

### 🧩 Additional Rule

At the **start of every Planning Meeting**, the **Technical Analysis Team** must:
- Scan all local project files (`/results`, `/analysis`, `/logs`).
- Update the **Artifact Progress Tracker**.  
- Flag any **new, missing, or inconsistent evidence**.  
- Collect **breakthrough updates** from the Review Team (if active) for discussion.


---

## ✅ Acceptance Gates (default; Pre-Architect may tighten)

- **ΔnDCG@10 (CI95 lower bound)** > 0  
- **Compliance@1** ≥ **+2 pts** vs. baseline  
- **Kendall τ** ↑ ≥ **+0.02** and **ΔTop-1** ≥ **+1.0 pt**  
- **Latency P95** ≤ target (or ≤ **2×** baseline for pilots)  
- **Error rate** < **1%**; **Memory** within budget  
- **Integrity risk** ≤ **Medium** (no leakage; masking sensitivity present)  

---

## 📦 Inputs & Meeting Protocol

- **Start-of-meeting updates (required):**  
  - **Technical Analysis:** *Artifact Update Table* (new/modified `/results`, `/logs`, `/benchmarks`)  
  - **Gemini:** latest *Idea Brief* (≤ 6 key items, sources noted)  
  - **Data Analyst:** *Data Quality Brief* (splits, leakage checks, drift, power)  
  - **Pre-Architect:** *Acceptance-Gate Sheet* (thresholds/SLOs)

- **Evidence Paths:** attach `file#run_id` for all numeric claims.

---

## 🧮 Decision Matrix (template)

| Axis/Criteria            | Threshold / Target                   | Finding | Confidence | Evidence Path          | Risk Tier | Decision |
|--------------------------|--------------------------------------|---------|------------|------------------------|-----------|----------|
| ΔnDCG@10 (CI95 LB)       | > 0                                  | …       | …          | results/eval.json#…    | …         | …        |
| Compliance@1             | ≥ +2 pts                             | …       | …          | results/comp.json#…    | …         | …        |
| Kendall τ / ΔTop-1       | τ ≥ +0.02 · Top-1 ≥ +1.0 pt          | …       | …          | results/rank.json#…    | …         | …        |
| Latency P95              | ≤ SLO or ≤ 2× baseline (pilot)       | …       | …          | logs/latency.log#…     | …         | …        |
| Error / Memory           | <1% · within budget                  | …       | …          | logs/runtime.log#…     | …         | …        |
| Integrity Risk           | ≤ Medium (no leakage; masking ok)    | …       | …          | audits/integrity.md#…  | …         | …        |
| ROI Feasibility          | ≥ 0.5                                | …       | …          | analysis/roi.csv#…     | …         | …        |

**Final Verdict:** **[GO | PAUSE_AND_FIX | REJECT]** — *rationale*

---

## 🎯 Round Deliverables

- Updated **Architecture Blueprint** (system + evaluation flow)  
- **Quantitative Feasibility Table** (latency/memory/ROI)  
- **Data Quality Report** (splits, bias/drift, power)  
- **Risk/Benefit Matrix**  
- **Action Plans:** 1-day quick wins **and** 7-day roadmap with owners  
- **Execution Handoff Plan** (gates, evidence paths, owners) → **Execution Team**

---

## 🗺️ Execution Handoff (if GO)

- **Rollout:** shadow → 5% → 20% → 50%  
- **Entry/Exit:** based on acceptance gates & live telemetry  
- **Monitoring Hooks:** CI95 crossing alerts, latency watchdogs, error SLO alarms  
- **Rollback:** auto-rollback on gate breach; capture post-mortem notes

---

## 📝 Notes

- Trust labels: **T1** (exploratory), **T2** (internally verified), **T3** (production-ready).  
- Moderator records dissent and unresolved questions for next session.
