Got it. Here’s a tight, end-to-end review that (1) checks for mistakes, (2) proposes concrete changes to the **current pipeline & plans**, and (3) evaluates the **V2 rescue plan** with clear pass/fail gates.

# 1) Mistake Map (what to fix now)

**Evidence integrity**

* **Over-precise / unsupported numbers (CoTRR):** “300.7×”, “2.3 GB” had no artifact path → **remove** or relabel as estimates with uncertainty.
* **Rounding inflation (V1.0):** +14.2% claimed vs **+13.8%** in `production_evaluation.json` → **standardize rounding** (one decimal; tie to run_id).
* **Demo/placeholder files:** Any non-critical mock artifacts must be **tagged `demo: true`** and excluded from summaries.

**Methodological risks**

* **Masking anomaly (V2):** Visual masking changed metrics by ~0.15% only → suggests weak visual dependence; must re-test with **stronger masks** and **feature-drop ablations**.
* **Score correlation (V1/V2):** r≈0.99 suggests a near-linear shift, not ranking quality → require **Kendall τ / ΔTop-1** checks, not just mean delta.
* **Small-n evaluations:** Most “wins” tested on ~120 queries → underpowered for subtle effects; CI95 OK but fragile → **increase to ≥500 queries**.

**Operational gaps**

* **Caching guardrails:** Risk of stale/poisoned cache affecting “wins” → add **TTL + versioned keys + feature-hash**.
* **Rollout hygiene:** Some proposals lacked explicit **rollback triggers** and **blast-radius** limits.

---

# 2) Pipeline Changes (immediate, low-risk)

**A. Evidence protocol (P0)**

* Every precise metric must carry an **evidence path**: `file#run_id#line_range` (or artifact hash).
* Enforce a **“trust tier”** stamp on every table: T3 (verified), T2 (partial), T1 (exploratory).

**B. Evaluation harness (P0)**

* Add **paired bootstrap (10k)** for nDCG@10 & C@1 with CI95.
* Add **paired permutation test** for listwise re-ranking.
* Always report: Δmean, **CI95**, **p**, **τ/Kendall** (rank agreement), and **ΔTop-1**.

**C. Leakage & masking checks (P0)**

* **Leakage battery:** query-level overlap check, label permutation (“label penetration”), and feature-freeze baseline.
* **Masking battery:** (i) random patch; (ii) saliency-guided patch; (iii) full-channel drop. Expect **meaningful degradation**; otherwise flag.

**D. Caching guardrails (P0)**

* Key = `model_ver + data_hash + feature_ver`.
* TTL per cohort; **invalidate on model/config change**.
* Log cache hit ratio per experiment (avoid hidden “wins”).

**E. Rollout contract (P0)**

* Gray rollout **5% → 20% → 50%**, ≥24h/step, with auto-rollback:

  * **C@1 drop ≥2 pts**, **nDCG@10 CI95 lower bound ≤0**, **P95 +10%**, or **Err >1%**.

**F. Reporting standard (P0)**

* Single “**Metrics Registry**” table auto-rendered from artifacts with rounding rules (1 decimal %; 3 decimals nDCG deltas).

---

# 3) Plan Updates by Track

## 3.1 V1.0 Production Line (status: ✅ keep investing)

* **Truth:** +13.8% compliance (artifact-backed), P95≈312 ms, err 0.3% → solid.
* **Next changes (safe):**

  1. Bake **Dual-Score** weights sweep (grid or Bayes) behind a flag; report ΔC@1, Δτ, ΔP95.
  2. Add **Subject-Object** failure audit to monitoring (top-N pairs & reasons).
  3. Ship **latency budget dashboard** (model vs I/O vs post-proc).
* **Go/No-Go:** **GO** with routine A/Bs, strict rollback gates.

## 3.2 CoTRR Lightweight Line (status: ⚡ pivot → **constrained**)

* **Truth:** Full CoTRR blows latency/mem; simplified variants are promising.
* **Required guardrails:**

  * ≤**2×** latency vs baseline, **<500 MB** mem, **ΔnDCG@10 ≥ +0.01** with CI95>0, **τ gain** or **ΔTop-1 ≥ +1.0 pt**.
  * Two-step reasoning max; **vectorized ops**; **no unbounded caches**.
* **Go/No-Go:** **Shadow-only** until it meets all above; if not, **PAUSE**.

## 3.3 V2.0 Multimodal Research Line (status: 🔴 **PAUSE_AND_FIX**)

* **Truth:** CI95 lower bound positive on ~120 queries, but **integrity not passed** (masking & correlation issues).
* **Fixes needed (before any “resume”):**

  1. **Data scale:** expand to **≥500 real queries** across ≥3 strata (e.g., cocktails/flowers/high_quality). Target power ≥0.8 for Δ=0.01.
  2. **Ablations:** (i) modality-only (I/T/M), (ii) pairwise combinations, (iii) feature-drop channels; require **monotonic** contribution pattern.
  3. **Correlation control:** report **Kendall τ** change; we want **τ↑** not just linear shift.
  4. **Integrity:** ban synthetic features from headline; allow only as T1 demos.
* **Go/No-Go:** Keep **PAUSE_AND_FIX**. Resume only if **all** acceptance gates below pass.

---

# 4) V2 Rescue Plan — Evaluation & Gates

**Rescue objective:** Determine if V2 (multimodal fusion) delivers **real** ranking gains (not correlation shifts) and is worth a controlled shadow rollout.

### A. Experiment Design (rigor)

* **Dataset:** ≥**500 queries**, stratified (≥3 strata), fixed candidate pools; log seed/splits.
* **Metrics:** ΔnDCG@10 (primary); C@1; **Kendall τ**; **ΔTop-1**.
* **Stats:** Paired bootstrap 10k (CI95), paired permutation test (p), report effect size.
* **Ablations:** I-only, T-only, M-only; I+T, I+M, T+M; full I+T+M. Expect **ordered performance**.

### B. Integrity & Dependence

* **Masking:** random & saliency-guided masks → expect **material drops** if vision really matters.
* **Leakage:** penetration test & overlap audit must be **clean**.
* **Cache isolation:** no reuse of V1 caches; version keys enforced.

### C. Acceptance Gates (must pass all)

1. **ΔnDCG@10 CI95 lower bound ≥ +0.015** (power adjusted for 500 queries).
2. **Kendall τ** improvement ≥ **+0.02** and **ΔTop-1 ≥ +1.0 pt**.
3. **Masking sensitivity:** ≥ **−1.0 pt** drop (proves vision contributes).
4. **Leakage battery:** **PASS** across all tests.
5. **Latency impact:** ≤ **1.5×** with batched inference plan (or distill/quantize to hit ≤2×).

**Decision outcomes**

* **GO (Shadow 5%)** if all pass.
* **PAUSE_AND_FIX** if metrics pass but integrity/masking fail.
* **REJECT** if CI95 crosses 0 or τ/Top-1 unchanged.

---

# 5) Minimal Worklist (execute in order)

**Within 24h (P0)**

1. Replace all over-precise CoTRR numbers with **“estimate”** labels or remove; fix V1.0 rounding to **+13.8%**.
2. Land **evaluation harness**: bootstrap/permutation/τ, masking & leakage batteries; emit a **Metrics Registry** markdown from artifacts.
3. Add **cache keying** (model_ver+data_hash+feature_ver) + TTL + hit-ratio logging.

**48–72h (P1)**
4) Run V2 **500-query** experiment with full ablations & masking; export report with CI95/τ/Top-1.
5) Profile CoTRR-lite; enforce ≤2× latency; if not met, return **PAUSE** with concrete hotspots.

**Week 1 (P2)**
6) If V2 passes gates → **shadow 5%** with auto-rollback; else keep paused and re-target data curation or candidate generation (higher ROI).
7) Ship V1 **Dual-Score** weight sweep via feature flag; dashboard latency budget.

---

# 6) Final Verdicts (today)

* **V1.0 Production:** **GO** (continue incremental optimization under strict rollout gates).
* **CoTRR-lite:** **SHADOW-ONLY** (must prove ≤2× latency & τ/Top-1 gains; otherwise PAUSE).
* **V2.0 Multimodal:** **PAUSE_AND_FIX** (resume only after 500-query, ablations, masking sensitivity, τ/Top-1 gains, and leakage pass).

