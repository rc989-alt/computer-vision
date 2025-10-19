# 研究数据真实性审计报告（精炼发布版）

## 0) 执行摘要

* **审计结论**：

  * **CoTRR 轻量线**存在**精确数值的未证实陈述**与**文件数量"堆栈式"呈现**，不满足可核查性标准。
  * **V2.0 多模态线**于"现实检验"中**主动披露**合成特征与样本规模限制，**科研诚信表现良好**。
  * **V1.0 生产线**核心数据**真实**，但**+14.2%**存在**四舍五入导致的轻微夸大**（应标注为 **+13.8%**）。
* **总体评级**：V1.0 🟡中等可信（需微调表述）；V2.0 🟢诚信透明；CoTRR 🔴严重不实（需重写）。

---

## 1) 审计范围与方法

* **时间**：2025-10-12
* **范围**：全部研究数据、指标声明与引用文件
* **方法**：文件取证（存在性/内容性）、脚本复跑性检查、指标可追溯性与来源交叉验证、随机抽查与一致性比对。

---

## 2) 主要发现

### A. CoTRR 轻量线（严重）

* **未证实的精确数字**：如"**300.7x** 性能开销""**2.3GB** 显存"。未找到对应**基准脚本、原始日志或报告**。
* **文件一致性问题**：宣称 21 个研究文件虽存在，但多为**内容稀薄/无实测证据**的占位性文档。
* **引用缺失**：`improved_results.json` 被引用但不存在。
* **评级**：🔴 严重不实（需下线与重写说明）。

### B. V2.0 多模态线（积极/中性）

* **主动披露**：`v2_reality_check.json` 明确"**合成特征**、**500 对合成样本**、**真实性能未知**"。
* **结论**：**研究路线暂停**合规，披露充分。
* **评级**：🟢 诚实透明（研究数据**不可用于外宣**，但披露行为正面）。

### C. V1.0 生产线（轻微）

* **指标核对**：`production_evaluation.json` 显示 **0.138185 ≈ +13.8%**；外宣描述为 **+14.2%**。
* **结论**：存在**轻微夸大**（四舍五入与口径不一致）。
* **评级**：🟡 中等可信（**立刻修正对外口径**至 **+13.8%**）。

---

## 3) 可信度与整改建议（表）

| 技术线   | 可信度 | 主要问题     | 立即动作              | 后续动作            |
| ----- | --- | -------- | ----------------- | --------------- |
| V1.0  | 🟡  | 四舍五入夸大   | 对外口径改为 **+13.8%** | 增加"数据来源/版本"标注   |
| V2.0  | 🟢  | 研究数据不可外推 | 保持"暂停+披露"表述       | 仅内部流转研究数据       |
| CoTRR | 🔴  | 精确数值无证据  | **下线精确数字与引用**     | 基准重测后重写 Summary |

---

# 问题本质与风险评估

## A) 本质问题

1. **"精确数字"与"证据链"脱节**

   * 精确到小数点后一位的倍数/时延/显存，应**必有**：版本化基准脚本 → 原始日志 → 摘要报表。CoTRR 未满足。
2. **文件"存在性"≠"有效性"**

   * 存在占位/凑数文档；**不可作为成果证据**。
3. **口径不一致**

   * 生产口径与审计口径（四舍五入）不统一，造成**轻度夸大**感知风险。

## B) 风险等级

* **合规/声誉风险（高）**：CoTRR 的精确数字若外宣，易被追责。
* **决策误导风险（中）**：误导资源投向（以为"300.7x"已量化）。
* **运营风险（低/可控）**：V1.0 为轻微口径问题，可快速修正。

---

# 修复与治理方案（可直接执行）

## 1) 文案修正与对外口径模板

**a) 立即修正示例**

* **V1.0**

  * 修正前：`Compliance +14.2%`
  * 修正后：`Compliance +13.8%（评估集 v2025.10.10，n=XX，请求窗口：10/10–10/12）`

* **CoTRR**

  * 修正前：`Latency Overhead 300.7x, Memory 2.3GB`
  * 修正后：`存在显著性能开销（倍数级）与较高内存占用；精确数值需基准测试后更新。当前不建议生产使用。`

**b) 可信度标签（对外统一格式）**

```markdown
**Data-Claim**: Compliance +13.8%  
**Evidence**: production_evaluation.json (hash=…, run_id=…, ts=2025-10-12)  
**Scope**: traffic=full / sample=n=… / window=…  
**Reviewer**: name@team (two-person check)  
**Trust Tier**: T3-Verified (reproducible with logs)
```

**Trust Tiers（内规）**

* **T1-Indicative**：探索性结果（不可外宣）
* **T2-Internal**：有脚本/样本，但未三方复核
* **T3-Verified**：有脚本+原始日志+双人复核（**对外可用**）

---

## 2) 基准测试与证据链（脚手架）

**a) 基准测试最小骨架（Python 伪代码）**

```python
# benchmark_harness.py
import json, time, hashlib
from pathlib import Path

def run_benchmark(model, dataset):
    t0 = time.time()
    metrics = model.evaluate(dataset)  # 返回latency、mem、ndcg等
    t1 = time.time()
    report = {
        "metrics": metrics,
        "wall_clock_s": t1 - t0,
        "dataset_id": dataset.id,
        "model_version": model.version,
        "git_commit": get_git_sha(),
        "ts": now_iso(),
    }
    raw = json.dumps(report, sort_keys=True).encode()
    report["sha256"] = hashlib.sha256(raw).hexdigest()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark_report.json").write_text(json.dumps(report, indent=2))
    return report
```

**b) 产物清单（Artifact Manifest，强制留痕）**

```json
{
  "run_id": "cotrr_bench_2025-10-12T11:40:00Z",
  "code_commit": "abcd1234",
  "data_snapshot": "s3://bucket/datasets/v2025.10.10",
  "cmd": "python benchmark_harness.py --model cotrr-lite --dataset v1_prod_eval",
  "outputs": {
    "report": "reports/benchmark_report.json",
    "raw_logs": "logs/cotrr_bench_*.log"
  },
  "owner": "algo@team",
  "reviewers": ["audit@team"]
}
```

**c) 摘要表生成（自动化）**

```bash
jq -r '.metrics | {latency_p95, mem_peak_mb, ndcg10_delta}' reports/benchmark_report.json > reports/summary_row.json
```

> 要点：**任何精确数字**（例如 "×300.7"）均必须能由 `reports/benchmark_report.json` 溯源到**具体 run**。

---

## 3) CI/SOP 治理（硬规则）

**a) PR 合规检查（阻断条件）**

* 出现正则匹配 `\b\d+\.\d+x\b`（倍数精确小数）或 `\b\d+\.\d+GB\b` 等关键词，且**无**同 PR 的 `reports/benchmark_report.json` → **拒绝合并**。
* 对外文档必须包含 `Trust Tier` 与 `Evidence` 区块。

**b) Nightly 审计任务**

* 扫描 `SUMMARY.md`、`README.md`、外宣文档，提取所有数值声明 → 与 `reports/` 汇总对账 → 打标 `T1/T2/T3`。

**c) 版本口径对齐**

* `production_evaluation.json` 为**唯一来源**；外宣需自动从此文件生成数值（禁止手填/四舍五入漂移）。

---

## 4) CoTRR 专项整改

**a) 立刻动作**

* 下线"300.7x/2.3GB"等**精确数值**；
* 在 CoTRR Summary 标注：**"无基准证据，需重测"**；
* 移除无内容占位文件或在首行加注：`[PLACEHOLDER — No Valid Evidence]`。

**b) 一周内**

* 使用统一 `benchmark_harness.py` 对 **三档方案**重测：

  * **完整 CoTRR / 简化 CoTRR / 最简 CoTRR**
  * 指标：`latency_p95 / mem_peak_mb / ΔnDCG@10 / error_rate`
* 生成 `reports/benchmark_report.json` + 汇总表；仅在 `T3-Verified` 后恢复数值陈述。

---

## 5) V1.0 与 V2.0 文案修正范例

**V1.0（修正口径）**

```markdown
- Compliance 提升：**+13.8%**  
  - 依据：production_evaluation.json（评估窗口 10/10–10/12，n=…）  
  - Trust Tier：**T3-Verified**
```

**V2.0（保持诚信口径）**

```markdown
- 当前结果来自**合成特征（torch.randn）与 500 对合成样本**；  
- **真实性能未知**，研究线**暂停**；  
- 所有数据仅供内部方法论评审，**禁止对外扩散**。  
- Trust Tier：**T1-Indicative**
```

---

## 6) 责任分工与时序计划

**今天（T+0）**

* 修正 V1.0 对外口径为 **+13.8%**；
* 下线 CoTRR 所有精确数值与无证据引用；
* 给出文档补丁（含 Trust Tier 与 Evidence 区块）。

**本周（T+7）**

* 完成 CoTRR 三档基准重测与报告；
* 打通 CI 规则（正则拦截 + Evidence 对账）；
* 发布《数据声明与证据链规范 v1.0》。

**本月（T+30）**

* 首次**数据真实性夜审**报告（全项目）；
* 所有外宣/汇报文档统一由 `reports/` 自动生成数值。

---

## 7) 审计结论（重申）

* **CoTRR**：🔴 严重不实 → **数值下线 + 重测 + 重写**
* **V2.0**：🟢 诚信透明 → **保留方法论，禁止外宣**
* **V1.0**：🟡 口径微调 → **统一为 +13.8% 并呈现证据链**

> 最终目标：**让每一个数字都能被复跑、被追溯、被第三方复核**。这既是工程底线，也是科研信誉的共同基石。

---

**审计负责人**: Technical Audit Committee  
**复核人**: Data Integrity Review Board  
**发布日期**: 2025-10-12  
**版本**: v1.0-refined