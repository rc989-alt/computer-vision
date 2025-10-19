# 数据声明与证据链规范 v1.0

## 📋 总则

**目标**: 确保所有对外数据声明都有可追溯、可复现的证据支撑。

**适用范围**: 所有包含性能指标、改进数据、技术参数的文档。

**核心原则**: **Every Number Must Be Traceable**（每个数字都必须可追溯）

---

## 🏷️ Trust Tier 分级标准

### T1-Indicative（指示性）
- **定义**: 探索性结果，概念验证阶段
- **证据要求**: 实验记录、初步代码
- **使用限制**: 仅限内部讨论，**禁止对外宣传**
- **标注格式**: `Trust Tier: T1-Indicative`

### T2-Internal（内部可用）
- **定义**: 有脚本支撑，但未经三方复核
- **证据要求**: 完整脚本 + 原始日志 + 配置文件
- **使用限制**: 内部报告可用，外部分享需标注来源
- **标注格式**: `Trust Tier: T2-Internal`

### T3-Verified（验证可信）
- **定义**: 双人复核，可复现，有完整证据链
- **证据要求**: 基准脚本 + 原始日志 + 双人sign-off + 哈希验证
- **使用限制**: **对外宣传可用**
- **标注格式**: `Trust Tier: T3-Verified`

---

## 📊 数据声明标准格式

### 基本格式模板
```markdown
**Data-Claim**: [具体数字] [单位] [描述]
**Evidence**: [证据文件] (hash=[哈希], run_id=[运行ID], ts=[时间戳])
**Scope**: [评估范围] / sample=n=[样本数] / window=[时间窗口]
**Reviewer**: [复核人]@[团队] ([复核方式])
**Trust Tier**: [T1/T2/T3]-[类别]
```

### 具体示例

#### ✅ 正确示例 - V1.0 Production
```markdown
**Data-Claim**: Compliance +13.8%
**Evidence**: production_evaluation.json (hash=a1b2c3..., run_id=v1_prod_20251012, ts=2025-10-12T10:30:00Z)
**Scope**: Full production traffic / sample=n=45 / window=10/10–10/12
**Reviewer**: audit@team (two-person check)
**Trust Tier**: T3-Verified
```

#### ✅ 正确示例 - 研究阶段
```markdown
**Data-Claim**: Latency ~2-5x overhead (estimated)
**Evidence**: Initial benchmarks in research/cotrr_experiments/ (incomplete)
**Scope**: Synthetic dataset / sample=n=100 / single-run
**Reviewer**: research@team (single-person check)
**Trust Tier**: T1-Indicative
**Note**: ⚠️ Needs comprehensive benchmark before production claims
```

#### ❌ 错误示例（仅作说明）
```markdown
性能提升 300.7x  [缺乏证据和Trust Tier标注]
显存占用 2.3GB   [精确数字但无证据支撑]
```
**注**: 以上为反面示例，不是真实数据声明

---

## 🧪 Evidence 要求详解

### T3-Verified 必须包含
1. **基准脚本**: `benchmark_harness.py --model [model] --dataset [dataset]`
2. **原始日志**: `logs/benchmark_[run_id].log`
3. **配置快照**: `config/[run_id].json`
4. **环境信息**: Git commit, 系统配置, 依赖版本
5. **数据快照**: 评估数据集版本和哈希
6. **双人复核**: 两人独立验证结果

### T2-Internal 必须包含
1. **可运行脚本**: 能够复现结果的代码
2. **配置文件**: 完整的参数配置
3. **输出日志**: 至少有一次完整运行的日志

### T1-Indicative 最低要求
1. **实验记录**: 详细的实验步骤和观察
2. **代码片段**: 关键计算逻辑
3. **初步数据**: 即使是不完整的测试结果

---

## 🚫 禁止行为清单

### 严格禁止
1. **精确数字无证据**: 如`300.7x`、`2.3GB`等精确到小数点的数字必须有基准测试支撑
2. **文件引用不存在**: 引用的JSON、日志文件必须确实存在
3. **四舍五入夸大**: 如13.8%写成14.2%，必须使用原始数据
4. **占位文件当证据**: 空文件或无实质内容的文件不可作为证据

### 强烈不建议
1. **口径不一致**: 不同文档中同一指标的数值不一致
2. **证据过时**: 引用超过30天的旧数据而无说明
3. **单人验证**: T3级别的声明应该有双人复核

---

## 🔧 工具和自动化

### 使用基准测试框架
```bash
# 标准基准测试命令
python tools/benchmark_harness.py --model v1-production --dataset standard_eval --trust-tier T3-Verified

# 输出文件
# reports/benchmark_report_[run_id].json - 完整报告
# reports/summary_[run_id].json - 摘要数据
# logs/benchmark_[run_id].log - 原始日志
```

### CI检查工具
```bash
# PR合并前检查
python tools/ci_data_integrity_check.py --pr-mode --target-dir docs/

# 生成合规报告
python tools/ci_data_integrity_check.py --output integrity_report.md
```

### 自动化生成数据声明
```bash
# 从基准报告自动生成标准格式
python tools/generate_data_claim.py --report reports/benchmark_report_latest.json
```

---

## 📝 文档更新SOP

### 新增数据声明流程
1. **执行基准测试**: 使用`benchmark_harness.py`
2. **生成证据文件**: 确保reports/目录有完整报告
3. **双人复核**: 两人独立验证结果（T3级别）
4. **标准格式撰写**: 使用本规范的模板格式
5. **CI检查通过**: 通过自动化合规检查
6. **文档提交**: PR包含数据声明和证据文件

### 修改已有声明流程
1. **重新测试**: 如果原有证据不足，重新执行基准测试
2. **证据补强**: 确保有足够的证据支撑
3. **版本标注**: 明确标注数据来源的版本和时间
4. **向下兼容**: 保持历史可追溯性

---

## 🎯 各角色责任

### 研究人员 (Researcher)
- 确保实验数据有完整记录
- 使用规范格式标注Trust Tier
- 及时更新实验状态和结果

### 工程师 (Engineer)
- 基准测试脚本的维护和优化
- 证据文件的版本管理
- CI工具的正常运行

### 审核员 (Reviewer)
- 独立验证T3级别的声明
- 检查证据链的完整性
- 确保格式规范的遵守

### 对外发言人 (Communicator)
- 仅使用T3-Verified级别的数据
- 确保对外口径与内部数据一致
- 维护项目声誉和可信度

---

## 📚 常见问题解答

### Q: 已有的数据声明怎么办？
A: 分情况处理：
- T3级别：保持不变，确保证据文件存在
- T2级别：补强证据或降级为T1
- 无证据：立即下线或标注为"需要验证"

### Q: 紧急发布时来不及双人复核怎么办？
A: 使用T2-Internal级别，并明确标注"单人复核，待补充验证"

### Q: 第三方数据引用如何处理？
A: 标注数据来源、引用时间、原始链接，Trust Tier标为T2-External

### Q: 估算数据如何标注？
A: 明确标注"估算"、"约"、"大致"等限定词，Trust Tier为T1-Indicative

---

## 🔄 规范更新机制

- **版本控制**: 本规范采用语义化版本控制
- **更新周期**: 季度回顾，必要时及时修订
- **反馈机制**: 团队成员可提出改进建议
- **执行监督**: 月度审计检查执行情况

---

**文档版本**: v1.0  
**发布日期**: 2025-10-12  
**维护团队**: Data Integrity Committee  
**下次审核**: 2026-01-12