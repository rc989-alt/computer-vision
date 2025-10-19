# 复核+运维维护 SOP 1.0 - 可直接落地版本

> **基于现状**: V1.0在产(+13.8%)、V2.0处于PAUSE_AND_FIX、CoTRR轻量化进行中  
> **文档规范**: 严格遵守 `docs/06_governance/data_claim_standards.md` Trust Tier标准

---

## 🚨 一、复核（Review）总流程 SOP 1.0

### 触发条件
- 任何影响"对外数字/线上用户"的更改：模型、特征、数据、指标口径、评测集
- 产物：审计可复跑证据链 + 结论页（1页）

### 1.1 准备阶段（T-0）- 环境锁定

**做什么**: 固化所有实验变量  
**怎么做**:
```bash
# 锁定数据源
cp production_evaluation.json data/frozen/evaluation_$(date +%Y%m%d).json
git add data/frozen/ && git commit -m "freeze: evaluation baseline"

# 环境冻结
echo "git_commit: $(git rev-parse HEAD)" > config/experiment_env.yaml
echo "dataset_id: v2025.10.12" >> config/experiment_env.yaml
echo "random_seed: 20251012" >> config/experiment_env.yaml
```
**过线阈值**: 必须有git_commit、dataset_id、seed三项，缺一不可

### 1.2 Sanity三件套（硬门槛）

#### 随机标签测试
**做什么**: 确保模型不是记忆训练数据  
**怎么做**:
```python
def random_label_sanity_check(model, dataset):
    shuffled_labels = np.random.permutation(dataset.labels)
    random_score = evaluate(model, dataset, shuffled_labels)
    return random_score
```
**过线阈值**: Top-1准确率必须≈随机水平(~20%)，若>30%则**拒绝**

#### 特征遮蔽测试  
**做什么**: 验证模型确实依赖关键特征  
**怎么做**:
```python
def feature_ablation_check(model, test_data):
    baseline_ndcg = evaluate_ndcg(model, test_data)
    masked_ndcg = evaluate_ndcg(model, mask_visual_features(test_data))
    delta = baseline_ndcg - masked_ndcg
    return delta
```
**过线阈值**: |ΔnDCG@10| ≥ 0.01，若性能不变则**拒绝**

#### 泄漏扫描
**做什么**: 确保训练/测试严格隔离  
**怎么做**:
```python
def leakage_scan(train_queries, test_queries):
    similarities = compute_similarity_matrix(train_queries, test_queries)
    leakage_pairs = similarities > 0.9
    return leakage_pairs.sum()
```
**过线阈值**: 相似度>0.9的样本数必须=0，发现即移出验证集

### 1.3 统计检验（科学性保证）

**做什么**: Bootstrap CI95 + 置换检验  
**怎么做**:
```python
def statistical_validation(baseline_scores, new_scores, n_bootstrap=10000):
    # Bootstrap 置信区间
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(baseline_scores), len(baseline_scores))
        diff = new_scores[sample_idx].mean() - baseline_scores[sample_idx].mean()
        bootstrap_diffs.append(diff)
    
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    # 置换检验
    combined = np.concatenate([baseline_scores, new_scores])
    observed_diff = new_scores.mean() - baseline_scores.mean()
    
    perm_diffs = []
    for _ in range(10000):
        np.random.shuffle(combined)
        perm_new = combined[:len(new_scores)]
        perm_baseline = combined[len(new_scores):]
        perm_diff = perm_new.mean() - perm_baseline.mean()
        perm_diffs.append(perm_diff)
    
    p_value = (np.array(perm_diffs) >= observed_diff).mean()
    
    return ci_lower, ci_upper, p_value
```
**过线阈值**: CI95下界>0 且 p<0.05，才能声称"有提升"

### 1.4 相关性审查
**做什么**: 检查是否只是线性偏移  
**怎么做**:
```python
correlation = np.corrcoef(old_scores, new_scores)[0,1]
```
**过线阈值**: 相关系数r < 0.98，否则怀疑只是线性偏移

### 1.5 性能&稳定性
**做什么**: P95延迟 + 错误率监控  
**过线阈值**: 
- P95延迟不劣于baseline +5%（或CoTRR-Lite额外≤150ms）
- 错误率不升（ΔError ≤ +0.2%）

### 1.6 复核结论页（1页标准格式）

```yaml
# === 复核结论页模板 ===
experiment_id: "v2_integrity_fix_20251012"
what_changed: "视觉特征遮蔽敏感性修复 + 相关性降低"
how_measured: "Bootstrap CI95 + 置换检验 + 特征消融"

# 核心结果（严格按 data_claim_standards.md 格式）
data_claims:
  - claim: "nDCG@10 +0.0085"
    evidence: "reports/integrity_fix_benchmark_20251012.json"
    scope: "test_set / n=500 / window=2025-10-12"
    reviewer: ["audit@team", "research@team"] 
    trust_tier: "T3-Verified"
    
  - claim: "Feature ablation Δ=0.015 (significant)"
    evidence: "reports/ablation_test_20251012.json"
    trust_tier: "T3-Verified"
    
  - claim: "Correlation r=0.94 (< 0.98 threshold)"
    evidence: "reports/correlation_analysis_20251012.json"
    trust_tier: "T3-Verified"

# Sanity检查结果
sanity_checks:
  random_labels: "PASS (accuracy=19.3%)"
  feature_ablation: "PASS (Δ=0.015)"
  leakage_scan: "PASS (0 leakage pairs)"

# 统计验证
statistical_validation:
  ci95_lower: "+0.0032"
  p_value: "0.0089"
  significant: true

# 风险评估
risks:
  - "样本量仍限于500，需扩展到1000+"
  - "子域cocktails仍有异常，需专项修复"

# Go/No-Go决策
decision: "CONDITIONAL_GO"
conditions:
  - "扩展样本到≥1000"
  - "cocktails子域CI95下界>0"
  - "P95延迟<+5%"

# 证据链
evidence_chain:
  git_commit: "abcdef123456"
  dataset_version: "v2025.10.12"
  config_file: "config/integrity_fix.yaml"
  reports_hash: "sha256:1a2b3c..."
```

---

## 🔧 二、维护（Maintenance）总计划

### 2.1 生产（V1.0）- 周期化维护

#### 周任务 (每周一执行)
**做什么**: 指标回归 + 漂移监测 + 失败分析  
**怎么做**:
```bash
# 指标回归检查
python tools/production_health_check.py --metrics P50,P95,P99,error_rate,throughput

# 漂移监测  
python tools/drift_monitor.py --window 7days --features text_embedding,image_embedding

# 失败案例分析
python tools/failure_analysis.py --top_n 20 --output weekly_failures.json
```
**过线阈值**:
- P95 < 300ms（当前基线）
- 错误率 < 0.8%
- 最差10%子域不能连续2周恶化

#### 月度任务 (每月1号执行)
**做什么**: 评测集升级 + 口径校准  
**怎么做**:
```bash
# 评测集滚动升级（新增≥10%难例）
python tools/evaluation_set_refresh.py --add_hard_cases 0.1

# 口径核对与自动生成外宣数据
python tools/generate_external_claims.py --source production_evaluation.json --trust_tier T3-Verified
```

#### SLO/SLA 标准
```yaml
# 生产SLO
production_slo:
  p95_latency: "<300ms"
  error_rate: "<0.8%"  
  compliance_improvement: "≥+10%"  # 当前+13.8%

# 预警阈值（15min均值）
alerts:
  warning:
    p95_latency: ">350ms"
    error_rate: ">0.8%"
  critical:
    p95_latency: ">450ms"  # 触发自动回滚
    error_rate: ">1.2%"
```

### 2.2 研究（V2.0）- PAUSE_AND_FIX期间

#### 专注修复清单
**做什么**: 完整性修复 + 数据扩容  
**怎么做**:
```bash
# 1. 完整性修复
python research/fix_feature_ablation.py --target_delta 0.01
python research/reduce_correlation.py --target_r 0.95

# 2. 数据扩容到500+
python research/expand_evaluation_set.py --target_size 500 --hard_ratio 0.3
```

#### 复活阈值（全部满足才重启）
```yaml
revival_criteria:
  statistical:
    ci95_lower: "≥+0.015"
    p_value: "<0.05"
  
  integrity:
    feature_ablation_delta: "≥0.01"
    correlation: "<0.98"
    leakage_pairs: "=0"
  
  performance:
    latency_overhead: "≤+2ms"
    total_p95: "≤+5%"
  
  coverage:
    sample_size: "≥500"
    subdomain_all_significant: true  # 三个子域CI95均>0
```

### 2.3 CoTRR-Lite - 轻量化上线路径

#### 技术闸门
**做什么**: Top-K预筛 + 轻推理 + 缓存优化  
**怎么做**:
```python
class CoTRRLite:
    def __init__(self):
        self.top_k = 20  # 预筛维度
        self.cache = LRUCache(maxsize=1000)  # LRU缓存
        self.target_hit_rate = 0.3
    
    def inference(self, query):
        cache_key = self.get_cache_key(query)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 轻量推理
        candidates = self.prefilter_topk(query)
        scores = self.lightweight_scoring(candidates)
        result = self.vectorized_ranking(scores)
        
        self.cache[cache_key] = result
        return result
```

#### 上线阈值
```yaml
cotrr_lite_gates:
  performance:
    additional_p95: "≤150ms"
    memory_peak: "≤512MB"
    
  quality:
    ndcg_delta: "≥-2.5pts"  # 最多损失2.5分
    ci95_bound: ">-3.0"     # CI95不能触及-3
    
  cache:
    hit_rate: "≥30%"
    
  rollout:
    stages: [5%, 20%, 50%]
    hold_time: "≥24h"       # 每档至少24小时
    min_requests: "50k"     # 每档至少5万请求
```

#### 自动回滚触发线
**任一条件触发即回退**:
- 延迟>150ms（5分钟均值）
- 显存>512MB（峰值）
- 错误率>1.2%（5分钟均值）  
- CI95触及负阈值（-3.0）

---

## 📊 三、证据链标准实施（Trust Tier严格执行）

### 3.1 当前数据声明重新标注

根据`data_claim_standards.md`，重新标注现有声明：

```yaml
# V1.0 生产数据（重新标注）
v1_production:
  - claim: "Compliance +13.8%"
    evidence: "production_evaluation.json"
    evidence_hash: "sha256:a1b2c3d4..."
    run_id: "v1_prod_20251012"
    timestamp: "2025-10-12T10:30:00Z"
    scope: "Full production traffic / sample=n=120 / window=10/10–10/12"
    reviewer: ["audit@team", "owner@team"]
    trust_tier: "T3-Verified"  # 双人复核，对外宣传可用
    
# V2.0 科学复核数据（重新标注）  
v2_scientific_review:
  - claim: "CI95 lower bound +0.0101"
    evidence: "v2_scientific_review_report.json"
    evidence_hash: "sha256:e5f6g7h8..."
    scope: "Test set / sample=n=120 / bootstrap=10k"
    reviewer: ["research@team"]
    trust_tier: "T2-Internal"  # 单人复核，内部可用但不可对外
    note: "Need dual verification for T3"
    
  - claim: "Feature ablation Δ≈0 (anomaly detected)"
    evidence: "v2_scientific_review_report.json#ablation_results"
    trust_tier: "T2-Internal"
    note: "⚠️ Integrity issue requiring fix"
    
  - claim: "Linear distillation ~88% retention"
    evidence: "Initial estimates in research folder"
    trust_tier: "T1-Indicative"  # 探索性结果
    note: "⚠️ Needs comprehensive benchmark before production"
```

### 3.2 新实验标准流程

```bash
# 标准实验执行（自动生成Trust Tier标注）
python tools/run_experiment.py \
  --config configs/v2_integrity_fix.yaml \
  --trust-tier T3-Verified \
  --dual-reviewer audit@team,research@team

# 自动生成标准格式报告
# → reports/experiment_report_20251012.json (含完整证据链)
# → reports/data_claims_20251012.yaml (标准格式声明)
```

### 3.3 CI审计工具

```python
# tools/ci_data_integrity_check.py
def check_data_claims(pr_files):
    """检查PR中的数据声明是否符合标准"""
    
    violations = []
    
    for file in pr_files:
        content = read_file(file)
        
        # 检查精确数字是否有证据支撑
        precise_numbers = re.findall(r'\b\d+\.\d+[xX%]\b', content)
        for number in precise_numbers:
            if not has_evidence_file(file, number):
                violations.append(f"Precise number {number} in {file} lacks evidence")
        
        # 检查Trust Tier标注
        claims = extract_data_claims(content)
        for claim in claims:
            if not claim.get('trust_tier'):
                violations.append(f"Missing trust_tier in {file}: {claim}")
    
    return violations

# PR检查钩子
if violations:
    print("❌ Data integrity violations found:")
    for v in violations:
        print(f"  - {v}")
    exit(1)
else:
    print("✅ All data claims meet integrity standards")
```

---

## 📈 四、观测仪表板（实时监控要点）

### 4.1 生产健康概览
```yaml
production_dashboard:
  core_metrics:
    - "P50/P95/P99 latency (target: <300ms)"
    - "Error rate (target: <0.8%)"  
    - "QPS throughput"
    - "CPU/GPU utilization"
    - "Memory peak (GPU VRAM)"
  
  quality_metrics:
    - "ΔnDCG@10 vs baseline (current: +13.8%)"
    - "Subdomain performance (worst 10%)"
    - "Compliance rate trends"
  
  explainability:
    - "Attribution breakdown (compliance/conflict/region)"
    - "Failure case top-N ranking"
    - "User query difficulty distribution"
```

### 4.2 实验追踪面板
```yaml
experiment_dashboard:
  active_experiments:
    - experiment_id: "v2_integrity_fix"
      status: "PAUSE_AND_FIX"
      trust_tier: "T2-Internal"
      progress: "60% (correlation reduced to 0.94)"
      
    - experiment_id: "cotrr_lite_v1"  
      status: "STAGING"
      trust_tier: "T1-Indicative"
      progress: "Tech gate: 40% (cache hit rate achieved)"
  
  pipeline_health:
    - "CI95 confidence intervals (real-time)"
    - "Rollback events timeline"
    - "Trust tier compliance rate"
```

### 4.3 缓存监控（安全栅栏）
```yaml
cache_monitoring:
  performance:
    - "Hit rate (target: ≥30%)"
    - "Cache vs recompute score deviation"
    - "TTL effectiveness (24h window)"
  
  safety:
    - "Version signature mismatches"
    - "Similarity threshold violations"
    - "Auto-disable trigger events"
  
  alerts:
    - "Score deviation >0.02 → auto-disable cache"
    - "Hit rate drop >20% → investigate"
    - "Memory usage >1GB → LRU aggressive cleanup"
```

---

## 👥 五、角色职责与节奏（RACI矩阵）

### 5.1 核心角色
```yaml
roles:
  owner_algorithm:
    responsible: ["实验设计", "证据链产出", "Trust Tier标注"]
    accountable: ["技术决策", "代码质量"]
    
  reviewer_audit:
    responsible: ["Sanity检查", "统计验证", "双人复核"]
    accountable: ["科学性把关", "Trust Tier升级审批"]
    
  ops_platform:
    responsible: ["监控告警", "自动回滚", "性能基准"]
    accountable: ["生产稳定性", "SLO达成"]
    
  communicator_product:
    responsible: ["外宣口径", "业务解释"]
    accountable: ["T3-Verified数据使用", "公信力维护"]
```

### 5.2 运营节奏
```yaml
daily:
  - "生产健康检查（5分钟）"
  - "实验进展更新（≤3行状态）"
  
weekly:
  - "Worst 10%子域复盘会议（30分钟）"
  - "实验Go/No-Go决策（基于Trust Tier）"
  - "失败案例库更新（≥20条）"
  
monthly:
  - "评测集校准与升级"
  - "外宣口径vs内部数据对账"
  - "Trust Tier合规审计"
  - "路线图滚动更新"
```

---

## 🎯 六、一句话行动指南

### 现状对应策略

**V1.0在产**: 
```yaml
action: "守住+13.8%的T3-Verified声明"
focus: ["P95<300ms", "最差子域盯防", "可解释性日志"]
kpi: "SLO达成率≥95%"
```

**V2.0 PAUSE_AND_FIX**:
```yaml  
action: "先修完整性再谈复活"
focus: ["特征遮蔽Δ≥0.01", "相关性r<0.98", "样本扩展≥500"]
gate: "全部修复完成才重新进入T3评审"
```

**CoTRR-Lite轻量化**:
```yaml
action: "阈值驱动上线，触线即回滚"  
focus: ["≤150ms", "≤512MB", "CI不触负阈"]
strategy: "命中即扩量(5%→20%→50%)"
```

---

## 🚨 七、紧急响应SOP

### 7.1 生产告警响应（5分钟内）
```bash
# P0级别（自动回滚触发）
if [[ $(check_p95_latency) -gt 450 ]] || [[ $(check_error_rate) -gt 1.2 ]]; then
    trigger_rollback "P0_PRODUCTION_DEGRADATION"
    notify_oncall_engineer
fi

# P1级别（人工介入）  
if [[ $(check_p95_latency) -gt 350 ]] || [[ $(check_error_rate) -gt 0.8 ]]; then
    alert_team "P1_SLO_BREACH"
    start_investigation_runbook
fi
```

### 7.2 数据完整性事件响应
```bash
# Trust Tier降级流程
if integrity_violation_detected; then
    downgrade_trust_tier "T3-Verified" "T1-Indicative" 
    quarantine_external_claims
    initiate_audit_review
fi
```

### 7.3 实验紧急停止
```bash
# CI95触及负阈值
if [[ $(check_ci95_lower) -lt -0.03 ]]; then
    stop_experiment_immediately
    preserve_evidence_chain
    document_failure_analysis
fi
```

---

## 📋 八、检查清单（每次使用）

### 实验启动前 ✅
- [ ] 配置文件锁定（git commit + dataset + seed）
- [ ] Trust Tier目标确定（T1/T2/T3）
- [ ] 证据文件路径规划（reports/, logs/, config/）
- [ ] 双人复核人员指定（T3级别）

### 实验执行中 ✅  
- [ ] Sanity三件套全部通过
- [ ] 统计检验CI95>0且p<0.05
- [ ] 性能测试P95不劣化>5%
- [ ] 相关性检查r<0.98

### 结果发布前 ✅
- [ ] 证据链完整（可复现命令行）
- [ ] Trust Tier标注正确
- [ ] 外宣口径与内部数据一致  
- [ ] CI审计检查通过

### 生产上线前 ✅
- [ ] T3-Verified级别证据支撑
- [ ] 自动回滚机制配置
- [ ] 监控告警阈值设定
- [ ] 应急响应SOP确认

---

**SOP版本**: v1.0  
**生效时间**: 2025-10-12  
**下次审核**: 2025-11-12  
**维护团队**: Data Integrity + Production Ops