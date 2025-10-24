# V1.0 生产回滚方案

## 🚨 回滚决策矩阵

### 立即回滚条件 (自动触发)
- **错误率 > 5%**: 系统功能异常
- **P95延迟 > 2.0ms**: 用户体验严重影响  
- **Compliance改进 < -0.05**: 核心指标恶化
- **Blossom→Fruit准确率 < 95%**: 基础准确性失效

### 建议回滚条件 (人工决策)
- **错误率 > 1%**: 系统稳定性风险
- **P95延迟 > 0.5ms**: 性能下降明显
- **Compliance改进 < +0.08**: 低于预期收益
- **用户投诉增加**: 主观体验恶化

---

## ⚡ 快速回滚流程 (5分钟内完成)

### 步骤1: 紧急流量切换 (1分钟)
```python
# 立即禁用增强器
def enhanced_search(candidates, query):
    # return enhancer.enhance_ranking(candidates, query)  # 注释掉
    return candidates  # 直接返回原始结果

# 或通过配置开关
ENABLE_V1_ENHANCER = False  # 设置为False

def enhanced_search(candidates, query):
    if ENABLE_V1_ENHANCER:
        return enhancer.enhance_ranking(candidates, query)
    else:
        return candidates
```

### 步骤2: 验证回滚效果 (2分钟)
```bash
# 检查流量是否已切换到原始系统
curl -X POST http://api/search -d '{"query": "test", "debug": true}' | grep "enhanced"
# 应该不包含"enhanced"字段

# 快速验证基础功能
curl -X POST http://api/search -d '{"query": "apple"}' | jq '.results | length'
# 应该返回正常结果数量
```

### 步骤3: 监控系统恢复 (2分钟)
```bash
# 检查关键指标
python -c "
import requests, time
start = time.time()
r = requests.post('http://api/search', json={'query': 'test'})
latency = (time.time() - start) * 1000
print(f'Latency: {latency:.2f}ms')
print(f'Status: {r.status_code}')
"
```

---

## 🔄 分层回滚策略

### Level 1: 配置回滚 (无需重启)
```json
// 修改production_config.json
{
  "feature_flags": {
    "enable_position_optimization": false,
    "enable_diversity_penalty": false,
    "enable_top_k_boost": false,
    "enable_quality_filtering": false
  }
}
```

### Level 2: 流量回滚 (逐步降低)
```python
# 将流量从100% → 50% → 20% → 5% → 0%
TRAFFIC_PERCENTAGE = 0  # 设置为0完全禁用

def enhanced_search(candidates, query):
    if random.random() < TRAFFIC_PERCENTAGE / 100:
        return enhancer.enhance_ranking(candidates, query)
    else:
        return candidates
```

### Level 3: 版本回滚 (回到上一稳定版本)
```bash
# 如果有前一版本备份
cp enhancer_v0.9.py enhancer_v1.py
cp production_config_v0.9.json production_config.json

# 重启相关服务
systemctl restart search-service
```

---

## 📊 回滚验证检查点

### 即时验证 (回滚后5分钟内)
- [ ] **基础功能**: 搜索API正常响应
- [ ] **延迟恢复**: P95延迟 < 0.1ms  
- [ ] **错误率**: < 0.1%
- [ ] **流量分配**: 确认100%流量走原始系统

### 短期验证 (回滚后1小时内)
- [ ] **用户体验**: 无新增用户投诉
- [ ] **业务指标**: 核心转化率稳定
- [ ] **系统稳定**: 无异常错误日志
- [ ] **资源使用**: CPU/内存恢复正常

### 长期验证 (回滚后24小时内)  
- [ ] **整体性能**: 所有指标恢复到V1.0部署前水平
- [ ] **数据一致性**: 无数据异常或损坏
- [ ] **监控告警**: 清除所有相关告警
- [ ] **文档更新**: 记录回滚原因和过程

---

## 🔍 回滚后问题排查

### 1. 确定回滚原因
```bash
# 分析V1.0部署期间的日志
grep -E "(ERROR|WARNING)" health_check.log | tail -20

# 查看性能指标变化
python -c "
import json
with open('health_report.json') as f:
    report = json.load(f)
    print('Performance:', report['checks']['performance'])
    print('Accuracy:', report['checks']['accuracy'])
"
```

### 2. 评估影响范围
```sql
-- 如果有数据库记录
SELECT 
    COUNT(*) as affected_queries,
    AVG(response_time) as avg_latency,
    SUM(CASE WHEN error_code IS NOT NULL THEN 1 ELSE 0 END) as error_count
FROM search_logs 
WHERE timestamp BETWEEN 'V1.0_deploy_time' AND 'rollback_time';
```

### 3. 生成回滚报告
```python
rollback_report = {
    'rollback_time': datetime.now().isoformat(),
    'rollback_reason': 'performance_degradation',  # 填写具体原因
    'affected_duration_hours': 2.5,
    'impact_assessment': {
        'affected_queries': 1250,
        'error_rate_peak': 0.025,
        'latency_peak_ms': 1.8,
        'user_complaints': 3
    },
    'recovery_metrics': {
        'rollback_duration_minutes': 4,
        'full_recovery_minutes': 15,
        'performance_restored': True
    }
}
```

---

## 🚨 紧急联系信息

### 技术负责人
- **主要联系人**: [姓名] [电话] [邮箱]
- **备用联系人**: [姓名] [电话] [邮箱]

### 业务负责人  
- **产品经理**: [姓名] [电话]
- **运营经理**: [姓名] [电话]

### 基础设施
- **SRE团队**: [24小时值班电话]
- **云服务支持**: [厂商支持热线]

---

## 📋 回滚后续行动

### 短期行动 (24小时内)
1. **根因分析**: 确定V1.0失败的技术原因
2. **影响评估**: 量化对用户和业务的影响
3. **修复计划**: 制定问题修复的时间表
4. **文档更新**: 更新部署和回滚文档

### 中期行动 (1周内)
1. **代码修复**: 修复识别的技术问题
2. **测试加强**: 增加更严格的预发布测试
3. **监控优化**: 改进监控和告警机制
4. **流程改进**: 优化部署和回滚流程

### 长期行动 (1个月内)
1. **架构评估**: 评估是否需要架构级改进
2. **工具建设**: 开发更好的部署和监控工具
3. **团队培训**: 加强团队的故障处理能力
4. **预案完善**: 完善各种异常场景的应对预案

---

## 📝 回滚记录模板

```
回滚日期: ____________
回滚人员: ____________  
回滚原因: ____________
回滚触发条件: ________
回滚耗时: ____________

回滚前状态:
- 错误率: ____%
- P95延迟: ____ms  
- Compliance: ____
  
回滚后状态:
- 错误率: ____%
- P95延迟: ____ms
- 系统恢复: ✅/❌

影响评估:
- 受影响查询数: ____
- 用户投诉数: ____
- 业务损失: ____

后续行动:
1. ________________
2. ________________
3. ________________

经验教训:
________________
```

---

**🎯 回滚目标**: 在5分钟内将系统恢复到V1.0部署前的稳定状态，确保用户体验不受持续影响。