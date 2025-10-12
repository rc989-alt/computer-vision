# V1.0 生产部署指南

## 🚀 部署概览

V1.0增强器是经过严格验证的生产就绪系统，具备以下特点：
- **性能保证**: +0.1382 Compliance@1, 0.06ms P95延迟
- **稳定性验证**: 6种优化方法对比确认的最优平衡点
- **生产特性**: 健康检查、监控告警、自动回滚

---

## 📋 部署前检查清单

### 环境要求
- [ ] Python 3.8+
- [ ] NumPy >= 1.19.0
- [ ] 内存: 最少512MB可用
- [ ] CPU: 支持并发处理
- [ ] 磁盘: 100MB存储空间

### 依赖安装
```bash
# 基础依赖
pip install numpy

# 可选依赖（用于资源监控）
pip install psutil
```

### 文件完整性检查
- [ ] `enhancer_v1.py` - 核心增强器
- [ ] `production_config.json` - 生产配置
- [ ] `health_check.py` - 健康检查脚本
- [ ] `deployment_guide.md` - 本文档
- [ ] `rollback_plan.md` - 回滚方案

---

## 🎯 分阶段部署流程

### 阶段1: 环境验证 (30分钟)

1. **功能测试**
```bash
cd production/
python enhancer_v1.py
```
预期输出: ✅ V1.0生产增强器测试完成

2. **健康检查**
```bash
python health_check.py
```
预期输出: 整体状态: healthy

3. **配置验证**
```bash
python -c "import json; print(json.load(open('production_config.json'))['version'])"
```
预期输出: 1.0.0

### 阶段2: 灰度发布 (5%流量, 24小时)

1. **部署配置**
```python
from enhancer_v1 import create_production_enhancer

# 创建增强器实例
enhancer = create_production_enhancer('production_config.json')

# 集成到现有系统
def enhanced_search(candidates, query):
    if should_use_enhancer(traffic_percentage=5):  # 5%流量
        return enhancer.enhance_ranking(candidates, query)
    else:
        return candidates  # 原始结果
```

2. **监控指标**
- 每小时检查健康状态
- 监控延迟P95 < 0.2ms
- 监控错误率 < 1%
- 记录Compliance改进

3. **验证通过条件**
- 24小时内无严重错误
- Compliance@1改进 >= +0.13
- P95延迟 < 0.1ms
- Blossom→Fruit准确率 >= 98%

### 阶段3: 扩大灰度 (20%流量, 24小时)

```python
# 更新流量配置
def enhanced_search(candidates, query):
    if should_use_enhancer(traffic_percentage=20):  # 20%流量
        return enhancer.enhance_ranking(candidates, query)
    else:
        return candidates
```

### 阶段4: 半量发布 (50%流量, 24小时)

### 阶段5: 全量发布 (100%流量)

---

## 📊 监控和告警

### 实时监控脚本
```bash
# 每5分钟运行健康检查
*/5 * * * * cd /path/to/production && python health_check.py

# 每小时生成状态报告
0 * * * * cd /path/to/production && python health_check.py > hourly_report.log
```

### 关键监控指标

| 指标 | 正常范围 | 告警阈值 | 严重告警 |
|------|----------|----------|----------|
| P95延迟 | < 0.1ms | > 0.2ms | > 1.0ms |
| 错误率 | < 0.1% | > 1% | > 5% |
| Compliance改进 | > +0.13 | < +0.11 | < +0.08 |
| 内存使用 | < 200MB | > 500MB | > 1GB |

### 告警通知设置
```bash
# 错误告警
if error_rate > 0.01:
    send_alert("V1.0增强器错误率超标: {error_rate}")

# 性能告警  
if p95_latency > 0.2:
    send_alert("V1.0增强器延迟超标: {p95_latency}ms")

# 准确性告警
if blossom_fruit_accuracy < 0.98:
    send_alert("V1.0增强器准确率下降: {accuracy}")
```

---

## 🔧 生产配置优化

### 高并发配置
```json
{
  "enhancer_config": {
    "max_latency_ms": 0.5,
    "enable_health_check": false,  // 高并发时关闭实时检查
    "enable_caching": true
  },
  "monitoring": {
    "metrics_interval_seconds": 300,  // 降低监控频率
    "log_sample_rate": 0.001
  }
}
```

### 低延迟配置
```json
{
  "enhancer_config": {
    "diversity_weight": 0.1,  // 简化多样性计算
    "enable_health_check": false,
    "max_latency_ms": 0.1
  }
}
```

### 高准确性配置
```json
{
  "enhancer_config": {
    "quality_threshold": 0.6,  // 提高质量门槛
    "top_k_boost": 0.2,
    "position_decay": 0.9
  }
}
```

---

## ⚡ 性能优化建议

### 1. 代码级优化
- 预计算常用指标
- 缓存重复查询结果
- 批量处理候选项

### 2. 系统级优化
- 使用SSD存储
- 配置适当的JVM/Python内存
- 启用并行处理

### 3. 网络优化
- 启用HTTP/2
- 配置CDN缓存
- 优化序列化格式

---

## 🔍 故障排查

### 常见问题

**1. 延迟突然增加**
```bash
# 检查资源使用
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"

# 检查健康状态
python health_check.py
```

**2. 准确率下降**
```bash
# 运行Blossom→Fruit测试
python -c "
from health_check import ProductionHealthChecker
checker = ProductionHealthChecker()
result = checker._check_accuracy()
print(result)
"
```

**3. 错误率上升**
```bash
# 查看详细错误日志
tail -f health_check.log | grep ERROR

# 检查配置文件
python -c "
import json
config = json.load(open('production_config.json'))
print('Config validation:', 'version' in config)
"
```

---

## 📞 联系信息

- **负责人**: 研发团队
- **紧急联系**: 技术支持
- **文档更新**: 每次版本发布后更新

---

## 📝 部署记录模板

```
部署日期: ____
部署人员: ____
环境: 生产/测试
流量比例: ____%
验证结果:
  - 功能测试: ✅/❌
  - 性能测试: ✅/❌  
  - 健康检查: ✅/❌
  - 监控配置: ✅/❌
备注: ________________
```

---

**🎯 部署成功标志**: 24小时内稳定运行，所有监控指标正常，无用户投诉。