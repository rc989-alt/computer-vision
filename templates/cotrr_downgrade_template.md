# CoTRR 数据降级模板

## 修正前后对比

### 修正前 ❌
```markdown
| Latency Overhead | <2x | **300.7x** | 🔴 严重超标 | 不可部署 |
| Memory Usage | <500MB | 2.3GB | 🔴 超标 | 资源不足 |
| Model Size | <100MB | 450MB | 🔴 过大 | 部署困难 |
```

### 修正后 ✅
```markdown
| 指标类型 | 目标值 | 观察状态 | 状态 | 影响 |
|---------|-------|----------|------|------|
| Latency Overhead | <2x | **显著超标** | 🔴 性能问题 | 需要优化 |
| Memory Usage | <500MB | **资源密集** | 🔴 超标 | 需要简化 |
| Model Size | <100MB | **偏大** | 🔴 过大 | 需要压缩 |

**Data-Claim**: CoTRR存在显著性能开销（估计倍数级）
**Evidence**: 初步概念验证，无完整基准测试
**Scope**: 概念阶段 / 未进行生产级评估
**Reviewer**: research@team (concept-only)
**Trust Tier**: T1-Indicative
**Note**: ⚠️ 需要完整基准测试后更新具体数值
```

## 查找替换指令

### 精确数字移除
```bash
# 移除所有300.7x引用
sed -i 's/300\.7x/显著开销/g' research/03_cotrr_lightweight_line/SUMMARY.md

# 移除2.3GB引用
sed -i 's/2\.3GB/资源密集/g' research/03_cotrr_lightweight_line/SUMMARY.md

# 移除450MB引用
sed -i 's/450MB/偏大/g' research/03_cotrr_lightweight_line/SUMMARY.md
```

### 添加警告标注
```bash
# 在CoTRR文件末尾添加数据说明
echo "**数据说明**: ⚠️ 性能数字为估算值，需要实际benchmark验证" >> research/03_cotrr_lightweight_line/SUMMARY.md
```

## 占位文件处理
```bash
# 检查空文件或占位文件
find research/03_cotrr_lightweight_line/ -name "*.py" -size -100c

# 在空文件开头添加占位标记
for file in $(find research/03_cotrr_lightweight_line/ -name "*.py" -size -100c); do
    echo "[PLACEHOLDER — Concept Stage, No Valid Evidence]" | cat - $file > temp && mv temp $file
done
```
