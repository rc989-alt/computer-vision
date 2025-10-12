# Colab夜间实验执行指南

## 🚀 快速启动 (推荐方式)

### 方式1: 一键启动脚本
```python
# 在Colab新建notebook，复制粘贴运行
!wget -q https://raw.githubusercontent.com/rc989-alt/computer-vision/main/colab_quick_start.py
!python colab_quick_start.py
```

### 方式2: 手动分步执行

1. **挂载Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **下载数据和脚本**
```python
# 直接从GitHub下载所需文件
!wget -q https://raw.githubusercontent.com/rc989-alt/computer-vision/main/colab_night_runner.py
!wget -q https://raw.githubusercontent.com/rc989-alt/computer-vision/main/production_dataset.json
print("✅ 文件下载完成!")
```

3. **启动实验**
```python
# 完整8小时实验 (推荐今晚使用)
!python colab_night_runner.py \
  --data production_dataset.json \
  --out_dir "/content/drive/MyDrive/v1_night_opt" \
  --hours_per_shard 2.0 \
  --total_shards 4

# 快速测试版 (30分钟验证用)
!python colab_night_runner.py \
  --data production_dataset.json \
  --out_dir "/content/drive/MyDrive/v1_night_test" \
  --hours_per_shard 0.25 \
  --total_shards 2
```

## 📊 实验配置

### 参数网格
- **MMR Alpha**: [0.70, 0.75, 0.80] (多样性权重)
- **主题槽位**: [0, 1, 2] (强制主题覆盖数)
- **数据分片**: 4片 (每片2小时，总计8小时)

### 统计标准
- **样本量**: 320 queries (统计功效95%+)
- **置信区间**: Bootstrap 95% CI
- **显著性**: p < 0.05 + CI下界 > 0

## 🛡️ 稳定性保障

### 分片机制
- ✅ 每片独立运行，断线不影响已完成分片
- ✅ 所有结果自动保存到Google Drive
- ✅ 支持断点续跑，重启后跳过已完成实验

### 监控要点
- **内存使用**: 每分片 < 2GB (Colab限制内避免OOM)
- **运行时长**: 每分片 < 2小时 (避免超时断线)  
- **输出频率**: 最小化print减少网络负载

## 📋 结果文件结构

```
/content/drive/MyDrive/v1_night_opt/
├── progress_20241012_220000.json      # 运行进度
├── morning_summary.json               # 汇总报告
├── shard_0_mmr_a0.70_s0/             # 分片实验结果
│   ├── results.json                   # 详细统计结果
│   └── experiment.log                 # 运行日志
├── shard_0_mmr_a0.70_s1/
├── shard_1_mmr_a0.70_s0/
└── ...                               # 全部36个实验配置
```

## 🎯 决策标准

### GO条件 (部署到生产)
- ✅ 95% CI下界 > 0 (统计显著)
- ✅ 平均改进 > 0.01 (实用显著)
- ✅ 至少2/3分片显示一致改进

### PAUSE条件 (需要更多数据)
- ⚠️ CI包含0但趋势正向
- ⚠️ 部分分片显示改进

### NO_GO条件 (回滚当前方案)
- ❌ 多数分片显示负面影响
- ❌ 统计不显著且改进 < 0.005

## ⏰ 时间规划

### 今晚 (22:00 - 06:00)
- 22:00-22:30: 环境配置和数据准备
- 22:30-06:30: 分片实验执行 (8小时)
- 06:30-07:00: 结果汇总和报告生成

### 明早决策点
- 07:00: 查看morning_summary.json
- 07:30: 基于统计结果做GO/NO_GO决策
- 08:00: 如果GO，开始生产部署准备

## 🔧 故障排除

### 常见问题
1. **OOM错误**: 减少分片大小或TOTAL_SHARDS
2. **网络超时**: 启用Colab Pro后台执行
3. **Drive空间不足**: 清理旧实验文件
4. **随机断线**: 依赖分片机制，重启继续

### 应急方案
- **轻量版**: 只跑最优1-2个配置组合
- **本地版**: 下载数据到本地GPU机器执行
- **延期版**: 分散到多个晚上执行

## 📞 支持联系

实验过程中如遇问题:
1. 检查progress文件确认已完成实验数
2. 查看最新的experiment.log定位错误
3. 必要时重启Colab会话继续分片执行

---
**记住**: 这是"可中断、可续跑、自动落盘"的设计，不要怕断线！