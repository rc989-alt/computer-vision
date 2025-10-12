#!/usr/bin/env python3
"""
CoTRR-Stable Stage 1 Progress Tracker
两周冲刺进度管理和任务跟踪系统

功能:
1. 任务进度跟踪
2. 性能指标记录
3. 里程碑检查点
4. 风险预警系统
5. 每日进度报告
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    description: str
    assigned_day: int  # Day 1-14
    estimated_hours: float
    dependencies: List[str]  # 依赖的task_id列表
    status: str = "pending"  # pending, in_progress, completed, blocked
    actual_hours: float = 0.0
    completion_percentage: int = 0
    notes: List[str] = None
    deliverables: List[str] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []
        if self.deliverables is None:
            self.deliverables = []

@dataclass
class Milestone:
    """里程碑定义"""
    id: str
    name: str
    target_date: str  # YYYY-MM-DD
    success_criteria: List[str]
    status: str = "pending"  # pending, achieved, missed
    actual_date: Optional[str] = None
    notes: str = ""

@dataclass
class PerformanceMetric:
    """性能指标记录"""
    timestamp: str
    metric_name: str
    value: float
    baseline_value: Optional[float] = None
    target_value: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    notes: str = ""

class CoTRRStageTracker:
    """CoTRR-Stable Stage 1 进度跟踪器"""
    
    def __init__(self, project_dir: str = "research/stage1_progress"):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # 项目开始时间
        self.start_date = datetime(2025, 10, 11)  # 今天
        self.current_day = 1
        
        # 任务和里程碑
        self.tasks = {}
        self.milestones = {}
        self.metrics = []
        self.daily_reports = []
        
        # 初始化任务计划
        self._initialize_stage1_tasks()
        self._initialize_milestones()
        
        logger.info(f"CoTRR-Stable Stage 1 跟踪器已初始化: {self.project_dir}")
    
    def _initialize_stage1_tasks(self):
        """初始化Stage 1任务计划"""
        stage1_tasks = [
            # Week 1: 核心架构实现
            Task(
                id="T001",
                name="Cross-Attention模型架构",
                description="实现Token化多模态编码器和轻量级Cross-Attention",
                assigned_day=1,
                estimated_hours=8.0,
                dependencies=[],
                deliverables=[
                    "TokenizedMultiModalEncoder类",
                    "LightweightCrossAttention类", 
                    "单元测试通过",
                    "性能基准测试"
                ]
            ),
            Task(
                id="T002", 
                name="ListMLE + Focal Loss实现",
                description="实现ListMLE和Focal Loss组合损失函数",
                assigned_day=2,
                estimated_hours=6.0,
                dependencies=["T001"],
                deliverables=[
                    "ListMLEWithFocalLoss类",
                    "损失函数单元测试",
                    "梯度检查通过"
                ]
            ),
            Task(
                id="T003",
                name="稳健训练Pipeline",
                description="实现两阶段训练：Pairwise Warmup → ListMLE Fine-tune",
                assigned_day=3,
                estimated_hours=10.0,
                dependencies=["T001", "T002"],
                deliverables=[
                    "StableTrainingPipeline类完善",
                    "训练循环实现",
                    "checkpoint保存/加载"
                ]
            ),
            Task(
                id="T004",
                name="等渗校准器",
                description="实现等渗回归校准和温度标定",
                assigned_day=4,
                estimated_hours=4.0,
                dependencies=["T003"],
                deliverables=[
                    "IsotonicCalibrator类",
                    "校准评测指标",
                    "ECE计算验证"
                ]
            ),
            Task(
                id="T005",
                name="Step5数据集成",
                description="完善scored.jsonl读取和特征提取",
                assigned_day=5,
                estimated_hours=6.0,
                dependencies=["T003"],
                deliverables=[
                    "Step5DataLoader完善",
                    "特征提取验证",
                    "数据质量分析报告"
                ]
            ),
            Task(
                id="T006",
                name="初步训练测试",
                description="使用mock数据进行完整训练流程测试",
                assigned_day=5,
                estimated_hours=4.0,
                dependencies=["T003", "T004", "T005"],
                deliverables=[
                    "训练流程成功运行",
                    "初步性能指标",
                    "问题识别和修复"
                ]
            ),
            
            # Week 2: 优化和A/B准备
            Task(
                id="T007",
                name="超参数调优",
                description="网格搜索最优超参数组合",
                assigned_day=8,
                estimated_hours=8.0,
                dependencies=["T006"],
                deliverables=[
                    "超参数搜索脚本",
                    "最优参数配置",
                    "性能对比报告"
                ]
            ),
            Task(
                id="T008",
                name="Hard Negative Mining",
                description="实现困难样本挖掘策略",
                assigned_day=9,
                estimated_hours=6.0,
                dependencies=["T007"],
                deliverables=[
                    "HardNegativeMiner类",
                    "采样策略验证",
                    "训练效果分析"
                ]
            ),
            Task(
                id="T009",
                name="评测框架完善",
                description="Bootstrap CI计算和失败分析系统",
                assigned_day=10,
                estimated_hours=8.0,
                dependencies=["T007"],
                deliverables=[
                    "BootstrapEvaluator完善",
                    "失败分析可视化",
                    "统计显著性检验"
                ]
            ),
            Task(
                id="T010",
                name="A/B测试接口",
                description="实现生产环境部署接口",
                assigned_day=12,
                estimated_hours=10.0,
                dependencies=["T008", "T009"],
                deliverables=[
                    "rerank_step4_candidates接口",
                    "Shadow mode支持",
                    "监控指标集成"
                ]
            ),
            Task(
                id="T011",
                name="性能验收测试",
                description="最终性能指标验证和门槛检查",
                assigned_day=13,
                estimated_hours=6.0,
                dependencies=["T010"],
                deliverables=[
                    "完整性能报告",
                    "置信区间计算",
                    "门槛达成确认"
                ]
            ),
            Task(
                id="T012",
                name="文档和交付",
                description="技术文档编写和代码整理",
                assigned_day=14,
                estimated_hours=6.0,
                dependencies=["T011"],
                deliverables=[
                    "技术文档",
                    "使用手册",
                    "部署指南"
                ]
            )
        ]
        
        for task in stage1_tasks:
            self.tasks[task.id] = task
    
    def _initialize_milestones(self):
        """初始化里程碑"""
        milestones = [
            Milestone(
                id="M1",
                name="Week 1 核心架构完成",
                target_date="2025-10-18",
                success_criteria=[
                    "所有核心组件实现完成",
                    "单元测试全部通过", 
                    "Mock数据训练成功",
                    "初步性能指标可测量"
                ]
            ),
            Milestone(
                id="M2", 
                name="性能目标达成",
                target_date="2025-10-23",
                success_criteria=[
                    "Compliance@1 ≥ +4pts (95% CI)",
                    "nDCG@10 ≥ +8pts (95% CI)",
                    "ECE ≤ 0.03",
                    "统计显著性 p < 0.05"
                ]
            ),
            Milestone(
                id="M3",
                name="A/B测试就绪",
                target_date="2025-10-25",
                success_criteria=[
                    "生产接口完成",
                    "Shadow mode测试通过",
                    "监控告警配置完成",
                    "回滚机制验证"
                ]
            )
        ]
        
        for milestone in milestones:
            self.milestones[milestone.id] = milestone
    
    def get_current_day(self) -> int:
        """获取当前项目日期"""
        current_date = datetime.now()
        days_passed = (current_date - self.start_date).days + 1
        return min(days_passed, 14)  # 项目总共14天
    
    def update_task_status(self, task_id: str, status: str, 
                          completion_percentage: int = None,
                          actual_hours: float = None,
                          notes: str = None):
        """更新任务状态"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        task.status = status
        
        if completion_percentage is not None:
            task.completion_percentage = completion_percentage
        
        if actual_hours is not None:
            task.actual_hours = actual_hours
        
        if notes:
            task.notes.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: {notes}")
        
        logger.info(f"Task {task_id} updated: {status} ({completion_percentage}%)")
        self._save_progress()
    
    def record_metric(self, metric_name: str, value: float,
                     baseline_value: float = None,
                     target_value: float = None,
                     confidence_interval: List[float] = None,
                     notes: str = ""):
        """记录性能指标"""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_name=metric_name,
            value=value,
            baseline_value=baseline_value,
            target_value=target_value,
            confidence_interval=confidence_interval,
            notes=notes
        )
        
        self.metrics.append(metric)
        logger.info(f"Metric recorded: {metric_name} = {value}")
        self._save_progress()
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """生成每日进度报告"""
        current_day = self.get_current_day()
        
        # 今日任务
        today_tasks = [task for task in self.tasks.values() 
                      if task.assigned_day == current_day]
        
        # 进度统计
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        in_progress_tasks = len([t for t in self.tasks.values() if t.status == "in_progress"])
        blocked_tasks = len([t for t in self.tasks.values() if t.status == "blocked"])
        
        # 时间统计
        total_estimated_hours = sum(t.estimated_hours for t in self.tasks.values())
        total_actual_hours = sum(t.actual_hours for t in self.tasks.values())
        
        # 即将到来的里程碑
        upcoming_milestones = [
            m for m in self.milestones.values()
            if m.status == "pending" and 
            datetime.strptime(m.target_date, "%Y-%m-%d") >= datetime.now()
        ]
        
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "project_day": current_day,
            "today_tasks": [asdict(task) for task in today_tasks],
            "progress_summary": {
                "completed_tasks": completed_tasks,
                "in_progress_tasks": in_progress_tasks,
                "blocked_tasks": blocked_tasks,
                "total_tasks": total_tasks,
                "completion_rate": completed_tasks / total_tasks * 100
            },
            "time_tracking": {
                "estimated_hours": total_estimated_hours,
                "actual_hours": total_actual_hours,
                "efficiency": total_actual_hours / total_estimated_hours * 100 if total_estimated_hours > 0 else 0
            },
            "upcoming_milestones": [asdict(m) for m in upcoming_milestones],
            "recent_metrics": [asdict(m) for m in self.metrics[-5:]],  # 最近5个指标
            "risks_and_blockers": self._identify_risks()
        }
        
        self.daily_reports.append(report)
        self._save_daily_report(report)
        
        return report
    
    def _identify_risks(self) -> List[str]:
        """识别风险和阻塞项"""
        risks = []
        current_day = self.get_current_day()
        
        # 检查延期任务
        overdue_tasks = [
            task for task in self.tasks.values()
            if task.assigned_day < current_day and task.status != "completed"
        ]
        
        if overdue_tasks:
            risks.append(f"{len(overdue_tasks)} 个任务延期")
        
        # 检查阻塞任务
        blocked_tasks = [task for task in self.tasks.values() if task.status == "blocked"]
        if blocked_tasks:
            risks.append(f"{len(blocked_tasks)} 个任务被阻塞")
        
        # 检查依赖关系
        for task in self.tasks.values():
            if task.status == "pending":
                unmet_deps = [
                    dep for dep in task.dependencies
                    if self.tasks[dep].status != "completed"
                ]
                if unmet_deps and task.assigned_day <= current_day:
                    risks.append(f"Task {task.id} 依赖未满足: {unmet_deps}")
        
        return risks
    
    def _save_progress(self):
        """保存进度到文件"""
        progress_data = {
            "tasks": {tid: asdict(task) for tid, task in self.tasks.items()},
            "milestones": {mid: asdict(milestone) for mid, milestone in self.milestones.items()},
            "metrics": [asdict(metric) for metric in self.metrics],
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.project_dir / "progress.json", "w") as f:
            json.dump(progress_data, f, indent=2)
    
    def _save_daily_report(self, report: Dict):
        """保存每日报告"""
        report_file = self.project_dir / f"daily_report_{report['date']}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # 同时保存为markdown格式
        self._save_markdown_report(report)
    
    def _save_markdown_report(self, report: Dict):
        """保存Markdown格式的每日报告"""
        md_content = f"""# CoTRR-Stable Daily Report - {report['date']}

## 📅 项目进度
- **项目第 {report['project_day']} 天** (共14天)
- **完成进度**: {report['progress_summary']['completion_rate']:.1f}% ({report['progress_summary']['completed_tasks']}/{report['progress_summary']['total_tasks']} 任务)

## 📋 今日任务
"""
        
        for task in report['today_tasks']:
            md_content += f"### {task['name']} ({task['id']})\n"
            md_content += f"- **状态**: {task['status']}\n"
            md_content += f"- **完成度**: {task['completion_percentage']}%\n"
            md_content += f"- **描述**: {task['description']}\n\n"
        
        md_content += f"""## 📊 进度统计
- ✅ 已完成: {report['progress_summary']['completed_tasks']} 任务
- 🔄 进行中: {report['progress_summary']['in_progress_tasks']} 任务
- ⛔ 阻塞中: {report['progress_summary']['blocked_tasks']} 任务

## ⏱️ 时间跟踪
- **预估总时间**: {report['time_tracking']['estimated_hours']:.1f} 小时
- **实际用时**: {report['time_tracking']['actual_hours']:.1f} 小时
- **效率**: {report['time_tracking']['efficiency']:.1f}%

## 🎯 即将到来的里程碑
"""
        
        for milestone in report['upcoming_milestones']:
            md_content += f"- **{milestone['name']}** (目标: {milestone['target_date']})\n"
        
        if report['risks_and_blockers']:
            md_content += "\n## ⚠️ 风险和阻塞项\n"
            for risk in report['risks_and_blockers']:
                md_content += f"- {risk}\n"
        
        if report['recent_metrics']:
            md_content += "\n## 📈 最新性能指标\n"
            for metric in report['recent_metrics']:
                md_content += f"- **{metric['metric_name']}**: {metric['value']:.3f}\n"
        
        md_file = self.project_dir / f"daily_report_{report['date']}.md"
        with open(md_file, "w") as f:
            f.write(md_content)
    
    def create_progress_visualization(self):
        """创建进度可视化图表"""
        # 任务完成状态饼图
        status_counts = {}
        for task in self.tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 任务状态分布
        axes[0, 0].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('任务状态分布')
        
        # 每日计划 vs 实际进度
        days = list(range(1, 15))
        planned_completion = []
        actual_completion = []
        
        for day in days:
            planned_tasks = [t for t in self.tasks.values() if t.assigned_day <= day]
            planned_completion.append(len(planned_tasks))
            
            completed_tasks = [t for t in self.tasks.values() 
                             if t.assigned_day <= day and t.status == "completed"]
            actual_completion.append(len(completed_tasks))
        
        axes[0, 1].plot(days, planned_completion, 'b-', label='计划完成')
        axes[0, 1].plot(days, actual_completion, 'r-', label='实际完成')
        axes[0, 1].set_xlabel('项目天数')
        axes[0, 1].set_ylabel('累计完成任务数')
        axes[0, 1].set_title('进度跟踪')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 性能指标趋势
        if self.metrics:
            metric_names = list(set(m.metric_name for m in self.metrics))
            for i, metric_name in enumerate(metric_names[:4]):  # 最多显示4个指标
                metric_data = [m for m in self.metrics if m.metric_name == metric_name]
                timestamps = [datetime.fromisoformat(m.timestamp) for m in metric_data]
                values = [m.value for m in metric_data]
                
                axes[1, 0].plot(timestamps, values, label=metric_name, marker='o')
            
            axes[1, 0].set_xlabel('时间')
            axes[1, 0].set_ylabel('指标值')
            axes[1, 0].set_title('性能指标趋势')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 工时统计
        task_hours = [(t.name, t.estimated_hours, t.actual_hours) for t in self.tasks.values()]
        if task_hours:
            task_names = [t[0][:15] for t in task_hours]  # 截断长名称
            estimated = [t[1] for t in task_hours]
            actual = [t[2] for t in task_hours]
            
            x = range(len(task_names))
            width = 0.35
            
            axes[1, 1].bar([i - width/2 for i in x], estimated, width, label='预估时间')
            axes[1, 1].bar([i + width/2 for i in x], actual, width, label='实际时间')
            axes[1, 1].set_xlabel('任务')
            axes[1, 1].set_ylabel('小时')
            axes[1, 1].set_title('工时对比')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(task_names, rotation=45)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.project_dir / 'progress_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"进度可视化已保存: {self.project_dir / 'progress_dashboard.png'}")

def main():
    """初始化项目跟踪"""
    tracker = CoTRRStageTracker()
    
    print("🚀 CoTRR-Stable Stage 1 项目跟踪系统已启动！")
    print(f"📅 项目开始日期: {tracker.start_date.strftime('%Y-%m-%d')}")
    print(f"📊 总任务数: {len(tracker.tasks)}")
    print(f"🎯 里程碑数: {len(tracker.milestones)}")
    
    # 生成今日报告
    report = tracker.generate_daily_report()
    print(f"\n📋 今日是项目第 {report['project_day']} 天")
    print(f"📈 当前完成进度: {report['progress_summary']['completion_rate']:.1f}%")
    
    # 显示今日任务
    if report['today_tasks']:
        print("\n📝 今日任务:")
        for task in report['today_tasks']:
            print(f"  - {task['name']} ({task['id']})")
    
    # 创建进度图表
    tracker.create_progress_visualization()
    
    print(f"\n💾 进度文件保存在: {tracker.project_dir}")
    print("📊 使用 tracker.generate_daily_report() 生成每日报告")
    print("📈 使用 tracker.update_task_status() 更新任务状态")
    print("📋 使用 tracker.record_metric() 记录性能指标")
    
    return tracker

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracker = main()