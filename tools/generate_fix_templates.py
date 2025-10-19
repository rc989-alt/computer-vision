#!/usr/bin/env python3
"""
Immediate Fix Templates Generator - 立即修复模板生成器
================================================================================
根据审计发现，生成可直接应用的文档修正模板
================================================================================

Usage:
    python generate_fix_templates.py --type v1-correction
    python generate_fix_templates.py --type cotrr-downgrade
    python generate_fix_templates.py --type evidence-annotation
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

class FixTemplateGenerator:
    """修复模板生成器"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def generate_v1_correction_template(self) -> str:
        """生成V1.0数据修正模板"""
        return """# V1.0 数据修正模板

## 修正前后对比

### 修正前 ❌
```markdown
- **生产部署**: ✅ **完成** - +14.2% compliance 提升
- Compliance Score | +10% | **+14.2%** | ✅ 超预期
```

### 修正后 ✅
```markdown
- **生产部署**: ✅ **完成** - +13.8% compliance 提升

**Data-Claim**: Compliance +13.8%
**Evidence**: research/day3_results/production_evaluation.json (hash=prod_eval_2025, run_id=v1_prod_eval, ts=2025-10-12)
**Scope**: Full evaluation dataset / sample=n=45 / window=2025-10-10 to 2025-10-12
**Reviewer**: production@team (automated from production_evaluation.json)
**Trust Tier**: T3-Verified

| Compliance Score | +10% | **+13.8%** | ✅ 超预期 | ↗️ 稳定上升 |
```

## 查找替换指令

### 全局替换命令
```bash
# 替换所有14.2%为13.8%
find . -name "*.md" -exec sed -i 's/+14\.2%/+13.8%/g' {} \\;

# 添加证据标注到V1.0相关文件
python tools/add_evidence_annotation.py --pattern "13.8%" --evidence "production_evaluation.json"
```

### 手动替换清单
1. `research/01_v1_production_line/SUMMARY.md` - 已修正 ✅
2. `docs/05_analysis_reports/12_innovation_research_comprehensive_analysis.md` - 已修正 ✅
3. 其他包含14.2%的文件 - 待检查

## 验证命令
```bash
# 检查是否还有14.2%残留
grep -r "14\.2%" docs/ research/

# 验证新的13.8%都有证据标注
python tools/ci_data_integrity_check.py --target-dir . | grep "13.8%"
```
"""
    
    def generate_cotrr_downgrade_template(self) -> str:
        """生成CoTRR降级模板"""
        return """# CoTRR 数据降级模板

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
"""
    
    def generate_evidence_annotation_template(self) -> str:
        """生成证据标注模板"""
        return """# 证据标注模板

## Trust Tier 标注模板

### T3-Verified 模板
```markdown
**Data-Claim**: [具体指标和数值]
**Evidence**: [文件路径] (hash=[文件哈希前8位], run_id=[运行ID], ts=[时间戳])
**Scope**: [评估范围] / sample=n=[样本数] / window=[时间窗口]
**Reviewer**: [复核人]@[团队] (two-person check)
**Trust Tier**: T3-Verified
```

### T2-Internal 模板
```markdown
**Data-Claim**: [具体指标和数值]
**Evidence**: [脚本或实验记录] (run_id=[运行ID], ts=[时间戳])
**Scope**: [评估范围] / sample=n=[样本数]
**Reviewer**: [复核人]@[团队] (single-person check)
**Trust Tier**: T2-Internal
**Note**: 内部使用，对外分享需标注来源
```

### T1-Indicative 模板
```markdown
**Data-Claim**: [数值] (估计值/概念阶段)
**Evidence**: [实验记录或概念验证] (preliminary)
**Scope**: [有限范围] / [概念验证]
**Reviewer**: [研究人员]@[团队] (concept-only)
**Trust Tier**: T1-Indicative
**Note**: ⚠️ 探索性结果，禁止对外宣传
```

## 批量添加脚本

### 为现有文件添加标注
```python
#!/usr/bin/env python3
import re
from pathlib import Path

def add_evidence_to_file(file_path, pattern, evidence_block):
    content = Path(file_path).read_text()
    
    # 在数值后添加证据块
    def replace_func(match):
        return match.group(0) + "\\n\\n" + evidence_block + "\\n"
    
    new_content = re.sub(pattern, replace_func, content)
    Path(file_path).write_text(new_content)
    print(f"Added evidence to {file_path}")

# 使用示例
add_evidence_to_file(
    "research/01_v1_production_line/SUMMARY.md",
    r"\\+13\\.8%",
    '''**Data-Claim**: Compliance +13.8%
**Evidence**: research/day3_results/production_evaluation.json
**Trust Tier**: T3-Verified'''
)
```

### 验证标注完整性
```bash
# 检查哪些数值缺乏证据标注
python tools/ci_data_integrity_check.py --target-dir research/ | grep "missing_evidence"

# 检查Trust Tier分布
grep -r "Trust Tier" research/ | cut -d: -f2 | sort | uniq -c
```

## 文件头部标注模板

### 高可信度文件
```markdown
---
data_integrity: verified
trust_tier: T3-Verified
last_audit: 2025-10-12
evidence_files: 
  - production_evaluation.json
  - benchmark_report_v1_prod.json
---
```

### 概念阶段文件
```markdown
---
data_integrity: indicative
trust_tier: T1-Indicative
last_audit: 2025-10-12
status: concept-only
warning: "Not for external use"
---
```

### 需要验证的文件
```markdown
---
data_integrity: needs_verification
trust_tier: unknown
last_audit: 2025-10-12
action_required: "Run benchmark and add evidence"
---
```
"""
    
    def generate_ci_setup_template(self) -> str:
        """生成CI设置模板"""
        return """# CI数据完整性检查设置

## GitHub Actions 配置

### .github/workflows/data-integrity.yml
```yaml
name: Data Integrity Check

on:
  pull_request:
    paths:
      - 'docs/**/*.md'
      - 'research/**/*.md'

jobs:
  data-integrity:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run data integrity check
      run: |
        python tools/ci_data_integrity_check.py --pr-mode --target-dir docs/
        python tools/ci_data_integrity_check.py --pr-mode --target-dir research/
    
    - name: Upload check report
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: integrity-report
        path: integrity_report.md
```

## Pre-commit Hook 配置

### .pre-commit-config.yaml
```yaml
repos:
- repo: local
  hooks:
  - id: data-integrity
    name: Data Integrity Check
    entry: python tools/ci_data_integrity_check.py --target-dir docs/ research/
    language: system
    files: \\.(md)$
    fail_fast: true
```

## 本地开发设置

### 安装pre-commit
```bash
pip install pre-commit
pre-commit install
```

### 手动检查命令
```bash
# 检查特定文件
python tools/ci_data_integrity_check.py --file docs/some_file.md

# 检查整个目录
python tools/ci_data_integrity_check.py --target-dir research/

# 生成报告
python tools/ci_data_integrity_check.py --output integrity_report.md
```

## 规则配置文件

### tools/integrity_rules.json
```json
{
  "precision_patterns": [
    "\\\\b\\\\d+\\\\.\\\\d+x\\\\b",
    "\\\\b\\\\d+\\\\.\\\\d+GB\\\\b", 
    "\\\\b\\\\d+\\\\.\\\\d+MB\\\\b",
    "\\\\b\\\\+\\\\d+\\\\.\\\\d+%\\\\b"
  ],
  "exempt_files": [
    "**/README.md",
    "**/audit*.md",
    "**/template*.md"
  ],
  "required_trust_tiers": ["T1-Indicative", "T2-Internal", "T3-Verified"],
  "evidence_sources": ["reports/", "research/day3_results/"]
}
```
"""
    
    def generate_all_templates(self) -> Dict[str, str]:
        """生成所有模板"""
        return {
            "v1_correction": self.generate_v1_correction_template(),
            "cotrr_downgrade": self.generate_cotrr_downgrade_template(),
            "evidence_annotation": self.generate_evidence_annotation_template(),
            "ci_setup": self.generate_ci_setup_template()
        }

def main():
    parser = argparse.ArgumentParser(description="Fix Templates Generator")
    parser.add_argument("--type", 
                       choices=["v1-correction", "cotrr-downgrade", "evidence-annotation", "ci-setup", "all"],
                       default="all",
                       help="Type of template to generate")
    parser.add_argument("--output-dir", default="templates", help="Output directory for templates")
    
    args = parser.parse_args()
    
    generator = FixTemplateGenerator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.type == "all":
        templates = generator.generate_all_templates()
        for template_type, content in templates.items():
            output_file = output_dir / f"{template_type}_template.md"
            output_file.write_text(content)
            print(f"Generated: {output_file}")
    else:
        template_method = f"generate_{args.type.replace('-', '_')}_template"
        if hasattr(generator, template_method):
            content = getattr(generator, template_method)()
            output_file = output_dir / f"{args.type}_template.md"
            output_file.write_text(content)
            print(f"Generated: {output_file}")
        else:
            print(f"Unknown template type: {args.type}")
            return 1
    
    print(f"\\n✅ Templates generated in {output_dir}/")
    print("📋 Available templates:")
    for file in output_dir.glob("*.md"):
        print(f"  - {file.name}")
    
    return 0

if __name__ == "__main__":
    exit(main())