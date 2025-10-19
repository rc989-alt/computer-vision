# 证据标注模板

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
        return match.group(0) + "\n\n" + evidence_block + "\n"
    
    new_content = re.sub(pattern, replace_func, content)
    Path(file_path).write_text(new_content)
    print(f"Added evidence to {file_path}")

# 使用示例
add_evidence_to_file(
    "research/01_v1_production_line/SUMMARY.md",
    r"\+13\.8%",
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
