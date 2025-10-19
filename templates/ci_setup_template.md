# CI数据完整性检查设置

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
    files: \.(md)$
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
    "\\b\\d+\\.\\d+x\\b",
    "\\b\\d+\\.\\d+GB\\b", 
    "\\b\\d+\\.\\d+MB\\b",
    "\\b\\+\\d+\\.\\d+%\\b"
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
