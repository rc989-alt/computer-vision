#!/usr/bin/env python3
"""
Data Integrity CI Checker - 数据声明合规性检查工具
================================================================================
用于PR合并前的数据声明验证，确保所有精确数字都有证据支撑
================================================================================

Usage:
    python ci_data_integrity_check.py --target-dir docs/
    python ci_data_integrity_check.py --file research/*/SUMMARY.md
    python ci_data_integrity_check.py --pr-mode --reports-dir reports/
"""

import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIntegrityChecker:
    """数据完整性CI检查器"""
    
    # Trust Tier 验证
    VALID_TRUST_TIERS = [
        "T1-Indicative", "T2-Internal", "T3-Verified"
    ]
    
    # 需要证据支撑的精确数字模式
    PRECISION_PATTERNS = [
        r'\b\d+\.\d+x\b',           # 倍数 (如 300.7x)
        r'\b\d+\.\d+X\b',           # 倍数大写 (如 300.7X)
        r'\b\d+\.\d+×\b',           # 倍数乘号 (如 300.7×)
        r'\b\d+\.\d+GB\b',          # 显存/内存 (如 2.3GB)
        r'\b\d+\.\d+MB\b',          # 内存 (如 180MB)
        r'\b\d+\.\d+ms\b',          # 延迟 (如 312.5ms)
        r'\b\+\d+\.\d+%\b',         # 百分比改进 (如 +14.2%)
        r'\b\d+\.\d+%\s*提升\b',     # 中文提升表述
        r'\b\d+\.\d+%\s*改进\b',     # 中文改进表述
    ]
    
    # Trust Tier 要求
    TRUST_TIER_PATTERN = r'Trust Tier[:\s]*([T]\d-\w+)'
    EVIDENCE_PATTERN = r'Evidence[:\s]*([^\n]+)'
    
    # 豁免文件（不需要检查的文件）
    EXEMPT_FILES = [
        'README.md',
        '**/audit*.md',
        '**/example*.md',
        '**/template*.md'
    ]
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.violations = []
        self.warnings = []
        
    def check_file(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """检查单个文件的数据完整性"""
        violations = []
        warnings = []
        
        if not file_path.exists():
            return violations, warnings
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, IOError) as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return violations, warnings
            
        lines = content.split('\n')
        
        # 检查精确数字
        for line_num, line in enumerate(lines, 1):
            for pattern in self.PRECISION_PATTERNS:
                matches = re.finditer(pattern, line)
                for match in matches:
                    # 计算全局位置
                    line_start_pos = sum(len(lines[i]) + 1 for i in range(line_num - 1))
                    match_pos = line_start_pos + match.start()
                    
                    # 检查是否在代码块或示例中
                    if self.is_in_code_block_or_example(content, match_pos):
                        continue  # 跳过代码块和示例中的数字
                    
                    # 检查是否有对应的证据
                    evidence = self.find_evidence_for_line(content, line_num, match.group())
                    trust_tier = self.find_trust_tier(content, line_num)
                    
                    if not evidence:
                        violations.append({
                            'file': str(file_path),
                            'line': line_num,
                            'content': line.strip(),
                            'match': match.group(),
                            'issue': 'missing_evidence',
                            'severity': 'high'
                        })
                    elif trust_tier not in ['T3-Verified']:
                        warnings.append({
                            'file': str(file_path),
                            'line': line_num,
                            'content': line.strip(),
                            'match': match.group(),
                            'trust_tier': trust_tier,
                            'issue': 'low_trust_tier',
                            'severity': 'medium'
                        })
        
        # 检查引用的文件是否存在
        self.check_file_references(file_path, content, violations)
        
        return violations, warnings
    
    def is_in_code_block_or_example(self, content: str, match_start: int) -> bool:
        """检查匹配是否在代码块或示例中"""
        # 检查是否在markdown代码块中
        before_match = content[:match_start]
        after_match = content[match_start:]
        
        # 计算```的数量，如果是奇数说明在代码块中
        code_block_starts = before_match.count('```')
        if code_block_starts % 2 == 1:
            return True
        
        # 检查是否在行内代码中
        line_start = before_match.rfind('\n')
        line_end = after_match.find('\n')
        if line_end == -1:
            line_end = len(after_match)
        
        current_line = content[line_start:match_start + line_end]
        
        # 检查行内代码
        backtick_before = before_match[line_start:].count('`') if line_start >= 0 else 0
        if backtick_before % 2 == 1:
            return True
        
        # 检查是否是示例或说明
        example_keywords = [
            '示例', 'example', '如', 'such as', '例如', '反面示例', 
            '错误示例', 'bad example', '仅作说明', 'for illustration',
            '❌ 错误示例', '✅ 正确示例'
        ]
        
        # 检查前后100字符
        context_start = max(0, match_start - 100)
        context_end = min(len(content), match_start + 100)
        context = content[context_start:context_end].lower()
        
        return any(keyword in context for keyword in example_keywords)
    
    def check_file_references(self, file_path: Path, content: str, violations: List[Dict]) -> None:
        """检查引用的文件是否存在"""
        file_refs = re.findall(r'([a-zA-Z0-9_/]+\.json)', content)
        for ref in file_refs:
            ref_path = Path(ref)
            # 检查多个可能的路径
            possible_paths = [
                ref_path,  # 绝对路径
                Path('research') / ref,  # research目录
                file_path.parent / ref,  # 同级目录
                Path('reports') / ref,  # reports目录
            ]
            
            if not any(p.exists() for p in possible_paths):
                violations.append({
                    'file': str(file_path),
                    'content': ref,
                    'issue': 'missing_referenced_file',
                    'severity': 'high'
                })
    
    def find_evidence_for_line(self, content: str, line_num: int, match: str) -> Optional[str]:
        """查找指定行附近的证据标注"""
        lines = content.split('\n')
        search_range = 5  # 在前后5行范围内查找
        
        start = max(0, line_num - search_range - 1)
        end = min(len(lines), line_num + search_range)
        
        for i in range(start, end):
            evidence_match = re.search(self.EVIDENCE_PATTERN, lines[i])
            if evidence_match:
                return evidence_match.group(1)
                
        return None
    
    def find_trust_tier(self, content: str, line_num: int) -> str:
        """查找Trust Tier标注"""
        lines = content.split('\n')
        search_range = 10
        
        start = max(0, line_num - search_range - 1)
        end = min(len(lines), line_num + search_range)
        
        for i in range(start, end):
            tier_match = re.search(self.TRUST_TIER_PATTERN, lines[i])
            if tier_match:
                return tier_match.group(1)
                
        return "unknown"
    
    def check_reports_consistency(self) -> List[Dict]:
        """检查reports目录中的数据一致性"""
        inconsistencies = []
        
        if not self.reports_dir.exists():
            return inconsistencies
        
        # 查找所有基准测试报告
        report_files = list(self.reports_dir.glob("benchmark_report_*.json"))
        
        for report_file in report_files:
            try:
                with open(report_file) as f:
                    report = json.load(f)
                
                # 验证报告完整性
                required_fields = ['run_id', 'metrics', 'timestamp', 'sha256']
                for field in required_fields:
                    if field not in report:
                        inconsistencies.append({
                            'file': str(report_file),
                            'issue': f'missing_field_{field}',
                            'severity': 'high'
                        })
                
                # 验证SHA256
                report_copy = dict(report)
                if 'sha256' in report_copy:
                    original_hash = report_copy.pop('sha256')
                    calculated_hash = self.calculate_report_hash(report_copy)
                    if original_hash != calculated_hash:
                        inconsistencies.append({
                            'file': str(report_file),
                            'issue': 'hash_mismatch',
                            'severity': 'high'
                        })
                        
            except (json.JSONDecodeError, IOError) as e:
                inconsistencies.append({
                    'file': str(report_file),
                    'issue': f'read_error: {e}',
                    'severity': 'high'
                })
        
        return inconsistencies
    
    def find_trust_tier(self, content: str, target_line: int) -> Optional[str]:
        """在指定行附近查找Trust Tier标注"""
        lines = content.split('\n')
        
        # 检查前后5行
        start_line = max(0, target_line - 5)
        end_line = min(len(lines), target_line + 5)
        
        trust_tier_patterns = [
            r'\*\*Trust Tier\*\*:\s*(.+)',
            r'trust_tier:\s*["\']?(.+?)["\']?',
            r'Trust Tier:\s*(.+)'
        ]
        
        for line_idx in range(start_line, end_line):
            line = lines[line_idx]
            for pattern in trust_tier_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    tier = match.group(1).strip().strip('"\'')
                    if tier in self.VALID_TRUST_TIERS:
                        return tier
        
        return None
    
    def check_trust_tier_validity(self, file_path: Path) -> List[Dict]:
        """检查Trust Tier标注的有效性"""
        violations = []
        
        if not file_path.exists():
            return violations
            
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            trust_tier_pattern = r'\*\*Trust Tier\*\*:\s*(.+)|trust_tier:\s*["\']?(.+?)["\']?'
            
            for line_num, line in enumerate(lines, 1):
                matches = re.finditer(trust_tier_pattern, line, re.IGNORECASE)
                for match in matches:
                    tier = match.group(1) or match.group(2)
                    tier = tier.strip().strip('"\'')
                    
                    if tier not in self.VALID_TRUST_TIERS:
                        violations.append({
                            'file': str(file_path),
                            'line': line_num,
                            'content': line.strip(),
                            'trust_tier': tier,
                            'issue': 'invalid_trust_tier',
                            'severity': 'high',
                            'valid_options': self.VALID_TRUST_TIERS
                        })
        
        except (UnicodeDecodeError, IOError) as e:
            logger.warning(f"Could not read {file_path}: {e}")
        
        return violations
    
    def check_data_claim_format(self, file_path: Path) -> List[Dict]:
        """检查数据声明格式是否符合标准"""
        violations = []
        
        if not file_path.exists():
            return violations
            
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # 查找Data-Claim模式
            claim_pattern = r'\*\*Data-Claim\*\*:\s*(.+)'
            claims = list(re.finditer(claim_pattern, content))
            
            for claim_match in claims:
                line_num = content[:claim_match.start()].count('\n') + 1
                claim_text = claim_match.group(1)
                
                # 提取声明块（到下一个段落）
                claim_block = self._extract_claim_block(content, claim_match.start())
                
                # 检查必需字段
                required_fields = ['Evidence', 'Trust Tier']
                missing_fields = []
                
                for field in required_fields:
                    if not re.search(rf'\*\*{field}\*\*:', claim_block, re.IGNORECASE):
                        missing_fields.append(field)
                
                if missing_fields:
                    violations.append({
                        'file': str(file_path),
                        'line': line_num,
                        'content': claim_text.strip(),
                        'issue': 'missing_required_fields',
                        'missing_fields': missing_fields,
                        'severity': 'high'
                    })
        
        except (UnicodeDecodeError, IOError) as e:
            logger.warning(f"Could not read {file_path}: {e}")
        
        return violations
    
    def _extract_claim_block(self, content: str, start_pos: int) -> str:
        """提取数据声明块（到下一个段落或声明）"""
        remaining = content[start_pos:]
        next_claim = re.search(r'\n\*\*Data-Claim\*\*:', remaining)
        double_newline = re.search(r'\n\n', remaining)
        
        if next_claim and double_newline:
            end_pos = min(next_claim.start(), double_newline.start())
        elif next_claim:
            end_pos = next_claim.start()
        elif double_newline:
            end_pos = double_newline.start()
        else:
            end_pos = len(remaining)
        
        return remaining[:end_pos]
    
    def check_directory(self, target_dir: Path) -> Tuple[List[Dict], List[Dict]]:
        """检查整个目录"""
        all_violations = []
        all_warnings = []
        
        # 查找所有markdown文件
        md_files = list(target_dir.rglob("*.md"))
        
        for md_file in md_files:
            # 检查是否在豁免列表中
            if any(md_file.match(pattern) for pattern in self.EXEMPT_FILES):
                continue
                
            violations, warnings = self.check_file(md_file)
            all_violations.extend(violations)
            all_warnings.extend(warnings)
        
        # 检查reports一致性
        report_issues = self.check_reports_consistency()
        all_violations.extend(report_issues)
        
        return all_violations, all_warnings
    
    def generate_report(self, violations: List[Dict], warnings: List[Dict]) -> str:
        """生成检查报告"""
        report = []
        report.append("# Data Integrity Check Report")
        report.append(f"Generated: {subprocess.check_output(['date'], text=True).strip()}")
        report.append("")
        
        if violations:
            report.append("## ❌ Violations (Must Fix)")
            report.append("")
            for v in violations:
                report.append(f"**File**: `{v['file']}`")
                if 'line' in v:
                    report.append(f"**Line**: {v['line']}")
                if 'content' in v:
                    report.append(f"**Content**: `{v['content']}`")
                if 'match' in v:
                    report.append(f"**Match**: `{v['match']}`")
                report.append(f"**Issue**: {v['issue']}")
                report.append(f"**Severity**: {v['severity']}")
                report.append("")
        else:
            report.append("## ✅ No Violations Found")
            report.append("")
        
        if warnings:
            report.append("## ⚠️ Warnings (Should Fix)")
            report.append("")
            for w in warnings:
                report.append(f"**File**: `{w['file']}`")
                if 'line' in w:
                    report.append(f"**Line**: {w['line']}")
                if 'content' in w:
                    report.append(f"**Content**: `{w['content']}`")
                if 'trust_tier' in w:
                    report.append(f"**Trust Tier**: {w['trust_tier']}")
                report.append(f"**Issue**: {w['issue']}")
                report.append("")
        
        # 总结
        report.append("## Summary")
        report.append(f"- Violations: {len(violations)}")
        report.append(f"- Warnings: {len(warnings)}")
        
        if violations:
            report.append("\n**❌ CI CHECK FAILED** - Fix violations before merging")
        else:
            report.append("\n**✅ CI CHECK PASSED** - Ready for merge")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Data Integrity CI Checker")
    parser.add_argument("--target-dir", help="Target directory to check")
    parser.add_argument("--file", help="Specific file to check")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory")
    parser.add_argument("--pr-mode", action="store_true", help="PR mode - exit with error if violations found")
    parser.add_argument("--output", help="Output report file")
    
    args = parser.parse_args()
    
    checker = DataIntegrityChecker(args.reports_dir or "reports")
    
    if args.file:
        violations, warnings = checker.check_file(Path(args.file))
    elif args.target_dir:
        violations, warnings = checker.check_directory(Path(args.target_dir))
    else:
        # 默认检查整个项目
        violations, warnings = checker.check_directory(Path("."))
    
    # 生成报告
    report_content = checker.generate_report(violations, warnings)
    
    if args.output:
        Path(args.output).write_text(report_content)
        logger.info(f"Report saved to: {args.output}")
    else:
        print(report_content)
    
    # PR模式下，如果有违规则失败
    if args.pr_mode and violations:
        logger.error(f"CI check failed: {len(violations)} violations found")
        return 1
    
    logger.info(f"CI check completed: {len(violations)} violations, {len(warnings)} warnings")
    return 0

if __name__ == "__main__":
    exit(main())