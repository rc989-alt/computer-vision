#!/usr/bin/env python3
"""
Day 3 诊断分析
分析轻量级增强器问题
"""

import json
import sys
sys.path.append('.')

from research.day3_lightweight_enhancer import LightweightPipelineEnhancer, OptimizationConfig
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def diagnose_enhancer():
    """诊断增强器问题"""
    print("🔍 Day 3 Lightweight Enhancer Diagnosis")
    print("="*60)
    
    # 加载测试数据
    with open("data/input/sample_input.json", 'r') as f:
        data = json.load(f)
    
    test_data = data.get('inspirations', [])
    
    if not test_data:
        print("❌ 没有找到测试数据")
        return
    
    item = test_data[0]
    query = item.get('query', '')
    candidates = item.get('candidates', [])
    
    print(f"📝 测试查询: '{query}'")
    print(f"🎯 候选项数量: {len(candidates)}")
    
    # 显示原始数据
    print("\\n📊 原始候选项:")
    for i, candidate in enumerate(candidates):
        print(f"   {i+1}. ID: {candidate.get('id', 'unknown')}")
        print(f"      分数: {candidate.get('score', 0)}")
        print(f"      标题: {candidate.get('title', 'N/A')}")
        print(f"      描述: {candidate.get('description', 'N/A')[:100]}...")
    
    # 创建增强器
    config = OptimizationConfig(
        compliance_weight=0.3,
        conflict_penalty_alpha=0.1,
        description_boost_weight=0.1
    )
    
    enhancer = LightweightPipelineEnhancer(config)
    
    print("\\n🔧 增强处理过程:")
    
    # 逐步处理分析
    enhanced_candidates = []
    
    for i, candidate in enumerate(candidates):
        print(f"\\n   处理候选项 {i+1}:")
        
        # 原始分数
        original_score = candidate.get('score', 0)
        print(f"      原始分数: {original_score}")
        
        # 复制候选项
        enhanced = candidate.copy()
        
        # 1. 基础增强
        base_enhance = enhancer._calculate_base_enhancement(query, enhanced)
        print(f"      基础增强: {base_enhance}")
        
        # 2. 描述提升
        description = enhanced.get('description', '')
        desc_boost = enhancer._analyze_description_quality(description, query)
        print(f"      描述提升: {desc_boost}")
        print(f"      描述内容: '{description[:50]}...'")
        
        # 3. Compliance分数
        compliance_score = enhancer.subject_object.check_constraint([enhanced], {'query': query}).get('compliance', 1.0)
        print(f"      Compliance: {compliance_score}")
        
        # 4. Conflict惩罚
        dual_results = enhancer.dual_score.score_candidates([enhanced], query)
        conflict_penalty = enhancer.conflict_penalty.calculate_penalty(dual_results)
        print(f"      冲突惩罚: {conflict_penalty}")
        
        # 5. 最终计算
        enhanced_score = original_score + base_enhance
        enhanced_score = enhanced_score * (1 + desc_boost * config.description_boost_weight)
        enhanced_score = enhanced_score * (config.compliance_weight * compliance_score + (1 - config.compliance_weight))
        enhanced_score = enhanced_score * (1 - conflict_penalty * config.conflict_penalty_alpha)
        
        print(f"      最终分数: {enhanced_score}")
        print(f"      改进量: {enhanced_score - original_score:+.4f}")
        
        enhanced['enhanced_score'] = enhanced_score
        enhanced_candidates.append(enhanced)
    
    print("\\n📈 汇总分析:")
    original_avg = sum(c.get('score', 0) for c in candidates) / len(candidates)
    enhanced_avg = sum(c.get('enhanced_score', 0) for c in enhanced_candidates) / len(enhanced_candidates)
    
    print(f"   原始平均分: {original_avg:.4f}")
    print(f"   增强平均分: {enhanced_avg:.4f}")
    print(f"   总体改进: {enhanced_avg - original_avg:+.4f}")
    
    print("\\n🧠 问题分析:")
    
    if enhanced_avg < original_avg:
        print("   ❌ 增强器正在降低分数")
        
        # 分析原因
        reasons = []
        
        # 检查描述质量分析
        for candidate in candidates:
            desc = candidate.get('description', '')
            if not desc or len(desc) < 20:
                reasons.append("描述内容过短或为空")
            
            # 检查查询匹配
            query_words = query.lower().split()
            desc_words = desc.lower().split()
            matches = sum(1 for word in query_words if word in desc_words)
            if matches == 0:
                reasons.append("描述与查询无关键词匹配")
        
        # 检查compliance
        if compliance_score < 1.0:
            reasons.append(f"Compliance分数过低: {compliance_score}")
        
        # 检查conflict penalty
        if conflict_penalty > 0:
            reasons.append(f"存在冲突惩罚: {conflict_penalty}")
        
        print("   可能原因:")
        for reason in reasons:
            print(f"     • {reason}")
    
    print("\\n💡 修复建议:")
    print("   1. 检查描述质量分析逻辑")
    print("   2. 调整权重参数，避免过度惩罚")
    print("   3. 改进查询-描述匹配算法")
    print("   4. 考虑添加正向激励机制")
    
    # 建议新的配置
    print("\\n🔧 建议配置调整:")
    print("   • compliance_weight: 0.8 → 1.0 (减少compliance影响)")
    print("   • description_boost_weight: 0.1 → 0.05 (减少描述惩罚)")
    print("   • conflict_penalty_alpha: 0.1 → 0.05 (减少冲突惩罚)")
    
    return enhanced_candidates

def test_fixed_approach():
    """测试修复后的方法"""
    print("\\n" + "="*60)
    print("🛠️ Testing Fixed Approach")
    print("="*60)
    
    # 加载数据
    with open("data/input/sample_input.json", 'r') as f:
        data = json.load(f)
    
    test_data = data.get('inspirations', [])
    item = test_data[0]
    query = item.get('query', '')
    candidates = item.get('candidates', [])
    
    # 修复的配置
    fixed_config = OptimizationConfig(
        compliance_weight=1.0,  # 不惩罚compliance
        conflict_penalty_alpha=0.01,  # 极小的冲突惩罚
        description_boost_weight=0.02  # 极小的描述权重
    )
    
    enhancer = LightweightPipelineEnhancer(fixed_config)
    enhanced_candidates = enhancer.enhance_candidates(query, candidates)
    
    print(f"查询: '{query}'")
    print("\\n结果对比:")
    
    original_scores = [c.get('score', 0) for c in candidates]
    enhanced_scores = [c.get('enhanced_score', 0) for c in enhanced_candidates]
    
    for i, (orig, enh) in enumerate(zip(original_scores, enhanced_scores)):
        improvement = enh - orig
        print(f"   候选项 {i+1}: {orig:.3f} → {enh:.3f} ({improvement:+.4f})")
    
    total_improvement = sum(enhanced_scores) - sum(original_scores)
    avg_improvement = total_improvement / len(candidates)
    
    print(f"\\n总改进: {avg_improvement:+.4f}")
    
    if avg_improvement > 0:
        print("✅ 修复成功！增强器现在能够提升分数")
        return True
    else:
        print("❌ 仍需进一步调整")
        return False

if __name__ == "__main__":
    # 诊断当前问题
    diagnose_enhancer()
    
    # 测试修复方案
    if test_fixed_approach():
        print("\\n🎯 建议下一步:")
        print("   1. 使用修复后的配置重新运行参数优化")
        print("   2. 扩展测试数据集进行更全面验证")
        print("   3. 考虑添加更多正向特征")
    else:
        print("\\n🔄 需要进一步分析:")
        print("   1. 重新设计增强算法")
        print("   2. 简化计算逻辑")
        print("   3. 考虑使用加法而非乘法组合")