#!/usr/bin/env python3
"""
Day 3 简化诊断
找出轻量级增强器问题的根本原因
"""

import json
import sys
sys.path.append('.')

from research.day3_lightweight_enhancer import LightweightPipelineEnhancer, OptimizationConfig

def simple_debug():
    """简化调试"""
    print("🔍 Day 3 简化诊断")
    print("="*50)
    
    # 加载测试数据
    with open("data/input/sample_input.json", 'r') as f:
        data = json.load(f)
    
    test_data = data.get('inspirations', [])
    item = test_data[0]
    query = item.get('query', '')
    candidates = item.get('candidates', [])
    
    print(f"查询: '{query}'")
    print(f"候选项: {len(candidates)}")
    
    # 显示原始数据结构
    print("\\n原始数据结构:")
    for i, candidate in enumerate(candidates):
        print(f"   候选项 {i+1}:")
        for key, value in candidate.items():
            if isinstance(value, str) and len(value) > 50:
                print(f"      {key}: '{value[:50]}...'")
            else:
                print(f"      {key}: {value}")
    
    # 创建增强器并处理
    config = OptimizationConfig(
        compliance_weight=0.5,
        conflict_penalty_alpha=0.1,
        description_boost_weight=0.1
    )
    
    enhancer = LightweightPipelineEnhancer(config)
    
    print("\\n增强器处理:")
    
    # 逐个处理
    for i, candidate in enumerate(candidates):
        print(f"\\n   处理候选项 {i+1}:")
        print(f"      原始分数: {candidate.get('score', 'N/A')}")
        
        # 检查关键字段
        description = candidate.get('alt_description', '')
        print(f"      描述长度: {len(description)}")
        print(f"      描述内容: '{description[:100]}...'")
        
        # 执行增强
        enhanced = enhancer._enhance_single_candidate(candidate, query)
        
        print(f"      增强分数: {enhanced.get('enhanced_score', 'N/A')}")
        print(f"      分数变化: {enhanced.get('enhanced_score', 0) - candidate.get('score', 0):+.4f}")
        
        # 显示增强细节
        details = enhanced.get('enhancement_details', {})
        print(f"      区域检测: {details.get('regions_detected', 0)}")
        print(f"      Compliance: {enhanced.get('compliance_score', 'N/A')}")
        print(f"      Conflict penalty: {enhanced.get('conflict_penalty', 'N/A')}")
        print(f"      描述提升: {enhanced.get('description_boost', 'N/A')}")
    
    print("\\n分析:")
    
    # 检查数据质量
    has_descriptions = all(c.get('alt_description') for c in candidates)
    avg_desc_length = sum(len(c.get('alt_description', '')) for c in candidates) / len(candidates)
    
    print(f"   所有候选项都有描述: {has_descriptions}")
    print(f"   平均描述长度: {avg_desc_length:.1f} 字符")
    
    if not has_descriptions:
        print("   ❌ 缺少alt_description字段！")
    
    if avg_desc_length < 50:
        print("   ⚠️  描述内容过短，可能影响分析质量")
    
    # 测试完整流程
    print("\\n完整流程测试:")
    enhanced_candidates = enhancer.enhance_candidates(query, candidates)
    
    original_avg = sum(c.get('score', 0) for c in candidates) / len(candidates)
    enhanced_avg = sum(c.get('enhanced_score', 0) for c in enhanced_candidates) / len(enhanced_candidates)
    
    print(f"   原始平均分: {original_avg:.4f}")
    print(f"   增强平均分: {enhanced_avg:.4f}")
    print(f"   改进量: {enhanced_avg - original_avg:+.4f}")
    
    if enhanced_avg > original_avg:
        print("   ✅ 增强器工作正常")
    else:
        print("   ❌ 增强器降低了分数")
        
        # 提供修复建议
        print("\\n修复建议:")
        print("   1. 检查alt_description字段是否存在且有内容")
        print("   2. 调整权重参数，减少惩罚性因子")
        print("   3. 改进描述分析逻辑")
        print("   4. 使用加法而非复杂的权重组合")

def test_simple_fix():
    """测试简单修复方案"""
    print("\\n" + "="*50)
    print("🛠️ 测试简单修复方案")
    print("="*50)
    
    # 简单的正向增强逻辑
    class SimpleEnhancer:
        def enhance(self, query, candidates):
            """超简单的增强逻辑"""
            enhanced = []
            
            for candidate in candidates:
                new_candidate = candidate.copy()
                original_score = candidate.get('score', 0)
                
                # 简单的正向增强
                boost = 0.01  # 固定的小幅提升
                
                # 如果有描述且包含查询词汇，额外提升
                description = candidate.get('alt_description', '').lower()
                query_words = query.lower().split()
                
                matches = sum(1 for word in query_words if word in description)
                if matches > 0:
                    boost += 0.02 * matches  # 每个匹配词汇额外提升
                
                new_score = original_score + boost
                new_candidate['enhanced_score'] = new_score
                new_candidate['score'] = new_score
                
                enhanced.append(new_candidate)
            
            return enhanced
    
    # 加载数据
    with open("data/input/sample_input.json", 'r') as f:
        data = json.load(f)
    
    test_data = data.get('inspirations', [])
    item = test_data[0]
    query = item.get('query', '')
    candidates = item.get('candidates', [])
    
    # 测试简单增强器
    simple_enhancer = SimpleEnhancer()
    enhanced = simple_enhancer.enhance(query, candidates)
    
    print(f"查询: '{query}'")
    print("\\n结果:")
    
    for i, (orig, enh) in enumerate(zip(candidates, enhanced)):
        orig_score = orig.get('score', 0)
        enh_score = enh.get('enhanced_score', 0)
        improvement = enh_score - orig_score
        
        print(f"   候选项 {i+1}: {orig_score:.3f} → {enh_score:.3f} ({improvement:+.4f})")
    
    # 计算总体改进
    original_avg = sum(c.get('score', 0) for c in candidates) / len(candidates)
    enhanced_avg = sum(c.get('enhanced_score', 0) for c in enhanced) / len(enhanced)
    total_improvement = enhanced_avg - original_avg
    
    print(f"\\n总体改进: {total_improvement:+.4f}")
    
    if total_improvement > 0:
        print("✅ 简单方案成功！")
        print("\\n关键洞察:")
        print("   • 保持简单的正向增强逻辑")
        print("   • 避免复杂的权重组合")
        print("   • 专注于明确的匹配信号")
        return True
    else:
        print("❌ 连简单方案也有问题")
        return False

if __name__ == "__main__":
    # 诊断当前问题
    simple_debug()
    
    # 测试简单修复
    if test_simple_fix():
        print("\\n🎯 下一步行动:")
        print("   1. 基于简单方案重新设计轻量级增强器")
        print("   2. 实现明确的正向增强逻辑")
        print("   3. 避免复杂的惩罚机制")
    else:
        print("\\n🚨 需要深入调查数据质量问题")