#!/usr/bin/env python3
"""
Simple Day 3 Test: 检查基本组件是否可以工作
"""

import json
import sys
import os
from pathlib import Path

# 测试1: 检查数据文件
print("🔍 Test 1: 检查测试数据")
test_data_path = "data/input/sample_input.json"
if os.path.exists(test_data_path):
    with open(test_data_path) as f:
        data = json.load(f)
    print(f"✅ 测试数据加载成功: {len(data.get('inspirations', []))} queries")
    for i, item in enumerate(data.get('inspirations', [])[:2]):
        print(f"   Query {i+1}: '{item.get('query', 'N/A')}' - {len(item.get('candidates', []))} candidates")
else:
    print(f"❌ 测试数据不存在: {test_data_path}")

# 测试2: 检查配置文件
print("\n🔍 Test 2: 检查配置文件")
config_path = "config/default.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    print(f"✅ 配置文件加载成功")
    print(f"   Detection threshold: {config.get('detection_threshold', 'N/A')}")
    print(f"   Family: {config.get('family', 'N/A')}")
else:
    print(f"❌ 配置文件不存在: {config_path}")

# 测试3: 检查CoTRR组件
print("\n🔍 Test 3: 检查CoTRR-Stable组件")
sys.path.append('research/src')
try:
    from step5_integration import CoTRRStableStep5Integration, IntegrationConfig
    print("✅ CoTRRStableStep5Integration 导入成功")
    
    from isotonic_calibration import IsotonicCalibrator
    print("✅ IsotonicCalibrator 导入成功")
    
    from cotrr_stable import StableCrossAttnReranker, StableConfig
    print("✅ StableCrossAttnReranker 导入成功")
    
    # 测试初始化
    config = IntegrationConfig(
        model_path="nonexistent.pt",
        calibrator_path="nonexistent.pkl",
        rollout_percentage=100.0
    )
    integration = CoTRRStableStep5Integration(config)
    print("✅ CoTRR-Stable集成器初始化成功")
    
    # 测试健康检查
    health = integration.health_check()
    print(f"✅ 健康检查通过: {health['status']}")
    
except ImportError as e:
    print(f"❌ CoTRR组件导入失败: {e}")
except Exception as e:
    print(f"❌ CoTRR初始化失败: {e}")

# 测试4: 检查核心模块
print("\n🔍 Test 4: 检查核心模块")
try:
    from src.subject_object import check_subject_object
    from src.conflict_penalty import conflict_penalty
    from src.dual_score import fuse_dual_score
    print("✅ 核心模块导入成功")
    
    # 测试调用
    mock_regions = [{'label': 'glass', 'type': 'crystal_glass', 'confidence': 0.9}]
    compliance, details = check_subject_object(regions=mock_regions)
    print(f"✅ Subject-object检查成功: compliance={compliance:.3f}")
    
    penalty, penalty_details = conflict_penalty(mock_regions, alpha=0.3)
    print(f"✅ Conflict penalty计算成功: penalty={penalty:.3f}")
    
    fused = fuse_dual_score(compliance, penalty, w_c=0.6, w_n=0.4)
    print(f"✅ Dual score融合成功: fused={fused:.3f}")
    
except ImportError as e:
    print(f"❌ 核心模块导入失败: {e}")
except Exception as e:
    print(f"❌ 核心模块测试失败: {e}")

# 测试5: 简单CoTRR测试
print("\n🔍 Test 5: 简单CoTRR功能测试")
try:
    if 'integration' in locals():
        # 创建测试数据
        mock_candidates = [{
            "candidate_id": "test_001",
            "text_features": [0.1] * 256,
            "image_features": [0.2] * 256,
            "original_score": 0.75
        }]
        
        query_data = {"query_id": "test_query", "query_text": "test cocktail"}
        
        result = integration.rerank_candidates(
            query_data, mock_candidates, {"return_scores": True}
        )
        
        print(f"✅ CoTRR重排序成功:")
        print(f"   Status: {result['metadata']['status']}")
        print(f"   Inference time: {result['metadata']['inference_time']:.4f}s")
        print(f"   Candidates processed: {len(result['candidates'])}")
        
        # 获取性能统计
        stats = integration.get_performance_stats()
        print(f"✅ 性能统计获取成功:")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"   Reranked queries: {stats['reranked_queries']}")
        print(f"   Error rate: {stats['error_rate']:.1%}")
        
except Exception as e:
    print(f"❌ CoTRR功能测试失败: {e}")

print("\n" + "="*50)
print("🎯 Day 3 Basic Component Test Summary")
print("="*50)
print("如果所有测试通过，则可以继续集成测试")
print("如果有失败项，需要先修复对应问题")