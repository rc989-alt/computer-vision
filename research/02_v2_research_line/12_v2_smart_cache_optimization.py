#!/usr/bin/env python3
"""
V2.0救援计划 - 智能缓存优化组件
================================================================================
目标: 解决V1/V2系统中的重复计算问题，通过智能缓存机制提升性能
重点: 输入变化检测、结果重用策略、多样化场景适配
应用: V1生产优化 + V2轻量化部署的关键支撑
================================================================================
"""

import hashlib
import json
import pickle
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import threading
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    result: Any
    timestamp: datetime
    hit_count: int = 0
    compute_time: float = 0.0
    input_signature: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CacheStats:
    """缓存统计"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_compute_time_saved: float = 0.0
    avg_hit_rate: float = 0.0
    
    def update_hit_rate(self):
        if self.total_requests > 0:
            self.avg_hit_rate = self.cache_hits / self.total_requests

class InputSignatureGenerator:
    """输入签名生成器 - 判断何时可以重用缓存"""
    
    @staticmethod
    def generate_signature(input_data: Any, granularity: str = "precise") -> str:
        """生成输入签名
        
        Args:
            input_data: 输入数据
            granularity: 精确度级别 ("precise", "approximate", "semantic")
        """
        if granularity == "precise":
            return InputSignatureGenerator._precise_signature(input_data)
        elif granularity == "approximate":
            return InputSignatureGenerator._approximate_signature(input_data)
        elif granularity == "semantic":
            return InputSignatureGenerator._semantic_signature(input_data)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
    
    @staticmethod
    def _precise_signature(input_data: Any) -> str:
        """精确签名 - 任何变化都重新计算"""
        if isinstance(input_data, dict):
            # 对字典按键排序确保一致性
            sorted_items = sorted(input_data.items())
            content = json.dumps(sorted_items, sort_keys=True, default=str)
        elif isinstance(input_data, (list, tuple)):
            content = json.dumps(input_data, default=str)
        elif isinstance(input_data, torch.Tensor):
            content = str(input_data.shape) + str(input_data.dtype) + str(input_data.sum().item())
        elif isinstance(input_data, np.ndarray):
            content = str(input_data.shape) + str(input_data.dtype) + str(input_data.sum())
        else:
            content = str(input_data)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def _approximate_signature(input_data: Any) -> str:
        """近似签名 - 小幅变化可以重用"""
        if isinstance(input_data, dict):
            # 忽略微小的数值差异
            normalized_data = {}
            for k, v in input_data.items():
                if isinstance(v, (int, float)):
                    # 保留3位小数精度
                    normalized_data[k] = round(float(v), 3)
                elif isinstance(v, str):
                    # 忽略大小写
                    normalized_data[k] = v.lower().strip()
                else:
                    normalized_data[k] = v
            content = json.dumps(normalized_data, sort_keys=True, default=str)
        elif isinstance(input_data, torch.Tensor):
            # 基于形状和近似统计
            stats = f"{input_data.shape}_{input_data.dtype}_{input_data.mean():.3f}_{input_data.std():.3f}"
            content = stats
        else:
            content = str(input_data)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def _semantic_signature(input_data: Any) -> str:
        """语义签名 - 基于语义相似性"""
        if isinstance(input_data, dict):
            # 提取关键语义字段
            semantic_fields = ['query', 'domain', 'category', 'type']
            semantic_data = {}
            for field in semantic_fields:
                if field in input_data:
                    value = input_data[field]
                    if isinstance(value, str):
                        # 提取关键词
                        keywords = set(value.lower().split())
                        semantic_data[field] = sorted(keywords)
                    else:
                        semantic_data[field] = value
            content = json.dumps(semantic_data, sort_keys=True)
        else:
            content = str(input_data)
        
        return hashlib.md5(content.encode()).hexdigest()

class SmartCacheManager:
    """智能缓存管理器"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: int = 3600,
                 enable_persistence: bool = True,
                 cache_dir: str = "cache"):
        """
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存存活时间
            enable_persistence: 是否持久化缓存
            cache_dir: 缓存目录
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        self.cache_dir = Path(cache_dir)
        
        # 内存缓存
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
        # 创建缓存目录
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
        
        logger.info(f"🗄️ 智能缓存管理器初始化完成")
        logger.info(f"   最大缓存: {max_size} 条目")
        logger.info(f"   TTL: {ttl_seconds} 秒")
        logger.info(f"   持久化: {'启用' if enable_persistence else '禁用'}")
    
    def get(self, 
            key: str, 
            input_data: Any = None,
            granularity: str = "precise") -> Optional[Any]:
        """获取缓存结果
        
        Args:
            key: 缓存键
            input_data: 输入数据（用于签名验证）
            granularity: 匹配精确度
        """
        with self.lock:
            self.stats.total_requests += 1
            
            # 检查内存缓存
            if key in self.cache:
                entry = self.cache[key]
                
                # 检查TTL
                if self._is_expired(entry):
                    self._remove_entry(key)
                    self.stats.cache_misses += 1
                    return None
                
                # 检查输入签名（如果提供）
                if input_data is not None:
                    current_signature = InputSignatureGenerator.generate_signature(
                        input_data, granularity
                    )
                    if entry.input_signature != current_signature:
                        logger.debug(f"🔄 签名不匹配，缓存失效: {key}")
                        self.stats.cache_misses += 1
                        return None
                
                # 缓存命中
                entry.hit_count += 1
                self.cache.move_to_end(key)  # LRU更新
                self.stats.cache_hits += 1
                self.stats.total_compute_time_saved += entry.compute_time
                self.stats.update_hit_rate()
                
                logger.debug(f"✅ 缓存命中: {key}, 节省时间: {entry.compute_time:.3f}s")
                return entry.result
            
            # 检查持久化缓存
            if self.enable_persistence:
                persistent_result = self._load_from_disk(key)
                if persistent_result is not None:
                    logger.debug(f"💾 从磁盘加载缓存: {key}")
                    self.stats.cache_hits += 1
                    self.stats.update_hit_rate()
                    return persistent_result
            
            self.stats.cache_misses += 1
            self.stats.update_hit_rate()
            return None
    
    def set(self, 
            key: str, 
            result: Any, 
            input_data: Any = None,
            compute_time: float = 0.0,
            granularity: str = "precise",
            metadata: Dict[str, Any] = None) -> None:
        """设置缓存结果"""
        with self.lock:
            # 生成输入签名
            input_signature = ""
            if input_data is not None:
                input_signature = InputSignatureGenerator.generate_signature(
                    input_data, granularity
                )
            
            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                result=result,
                timestamp=datetime.now(),
                compute_time=compute_time,
                input_signature=input_signature,
                metadata=metadata or {}
            )
            
            # 添加到内存缓存
            if key in self.cache:
                self.cache.pop(key)
            
            self.cache[key] = entry
            self.cache.move_to_end(key)
            
            # 检查缓存大小限制
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                self._remove_entry(oldest_key)
            
            # 持久化到磁盘
            if self.enable_persistence:
                self._save_to_disk(key, entry)
            
            logger.debug(f"💾 缓存设置: {key}, 计算时间: {compute_time:.3f}s")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查缓存是否过期"""
        return (datetime.now() - entry.timestamp).total_seconds() > self.ttl_seconds
    
    def _remove_entry(self, key: str) -> None:
        """移除缓存条目"""
        if key in self.cache:
            del self.cache[key]
        
        if self.enable_persistence:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """从磁盘加载缓存"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                if not self._is_expired(entry):
                    # 加载到内存缓存
                    self.cache[key] = entry
                    self.cache.move_to_end(key)
                    return entry.result
                else:
                    # 过期，删除文件
                    cache_file.unlink()
        except Exception as e:
            logger.warning(f"从磁盘加载缓存失败 {key}: {e}")
        
        return None
    
    def _save_to_disk(self, key: str, entry: CacheEntry) -> None:
        """保存缓存到磁盘"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"保存缓存到磁盘失败 {key}: {e}")
    
    def _load_persistent_cache(self) -> None:
        """加载持久化缓存"""
        if not self.cache_dir.exists():
            return
        
        loaded_count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                if not self._is_expired(entry):
                    key = cache_file.stem
                    self.cache[key] = entry
                    loaded_count += 1
                else:
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"加载持久化缓存失败 {cache_file}: {e}")
                cache_file.unlink()
        
        logger.info(f"📂 加载持久化缓存: {loaded_count} 条目")
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
            
            if self.enable_persistence:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
        
        logger.info("🗑️ 缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'total_requests': self.stats.total_requests,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'hit_rate': self.stats.avg_hit_rate,
                'total_time_saved': self.stats.total_compute_time_saved,
                'avg_time_saved_per_hit': (
                    self.stats.total_compute_time_saved / max(self.stats.cache_hits, 1)
                )
            }

class V1CacheOptimizer:
    """V1.0系统缓存优化器"""
    
    def __init__(self):
        self.clip_cache = SmartCacheManager(
            max_size=500, 
            ttl_seconds=7200,  # 2小时
            cache_dir="cache/v1_clip"
        )
        self.yolo_cache = SmartCacheManager(
            max_size=300,
            ttl_seconds=3600,  # 1小时  
            cache_dir="cache/v1_yolo"
        )
        self.dual_score_cache = SmartCacheManager(
            max_size=1000,
            ttl_seconds=1800,  # 30分钟
            cache_dir="cache/v1_dual_score"
        )
        
        logger.info("🚀 V1.0缓存优化器初始化完成")
    
    def cached_clip_inference(self, image_url: str, text_queries: List[str]) -> Dict[str, float]:
        """缓存的CLIP推理"""
        # 创建缓存键
        cache_key = f"clip_{hashlib.md5((image_url + str(sorted(text_queries))).encode()).hexdigest()}"
        
        # 输入数据用于签名验证
        input_data = {
            'image_url': image_url,
            'text_queries': sorted(text_queries)
        }
        
        # 尝试从缓存获取
        cached_result = self.clip_cache.get(cache_key, input_data, granularity="approximate")
        if cached_result is not None:
            return cached_result
        
        # 执行实际计算
        start_time = time.time()
        result = self._execute_clip_inference(image_url, text_queries)
        compute_time = time.time() - start_time
        
        # 保存到缓存
        self.clip_cache.set(
            cache_key, 
            result, 
            input_data, 
            compute_time,
            granularity="approximate",
            metadata={
                'model': 'CLIP-ViT-B/32',
                'image_url': image_url,
                'query_count': len(text_queries)
            }
        )
        
        return result
    
    def cached_yolo_detection(self, image_url: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """缓存的YOLO目标检测"""
        cache_key = f"yolo_{hashlib.md5(image_url.encode()).hexdigest()}_{confidence_threshold}"
        
        input_data = {
            'image_url': image_url,
            'confidence_threshold': confidence_threshold
        }
        
        cached_result = self.yolo_cache.get(cache_key, input_data, granularity="precise")
        if cached_result is not None:
            return cached_result
        
        start_time = time.time()
        result = self._execute_yolo_detection(image_url, confidence_threshold)
        compute_time = time.time() - start_time
        
        self.yolo_cache.set(
            cache_key,
            result,
            input_data,
            compute_time,
            metadata={
                'model': 'YOLOv8',
                'confidence_threshold': confidence_threshold,
                'detection_count': len(result)
            }
        )
        
        return result
    
    def cached_dual_score_computation(self, 
                                    compliance_score: float,
                                    conflict_score: float,
                                    w_c: float = 0.7,
                                    w_n: float = 0.3) -> float:
        """缓存的双分数计算"""
        # 对于简单计算，使用近似签名
        cache_key = f"dual_score_{compliance_score:.3f}_{conflict_score:.3f}_{w_c:.3f}_{w_n:.3f}"
        
        input_data = {
            'compliance_score': compliance_score,
            'conflict_score': conflict_score,
            'w_c': w_c,
            'w_n': w_n
        }
        
        cached_result = self.dual_score_cache.get(cache_key, input_data, granularity="approximate")
        if cached_result is not None:
            return cached_result
        
        start_time = time.time()
        result = self._execute_dual_score_computation(compliance_score, conflict_score, w_c, w_n)
        compute_time = time.time() - start_time
        
        self.dual_score_cache.set(
            cache_key,
            result,
            input_data,
            compute_time,
            granularity="approximate"
        )
        
        return result
    
    def _execute_clip_inference(self, image_url: str, text_queries: List[str]) -> Dict[str, float]:
        """执行实际CLIP推理 (模拟)"""
        logger.info(f"🔄 执行CLIP推理: {image_url}")
        time.sleep(0.1)  # 模拟计算时间
        
        # 模拟结果
        results = {}
        for query in text_queries:
            results[query] = np.random.uniform(0.1, 0.9)
        
        return results
    
    def _execute_yolo_detection(self, image_url: str, confidence_threshold: float) -> List[Dict]:
        """执行实际YOLO检测 (模拟)"""
        logger.info(f"🔄 执行YOLO检测: {image_url}")
        time.sleep(0.2)  # 模拟计算时间
        
        # 模拟检测结果
        detections = []
        for i in range(np.random.randint(0, 5)):
            detections.append({
                'class': f'object_{i}',
                'confidence': np.random.uniform(confidence_threshold, 1.0),
                'bbox': [
                    np.random.randint(0, 800),
                    np.random.randint(0, 600),
                    np.random.randint(50, 200),
                    np.random.randint(50, 200)
                ]
            })
        
        return detections
    
    def _execute_dual_score_computation(self, compliance: float, conflict: float, 
                                      w_c: float, w_n: float) -> float:
        """执行实际双分数计算"""
        time.sleep(0.001)  # 模拟计算时间
        return w_c * compliance + w_n * (1 - conflict)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合缓存统计"""
        return {
            'clip_cache': self.clip_cache.get_stats(),
            'yolo_cache': self.yolo_cache.get_stats(),
            'dual_score_cache': self.dual_score_cache.get_stats()
        }

class V2CacheOptimizer:
    """V2.0系统缓存优化器 - 针对多模态融合的特殊需求"""
    
    def __init__(self):
        self.feature_cache = SmartCacheManager(
            max_size=800,
            ttl_seconds=3600,
            cache_dir="cache/v2_features"
        )
        self.fusion_cache = SmartCacheManager(
            max_size=600,
            ttl_seconds=1800,
            cache_dir="cache/v2_fusion"
        )
        self.inference_cache = SmartCacheManager(
            max_size=400,
            ttl_seconds=1200,
            cache_dir="cache/v2_inference"
        )
        
        logger.info("🔬 V2.0缓存优化器初始化完成")
    
    def cached_multimodal_feature_extraction(self, 
                                           image_url: str,
                                           text_description: str,
                                           metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """缓存的多模态特征提取"""
        # 对于V2.0，使用语义签名来允许更好的重用
        cache_key = f"features_{hashlib.md5((image_url + text_description).encode()).hexdigest()}"
        
        input_data = {
            'image_url': image_url,
            'text_description': text_description,
            'metadata': metadata
        }
        
        # 使用语义签名，允许相似内容重用
        cached_result = self.feature_cache.get(cache_key, input_data, granularity="semantic")
        if cached_result is not None:
            return cached_result
        
        start_time = time.time()
        result = self._execute_multimodal_feature_extraction(image_url, text_description, metadata)
        compute_time = time.time() - start_time
        
        self.feature_cache.set(
            cache_key,
            result,
            input_data,
            compute_time,
            granularity="semantic",
            metadata={
                'feature_dims': {k: v.shape for k, v in result.items()},
                'extraction_time': compute_time
            }
        )
        
        return result
    
    def cached_attention_fusion(self, 
                              visual_features: torch.Tensor,
                              text_features: torch.Tensor,
                              attr_features: torch.Tensor) -> torch.Tensor:
        """缓存的注意力融合"""
        # 为张量创建签名
        feature_signature = (
            f"{visual_features.shape}_{visual_features.sum().item():.6f}_"
            f"{text_features.shape}_{text_features.sum().item():.6f}_"
            f"{attr_features.shape}_{attr_features.sum().item():.6f}"
        )
        cache_key = f"fusion_{hashlib.md5(feature_signature.encode()).hexdigest()}"
        
        input_data = {
            'visual_features': visual_features,
            'text_features': text_features,
            'attr_features': attr_features
        }
        
        # 对于注意力融合，使用近似签名
        cached_result = self.fusion_cache.get(cache_key, input_data, granularity="approximate")
        if cached_result is not None:
            return cached_result
        
        start_time = time.time()
        result = self._execute_attention_fusion(visual_features, text_features, attr_features)
        compute_time = time.time() - start_time
        
        self.fusion_cache.set(
            cache_key,
            result,
            input_data,
            compute_time,
            granularity="approximate",
            metadata={
                'input_shapes': [visual_features.shape, text_features.shape, attr_features.shape],
                'output_shape': result.shape,
                'fusion_time': compute_time
            }
        )
        
        return result
    
    def _execute_multimodal_feature_extraction(self, 
                                             image_url: str,
                                             text_description: str,
                                             metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """执行实际多模态特征提取 (模拟)"""
        logger.info(f"🔄 执行多模态特征提取: {image_url[:50]}...")
        time.sleep(0.3)  # 模拟复杂计算
        
        return {
            'visual': torch.randn(512),
            'text': torch.randn(384),
            'attributes': torch.randn(64)
        }
    
    def _execute_attention_fusion(self, 
                                visual_features: torch.Tensor,
                                text_features: torch.Tensor,
                                attr_features: torch.Tensor) -> torch.Tensor:
        """执行实际注意力融合 (模拟)"""
        time.sleep(0.05)  # 模拟注意力计算
        
        # 简化的融合逻辑
        combined = torch.cat([
            visual_features[:256],
            text_features[:256], 
            attr_features[:64]
        ])
        
        return combined

class CacheOrchestrator:
    """缓存编排器 - 协调V1和V2的缓存策略"""
    
    def __init__(self):
        self.v1_optimizer = V1CacheOptimizer()
        self.v2_optimizer = V2CacheOptimizer()
        self.system_mode = "v1"  # 当前系统模式
        
        logger.info("🎼 缓存编排器初始化完成")
    
    def set_system_mode(self, mode: str):
        """设置系统模式"""
        if mode in ["v1", "v2", "hybrid"]:
            self.system_mode = mode
            logger.info(f"🔄 切换到 {mode.upper()} 模式")
        else:
            raise ValueError(f"Unknown system mode: {mode}")
    
    def smart_pipeline_cache(self, 
                           image_url: str,
                           query: str,
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """智能管道缓存 - 根据模式选择最优缓存策略"""
        metadata = metadata or {}
        
        if self.system_mode == "v1":
            return self._v1_pipeline_cache(image_url, query, metadata)
        elif self.system_mode == "v2":
            return self._v2_pipeline_cache(image_url, query, metadata)
        elif self.system_mode == "hybrid":
            return self._hybrid_pipeline_cache(image_url, query, metadata)
    
    def _v1_pipeline_cache(self, image_url: str, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """V1.0管道缓存"""
        results = {}
        
        # CLIP推理
        clip_results = self.v1_optimizer.cached_clip_inference(
            image_url, [query]
        )
        results['clip'] = clip_results
        
        # YOLO检测
        yolo_results = self.v1_optimizer.cached_yolo_detection(image_url)
        results['yolo'] = yolo_results
        
        # 双分数计算
        compliance_score = metadata.get('compliance_score', 0.8)
        conflict_score = metadata.get('conflict_score', 0.2)
        
        dual_score = self.v1_optimizer.cached_dual_score_computation(
            compliance_score, conflict_score
        )
        results['dual_score'] = dual_score
        
        return results
    
    def _v2_pipeline_cache(self, image_url: str, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """V2.0管道缓存"""
        results = {}
        
        # 多模态特征提取
        features = self.v2_optimizer.cached_multimodal_feature_extraction(
            image_url, query, metadata
        )
        results['features'] = features
        
        # 注意力融合
        fused_features = self.v2_optimizer.cached_attention_fusion(
            features['visual'], features['text'], features['attributes']
        )
        results['fused_features'] = fused_features
        
        return results
    
    def _hybrid_pipeline_cache(self, image_url: str, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """混合模式管道缓存"""
        results = {}
        
        # 同时运行V1和V2，对比结果
        v1_results = self._v1_pipeline_cache(image_url, query, metadata)
        v2_results = self._v2_pipeline_cache(image_url, query, metadata)
        
        results['v1'] = v1_results
        results['v2'] = v2_results
        results['mode'] = 'hybrid'
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统级缓存统计"""
        return {
            'system_mode': self.system_mode,
            'v1_stats': self.v1_optimizer.get_comprehensive_stats(),
            'v2_stats': {
                'feature_cache': self.v2_optimizer.feature_cache.get_stats(),
                'fusion_cache': self.v2_optimizer.fusion_cache.get_stats(),
                'inference_cache': self.v2_optimizer.inference_cache.get_stats()
            }
        }
    
    def optimize_cache_strategy(self) -> Dict[str, str]:
        """基于统计数据优化缓存策略"""
        stats = self.get_system_stats()
        recommendations = {}
        
        # 分析V1缓存效率
        v1_stats = stats['v1_stats']
        for cache_type, cache_stat in v1_stats.items():
            hit_rate = cache_stat['hit_rate']
            if hit_rate < 0.3:
                recommendations[f'v1_{cache_type}'] = "考虑增加TTL或使用近似签名"
            elif hit_rate > 0.8:
                recommendations[f'v1_{cache_type}'] = "缓存效率很好，可以适当增加缓存大小"
        
        # 分析V2缓存效率
        v2_stats = stats['v2_stats']
        for cache_type, cache_stat in v2_stats.items():
            hit_rate = cache_stat['hit_rate']
            if hit_rate < 0.2:
                recommendations[f'v2_{cache_type}'] = "考虑使用语义签名增强重用性"
            elif hit_rate > 0.7:
                recommendations[f'v2_{cache_type}'] = "缓存效率良好"
        
        return recommendations

def demonstrate_cache_optimization():
    """演示缓存优化效果"""
    print("🚀 V2救援计划 - 智能缓存优化演示")
    print("=" * 80)
    
    # 创建缓存编排器
    orchestrator = CacheOrchestrator()
    
    # 模拟请求场景
    test_scenarios = [
        {
            'image_url': 'https://example.com/cocktail1.jpg',
            'query': 'pink floral cocktail',
            'metadata': {'domain': 'cocktails', 'quality_tier': 'high'}
        },
        {
            'image_url': 'https://example.com/cocktail1.jpg',  # 相同图片
            'query': 'pink flower drink',  # 相似查询
            'metadata': {'domain': 'cocktails', 'quality_tier': 'high'}
        },
        {
            'image_url': 'https://example.com/cocktail2.jpg',
            'query': 'pink floral cocktail',  # 相同查询，不同图片
            'metadata': {'domain': 'cocktails', 'quality_tier': 'medium'}
        }
    ]
    
    # 测试不同系统模式
    for mode in ['v1', 'v2', 'hybrid']:
        print(f"\n🔄 测试 {mode.upper()} 模式")
        print("-" * 40)
        
        orchestrator.set_system_mode(mode)
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\n📝 场景 {i+1}: {scenario['query'][:30]}...")
            
            start_time = time.time()
            results = orchestrator.smart_pipeline_cache(
                scenario['image_url'],
                scenario['query'],
                scenario['metadata']
            )
            process_time = time.time() - start_time
            
            print(f"   ⏱️ 处理时间: {process_time:.3f}s")
            print(f"   📊 结果键: {list(results.keys())}")
    
    # 显示缓存统计
    print(f"\n📊 缓存统计总结")
    print("=" * 40)
    
    stats = orchestrator.get_system_stats()
    
    print(f"V1 缓存命中率:")
    for cache_type, cache_stat in stats['v1_stats'].items():
        print(f"   {cache_type}: {cache_stat['hit_rate']:.1%} "
              f"({cache_stat['cache_hits']}/{cache_stat['total_requests']})")
    
    print(f"\nV2 缓存命中率:")
    for cache_type, cache_stat in stats['v2_stats'].items():
        print(f"   {cache_type}: {cache_stat['hit_rate']:.1%} "
              f"({cache_stat['cache_hits']}/{cache_stat['total_requests']})")
    
    # 优化建议
    print(f"\n💡 优化建议:")
    recommendations = orchestrator.optimize_cache_strategy()
    for component, suggestion in recommendations.items():
        print(f"   {component}: {suggestion}")
    
    print(f"\n✅ 缓存优化演示完成")
    print("🎯 关键洞察:")
    print("   • 智能签名机制实现了精确和近似匹配的平衡")
    print("   • 多层缓存架构适应V1/V2不同的计算特点")
    print("   • 持久化缓存减少了冷启动时间")
    print("   • 统计驱动的优化确保缓存策略持续改进")

if __name__ == "__main__":
    demonstrate_cache_optimization()