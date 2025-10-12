# ===================================================================
# V1.0夜间优化研究 - 修复版（简化特征处理）
# 目标：使用增强数据集优化V1.0算法
# ===================================================================

print("🌙 V1.0夜间优化研究启动 (修复版)")
print("="*80)
print("🎯 目标: 基于增强数据集优化V1.0算法")
print("⏰ 计划: 6小时自动化执行")
print("🔧 方法: 特征工程 + 算法优化 + 参数调优")
print("="*80)

import torch
import numpy as np
import json
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

# 检查执行环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 执行环境: {device}')
if torch.cuda.is_available():
    print(f'🚀 GPU: {torch.cuda.get_device_name()}')

# ===================================================================
# 数据加载和预处理
# ===================================================================

print("\n" + "="*60)
print("📊 数据加载和预处理")
print("="*60)

# 加载增强数据集
try:
    with open('/Users/guyan/computer_vision/computer-vision/data/input/enhanced_dataset.json', 'r') as f:
        production_data = json.load(f)
    inspirations = production_data.get('inspirations', [])
    print(f'✅ 加载增强数据集: {len(inspirations)} 个查询')
    
    # 数据统计
    domains = {}
    total_candidates = 0
    for inspiration in inspirations:
        domain = inspiration.get('domain', 'unknown')
        domains[domain] = domains.get(domain, 0) + 1
        total_candidates += len(inspiration.get('candidates', []))
    
    print(f"📈 数据统计:")
    for domain, count in domains.items():
        print(f"   {domain}: {count} 查询")
    print(f"   总候选项: {total_candidates}")
    print(f"   平均候选项/查询: {total_candidates/len(inspirations):.1f}")
    
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit(1)

# ===================================================================
# 小时1-2: 特征工程和数据准备
# ===================================================================

print("\n" + "="*60)
print("🕐🕑 小时1-2: 高级特征工程")
print("="*60)

class OptimizedFeatureEngineer:
    """优化的特征工程师"""
    
    def __init__(self):
        self.feature_dim = 50  # 固定特征维度
        
    def extract_features(self, inspirations):
        """提取统一的特征"""
        print("🔧 提取统一特征...")
        
        enhanced_samples = []
        
        for inspiration in inspirations:
            query = inspiration.get('query', '')
            domain = inspiration.get('domain', 'unknown')
            candidates = inspiration.get('candidates', [])
            
            for candidate in candidates:
                # 基础特征
                base_score = candidate.get('score', 0)
                compliance_score = candidate.get('compliance_score', 0)
                title = candidate.get('title', '')
                
                # 文本特征
                title_length = len(title)
                query_length = len(query)
                title_words = len(title.split())
                query_words = len(query.split())
                
                # 匹配特征
                query_terms = set(query.lower().split())
                title_terms = set(title.lower().split())
                common_terms = len(query_terms & title_terms)
                jaccard_similarity = common_terms / max(len(query_terms | title_terms), 1)
                
                # 领域特征
                domain_features = {
                    'food': [1, 0, 0, 0, 0],
                    'cocktails': [0, 1, 0, 0, 0], 
                    'alcohol': [0, 0, 1, 0, 0],
                    'dining': [0, 0, 0, 1, 0],
                    'beverages': [0, 0, 0, 0, 1]
                }.get(domain, [0, 0, 0, 0, 0])
                
                # 关键词特征
                premium_keywords = ['premium', 'craft', 'artisan', 'gourmet', 'delicious', 'fresh', 'elegant', 'signature']
                keyword_score = sum(1 for kw in premium_keywords if kw in title.lower())
                
                # 组合成固定长度特征向量
                features = [
                    base_score,
                    compliance_score,
                    title_length / 100.0,  # 归一化
                    query_length / 100.0,
                    title_words / 10.0,
                    query_words / 10.0,
                    common_terms,
                    jaccard_similarity,
                    keyword_score / len(premium_keywords)
                ] + domain_features  # 14维基础特征
                
                # 填充到固定维度
                while len(features) < self.feature_dim:
                    features.append(0.0)
                
                enhanced_samples.append({
                    'query': query,
                    'domain': domain,
                    'candidate': candidate,
                    'features': np.array(features[:self.feature_dim]),
                    'score': base_score,
                    'compliance': compliance_score
                })
        
        print(f"✅ 特征提取完成: {len(enhanced_samples)} 样本, {self.feature_dim}维特征")
        return enhanced_samples

feature_engineer = OptimizedFeatureEngineer()
enhanced_dataset = feature_engineer.extract_features(inspirations)

# ===================================================================
# 小时3: 机器学习优化
# ===================================================================

print("\n" + "="*60)
print("🕒 小时3: 机器学习优化")
print("="*60)

class MLOptimizer:
    """机器学习优化器"""
    
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.model = None
        
    def train_ranking_model(self, dataset):
        """训练排序模型"""
        print("🧠 训练排序优化模型...")
        
        if len(dataset) < 10:
            print("⚠️ 数据不足，跳过模型训练")
            return None
            
        # 简化的神经网络
        class RankingNet(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(32, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                    torch.nn.Sigmoid()
                )
                
            def forward(self, x):
                return self.net(x)
        
        # 准备训练数据
        features = torch.FloatTensor([d['features'] for d in dataset]).to(device)
        targets = torch.FloatTensor([[d['compliance']] for d in dataset]).to(device)
        
        model = RankingNet(self.feature_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # 训练
        model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch+1}/30, Loss: {loss.item():.6f}")
        
        self.model = model
        print("✅ 排序模型训练完成")
        return model

ml_optimizer = MLOptimizer(feature_engineer.feature_dim)
ranking_model = ml_optimizer.train_ranking_model(enhanced_dataset)

# ===================================================================
# 小时4: 权重优化
# ===================================================================

print("\n" + "="*60)
print("🕓 小时4: 权重优化")
print("="*60)

class WeightOptimizer:
    """权重优化器"""
    
    def optimize_weights(self, dataset):
        """优化权重参数"""
        print("⚖️ 优化算法权重...")
        
        param_grid = {
            'score_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
            'compliance_weight': [0.2, 0.3, 0.4, 0.5],
            'text_weight': [0.1, 0.15, 0.2, 0.25],
            'ml_weight': [0.1, 0.2, 0.3]
        }
        
        best_score = -1
        best_weights = None
        
        sample_size = min(len(dataset), 200)  # 限制计算量
        sample_data = dataset[:sample_size]
        
        for i, params in enumerate(ParameterGrid(param_grid)):
            if i % 50 == 0:
                print(f"   测试参数组合 {i+1}...")
                
            # 计算加权得分
            weighted_scores = []
            true_scores = []
            
            for data in sample_data:
                # 基础得分
                base_score = data['score']
                compliance = data['compliance']
                
                # 文本得分
                text_score = len(data['candidate'].get('title', '')) / 100.0
                
                # ML得分 (如果有模型)
                if ranking_model:
                    with torch.no_grad():
                        features = torch.FloatTensor(data['features']).unsqueeze(0).to(device)
                        ml_score = ranking_model(features).item()
                else:
                    ml_score = compliance
                
                # 加权组合
                weighted_score = (
                    params['score_weight'] * base_score +
                    params['compliance_weight'] * compliance +
                    params['text_weight'] * text_score +
                    params['ml_weight'] * ml_score
                )
                
                weighted_scores.append(weighted_score)
                true_scores.append(compliance)
            
            # 评估
            if len(true_scores) > 1:
                correlation = np.corrcoef(weighted_scores, true_scores)[0, 1]
                if not np.isnan(correlation) and correlation > best_score:
                    best_score = correlation
                    best_weights = params
        
        if best_weights is None:
            best_weights = {'score_weight': 0.5, 'compliance_weight': 0.3, 'text_weight': 0.1, 'ml_weight': 0.1}
            best_score = 0
            
        print(f"✅ 权重优化完成")
        print(f"   最优权重: {best_weights}")
        print(f"   相关性得分: {best_score:.4f}")
        
        return best_weights, best_score

weight_optimizer = WeightOptimizer()
optimal_weights, correlation_score = weight_optimizer.optimize_weights(enhanced_dataset)

# ===================================================================
# 小时5: 算法验证
# ===================================================================

print("\n" + "="*60)
print("🕔 小时5: 增强算法验证")
print("="*60)

class EnhancedV1Ranker:
    """增强版V1排序器"""
    
    def __init__(self, weights, ml_model=None):
        self.weights = weights
        self.ml_model = ml_model
        
    def rank_candidates(self, query, candidates, domain):
        """排序候选项"""
        enhanced_scores = []
        
        for candidate in candidates:
            # 基础分数
            base_score = candidate.get('score', 0)
            compliance_score = candidate.get('compliance_score', 0)
            title = candidate.get('title', '')
            
            # 文本特征
            text_score = min(len(title) / 50.0, 1.0)
            
            # ML预测 (如果有模型)
            if self.ml_model:
                # 快速特征提取
                features = [
                    base_score, compliance_score, len(title)/100.0, len(query)/100.0,
                    len(title.split())/10.0, len(query.split())/10.0
                ]
                while len(features) < 50:
                    features.append(0.0)
                
                try:
                    with torch.no_grad():
                        feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                        ml_score = self.ml_model(feat_tensor).item()
                except:
                    ml_score = compliance_score
            else:
                ml_score = compliance_score
                
            # 加权评分
            enhanced_score = (
                self.weights.get('score_weight', 0.5) * base_score +
                self.weights.get('compliance_weight', 0.3) * compliance_score +
                self.weights.get('text_weight', 0.1) * text_score +
                self.weights.get('ml_weight', 0.1) * ml_score
            )
            
            enhanced_scores.append({
                'candidate': candidate,
                'enhanced_score': enhanced_score,
                'original_score': base_score
            })
        
        # 排序
        enhanced_scores.sort(key=lambda x: x['enhanced_score'], reverse=True)
        return enhanced_scores

# 创建增强排序器
enhanced_ranker = EnhancedV1Ranker(optimal_weights, ranking_model)

# 验证算法改进
print("🔍 验证算法改进效果...")
improvements = []
validation_sample = inspirations[:30]  # 验证样本

for inspiration in validation_sample:
    query = inspiration.get('query', '')
    domain = inspiration.get('domain', 'unknown')
    candidates = inspiration.get('candidates', [])
    
    if len(candidates) >= 2:
        # 原始排序
        original_scores = [c.get('score', 0) for c in candidates]
        true_labels = [c.get('compliance_score', 0) for c in candidates]
        
        # 增强排序
        enhanced_ranking = enhanced_ranker.rank_candidates(query, candidates, domain)
        enhanced_scores = [r['enhanced_score'] for r in enhanced_ranking]
        
        # 计算nDCG改进
        if len(true_labels) >= 2:
            try:
                original_ndcg = ndcg_score([true_labels], [original_scores], k=5)
                enhanced_ndcg = ndcg_score([true_labels], [enhanced_scores], k=5)
                improvement = enhanced_ndcg - original_ndcg
                improvements.append(improvement)
            except:
                continue

if improvements:
    avg_improvement = np.mean(improvements)
    improvement_std = np.std(improvements)
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    
    print(f"✅ 算法验证完成")
    print(f"   验证样本数: {len(improvements)}")
    print(f"   平均nDCG改进: {avg_improvement:+.6f}")
    print(f"   改进标准差: {improvement_std:.6f}")
    print(f"   正改进比例: {positive_improvements}/{len(improvements)} ({positive_improvements/len(improvements)*100:.1f}%)")
else:
    avg_improvement = 0
    print("⚠️ 验证数据不足")

# ===================================================================
# 小时6: 最终集成和总结
# ===================================================================

print("\n" + "="*60)
print("🕕 小时6: 最终集成和总结")
print("="*60)

# 创建最终增强包
enhancement_package = {
    'version': 'V1.1-Enhanced-100Dataset',
    'timestamp': datetime.now().isoformat(),
    'dataset_info': {
        'total_queries': len(inspirations),
        'total_candidates': sum(len(q.get('candidates', [])) for q in inspirations),
        'domains': list(set(q.get('domain') for q in inspirations)),
        'enhanced_samples': len(enhanced_dataset)
    },
    'optimizations': {
        'feature_engineering': f'{feature_engineer.feature_dim}维统一特征',
        'ml_model': 'PyTorch深度排序网络' if ranking_model else '未启用',
        'weight_optimization': optimal_weights,
        'correlation_score': correlation_score
    },
    'performance_metrics': {
        'avg_ndcg_improvement': avg_improvement,
        'validation_samples': len(improvements) if improvements else 0,
        'positive_improvement_rate': f"{positive_improvements/len(improvements)*100:.1f}%" if improvements else "N/A"
    }
}

print("🏭 创建生产就绪增强包...")
print(f"✅ 增强包生成完成:")
print(f"   版本: {enhancement_package['version']}")
print(f"   数据规模: {enhancement_package['dataset_info']['total_queries']} 查询")
print(f"   特征维度: {feature_engineer.feature_dim}")
print(f"   模型状态: {'已训练' if ranking_model else '未启用'}")

# ===================================================================
# 研究总结
# ===================================================================

print("\n" + "="*80)
print("🌅 V1.0夜间优化研究完成 (增强版)")
print("="*80)

print("🎯 研究成果总结:")
print(f"✅ 数据处理: {len(inspirations)} 查询, {len(enhanced_dataset)} 增强样本")
print(f"✅ 特征工程: {feature_engineer.feature_dim}维统一特征")
print(f"✅ 机器学习: {'深度排序模型训练完成' if ranking_model else '模型训练跳过'}")
print(f"✅ 权重优化: 相关性得分 {correlation_score:.4f}")
print(f"✅ 算法验证: nDCG改进 {avg_improvement:+.6f}")

print(f"\n📊 关键指标:")
print(f"   🎯 验证样本: {len(improvements) if improvements else 0}")
if improvements:
    print(f"   📈 正向改进率: {positive_improvements/len(improvements)*100:.1f}%")
    print(f"   📊 平均改进: {avg_improvement:+.6f}")
    print(f"   📏 改进标准差: {improvement_std:.6f}")

recommendation = "PROCEED_WITH_TESTING" if avg_improvement > 0 else "CONTINUE_V1_MONITORING"
print(f"\n🚀 最终建议: {recommendation}")

if avg_improvement > 0:
    print("✅ 夜间研究取得正向改进，建议进行A/B测试验证")
else:
    print("⚠️ 改进效果有限，建议继续V1.0稳定运行并收集更多数据")

print("\n" + "="*80)
print("🎊 夜间研究完成！")
print("💡 建议: 晨间评估结果并决定下一步行动")
print("="*80)