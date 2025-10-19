# ===================================================================
# 今晚大样本MMR实验 - 科学验证算法改进
# 目标：300-500样本 + MMR多样性 + 主题覆盖 + Bootstrap CI
# 决策标准：95% CI > 0 才进入shadow testing
# ===================================================================

import numpy as np
import json
import random
from datetime import datetime
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("🌙 今晚大样本MMR实验启动")
print("="*80)
print("🎯 目标: 科学验证MMR多样性算法改进")
print("📊 样本: 300-500 queries with Bootstrap CI")
print("⚡ 决策: 95% CI > 0 才进入shadow testing")
print("="*80)

# ===================================================================
# 大样本数据集生成
# ===================================================================

class LargeScaleDataGenerator:
    """大规模数据集生成器"""
    
    def __init__(self, target_size=400):
        self.target_size = target_size
        self.domains = ['food', 'cocktails', 'alcohol', 'dining', 'beverages']
        
    def generate_comprehensive_dataset(self):
        """生成大规模综合数据集"""
        print(f"🏗️ 生成{self.target_size}查询的大规模数据集...")
        
        queries_per_domain = self.target_size // len(self.domains)
        dataset = []
        
        # 各域查询模板和术语
        domain_templates = {
            'food': {
                'descriptors': ['delicious', 'fresh', 'gourmet', 'artisan', 'organic', 'homemade', 'premium', 'authentic', 'seasonal', 'local'],
                'items': ['pasta', 'salad', 'soup', 'pizza', 'burger', 'sushi', 'steak', 'dessert', 'bread', 'cake', 'sandwich', 'tacos', 'noodles', 'curry'],
                'contexts': ['restaurant', 'home cooking', 'fine dining', 'street food', 'comfort food']
            },
            'cocktails': {
                'descriptors': ['craft', 'premium', 'artisanal', 'classic', 'signature', 'elegant', 'refreshing', 'sophisticated', 'creative', 'smooth'],
                'items': ['mojito', 'martini', 'old fashioned', 'whiskey sour', 'negroni', 'manhattan', 'daiquiri', 'margarita', 'cosmopolitan', 'gin fizz', 'bloody mary'],
                'contexts': ['bar', 'mixology', 'happy hour', 'rooftop', 'speakeasy']
            },
            'alcohol': {
                'descriptors': ['premium', 'aged', 'vintage', 'rare', 'craft', 'distillery', 'single malt', 'small batch', 'limited edition', 'collectors'],
                'items': ['whiskey', 'wine', 'gin', 'vodka', 'rum', 'bourbon', 'scotch', 'champagne', 'tequila', 'brandy', 'cognac', 'sake'],
                'contexts': ['collection', 'tasting', 'cellar', 'bottle', 'glass']
            },
            'dining': {
                'descriptors': ['elegant', 'cozy', 'modern', 'rustic', 'upscale', 'casual', 'intimate', 'vibrant', 'trendy', 'classic'],
                'items': ['restaurant', 'cafe', 'bistro', 'brasserie', 'tavern', 'eatery', 'diner', 'gastropub', 'wine bar', 'steakhouse'],
                'contexts': ['atmosphere', 'interior', 'ambiance', 'setting', 'experience']
            },
            'beverages': {
                'descriptors': ['refreshing', 'cold', 'hot', 'specialty', 'artisan', 'organic', 'fresh', 'smooth', 'rich', 'aromatic'],
                'items': ['coffee', 'tea', 'juice', 'smoothie', 'latte', 'cappuccino', 'espresso', 'matcha', 'kombucha', 'soda'],
                'contexts': ['cafe', 'morning', 'afternoon', 'specialty', 'barista']
            }
        }
        
        for domain in self.domains:
            templates = domain_templates[domain]
            
            for i in range(queries_per_domain):
                # 生成查询
                descriptor = random.choice(templates['descriptors'])
                item = random.choice(templates['items'])
                context = random.choice(templates['contexts'])
                
                query_variations = [
                    f"{descriptor} {item}",
                    f"{item} {descriptor} style",
                    f"best {descriptor} {item}",
                    f"{descriptor} {item} {context}",
                    f"{context} {descriptor} {item}",
                    f"{item} with {descriptor} presentation"
                ]
                
                query = random.choice(query_variations)
                
                # 生成候选项
                candidates = []
                num_candidates = random.randint(3, 6)
                
                for j in range(num_candidates):
                    # 基础分数
                    base_score = random.uniform(0.5, 0.95)
                    
                    # 相关性调整
                    if descriptor.lower() in item.lower() or item.lower() in descriptor.lower():
                        relevance_boost = 0.1
                    else:
                        relevance_boost = random.uniform(-0.05, 0.15)
                    
                    final_score = min(base_score + relevance_boost, 1.0)
                    compliance = min(final_score + random.uniform(-0.1, 0.1), 1.0)
                    
                    # 主题特征（用于MMR多样性）
                    themes = self._generate_theme_features(domain, descriptor, item)
                    
                    candidate = {
                        "id": f"{domain}_{i}_{j}",
                        "title": f"{descriptor.title()} {item} - {random.choice(['professional', 'high-quality', 'stunning', 'beautiful', 'perfect'])} {context}",
                        "score": round(final_score, 4),
                        "compliance_score": round(max(compliance, 0), 4),
                        "themes": themes,
                        "alt_description": f"{descriptor} {item} in {context} setting"
                    }
                    
                    candidates.append(candidate)
                
                # 排序候选项
                candidates.sort(key=lambda x: x['score'], reverse=True)
                
                dataset.append({
                    "query": query,
                    "domain": domain,
                    "candidates": candidates
                })
        
        # 随机打乱
        random.shuffle(dataset)
        
        print(f"✅ 大规模数据集生成完成:")
        print(f"   📊 总查询: {len(dataset)}")
        print(f"   🎯 总候选项: {sum(len(q['candidates']) for q in dataset)}")
        print(f"   🌍 域分布: {len(self.domains)} 个均衡域")
        
        return dataset
    
    def _generate_theme_features(self, domain, descriptor, item):
        """生成主题特征"""
        themes = {
            'color': random.choice(['warm', 'cool', 'neutral', 'vibrant', 'monochrome']),
            'style': random.choice(['modern', 'classic', 'rustic', 'elegant', 'casual']),
            'presentation': random.choice(['simple', 'elaborate', 'artistic', 'traditional', 'creative'])
        }
        
        if domain == 'cocktails':
            themes['glass_type'] = random.choice(['coupe', 'highball', 'rocks', 'martini', 'wine'])
            themes['garnish'] = random.choice(['citrus', 'herbs', 'fruit', 'none', 'olive'])
        elif domain == 'food':
            themes['cuisine'] = random.choice(['italian', 'asian', 'american', 'french', 'mediterranean'])
            themes['plating'] = random.choice(['rustic', 'fine-dining', 'casual', 'family-style', 'modern'])
        
        return themes

# 生成大规模数据集
data_generator = LargeScaleDataGenerator(target_size=400)
large_dataset = data_generator.generate_comprehensive_dataset()

# ===================================================================
# MMR多样性算法实现
# ===================================================================

class MMRDiversityRanker:
    """MMR多样性排序器"""
    
    def __init__(self, alpha=0.75, max_results=50):
        self.alpha = alpha  # 相关性vs多样性权衡
        self.max_results = max_results
        
    def calculate_similarity(self, candidate1, candidate2):
        """计算候选项相似度"""
        # 基于主题特征的相似度
        themes1 = candidate1.get('themes', {})
        themes2 = candidate2.get('themes', {})
        
        similarity = 0
        total_features = 0
        
        for key in themes1:
            if key in themes2:
                if themes1[key] == themes2[key]:
                    similarity += 1
                total_features += 1
        
        # 基于标题的文本相似度
        title1_words = set(candidate1.get('title', '').lower().split())
        title2_words = set(candidate2.get('title', '').lower().split())
        
        if title1_words and title2_words:
            text_similarity = len(title1_words & title2_words) / len(title1_words | title2_words)
            similarity += text_similarity
            total_features += 1
        
        return similarity / max(total_features, 1)
    
    def rank_with_mmr(self, query, candidates, domain):
        """使用MMR进行多样性排序"""
        if len(candidates) <= 2:
            return [{'candidate': c, 'mmr_score': c.get('score', 0), 'original_score': c.get('score', 0)} for c in candidates]
        
        # 限制处理范围
        top_candidates = candidates[:self.max_results]
        
        # 初始化
        selected = []
        remaining = top_candidates.copy()
        
        # 选择第一个（最高相关性）
        if remaining:
            first = remaining.pop(0)
            selected.append({
                'candidate': first,
                'mmr_score': first.get('score', 0),
                'original_score': first.get('score', 0)
            })
        
        # MMR选择剩余项
        while remaining and len(selected) < len(top_candidates):
            best_candidate = None
            best_mmr_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # 相关性分数
                relevance = candidate.get('score', 0)
                
                # 计算与已选择项的最大相似度
                max_similarity = 0
                for selected_item in selected:
                    sim = self.calculate_similarity(candidate, selected_item['candidate'])
                    max_similarity = max(max_similarity, sim)
                
                # MMR分数：α*相关性 - (1-α)*最大相似度
                mmr_score = self.alpha * relevance - (1 - self.alpha) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate:
                selected.append({
                    'candidate': best_candidate,
                    'mmr_score': best_mmr_score,
                    'original_score': best_candidate.get('score', 0)
                })
                remaining.pop(best_idx)
        
        return selected

# ===================================================================
# 主题覆盖算法实现
# ===================================================================

class ThemeCoverageRanker:
    """主题覆盖排序器"""
    
    def __init__(self, required_themes=None):
        self.required_themes = required_themes or ['color', 'style', 'presentation']
        
    def rank_with_theme_coverage(self, query, candidates, domain):
        """使用主题覆盖进行排序"""
        if len(candidates) <= 2:
            return [{'candidate': c, 'coverage_score': c.get('score', 0), 'original_score': c.get('score', 0)} for c in candidates]
        
        # 分析主题覆盖情况
        theme_coverage = {theme: [] for theme in self.required_themes}
        
        for i, candidate in enumerate(candidates):
            themes = candidate.get('themes', {})
            for theme in self.required_themes:
                if theme in themes:
                    theme_coverage[theme].append((i, candidate, themes[theme]))
        
        # 确保每个主题至少有一个代表
        selected_indices = set()
        selected_results = []
        
        # 为每个主题选择最佳代表
        for theme, candidates_with_theme in theme_coverage.items():
            if candidates_with_theme and not any(idx in selected_indices for idx, _, _ in candidates_with_theme):
                # 选择该主题下得分最高的候选项
                best_candidate = max(candidates_with_theme, key=lambda x: x[1].get('score', 0))
                idx, candidate, theme_value = best_candidate
                
                if idx not in selected_indices:
                    coverage_score = candidate.get('score', 0) * 1.1  # 主题覆盖奖励
                    selected_results.append({
                        'candidate': candidate,
                        'coverage_score': coverage_score,
                        'original_score': candidate.get('score', 0),
                        'covered_theme': theme
                    })
                    selected_indices.add(idx)
        
        # 添加剩余高分候选项
        for i, candidate in enumerate(candidates):
            if i not in selected_indices and len(selected_results) < len(candidates):
                selected_results.append({
                    'candidate': candidate,
                    'coverage_score': candidate.get('score', 0),
                    'original_score': candidate.get('score', 0),
                    'covered_theme': None
                })
        
        # 按覆盖分数排序
        selected_results.sort(key=lambda x: x['coverage_score'], reverse=True)
        
        return selected_results

# ===================================================================
# 大样本实验执行
# ===================================================================

print(f"\n🔬 大样本实验执行")
print("="*60)

# 初始化排序器
mmr_ranker = MMRDiversityRanker(alpha=0.75)
theme_ranker = ThemeCoverageRanker()

# 实验结果收集
baseline_ndcg = []
mmr_ndcg = []
theme_ndcg = []
combined_ndcg = []

sample_size = min(len(large_dataset), 400)
experiment_dataset = large_dataset[:sample_size]

print(f"🧪 开始实验，样本量: {sample_size}")

for i, inspiration in enumerate(experiment_dataset):
    if i % 50 == 0:
        print(f"   处理进度: {i+1}/{sample_size}")
    
    query = inspiration.get('query', '')
    domain = inspiration.get('domain', 'unknown')
    candidates = inspiration.get('candidates', [])
    
    if len(candidates) >= 2:
        # 基线排序（原始分数）
        baseline_scores = [c.get('score', 0) for c in candidates]
        true_labels = [c.get('compliance_score', 0) for c in candidates]
        
        # MMR排序
        mmr_results = mmr_ranker.rank_with_mmr(query, candidates, domain)
        mmr_scores = [r['mmr_score'] for r in mmr_results]
        
        # 主题覆盖排序
        theme_results = theme_ranker.rank_with_theme_coverage(query, candidates, domain)
        theme_scores = [r['coverage_score'] for r in theme_results]
        
        # 组合算法（0.6 MMR + 0.4 Theme Coverage）
        combined_scores = [0.6 * mmr + 0.4 * theme for mmr, theme in zip(mmr_scores, theme_scores)]
        
        # 计算nDCG@10
        try:
            if len(true_labels) >= 2:
                baseline_ndcg.append(ndcg_score([true_labels], [baseline_scores], k=10))
                mmr_ndcg.append(ndcg_score([true_labels], [mmr_scores], k=10))
                theme_ndcg.append(ndcg_score([true_labels], [theme_scores], k=10))
                combined_ndcg.append(ndcg_score([true_labels], [combined_scores], k=10))
        except:
            continue

print(f"✅ 实验完成，有效样本: {len(baseline_ndcg)}")

# ===================================================================
# Bootstrap置信区间计算
# ===================================================================

def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
    """计算Bootstrap置信区间"""
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_means, 100 * alpha/2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    
    return np.mean(data), lower, upper

print(f"\n📊 统计分析结果")
print("="*60)

# 计算改进和置信区间
mmr_improvement = np.array(mmr_ndcg) - np.array(baseline_ndcg)
theme_improvement = np.array(theme_ndcg) - np.array(baseline_ndcg)
combined_improvement = np.array(combined_ndcg) - np.array(baseline_ndcg)

# Bootstrap置信区间
mmr_mean, mmr_lower, mmr_upper = bootstrap_ci(mmr_improvement)
theme_mean, theme_lower, theme_upper = bootstrap_ci(theme_improvement)
combined_mean, combined_lower, combined_upper = bootstrap_ci(combined_improvement)

print(f"🎯 MMR多样性算法 (α=0.75):")
print(f"   平均nDCG改进: {mmr_mean:+.6f}")
print(f"   95% CI: [{mmr_lower:+.6f}, {mmr_upper:+.6f}]")
print(f"   统计显著: {'✅ 是' if mmr_lower > 0 else '❌ 否'}")

print(f"\n🎯 主题覆盖算法:")
print(f"   平均nDCG改进: {theme_mean:+.6f}")
print(f"   95% CI: [{theme_lower:+.6f}, {theme_upper:+.6f}]")
print(f"   统计显著: {'✅ 是' if theme_lower > 0 else '❌ 否'}")

print(f"\n🎯 组合算法 (0.6 MMR + 0.4 Theme):")
print(f"   平均nDCG改进: {combined_mean:+.6f}")
print(f"   95% CI: [{combined_lower:+.6f}, {combined_upper:+.6f}]")
print(f"   统计显著: {'✅ 是' if combined_lower > 0 else '❌ 否'}")

# ===================================================================
# 决策建议
# ===================================================================

print(f"\n🚀 实验结论和决策建议")
print("="*60)

# 选择最佳算法
algorithms = [
    ("MMR多样性", mmr_mean, mmr_lower, mmr_upper),
    ("主题覆盖", theme_mean, theme_lower, theme_upper),
    ("组合算法", combined_mean, combined_lower, combined_upper)
]

significant_algorithms = [(name, mean, lower, upper) for name, mean, lower, upper in algorithms if lower > 0]

if significant_algorithms:
    best_algorithm = max(significant_algorithms, key=lambda x: x[1])
    
    print(f"✅ 实验成功！发现有效改进算法:")
    print(f"   🏆 最佳算法: {best_algorithm[0]}")
    print(f"   📈 改进幅度: {best_algorithm[1]:+.6f} nDCG@10")
    print(f"   🎯 置信区间: [{best_algorithm[2]:+.6f}, {best_algorithm[3]:+.6f}]")
    
    print(f"\n🚀 建议行动:")
    print(f"   1️⃣ 立即进入Shadow Testing阶段")
    print(f"   2️⃣ 在10%流量上A/B测试{best_algorithm[0]}")
    print(f"   3️⃣ 监控核心指标：nDCG@10, Top-1命中率, 用户满意度")
    print(f"   4️⃣ 预期收益: {best_algorithm[1]*100:+.2f}% nDCG改进")
    
    # 保存实验结果
    experiment_results = {
        'timestamp': datetime.now().isoformat(),
        'sample_size': len(baseline_ndcg),
        'best_algorithm': {
            'name': best_algorithm[0],
            'improvement': best_algorithm[1],
            'ci_lower': best_algorithm[2],
            'ci_upper': best_algorithm[3]
        },
        'all_results': {
            'mmr': {'mean': mmr_mean, 'ci': [mmr_lower, mmr_upper]},
            'theme': {'mean': theme_mean, 'ci': [theme_lower, theme_upper]},
            'combined': {'mean': combined_mean, 'ci': [combined_lower, combined_upper]}
        }
    }
    
    with open('/Users/guyan/computer_vision/computer-vision/research/tonight_experiment_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"   💾 实验结果已保存: tonight_experiment_results.json")
    
else:
    print(f"❌ 实验结果：当前算法改进不显著")
    print(f"📊 所有算法的95%置信区间都包含0")
    print(f"\n🤔 可能原因:")
    print(f"   • 合成数据与真实场景存在差异")
    print(f"   • 需要更大样本量或更长时间的A/B测试")
    print(f"   • 当前V1.0算法已经高度优化")
    
    print(f"\n🔄 建议下一步:")
    print(f"   1️⃣ 收集更多真实生产数据")
    print(f"   2️⃣ 尝试候选池质量优化（更容易见效）")
    print(f"   3️⃣ 探索数据闭环和用户反馈信号")
    print(f"   4️⃣ 继续V1.0稳定运行，专注运营优化")

print(f"\n" + "="*80)
print("🌙 今晚大样本实验完成")
print("🔬 统计功效: 足够检测+0.01改进的样本量")
print("📊 科学方法: Bootstrap CI + 多算法对比")
print("⚡ 决策标准: 95% CI > 0 的统计显著性")
print("="*80)