# ===================================================================
# 数据增强工具 - 为V1优化研究创建更大的测试数据集
# 目标: 生成至少100个多样化的查询样本用于算法优化
# ===================================================================

import json
import random
import numpy as np
from datetime import datetime

print("🔧 V1优化数据增强工具")
print("="*60)
print("🎯 目标: 生成100+多样化查询样本")
print("🌍 覆盖: 多个领域和查询类型")
print("="*60)

class DataAugmentationEngine:
    """数据增强引擎"""
    
    def __init__(self):
        self.domains = ['food', 'cocktails', 'alcohol', 'dining', 'beverages']
        self.base_queries = []
        self.enhanced_dataset = []
    
    def generate_food_queries(self, count=30):
        """生成食物相关查询"""
        food_terms = [
            'delicious', 'fresh', 'organic', 'gourmet', 'artisan',
            'homemade', 'seasonal', 'local', 'premium', 'authentic'
        ]
        
        food_types = [
            'pasta', 'salad', 'soup', 'pizza', 'sandwich',
            'burger', 'sushi', 'steak', 'seafood', 'dessert',
            'bread', 'cheese', 'vegetables', 'fruits', 'cake'
        ]
        
        queries = []
        for i in range(count):
            term = random.choice(food_terms)
            food = random.choice(food_types)
            query_templates = [
                f"{term} {food}",
                f"{food} with {term} ingredients",
                f"best {term} {food} recipe",
                f"{term} {food} photography",
                f"restaurant quality {term} {food}"
            ]
            query = random.choice(query_templates)
            
            # 生成候选项
            candidates = []
            for j in range(random.randint(2, 5)):
                score = random.uniform(0.6, 0.98)
                compliance = min(score + random.uniform(-0.1, 0.1), 1.0)
                candidates.append({
                    "id": f"food_{i}_{j}",
                    "title": f"{term.title()} {food} - {random.choice(['restaurant style', 'homemade', 'professional', 'gourmet', 'artisan'])}",
                    "score": round(score, 3),
                    "compliance_score": round(max(compliance, 0), 3),
                    "alt_description": f"High quality {term} {food} image"
                })
            
            queries.append({
                "query": query,
                "domain": "food",
                "candidates": candidates
            })
        
        return queries
    
    def generate_cocktail_queries(self, count=30):
        """生成鸡尾酒相关查询"""
        cocktail_terms = [
            'craft', 'premium', 'artisanal', 'classic', 'signature',
            'elegant', 'refreshing', 'sophisticated', 'creative', 'smooth'
        ]
        
        cocktail_types = [
            'mojito', 'martini', 'old fashioned', 'whiskey sour', 'negroni',
            'manhattan', 'daiquiri', 'margarita', 'gin fizz', 'cosmopolitan',
            'bloody mary', 'mai tai', 'piña colada', 'moscow mule', 'aperol spritz'
        ]
        
        queries = []
        for i in range(count):
            term = random.choice(cocktail_terms)
            cocktail = random.choice(cocktail_types)
            query_templates = [
                f"{term} {cocktail}",
                f"{cocktail} with {term} garnish",
                f"best {term} {cocktail} recipe",
                f"{term} {cocktail} photography",
                f"bar quality {term} {cocktail}"
            ]
            query = random.choice(query_templates)
            
            # 生成候选项
            candidates = []
            for j in range(random.randint(2, 5)):
                score = random.uniform(0.65, 0.95)
                compliance = min(score + random.uniform(-0.08, 0.12), 1.0)
                candidates.append({
                    "id": f"cocktail_{i}_{j}",
                    "title": f"{term.title()} {cocktail} - {random.choice(['bar style', 'craft recipe', 'premium', 'signature', 'classic'])}",
                    "score": round(score, 3),
                    "compliance_score": round(max(compliance, 0), 3),
                    "alt_description": f"Professional {term} {cocktail} cocktail"
                })
            
            queries.append({
                "query": query,
                "domain": "cocktails",
                "candidates": candidates
            })
        
        return queries
    
    def generate_alcohol_queries(self, count=20):
        """生成酒类相关查询"""
        alcohol_terms = [
            'premium', 'aged', 'single malt', 'vintage', 'rare',
            'craft', 'distillery', 'collection', 'boutique', 'artisan'
        ]
        
        alcohol_types = [
            'whiskey', 'wine', 'gin', 'vodka', 'rum',
            'bourbon', 'scotch', 'champagne', 'tequila', 'brandy'
        ]
        
        queries = []
        for i in range(count):
            term = random.choice(alcohol_terms)
            alcohol = random.choice(alcohol_types)
            query_templates = [
                f"{term} {alcohol}",
                f"{alcohol} {term} bottle",
                f"best {term} {alcohol} collection",
                f"{term} {alcohol} photography",
                f"luxury {term} {alcohol}"
            ]
            query = random.choice(query_templates)
            
            # 生成候选项
            candidates = []
            for j in range(random.randint(2, 4)):
                score = random.uniform(0.7, 0.97)
                compliance = min(score + random.uniform(-0.05, 0.08), 1.0)
                candidates.append({
                    "id": f"alcohol_{i}_{j}",
                    "title": f"{term.title()} {alcohol} - {random.choice(['distillery special', 'limited edition', 'collector grade', 'premium quality'])}",
                    "score": round(score, 3),
                    "compliance_score": round(max(compliance, 0), 3),
                    "alt_description": f"High-end {term} {alcohol} bottle"
                })
            
            queries.append({
                "query": query,
                "domain": "alcohol",
                "candidates": candidates
            })
        
        return queries
    
    def generate_dining_queries(self, count=20):
        """生成餐饮相关查询"""
        dining_contexts = [
            'restaurant', 'cafe', 'bistro', 'fine dining', 'casual dining',
            'brunch', 'dinner', 'lunch', 'breakfast', 'happy hour'
        ]
        
        dining_descriptors = [
            'elegant', 'cozy', 'modern', 'rustic', 'upscale',
            'casual', 'intimate', 'vibrant', 'trendy', 'classic'
        ]
        
        queries = []
        for i in range(count):
            context = random.choice(dining_contexts)
            descriptor = random.choice(dining_descriptors)
            query_templates = [
                f"{descriptor} {context}",
                f"{context} {descriptor} atmosphere",
                f"best {descriptor} {context} experience",
                f"{descriptor} {context} photography",
                f"{descriptor} {context} interior"
            ]
            query = random.choice(query_templates)
            
            # 生成候选项
            candidates = []
            for j in range(random.randint(2, 4)):
                score = random.uniform(0.68, 0.94)
                compliance = min(score + random.uniform(-0.07, 0.1), 1.0)
                candidates.append({
                    "id": f"dining_{i}_{j}",
                    "title": f"{descriptor.title()} {context} - {random.choice(['premium experience', 'stylish ambiance', 'perfect setting', 'memorable dining'])}",
                    "score": round(score, 3),
                    "compliance_score": round(max(compliance, 0), 3),
                    "alt_description": f"Beautiful {descriptor} {context} setting"
                })
            
            queries.append({
                "query": query,
                "domain": "dining",
                "candidates": candidates
            })
        
        return queries
    
    def create_enhanced_dataset(self):
        """创建增强数据集"""
        print("🏗️ 生成增强数据集...")
        
        # 生成各类查询
        food_queries = self.generate_food_queries(30)
        cocktail_queries = self.generate_cocktail_queries(30)
        alcohol_queries = self.generate_alcohol_queries(20)
        dining_queries = self.generate_dining_queries(20)
        
        # 合并所有查询
        all_queries = food_queries + cocktail_queries + alcohol_queries + dining_queries
        
        # 随机打乱
        random.shuffle(all_queries)
        
        self.enhanced_dataset = all_queries
        
        print(f"✅ 数据集生成完成:")
        print(f"   📊 总查询数: {len(all_queries)}")
        print(f"   🍽️ 食物查询: {len(food_queries)}")
        print(f"   🍸 鸡尾酒查询: {len(cocktail_queries)}")
        print(f"   🍷 酒类查询: {len(alcohol_queries)}")
        print(f"   🏪 餐厅查询: {len(dining_queries)}")
        
        return all_queries
    
    def save_enhanced_dataset(self, filename):
        """保存增强数据集"""
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_queries": len(self.enhanced_dataset),
                "domains": list(set(q['domain'] for q in self.enhanced_dataset)),
                "purpose": "V1 algorithm optimization research"
            },
            "inspirations": self.enhanced_dataset
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 数据集已保存到: {filename}")
        return filename

# 执行数据增强
augmenter = DataAugmentationEngine()
enhanced_queries = augmenter.create_enhanced_dataset()

# 保存增强数据集
enhanced_file = '/Users/guyan/computer_vision/computer-vision/data/input/enhanced_dataset.json'
augmenter.save_enhanced_dataset(enhanced_file)

# 统计信息
print(f"\n📈 数据集统计:")
domain_counts = {}
total_candidates = 0
for query in enhanced_queries:
    domain = query['domain']
    domain_counts[domain] = domain_counts.get(domain, 0) + 1
    total_candidates += len(query['candidates'])

for domain, count in domain_counts.items():
    print(f"   📊 {domain}: {count} 查询")

print(f"   🎯 总候选项: {total_candidates}")
print(f"   📏 平均候选项/查询: {total_candidates/len(enhanced_queries):.1f}")

print(f"\n✅ 数据增强完成")
print(f"🚀 现在可以重新运行优化算法进行更可靠的测试")
print(f"📁 数据文件: {enhanced_file}")