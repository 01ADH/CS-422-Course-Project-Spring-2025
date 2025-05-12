import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split

# 读取数据集
df = pd.read_csv(r"C:\Users\Dell\Desktop\E-commerce04.csv")


# 定义年龄分箱函数
def get_age_group(age):
    if age < 30:
        return '<30岁'
    elif 30 <= age <= 50:
        return '30-50岁'
    else:
        return '>50岁'


# 构建事务数据集
transactions = []
for idx, row in df.iterrows():
    # 提取用户属性
    gender = row['Gender']
    income_tier = row['Income Tier']
    region = row['Region']
    time_on_site_tier = row['Time on Site Tier']
    age_group = get_age_group(row['Age'])

    # 解析购买历史并去重
    purchase_cats = list({item['Product Category'] for item in json.loads(row['Purchase History'])})

    # 组合属性项和购买类别项
    transaction = [
        gender, income_tier, region, time_on_site_tier, age_group,
        *purchase_cats  # 展开购买类别项
    ]
    # 清理数据，确保所有数据都是字符串类型
    transaction = [str(item) for item in transaction if pd.notna(item)]
    transactions.append(transaction)

# 生成用户ID列表（假设用户ID唯一）
user_ids = df['Customer ID'].tolist()
# 划分数据集（80%训练，20%验证）
train_users, val_users = train_test_split(user_ids, test_size=0.2, random_state=42)

# 筛选训练集和验证集事务
train_transactions = [t for u, t in zip(user_ids, transactions) if u in train_users]
val_transactions = [t for u, t in zip(user_ids, transactions) if u in val_users]

# 初始化事务编码器
te = TransactionEncoder()
te_ary = te.fit_transform(train_transactions)
train_df = pd.DataFrame(te_ary, columns=te.columns_)

# 生成频繁项集（最小支持度设为5%）
frequent_itemsets = apriori(
    train_df,
    min_support=0.05,
    use_colnames=True,
    max_len=5  # 限制项集长度（前件+后件总项数）
)

# 计算规则指标（支持度、置信度、提升度）
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5  # 最小置信度50%
)

# 筛选规则：后件为单一购买类别，前件包含至少一个属性项
attribute_items = {'Male', 'Female', '<30岁', '30-50岁', '>50岁',
                   'Low', 'Medium', 'High', 'Very High',
                   'Region 1', 'Region 2', 'Region 3', 'Region 4',
                   'Region 5', 'Region 6', 'Region 7', 'Region 8',
                   'Short', 'Medium', 'Long', 'Very Long'}

valid_rules = rules[
    (rules['consequents'].apply(lambda x: len(x) == 1)) &  # 后件为单一类别
    (rules['antecedents'].apply(lambda x: not x.isdisjoint(attribute_items)))  # 前件包含属性项
    ].copy()

# 正确计算提升度
valid_rules['Lift'] = valid_rules['confidence'] / valid_rules['consequent support']

# 筛选有效规则（提升度>1）
final_rules = valid_rules[valid_rules['Lift'] > 1].sort_values(by='confidence', ascending=False)

print(f"总事务数（训练集）：{len(train_transactions)}")
print(f"生成规则数：{len(final_rules)}")
print("\nTop 5 有效规则：")
print(final_rules[['antecedents', 'consequents', 'Lift']].head())

# 按核心属性分组（如收入层级）减少搜索空间
for tier in df['Income Tier'].unique():
    tier_transactions = [t for t in train_transactions if tier in t]
    # 对每个分组单独应用Apriori算法

# 对验证集进行编码
val_te_ary = te.transform(val_transactions)  # 使用训练集的编码器
val_df = pd.DataFrame(val_te_ary, columns=te.columns_)


# 计算规则在验证集上的覆盖率
def rule_coverage(rule, val_df):
    antecedent = set(rule['antecedents'])
    consequent = set(rule['consequents'])
    antecedent_mask = val_df[list(antecedent)].all(axis=1)
    return val_df[antecedent_mask][list(consequent)].any(axis=1).mean()


final_rules['validation_coverage'] = final_rules.apply(
    lambda x: rule_coverage(x, val_df), axis=1
)


# 计算精确率、召回率、F1值
def calculate_precision_recall_f1(test_df, rules):
    precision_numerator = 0  # ∑|R(u)∩T(u)|
    precision_denominator = 0  # ∑|R(u)|
    recall_denominator = 0  # ∑|T(u)|

    for idx, row in test_df.iterrows():
        # 提取用户属性（前件候选）
        gender = row['Gender']
        income_tier = row['Income Tier']
        region = row['Region']
        time_on_site_tier = row['Time on Site Tier']
        age_group = get_age_group(row['Age'])
        user_attributes = {gender, income_tier, region, time_on_site_tier, age_group}

        # 解析真实购买类别 T(u)
        purchase_history = json.loads(row['Purchase History'])
        T_u = {item['Product Category'] for item in purchase_history}
        recall_denominator += len(T_u)

        # 生成推荐列表 R(u)：匹配所有前件是用户属性子集的规则
        R_u = set()
        for _, rule in rules.iterrows():
            antecedent = set(rule['antecedents'])
            if antecedent.issubset(user_attributes):
                # 使用 next(iter()) 从 frozenset 中提取唯一元素
                consequent_item = next(iter(rule['consequents']))
                R_u.add(consequent_item)

        precision_denominator += len(R_u)
        intersection = R_u & T_u
        precision_numerator += len(intersection)
    precision = precision_numerator / precision_denominator if precision_denominator != 0 else 0
    recall = precision_numerator / recall_denominator if recall_denominator != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
test_df = pd.read_csv(r"C:\Users\Dell\Desktop\E-commerce06.csv")
metrics = calculate_precision_recall_f1(test_df, final_rules)
print("\n推荐性能指标：")
print(f"精确率 (Precision): {metrics['Precision']:.4f}")
print(f"召回率 (Recall): {metrics['Recall']:.4f}")
print(f"F1值 (F1): {metrics['F1']:.4f}")
# ====================== 可视化部分 ======================
# 1. 关联规则可视化（支持度-置信度-提升度气泡图）
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=final_rules,
    x='support', y='confidence',
    size='Lift', alpha=0.7,
    palette='viridis'
)
plt.xlabel('支持度 (Support)')
plt.ylabel('置信度 (Confidence)')
plt.title('关联规则可视化（支持度 vs 置信度，气泡大小表示提升度）')
plt.legend(title='提升度 (Lift)', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()



# 3. 规则后件分布（购买类别推荐频率）
consequent_counts = final_rules['consequents'].apply(lambda x: next(iter(x))).value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=consequent_counts.values, y=consequent_counts.index, palette='plasma')
plt.xlabel('规则数量')
plt.ylabel('推荐商品类别')
plt.title('推荐商品类别分布')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 显示所有图表
plt.show()