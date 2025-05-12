import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# 读取数据
data = pd.read_csv("C:/Users/Dell/Desktop/E-commerce04.csv")

# 提取用户 ID 和购买历史
user_ids = data['Customer ID'].tolist()
purchase_history = data['Purchase History'].tolist()

# 收集所有商品类别
all_product_categories = set()
for history in purchase_history:
    purchases = eval(history)
    for purchase in purchases:
        all_product_categories.add(purchase['Product Category'])

# 创建商品-用户交互矩阵
interaction_matrix = pd.DataFrame(index=user_ids, columns=list(all_product_categories), dtype=float)
interaction_matrix = interaction_matrix.fillna(0)

# 填充矩阵元素，以评分作为元素
for i, user_id in enumerate(user_ids):
    purchases = eval(purchase_history[i])
    for purchase in purchases:
        category = purchase['Product Category']
        rating = purchase['Product Review']['Rating']
        if category in interaction_matrix.columns:
            interaction_matrix.loc[user_id, category] = rating

# 划分为训练集和验证集
train_data, validation_data = train_test_split(data, test_size=0.3, random_state=42)

# 为验证集用户随机移除50%的购买记录
for index, row in validation_data.iterrows():
    purchases = eval(row['Purchase History'])
    if purchases:  # 确保购买记录不为空
        # 随机移除50%的购买记录
        purchases_to_keep = np.random.choice(purchases, size=int(len(purchases) * 0.5), replace=False)
        validation_data.at[index, 'Purchase History'] = str(purchases_to_keep.tolist())


# 使用KNN算法找到相似用户
def find_similar_users(user_id, interaction_matrix, n_neighbors=5):
    # 提取用户交互矩阵中的评分数据
    user_ratings = interaction_matrix.loc[user_id].values.reshape(1, -1)

    # 使用KNN算法找到相似用户
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(interaction_matrix.fillna(0).values)

    distances, indices = knn.kneighbors(user_ratings)

    # 获取相似用户的索引
    similar_user_indices = indices[0][1:]  # 排除自身

    # 将索引映射回用户ID
    similar_users = [interaction_matrix.index[i] for i in similar_user_indices]

    return similar_users


# 为验证集用户生成推荐列表
def generate_recommendations(validation_data, interaction_matrix, top_n=3):
    recommendations = {}
    for user_id in validation_data['Customer ID']:
        similar_users = find_similar_users(user_id, interaction_matrix)

        # 收集相似用户的购买记录
        recommended_items = {}
        for similar_user in similar_users:
            if similar_user in interaction_matrix.index:
                similar_user_ratings = interaction_matrix.loc[similar_user].dropna()
                for item in similar_user_ratings.index:
                    if pd.notna(similar_user_ratings[item]):
                        if item in recommended_items:
                            recommended_items[item] += 1
                        else:
                            recommended_items[item] = 1

        # 获取用户已经购买的商品（移除后的购买记录）
        purchases = eval(validation_data[validation_data['Customer ID'] == user_id]['Purchase History'].values[0])
        purchased_items = set(purchase['Product Category'] for purchase in purchases)

        # 剔除用户已经购买的商品
        recommended_items = {item: count for item, count in recommended_items.items() if item not in purchased_items}

        # 将推荐商品按出现次数从高到低排序
        sorted_recommended_items = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)

        # 提取商品名称并限制推荐列表的大小
        sorted_recommended_items = [item for item, count in sorted_recommended_items][:top_n]

        recommendations[user_id] = sorted_recommended_items

    return recommendations


# 计算精确率、召回率和F1分数
def calculate_metrics(recommendations, validation_data):
    precision_total = 0
    recall_total = 0
    total_users = len(recommendations)

    for user_id in recommendations:
        recommended_items = set(recommendations[user_id])
        actual_items = set()

        # 获取该用户的真实购买记录（完整的购买历史）
        original_purchases = eval(data[data['Customer ID'] == user_id]['Purchase History'].values[0])
        for purchase in original_purchases:
            actual_items.add(purchase['Product Category'])

        # 计算交集
        intersection = recommended_items.intersection(actual_items)

        # 计算精确率和召回率
        precision = len(intersection) / len(recommended_items) if len(recommended_items) > 0 else 0
        recall = len(intersection) / len(actual_items) if len(actual_items) > 0 else 0

        precision_total += precision
        recall_total += recall

    # 计算平均精确率和召回率
    avg_precision = precision_total / total_users
    avg_recall = recall_total / total_users

    # 计算F1分数
    f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return avg_precision, avg_recall, f1


# 生成用户相似度矩阵
def generate_user_similarity_matrix(interaction_matrix):
    # 填充缺失值为 0
    filled_matrix = interaction_matrix.fillna(0)
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(filled_matrix)
    # 转换为 DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=interaction_matrix.index, columns=interaction_matrix.index)
    return similarity_df


# 生成推荐列表
recommendations = generate_recommendations(validation_data, interaction_matrix)

# 打印调试信息
print("推荐列表示例:")
for user_id, items in list(recommendations.items())[:5]:
    print(f"用户 {user_id} 的推荐列表: {items}")

# 计算指标
precision, recall, f1 = calculate_metrics(recommendations, validation_data)

# 输出结果
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# 可视化用户-商品类别交互矩阵
def visualize_interaction_matrix(interaction_matrix):
    # 使用PCA降维
    pca = PCA(n_components=2)
    interaction_matrix_filled = interaction_matrix.fillna(0)
    user_embeddings = pca.fit_transform(interaction_matrix_filled)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(user_embeddings[:, 0], user_embeddings[:, 1], alpha=0.5)
    plt.title('User Embeddings using PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


# 可视化推荐的准确性
def visualize_recommendation_accuracy(recommendations, validation_data, interaction_matrix, data):
    # 收集实际购买类别和推荐类别
    actual_categories = []
    recommended_categories = []

    for user_id in recommendations:
        # 获取实际购买记录
        actual_purchases = eval(data[data['Customer ID'] == user_id]['Purchase History'].values[0])
        actual_categories.extend([purchase['Product Category'] for purchase in actual_purchases])

        # 获取推荐记录
        recommended_categories.extend(recommendations[user_id])

    # 统计频次
    actual_counts = pd.Series(actual_categories).value_counts().sort_index()
    recommended_counts = pd.Series(recommended_categories).value_counts().sort_index()

    # 绘制对比图
    plt.figure(figsize=(12, 8))

    # 实际购买类别分布
    ax1 = plt.subplot(2, 1, 1)
    actual_counts.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title('Actual Purchase Distribution')
    ax1.set_ylabel('Frequency')

    # 推荐类别分布
    ax2 = plt.subplot(2, 1, 2)
    recommended_counts.plot(kind='bar', color='orange', ax=ax2)
    ax2.set_title('Recommended Distribution')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# 执行可视化
visualize_interaction_matrix(interaction_matrix)
visualize_recommendation_accuracy(recommendations, validation_data, interaction_matrix, data)