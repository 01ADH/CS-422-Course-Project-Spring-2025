import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/Dell/Desktop/E-commerce03.csv")
data['Annual Income'] = data['Annual Income'].fillna(data['Annual Income'].median())
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Time on Site'] = data['Time on Site'].fillna(data['Time on Site'].median())
data['Location'] = data['Location'].fillna(data['Location'].mode()[0])
data = data[(data['Age'] >= 18) & (data['Age'] <= 100)]
data['Income Tier'] = pd.cut(data['Annual Income'], bins=[0, 50000, 80000, 110000, np.inf],
                             labels=['Low', 'Medium', 'High', 'Very High'])
region_mapping = {
    'City P': 'Region 1', 'City N': 'Region 2', 'City V': 'Region 3', 'City W': 'Region 4',
    'City Q': 'Region 6', 'City F': 'Region 7', 'City B': 'Region 8',
    'City A': 'Region 1', 'City T': 'Region 2', 'City U': 'Region 3', 'City Z': 'Region 4',
    'City X': 'Region 5', 'City Y': 'Region 6', 'City L': 'Region 7', 'City M': 'Region 8',
    'City C': 'Region 1', 'City D': 'Region 2', 'City S': 'Region 3', 'City J': 'Region 4',
    'City O': 'Region 5', 'City E': 'Region 8'
}
data['Region'] = data['Location'].map(region_mapping)
data['Time on Site Tier'] = pd.cut(data['Time on Site'], bins=[0, 100, 200, 300, np.inf],
                                   labels=['Short', 'Medium', 'Long', 'Very Long'])

try:
    data.to_csv("C:/Users/Dell/Desktop/E-commerce04.csv", index=False)
    print("数据已成功保存到文件。")
except PermissionError:
    print("没有权限写入文件，请确保文件未被其他程序占用，并且你有足够的权限。")
except Exception as e:
    print(f"保存文件时出现其他错误: {e}")