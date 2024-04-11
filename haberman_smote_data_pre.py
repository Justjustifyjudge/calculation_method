import pandas as pd
from imblearn.over_sampling import SMOTE

# 读取数据
data = pd.read_csv(r"lin_-\haberman+s+survival\haberman.data", header=None)

# 处理缺失值
data.fillna(method='ffill', inplace=True)

# 添加列名
data.columns = ['Age', 'Year', 'Nodes', 'Survived']

# 分离特征和标签
X = data[['Age', 'Year', 'Nodes']]
y = data['Survived']

# 应用 SMOTE 方法
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 将扩充后的数据保存到 .csv 文件中
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
resampled_data.to_csv(r"lin_-\haberman+s+survival\smote_data.csv", index=False)
data.to_csv(r"lin_-\haberman+s+survival\original_data.csv", index=False)