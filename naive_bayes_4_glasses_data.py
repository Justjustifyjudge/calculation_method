import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# 读取扩充后的数据集
expanded_file_path = "C:/Users/linyiwu\\Desktop\\计算方法实验\\lin_-/glass+identification\\my_using_data/smote-glasses-data.csv"
expanded_thyroid_data = pd.read_csv(expanded_file_path)

# 划分特征和标签
X = expanded_thyroid_data.drop('Class', axis=1)
y = expanded_thyroid_data['Class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化朴素贝叶斯分类器
gnb = GaussianNB()

# 训练模型
gnb.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = gnb.predict(X_test)

# 创建 DataFrame 存储预测结果和真实标签
results_df = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})

# 打印每一列的预测结果
print("Predicted Labels for Each Column:")
for col in results_df.columns:
    print(f"Column: {col}")
    print(results_df[col].value_counts())
    print()

# 计算 F1-measure
f1_measure = f1_score(y_test, y_pred, average='weighted')
print("F1-measure:", f1_measure)

# 显示分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()