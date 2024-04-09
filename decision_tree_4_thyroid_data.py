import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score

# 读取扩充后的数据集
expanded_file_path = "C:/Users/linyiwu\\Desktop\\计算方法实验\\lin_-/glass+identification\\my_using_data/smote-glasses-data.csv"
expanded_thyroid_data = pd.read_csv(expanded_file_path)

# 划分特征和标签
X = expanded_thyroid_data.drop('Class', axis=1)
y = expanded_thyroid_data['Class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化决策树分类器
clf = DecisionTreeClassifier(random_state=42, criterion='entropy')

# 训练模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

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
# f1_measure = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
print("F1-measure:", f1_measure)

# 显示分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))
