import pandas as pd
from imblearn.over_sampling import SMOTE

# 读取数据文件
file_path = "C:/Users/linyiwu/Desktop/计算方法实验/lin_-/glass+identification/glass.data"
columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
thyroid_data = pd.read_csv(file_path, header=None, names=columns)
glasses_file_path = "C:/Users/linyiwu/Desktop/计算方法实验/lin_-/glass+identification/my_using_data/row-glasses-data.csv"
thyroid_data.to_csv(glasses_file_path, index=False)
# 划分特征和标签
X = thyroid_data.drop('Class', axis=1)
y = thyroid_data['Class']

# 使用SMOTE方法对数据集进行扩充
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 将扩充后的数据重新组合为DataFrame
expanded_thyroid_data = pd.DataFrame(X_resampled, columns=X.columns)
expanded_thyroid_data['Class'] = y_resampled

# 扩充后的数据集保存到新文件中
expanded_file_path = "C:/Users/linyiwu/Desktop/计算方法实验/lin_-/glass+identification/my_using_data/smote-glasses-data.csv"
expanded_thyroid_data.to_csv(expanded_file_path, index=False)

# 统计各类样本数量
class_counts = expanded_thyroid_data['Class'].value_counts()

# 计算各类数据百分比
total_samples = len(expanded_thyroid_data)
class_percentages = (class_counts / total_samples) * 100

# 显示各类数据百分比
print("Class Counts:")
print(class_counts)
print("\nClass Percentages (%):")
print(class_percentages)
