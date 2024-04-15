# 1.1 用线性插值法对Thyroid Disease Data Set数据集进行扩充
import pandas as pd
import numpy as np

# 读取数据文件
file_path = "C:/Users/linyiwu/Desktop/计算方法实验/lin_-/thyroid+disease/new-thyroid.data"
columns = ['Class', 'T3_resin', 'Total_Serum_thyroxin', 'Total_serum_triiodothyronine', 'TSH', 'Max_TSH_difference']
thyroid_data = pd.read_csv(file_path, header=None, names=columns)

# 选择需要扩充的少数类样本
minority_data = thyroid_data[(thyroid_data['Class'] == 2) | (thyroid_data['Class'] == 3)]

# 线性插值函数
def linear_interpolation(row, prev_row, next_row):
    interpolated_row = {}
    for col in row.index:
        if col != 'Class':
            interpolated_row[col] = (prev_row[col] + next_row[col]) / 2.0
    return pd.Series(interpolated_row)

# 对少数类样本进行线性插值扩充
new_samples = []
for index, row in minority_data.iterrows():
    if index > 0 and index < len(minority_data) - 1:
        prev_row = minority_data.iloc[index - 1]
        next_row = minority_data.iloc[index + 1]
        interpolated_sample = linear_interpolation(row, prev_row, next_row)
        interpolated_sample['Class'] = row['Class']
        new_samples.append(interpolated_sample)

print(new_samples)

# 将新样本添加到原始数据集中
expanded_thyroid_data = pd.concat([thyroid_data, pd.DataFrame(new_samples)], ignore_index=True)

# 扩充后的数据集保存到新文件中
expanded_file_path = "C:/Users/linyiwu/Desktop/计算方法实验/lin_-/thyroid+disease/my_using_data/expanded-thyroid-data.csv"
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
