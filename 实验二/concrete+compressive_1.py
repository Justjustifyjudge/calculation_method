import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_excel(r'lin_-\实验二\concrete+compressive+strength\Concrete_Data.xls')

# 准备数据
X = df.iloc[:, :-1]  # 输入特征
y = df.iloc[:, -1]   # 输出目标

# 使用5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 建立多变量线性模型并进行交叉验证
model = LinearRegression()
mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
avg_mse = -mse_scores.mean()

# 打印测试集平方误差的平均值
print(f'测试集平方误差的平均值（5折交叉验证）：{avg_mse:.2f}')

# 绘制结果图表
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), -mse_scores, marker='o', linestyle='--')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Mean Squared Error')
plt.grid(True)
plt.show()
