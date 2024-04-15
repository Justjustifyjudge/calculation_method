import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载CSV文件，从第二行开始读取数据（第一行为列名），并使用分号分隔字段
df = pd.read_csv(r'C:\Users\linyiwu\Desktop\计算方法实验\lin_-\实验二\wine+quality\winequality-red.csv', sep=';', header=0)

# 将数据拆分为特征和目标变量
X = df.iloc[:, :-1]  # 输入特征
y = df.iloc[:, -1]   # 输出目标

# 设置不同的lambda参数进行实验
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # 正则化参数

mse_scores_with_regularization = []
mse_scores_without_regularization = []

for alpha in alphas:
    # 使用5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 建立带L2正则化的多变量线性模型并进行交叉验证
    model_with_regularization = Ridge(alpha=alpha)
    mse_scores_with_reg = cross_val_score(model_with_regularization, X, y, scoring='neg_mean_squared_error', cv=kf)
    mse_scores_with_regularization.append(-mse_scores_with_reg.mean())
    
    # 建立不带正则化的多变量线性模型并进行交叉验证
    model_without_regularization = Ridge(alpha=0)  # alpha=0表示无正则化
    mse_scores_without_reg = cross_val_score(model_without_regularization, X, y, scoring='neg_mean_squared_error', cv=kf)
    mse_scores_without_regularization.append(-mse_scores_without_reg.mean())

# 绘制结果图表
plt.figure(figsize=(10, 6))
plt.plot(alphas, mse_scores_with_regularization, marker='o', label='With Regularization')
plt.plot(alphas, mse_scores_without_regularization, marker='x', label='Without Regularization')
plt.xlabel('Lambda (Alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Regularization on Mean Squared Error')
plt.xscale('log')
plt.legend()
plt.grid(True)

# 保存图表为图片
plt.savefig(r'lin_-\实验二\photos\wine_table_and_chart.png')
plt.clf()
# 创建并保存表格图片
table_data = {
    'Lambda (Alpha)': alphas,
    'With Regularization': mse_scores_with_regularization,
    'Without Regularization': mse_scores_without_regularization
}
table_df = pd.DataFrame(table_data)
table = plt.table(cellText=table_df.values, colLabels=table_df.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # 调整表格大小
plt.axis('off')  # 隐藏坐标轴
plt.savefig(r'lin_-\实验二\photos\wine_red_2_table.png', bbox_inches='tight')
