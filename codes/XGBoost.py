import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
dataset.info()

# 删除包含缺失值的行
# dataset.dropna(inplace=True)

# 进行归一化
columns = ['Corn', 'TR', 'IR', 'ER', 'ine', 'FP.CFI', 'CFD', 'GPR', 'EPU', 'corn']
for col in columns:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))

# 特征数据集
X = dataset.drop(columns=['Corn'], axis=1)
# 标签数据集
y = dataset['Corn']

# 数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)

# 构造特征数据集
def create_dataset(X, y, seq_len=5):
    features = []
    targets = []
    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i + seq_len]  # 序列数据
        label = y.iloc[i + seq_len]  # 标签数据
        features.append(data.values.flatten())  # 展平为二维特征矩阵
        targets.append(label)
    return np.array(features), np.array(targets)

# 构造训练特征数据集
train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
# 构造测试特征数据集
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)

# 训练 XGBoost 模型
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.15, max_depth=5, random_state=42)
xgb_model.fit(train_dataset, train_labels)

# 预测
test_preds = xgb_model.predict(test_dataset)

# 计算指标
print('----' * 180)
print("r^2 值为：", r2_score(test_labels, test_preds))
print("MAE:", mean_absolute_error(test_labels, test_preds))
print("MSE:", mean_squared_error(test_labels, test_preds))
print("RMSE:", sqrt(mean_squared_error(test_labels, test_preds)))
print('MAPE:', np.mean(np.abs((test_preds - test_labels) / test_labels)))

# 导出预测值
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
predict = pd.DataFrame(test_preds)
predict.to_csv(results_directory + "XGBoost-y_predict.csv", header=["y_predict"])

# 反归一化
test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1))
test_preds = scaler.inverse_transform(test_preds.reshape(-1, 1))

# 绘制预测与真值结果
plt.figure(figsize=(7, 4))
plt.plot(test_labels, label="True value")
plt.plot(test_preds, label="Pred value")
plt.title("XGBoost", fontsize=16)
plt.xlabel("number of days", fontsize=16)
plt.ylabel("Price", fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.tick_params(labelsize=16)
plt.show(block=True)
