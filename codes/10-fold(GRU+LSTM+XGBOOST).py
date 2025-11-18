import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import GRU, LSTM, Dropout, Dense
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print(dataset.info())

# 数据预处理
columns = ['Corn', 'TR', 'IR', 'ER', 'ine', 'FP.CFI', 'CFD', 'GPR', 'EPU', 'corn']
for col in columns:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))

# 特征数据集和标签数据集
X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

# 将数据和标签分割成步长为 10 的序列
steps = 10
X_seq = [X[i:i + steps] for i in range(len(X) - steps)]
y_seq = y[steps:]

# 转换成 numpy 数组
X_seq = np.asarray(X_seq)
y_seq = np.asarray(y_seq)

# 定义 GRU + LSTM 模型
def create_gru_lstm_model():
    model = Sequential([
        GRU(64, input_shape=(steps, X.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# 初始化 MSE、RMSE、MAE、MAPE
mse_list, mae_list, rmse_list, mape_list = [], [], [], []

# 10 折交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
for train, test in kfold.split(X_seq, y_seq):
    # 训练和测试数据
    X_train, X_test = X_seq[train], X_seq[test]
    y_train, y_test = y_seq[train], y_seq[test]

    # 训练 GRU + LSTM 模型
    gru_lstm_model = create_gru_lstm_model()
    gru_lstm_model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

    # 使用 GRU + LSTM 模型进行预测
    gru_lstm_preds = gru_lstm_model.predict(X_test).flatten()

    # 将 GRU + LSTM 的预测结果作为特征
    stacked_features = np.column_stack((gru_lstm_preds,))

    # 训练 XGBoost 模型
    meta_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    meta_model.fit(stacked_features, y_test)

    # 使用 XGBoost 模型进行最终预测
    final_preds = meta_model.predict(stacked_features)

    # 计算指标
    mse_list.append(mean_squared_error(y_test, final_preds))
    mae_list.append(mean_absolute_error(y_test, final_preds))
    rmse_list.append(sqrt(mean_squared_error(y_test, final_preds)))
    mape_list.append(np.mean(np.abs((final_preds - y_test) / y_test)) * 100)

# 输出平均 MSE、MAE、MAPE
print("MAE:", np.mean(mae_list))
print("MSE:", np.mean(mse_list))
print("RMSE:", np.mean(rmse_list))
print("MAPE:", np.mean(mape_list))
