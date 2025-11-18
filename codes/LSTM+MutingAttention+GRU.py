import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from tensorflow.keras.layers import  Input,Dropout,Dense,GRU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from  sklearn.cluster import k_means
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential, layers, utils, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
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

# 数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)

# 构造特征数据集
def create_dataset(X, y, seq_len=5):
    features = []
    targets = []
    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i + seq_len]  # 序列数据
        label = y.iloc[i + seq_len]  # 标签数据
        features.append(data)
        targets.append(label)
    return np.array(features), np.array(targets)

# 构造训练和测试特征数据集
train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)

# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)

# 训练和测试批数据
train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# GRU 模型
gru_model = Sequential()
gru_model.add(GRU(64, input_shape=(5, 9), return_sequences=True))
gru_model.add(Dropout(0.01))
gru_model.add(GRU(16, input_shape=(5, 9), return_sequences=False))
gru_model.add(Dropout(0.01))
gru_model.add(Dense(1, activation='relu'))
gru_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
gru_model.fit(train_batch_dataset, epochs=200, validation_data=test_batch_dataset, verbose=0)

# LSTM 模型
lstm_model = Sequential([
    layers.LSTM(units=100, input_shape=(5, 9)),
    layers.Dense(4)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(train_batch_dataset, epochs=200, validation_data=test_batch_dataset, verbose=0)

# 使用GRU和LSTM模型进行预测
gru_preds = gru_model.predict(test_dataset, verbose=0)
gru_preds = gru_preds[:, 0]

lstm_preds = lstm_model.predict(test_dataset, verbose=0)
lstm_preds = lstm_preds[:, 0]

# 将GRU和LSTM的预测结果作为特征
stacked_features = np.column_stack((gru_preds, lstm_preds))

# 训练元模型（线性回归）
from sklearn.linear_model import LinearRegression
meta_model = LinearRegression()
meta_model.fit(stacked_features, test_labels)

# 使用元模型进行最终预测
final_preds = meta_model.predict(stacked_features)

# 计算指标
print('----' * 180)
print("r^2 值为：", r2_score(test_labels, final_preds))
print("MAE:", mean_absolute_error(test_labels, final_preds))
print("MSE:", mean_squared_error(test_labels, final_preds))
print("RMSE:", sqrt(mean_squared_error(test_labels, final_preds)))
print('MAPE:', np.mean(np.abs((final_preds - test_labels) / test_labels)))

# 导出预测值
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
predict = pd.DataFrame(final_preds)
predict.to_csv(results_directory + "Stacked-y_predict.csv", header=["y_predict"])

# 反归一化处理
test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1))
final_preds = scaler.inverse_transform(final_preds.reshape(-1, 1))

# 绘制预测与真值结果
plt.figure(figsize=(7, 4))
plt.plot(test_labels, label="True value")
plt.plot(final_preds, label="Pred value")
plt.title("Stacked Model (GRU + LSTM)", fontsize=16)
plt.xlabel("number of days", fontsize=16)
plt.ylabel("Price", fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.tick_params(labelsize=16)
plt.show(block=True)
