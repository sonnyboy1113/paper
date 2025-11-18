import os
import math
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import r2_score
from  sklearn.cluster import k_means
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
dataset.info()
#删除包含缺失值的行
#dataset.dropna(inplace=True)
# 进行归一化
columns = ['Corn','TR','IR','ER','ine','FP.CFI','CFD','GPR','EPU','corn']
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
        data = X.iloc[i:i + seq_len]
        label = y.iloc[i + seq_len]
        features.append(data)
        targets.append(label)
    return np.array(features), np.array(targets)
# 构造训练特征数据集
train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
# 构造测试特征数据集
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)
# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=100, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)
# 训练批数据
train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
# 测试批数据
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# 模型
model = tf.keras.Sequential([
    SimpleRNN(100, input_shape=(5, 9), return_sequences=True),
    Dropout(0.01),
    SimpleRNN(100),
    Dropout(0.01),
    Dense(1)])
model.compile(optimizer='adam',loss='mse')
model.fit(train_batch_dataset, epochs=200, validation_data=test_batch_dataset,validation_freq=1)

#预测
test_preds = model.predict(test_dataset, verbose=2)

# 计算指标
print('----'*180)
print("r^2 值为：", r2_score(test_labels, test_preds))
print("MAE:", mean_absolute_error(test_labels,  test_preds))
print("MSE:", mean_squared_error(test_labels, test_preds))
print("RMSE:", sqrt(mean_squared_error(test_labels, test_preds)))
print('MAPE:', np.mean(np.abs((test_preds-test_labels)/test_labels)))

#导出预测值
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
    predict = pd.DataFrame(test_preds)
    predict.to_csv(results_directory + f"RNN-y_predict.csv", header=["y_predict"])

# 反归一化
test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1))
test_preds = scaler.inverse_transform(test_preds.reshape(-1, 1))

# 绘制 预测与真值结果
plt.figure(figsize=(7, 4))
plt.plot(test_labels, label="True value")
plt.plot(test_preds, label="Pred value")
plt.title("RNN",fontsize=16)
plt.xlabel("number of days",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.legend(loc='best',fontsize=16)
plt.tick_params(labelsize=16)
plt.show(block=True)