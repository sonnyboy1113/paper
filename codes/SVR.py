
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
from sklearn.svm import SVR
from  sklearn.cluster import k_means
# from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, utils, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
dataset.info()
#删除包含缺失值的行
dataset.dropna(inplace=True)
# 进行归一化
columns = ['Corn','TR','IR','ER','ine','FP.CFI','CFD','GPR','EPU','corn']
for col in columns:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))
# 特征数据集
x = dataset.drop(columns=['Corn'], axis=1)
# 标签数据集
y = dataset['Corn']

# 分割训练数据和测试数据
seq_length=5
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY) * 0.8)
test_size = len(dataY) - train_size
x_train, x_test = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
y_train, y_test = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])
na, nb, nc = x_train.shape
x_train = x_train.reshape(na, nb*nc)
na, nb, nc = x_test.shape
x_test = x_test.reshape(na, nb*nc)

#模型
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)
poly_svr = SVR(kernel="rbf")
poly_svr.fit(x_train, y_train)

# 预测
poly_svr_y_predict = linear_svr.predict(x_test)

# 计算指标
print('----'*180)
print("r^2 值为：", r2_score(poly_svr_y_predict, y_test))
print("MAE:", mean_absolute_error(poly_svr_y_predict, y_test))
print("MSE:", mean_squared_error(poly_svr_y_predict, y_test))
print("RMSE:", sqrt(mean_squared_error(poly_svr_y_predict, y_test)))
print('MAPE:', np.mean(np.abs((y_test-poly_svr_y_predict)/poly_svr_y_predict)))

#导出预测值
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
    predict = pd.DataFrame(poly_svr_y_predict)
    predict.to_csv(results_directory + f"SVR-y_predict.csv", header=["y_predict"])

# 反归一化
test_labels = scaler.inverse_transform(poly_svr_y_predict.reshape(-1, 1))
test_preds = scaler.inverse_transform(y_test.reshape(-1, 1))

# 绘制预测与真值结果
plt.figure(figsize=(7, 4))
plt.plot(test_labels, label="True value")
plt.plot(test_preds, label="Pred value")
plt.title("SVR", fontsize = 16)
plt.xlabel("number of days", fontsize = 16)
plt.ylabel("Price", fontsize = 16)
plt.legend(loc='best', fontsize = 16)
plt.tick_params(labelsize=14)
plt.show(block=True)
