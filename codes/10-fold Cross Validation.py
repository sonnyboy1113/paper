# from sklearn.model_selection import KFold
#
# # 初始化MSE、RMSE、MAE、MAPE
# mse_list, mae_list, rmse_list, mape_list = [], [],  [], []
# kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# cvscores = []
# for train, test in kfold.split(X, y):
#     model = create_model()
#     X_train, X_test = X[train], X[test]
#     y_train, y_test = y[train], y[test]
#     model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), batch_size=32)
#     y_predict = model.predict(X_test)
#     mse_list.append(mean_squared_error(y_test, y_predict))
#     mae_list.append(mean_absolute_error(y_test, y_predict))
#     rmse_list.append(sqrt(mean_squared_error(y_test, y_predict)))
#     mape_list.append(np.mean(np.abs((y_test - y_predict) / y_test)) * 100)
#
# # 输出平均MSE、RMSE、MAE、MAPE
# print("MAE:", np.mean(mae_list))
# print("MSE:", np.mean(mse_list))
# print("RMSE:", np.mean(rmse_list))
# print("MAPE:", np.mean(mape_list))

#example
import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.cluster import k_means
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import  Input,Dropout,Dense,GRU,LSTM
from tensorflow.keras import Sequential, layers, utils, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings
warnings.filterwarnings('ignore')
#%%
# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
dataset.info()
# 进行归一化
columns = ['Corn','TR','IR','ER','ine','FP.CFI','CFD','GPR','EPU','corn']
for col in columns:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))
# 特征数据集
X = dataset.drop(columns=['Corn'], axis=1)
# 标签数据集
y = dataset['Corn']
# 将数据和标签分割成步长为N的序列
steps = 10
X  = [X[i:i+steps] for i in range(len(X)-steps)]
y = y[steps:]
# 转换成numpy数组
X = np.asarray(X)
y = np.asarray(y)
# 模型
def create_model():
    model = Sequential([
        LSTM(64, input_shape=(10, 9), return_sequences=True),
        Dropout(0.5),
        LSTM(16, input_shape=(10, 9), return_sequences=False),
        Dropout(0.5),
      Dense(1, activation='relu')])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
# 初始化MSE、RMSE、MAE、MAPE
mse_list, mae_list, rmse_list,mape_list = [], [],  [],[]
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cvscores = []
for train, test in kfold.split(X, y):
    model = create_model()
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), batch_size=32)
    y_predict = model.predict(X_test)
    mse_list.append(mean_squared_error(y_test, y_predict))
    mae_list.append(mean_absolute_error(y_test, y_predict))
    rmse_list.append(sqrt(mean_squared_error(y_test, y_predict)))
    mape_list.append(np.mean(np.abs((y_test - y_predict) / y_test)) * 100)
# 输出平均MSE、MAE、MAPE
print("MAE:", np.mean(mae_list))
print("MSE:", np.mean(mse_list))
print("RMSE:", np.mean(rmse_list))
print("MAPE:", np.mean(mape_list))