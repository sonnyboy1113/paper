import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
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
x = dataset.drop(columns=['Corn'], axis=1)
# 标签数据集
y = dataset['Corn']
# build a dataset
seq_length=5
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY) * 0.8)
test_size = len(dataY) - train_size
X_train, X_test = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
y_train, y_test = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])
na, nb, nc = X_train.shape
X_train = X_train.reshape(na, nb*nc)
na, nb, nc = X_test.shape
X_test = X_test.reshape(na, nb*nc)
#建立bp模型
model = Sequential()
model.add(Dense(16,input_dim=50,kernel_initializer='uniform'))
model.add(Activation('relu'))
model.add(Dense(4,kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(1))  #输出层
model.compile(loss='mean_squared_error', optimizer='Adam')
model.fit(X_train, y_train, epochs = 200, batch_size = 32)

#在测试集上的预测
y_test_predict=model.predict(X_test)
y_test_predict=y_test_predict[:,0]

# 计算指标
print('----'*180)
print("r^2 值为：", r2_score(y_test, y_test_predict))
print("MAE:", mean_absolute_error(y_test, y_test_predict))
print("MSE:", mean_squared_error(y_test, y_test_predict))
print("RMSE:", sqrt(mean_squared_error(y_test, y_test_predict)))
print('MAPE:', np.mean(np.abs((y_test_predict-y_test)/y_test)))

#导出预测值
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
    predict = pd.DataFrame(y_test_predict)
    predict.to_csv(results_directory + f"BP-y_predict.csv", header=["y_predict"])
#反归一化处理
test_labels = scaler.inverse_transform(y_test.reshape(-1, 1))
test_preds = scaler.inverse_transform(y_test_predict.reshape(-1, 1))

# 绘制预测与真值结果
plt.figure(figsize=(7, 4))
plt.plot(test_labels, label="True value")
plt.plot(test_preds, label="Pred value")
plt.title("BP",fontsize=16)
plt.xlabel("number of days",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.legend(loc='best',fontsize=16)
plt.tick_params(labelsize=16)
plt.show(block=True)

