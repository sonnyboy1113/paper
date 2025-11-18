import numpy as np
import pandas as pd
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.utils import *
#from keras.utils import multi_gpu_model
from keras import regularizers  # 正则化
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 假设数据已经准备好，存储在DataFrame中
data = pd.read_csv('Corn-new.csv')

# 提取日期和价格数据
dates = pd.to_datetime(data['Date'])
prices = data['Corn']

# 绘制时间序列图

plt.figure(figsize=(10, 5))
plt.plot(dates, prices, linestyle='-', color='b',label="Corn")
plt.legend(loc='best',fontsize=16)

# 添加标题和轴标签
plt.title('Corn Futures Price Time Series')
plt.xlabel('Date')
plt.ylabel('Price')

# 显示图表
plt.show()