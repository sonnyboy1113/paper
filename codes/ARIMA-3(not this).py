from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #ACF与PACF
from statsmodels.tsa.arima.model import ARIMA #ARIMA模型
from scipy import stats
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import itertools
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
Cornfutures = pd.read_csv('Corn-ARIMA.csv',index_col = 'Date',parse_dates=['Date'])

#Cornfutures.index = pd.to_datetime(Cornfutures.index)
sub = Cornfutures['2018-03':'2023-02']['Corn']
train = sub.loc['2018-03':'2022-03']
test = sub.loc['2022-04':'2023-02']
plt.figure(figsize=(10,10))
print(train)
plt.plot(train)
plt.title("Corn")
plt.show()

Cornfutures['Corn_diff_1'] = Cornfutures['Corn'].diff(1)
Cornfutures['Corn_diff_2'] = Cornfutures['Corn_diff_1'].diff(1)
fig = plt.figure(figsize=(20,6))
ax1 = fig.add_subplot(131)
ax1.plot(Cornfutures['Corn'])
ax2 = fig.add_subplot(132)
ax2.plot(Cornfutures['Corn_diff_1'])
ax3 = fig.add_subplot(133)
ax3.plot(Cornfutures['Corn_diff_2'])
plt.show()
#分图显示
# plt.figure(figsize=(10,10))
# plt.plot(Cornfutures['Corn_diff_1'])
# plt.show()
# plt.figure(figsize=(10,10))
# plt.plot(Cornfutures['Corn_diff_2'])
# plt.show()


fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
plt.show()

# 遍历，寻找适宜的参数


p_min = 0
d_min = 0
q_min = 0
p_max = 5
d_max = 0
q_max = 5

# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                           columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

for p, d, q in itertools.product(range(p_min, p_max + 1),
                                 range(d_min, d_max + 1),
                                 range(q_min, q_max + 1)):
    if p == 0 and d == 0 and q == 0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue

    try:
        model = sm.tsa.ARIMA(train, order=(p, d, q),
                             # enforce_stationarity=False,
                             # enforce_invertibility=False,
                             )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 cmap="Purples"
                 )
ax.set_title('BIC')
plt.show()

#results_bic.stack().idxmin()
train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)

print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

model = sm.tsa.ARIMA(sub, order=(0, 2, 0))
results = model.fit()
resid = results.resid #赋值
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40)
plt.show()

predict_sunspots = results.predict(dynamic=False)
#predict_sunspots = predict_sunspots.iloc[1:]
print(predict_sunspots)

#查看测试集的时间序列与数据（只包含测试集）
plt.figure(figsize=(12,6))
plt.plot(train,label="真实值")
plt.xticks(rotation=45)#旋转45度
plt.plot(predict_sunspots,label="预测值")
plt.legend(loc='best',fontsize=16)
plt.show()

#绘图
fig,ax =plt.subplots(figsize=(12,6))
ax = sub.plot(ax=ax)
#预测数据
predict_sunspots.plot(ax=ax)
plt.show()


print('----'*180)
print("MAE:", mean_absolute_error(train, predict_sunspots))
print("MSE:", mean_squared_error(train, predict_sunspots))
print("RMSE:", sqrt(mean_squared_error(train, predict_sunspots)))
print('MAPE:', np.mean(np.abs((predict_sunspots-train)/train)))