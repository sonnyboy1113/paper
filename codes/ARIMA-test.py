from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #ACF与PACF
from statsmodels.tsa.arima.model import ARIMA #ARIMA模型
from statsmodels.graphics.api import qqplot  #qq图
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')


dataset = pd.read_csv('Corn-ARIMA.csv')
df = dataset
# 更改列名
#df.rename(columns={'Date':'deal_data', 'Corn':'time_data'}, inplace = True)
# 设置索引
df.set_index(['Date'], inplace=True)

data = df
data.info()


# 绘制时序图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 绘图
data.plot()
# 图片展示

plt.show()

# 绘制自相关图
plot_acf(data).show()
# 绘制偏自相关图
plot_pacf(data).show()
#平稳性检测

print(u'原始序列的ADF检验结果为：', ADF(data[u'Corn']))

tmp_data = data.diff().dropna()  # 一阶差分并去空列
D_data = tmp_data.diff().dropna()  # 二阶差分
tmp_data.columns = [u'差分']  # 取列名
D_data.columns = [u'差分']
# 时序图
D_data.plot()
plt.show()
# 自相关图
plot_acf(D_data).show()
# 偏自相关图
plot_pacf(D_data).show()

print(u'一阶差分序列的ADF检验结果为：', ADF(tmp_data[u'差分']))  # 平稳性检测
print(u'二阶差分序列的ADF检验结果为：', ADF(D_data[u'差分']))  # 平稳性检测
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值

AIC = sm.tsa.stattools.arma_order_select_ic(D_data, max_ar=4, max_ma=4, ic='aic')['aic_min_order']
# BIC
BIC = sm.tsa.stattools.arma_order_select_ic(D_data, max_ar=4, max_ma=4, ic='bic')['bic_min_order']
print('---AIC与BIC准则定阶---')
print('the AIC is{}\nthe BIC is{}\n'.format(AIC, BIC), end='')
p = BIC[0]
q = BIC[1]
diff_num = 2

model = ARIMA(data, order=(p, diff_num, q)).fit()  # 建立ARIMA(p, diff+num, q)模型
print('模型报告为：\n', model.summary())
forecast_num = 10#设置预测步数
print("预测结果：\n",model.forecast(forecast_num))


print("预测结果(详细版)：\n")
forecast = model.get_forecast(steps=forecast_num)
table = pd.DataFrame(forecast.summary_frame())
print(table)

def Model_checking(model):
    # 残差检验:检验残差是否服从正态分布，画图查看，然后检验
    # 绘制残差图
    model.resid.plot(figsize=(10, 3))
    plt.title("残差图")
    plt.show()

    print('------------残差检验-----------')
    # model.resid：残差 = 实际观测值 – 模型预测值
    print(stats.normaltest(model.resid))

    # QQ图看正态性
    qqplot(model.resid, line="q", fit=True)
    plt.title("Q-Q图")
    plt.show()
    # 绘制直方图
    plt.figure()
    plt.hist(model.resid, bins=50)
    plt.title("Histogram")
    plt.show()

    # 进行Jarque-Bera检验:判断数据是否符合总体正态分布
    jb_test = sm.stats.stattools.jarque_bera(model.resid)
    print("==================================================")
    print('------------Jarque-Bera检验-----------')
    print('Jarque-Bera test:')
    print('JB:', jb_test[0])
    print('p-value:', jb_test[1])
    print('Skew:', jb_test[2])
    print('Kurtosis:', jb_test[3])

    # 残差序列自相关：残差序列是否独立
    print('------DW检验:残差序列自相关----')
    print(sm.stats.stattools.durbin_watson(model.resid.values))

Model_checking(model)
