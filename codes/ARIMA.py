from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from scipy import stats
import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')

import warnings

warnings.filterwarnings("ignore")

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 模型检测
def Model_checking(model) -> None:
    print('------------残差检验-----------')
    print(stats.normaltest(model.resid))
    qqplot(model.resid, line="q", fit=True)
    plt.title("Q-Q图")
    plt.show()

    plt.figure()
    plt.hist(model.resid, bins=50)
    plt.title("Histogram")
    plt.show()

    jb_test = sm.stats.stattools.jarque_bera(model.resid)
    print("==================================================")
    print('------------Jarque-Bera检验-----------')
    print('JB:', jb_test[0])
    print('p-value:', jb_test[1])
    print('Skew:', jb_test[2])
    print('Kurtosis:', jb_test[3])

    print("==================================================")
    print('------DW检验:残差序列自相关----')
    print(sm.stats.stattools.durbin_watson(model.resid.values))


# 计算误差指标
def calculate_metrics(test: pd.Series, forecast: pd.Series) -> None:
    # 确保没有缺失值
    test = test.dropna()
    forecast = forecast[test.index]  # 只取测试集对应的预测值

    if len(test) == 0 or len(forecast) == 0:
        print("无法计算指标，测试集或预测集为空。")
        return

    mae = np.mean(np.abs(test - forecast))
    mse = np.mean((test - forecast) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")


# 计算时序序列模型
def cal_time_series(data, forecast_num=3) -> None:
    # 时间序列数据分割
    train_size = int(len(data) * 0.9)  # 90%作为训练集
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # 绘制时序图
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train[u'Corn'], label='训练集', color='blue')
    plt.plot(test.index, test[u'Corn'], label='测试集', color='orange')
    plt.title('训练集与测试集')
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置最多显示6个标签
    plt.legend()
    plt.tight_layout()  # 防止标签重叠
    plt.show()

    # ADF检验
    original_ADF = ADF(train[u'Corn'])
    print(u'原始序列的ADF检验结果为：', original_ADF)

    # 差分处理
    diff_num = 1
    diff_data = train
    ADF_p_value = ADF(train[u'Corn'])[1]
    while ADF_p_value > 0.01:
        diff_data = diff_data.diff(periods=1).dropna()
        diff_num += 1
        ADF_result = ADF(diff_data[u'Corn'])
        ADF_p_value = ADF_result[1]
        print("ADF_p_value:", ADF_p_value)
        print(u'{diff_num}差分的ADF检验结果为：'.format(diff_num=diff_num), ADF_result)

    # 模型构建
    model = ARIMA(train, order=(0, diff_num, 1)).fit()
    print('模型报告为：\n', model.summary())

    # 预测
    forecast = model.get_forecast(steps=len(test))
    forecast_index = test.index
    forecast_mean = forecast.predicted_mean

    # 确保预测值的索引与测试集索引一致
    forecast_mean.index = forecast_index

    # 可视化预测结果
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train[u'Corn'], label='训练集', color='blue')
    plt.plot(test.index, test[u'Corn'], label='测试集', color='orange')
    plt.plot(forecast_index, forecast_mean, label='预测值', color='green')

    plt.fill_between(forecast_index,
                     forecast.conf_int().iloc[:, 0],
                     forecast.conf_int().iloc[:, 1],
                     color='k', alpha=.15)

    plt.title('预测结果')
    plt.xlabel('Date')
    plt.ylabel('Corn Price')
    plt.xticks(rotation=45)  # 旋转X轴标签
    # 使用 MaxNLocator 设置最多显示的标签数量
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置最多显示6个标签
    plt.legend()
    plt.tight_layout()  # 防止标签重叠
    plt.show()

    # 计算评价指标
    calculate_metrics(test[u'Corn'], forecast_mean)


    # 模型检查
    Model_checking(model)


if __name__ == '__main__':
    dataset = pd.read_csv('Corn-ARIMA.csv')
    print(dataset.columns)
    df = dataset
    df.set_index(['Date'], inplace=True)

    data = df
    data.info()
    cal_time_series(df, 7)  # 模型调用
