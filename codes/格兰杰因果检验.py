import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# 从CSV文件读取数据
file_path = 'Corn-new.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 假设CSV文件中有9列，选择其中两列进行分析
# 例如，选择第1列和第2列，列名分别为 'Column1' 和 'Column2'
# 请根据实际列名替换

x_series = data['corn']  # 替换为实际的列名
y_series = data['ine']  # 替换为实际的列名
x_name ='corn'
y_name ='ine'
# 平稳性检验函数
def check_stationarity(series):
    result = adfuller(series.dropna())  # 处理缺失值
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return result[1] > 0.05  # 如果p-value大于0.05，则认为序列是非平稳的

# 对数据进行平稳性检验
print("检验X的平稳性:")
x_stationary = not check_stationarity(x_series)

print("\n检验Y的平稳性:")
y_stationary = not check_stationarity(y_series)

# 对非平稳序列进行差分
if not x_stationary:
    x_diff = x_series.diff().dropna()
    print("\n对X进行差分后的平稳性检验:")
    check_stationarity(x_diff)

if not y_stationary:
    y_diff = y_series.diff().dropna()
    print("\n对Y进行差分后的平稳性检验:")
    check_stationarity(y_diff)

# 准备用于格兰杰因果检验的数据
if not x_stationary and not y_stationary:
    gc_data = pd.DataFrame({'X_diff': x_diff, 'Y_diff': y_diff})
elif not x_stationary:
    gc_data = pd.DataFrame({'X_diff': x_diff, 'Y': y_series.dropna()})
elif not y_stationary:
    gc_data = pd.DataFrame({'X': x_series.dropna(), 'Y_diff': y_diff})
else:
    gc_data = pd.DataFrame({'X': x_series, 'Y': y_series})

# 确保数据没有缺失值
gc_data = gc_data.dropna()

# 格兰杰因果检验
print("\n进行格兰杰因果检验:")
granger_test = grangercausalitytests(gc_data, maxlag=4)  # maxlag可以根据需要调整

# 定义显著性水平
alpha = 0.1

# 打印格兰杰因果检验结果
for lag, test_results in granger_test.items():
    print(f"\nLag: {lag}")
    ssr_ftest_pvalue = test_results[0]['ssr_ftest'][1]  # 获取p值
    if ssr_ftest_pvalue is not None and ssr_ftest_pvalue < alpha:
        print(f'{x_name} is a Granger Cause of {y_name} (p-value: {ssr_ftest_pvalue:.4f})')
    else:
        print(f'{x_name} is not a Granger Cause of {y_name} (p-value: {ssr_ftest_pvalue:.4f})')