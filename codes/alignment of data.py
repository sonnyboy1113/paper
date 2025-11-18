import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 1. 读取CSV文件并进行日期对齐（如前所述）
data = pd.read_csv('alignment.csv')
data['date1'] = pd.to_datetime(data['date1'], dayfirst=True)

data['date2'] = pd.to_datetime(data['date2'], dayfirst=True)

data['date1'] = pd.to_datetime(data['date1'])
data['date2'] = pd.to_datetime(data['date2'])
series1 = pd.Series(data['price1'].values, index=data['date1'])
series2 = pd.Series(data['price2'].values, index=data['date2'])
aligned_series1, aligned_series2 = series1.align(series2, join='outer')

# 2. 创建对齐后的DataFrame
aligned_data = pd.DataFrame({
    'date': aligned_series1.index,
    'price1': aligned_series1.values,
    'price2': aligned_series2.values
})
# 删除'price1'和'price2'列均为NaN的行
aligned_data.dropna(subset=['price1', 'price2'], how='all', inplace=True)

print(aligned_data)
# 7. 保存预处理后的数据到新的CSV文件
aligned_data.to_csv('preprocessed_time_series_data.csv', index=False)
