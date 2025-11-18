import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 1. 读取CSV文件并进行日期对齐
data = pd.read_csv('alignment.csv')
data['date1'] = pd.to_datetime(data['date1'], dayfirst=True)
data['date2'] = pd.to_datetime(data['date2'], dayfirst=True)

series1 = pd.Series(data['price1'].values, index=data['date1'])
series2 = pd.Series(data['price2'].values, index=data['date2'])
aligned_series1, aligned_series2 = series1.align(series2, join='outer')

# 2. 创建对齐后的DataFrame
aligned_data = pd.DataFrame({
    'date': aligned_series1.index,
    'price1': aligned_series1.values,
    'price2': aligned_series2.values
})

# 3. 处理缺失值
aligned_data['price1'].ffill(inplace=True)
aligned_data['price2'].ffill(inplace=True)

# 4. 数据标准化
scaler = StandardScaler()
aligned_data[['price1', 'price2']] = scaler.fit_transform(aligned_data[['price1', 'price2']])

# 5. 异常值检测与处理
z_scores = np.abs(stats.zscore(aligned_data[['price1', 'price2']]))
threshold = 3
outliers = (z_scores > threshold).any(axis=1)
aligned_data = aligned_data[~outliers]

# 6. 数据转换（例如对数转换）
aligned_data['price1'] = np.log1p(aligned_data['price1'])
aligned_data['price2'] = np.log1p(aligned_data['price2'])
print(aligned_data)
# 7. 保存预处理后的数据到新的CSV文件
aligned_data.to_csv('preprocessed_time_series_data2.csv', index=False)