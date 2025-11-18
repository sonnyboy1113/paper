import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 加载数据集
df = pd.read_csv('Oil-Gold.csv', parse_dates=['Date'], index_col=['Date'])
# 进行归一化
columns = ['Gold','Oil',  'GPR', 'MPU', 'EMV','EPU', 'CPU']
for col in columns:
    scaler = MinMaxScaler()
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
#绘制
cm = df.columns.tolist()
xcorr = df.corr()
mask = np.zeros_like(xcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
g = sns.heatmap(xcorr,
                annot=True,
                center=0.5,
                mask=mask,
                cmap="OrRd",
                square=True,
                robust=True,
                fmt='0.2f')
plt.xticks(fontsize=12)
plt.title("Pearson Heat Map", fontsize = 14)
plt.yticks(fontsize=12,rotation=360)

plt.show(block=True)