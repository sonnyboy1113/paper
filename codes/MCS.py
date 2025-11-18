from arch.bootstrap import MCS
import pandas as pd

# mae
losses = pd.read_csv('MAE.csv')
losses.info()
columns = ['MLP','CNN','BP','RNN','LSTM','GRU','XGBoost','GRU+LSTM+Xgboost']
#MCS检验
mcs = MCS(losses, size=1)
mcs.compute()
# 显示各模型的MSC检验的P值
print("MCS P-values")
print(mcs.pvalues)

