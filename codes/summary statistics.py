from IPython.display import display
import numpy as np
import pandas as pd
data = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
data.drop(['Corn'], axis=1, inplace=True) #删除无关的列
df = pd.DataFrame(data)
df_form=df.describe()
print(df_form)
