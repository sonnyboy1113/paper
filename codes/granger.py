import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# 读取CSV数据
data = pd.read_csv('Corn-new.csv')

# 检查数据列名称
print(data.columns)

# 提取变量
x = data['corn']
y = data['ine']
x_name ='corn'
y_name ='ine'
print(x_name)
print(y_name)
# 进行Granger因果检验
# 例如，进行最多4阶的检验
results = grangercausalitytests(data[['corn', 'ine']], maxlag=4, verbose=True)

# 打印检验结果结构以检查键名
print(results)

# 打印检验结果
for lag in results:
    print(f'Lag {lag}:')
    ssr_ftest_pvalue = results[lag][0]['ssr_ftest'][1] if 'ssr_ftest' in results[lag][0] else None
    lrtest_pvalue = results[lag][0]['lrtest'][1] if 'lrtest' in results[lag][0] else None
    params_ftest_pvalue = results[lag][0]['params_ftest'][1] if 'params_ftest' in results[lag][0] else None

    # 打印各项检验的p值，如果存在
    if ssr_ftest_pvalue is not None:
        print(f'F-test p-value: {ssr_ftest_pvalue}')
    if lrtest_pvalue is not None:
        print(f'Likelihood ratio test p-value: {lrtest_pvalue}')
    if params_ftest_pvalue is not None:
        print(f'Granger Causality test p-value: {params_ftest_pvalue}')
    print('-'*50)

# 检查是否有因果关系
# 如果p值小于显著性水平（例如0.1），则拒绝原假设，认为存在因果关系
# 这里以0.1作为显著性水平
alpha = 0.1
if ssr_ftest_pvalue is not None and ssr_ftest_pvalue < alpha:
    print(f'{x_name} is a Granger Cause of {y_name} (p-value: {ssr_ftest_pvalue:.4f})')
else:
    print(f'{x_name} is not a Granger Cause of {y_name} (p-value: {ssr_ftest_pvalue:.4f})')
