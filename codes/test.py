import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 生成随机数据
x = np.random.rand(100)
y = np.random.rand(100)

# 计算相关系数
correlation = np.corrcoef(x, y)[0, 1]

# 绘制散点图
plt.scatter(x, y)
plt.title('Correlation Coefficient: ' + str(correlation))
plt.show(block=True)
