import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib后端为TkAgg

import matplotlib.pyplot as plt

# 假设的数据，对应您的图片内容
models = ['MLP','RNN', 'BP', 'CNN', 'XGBoost','GRU', 'LSTM','LSTM+GRU+XGBoost']
performance = [0.097475066,0.091848313,0.022760068,0.026710966,0.037917122,0.020154023,0.022089944,0.006771497]

# 创建条形图
fig, ax = plt.subplots()
bar_width = 0.8  # 条形的宽度
index = range(len(models))

# 颜色列表
colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#C2C2F0', '#FFB3E6','#8B4C39','#FF0000']

# 绘制条形图，并为每个柱形指定不同的颜色和图例标签
bars = []
for i in range(len(models)):
    bars.append(ax.bar(index[i], performance[i], bar_width, color=colors[i], label=models[i]))

# 添加一些文本标签
for i, performance_value in enumerate(performance):
    ax.text(index[i], performance_value, round(performance_value, 4), ha='center', va='bottom')

# 设置图表标题和坐标轴标签
ax.set_xlabel('MAPE')
ax.set_ylabel('Error')
ax.set_xticks(index)
ax.set_xticklabels(models, rotation=45, ha='right')  # 旋转标签以便更好地显示

# 显示图例
#ax.legend()

# 显示图表
plt.tight_layout()  # 调整布局以适应标签
plt.show()
