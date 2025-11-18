import numpy as np
import pandas as pd
from scipy import stats
from math import sqrt
import matplotlib
matplotlib.use('TkAgg')



def dm_test(actual, pred1, pred2, h=1, loss='MSE'):
    """
    Diebold-Mariano检验

    参数:
    actual: 真实值
    pred1: 模型1的预测值（集成模型）
    pred2: 模型2的预测值（GRU模型）
    h: 预测步长，默认为1
    loss: 损失函数类型，'MSE' 或 'MAE'

    返回:
    DM统计量和p值
    """
    # 确保输入是numpy数组
    actual = np.array(actual).flatten()
    pred1 = np.array(pred1).flatten()
    pred2 = np.array(pred2).flatten()

    # 计算预测误差
    e1 = actual - pred1
    e2 = actual - pred2

    # 根据损失函数计算损失差异
    if loss == 'MSE':
        d = e1 ** 2 - e2 ** 2
    elif loss == 'MAE':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("损失函数必须是 'MSE' 或 'MAE'")

    # 计算损失差异的均值
    mean_d = np.mean(d)

    # 计算方差（考虑自相关）
    def autocovariance(x, lag):
        """计算自协方差"""
        n = len(x)
        x_mean = np.mean(x)

        if lag == 0:
            # lag=0时，计算方差
            return np.sum((x - x_mean) ** 2) / n
        else:
            # lag>0时，计算自协方差
            return np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean)) / n

    # 计算长期方差
    gamma_0 = autocovariance(d, 0)
    gamma_sum = 0

    # 考虑到h步预测的自相关
    for lag in range(1, h):
        gamma_lag = autocovariance(d, lag)
        gamma_sum += gamma_lag

    # 计算方差
    variance = (gamma_0 + 2 * gamma_sum) / len(d)

    # 防止方差为负或为零
    if variance <= 0:
        variance = gamma_0 / len(d)

    # 计算DM统计量
    dm_stat = mean_d / sqrt(variance)

    # 计算p值（双侧检验）
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


# ==================== 主程序 ====================

# 1. 加载预测结果
results_directory = "./Predict/"

print("正在加载预测结果...")

# 加载真实值和预测值（原始尺度）
stacked_preds = pd.read_csv(results_directory + "Stacked-y_predict_original.csv", index_col=0).values.flatten()
gru_preds = pd.read_csv(results_directory + "GRU-y_predict_original.csv", index_col=0).values.flatten()

# 确保两个预测结果长度一致
min_len = min(len(stacked_preds), len(gru_preds))
stacked_preds = stacked_preds[:min_len]
gru_preds = gru_preds[:min_len]

# 需要真实值（从原始数据重新计算）
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'], dayfirst=False)
y = dataset['Corn']
from sklearn.model_selection import train_test_split

_, y_test = train_test_split(y, test_size=0.2, shuffle=False, random_state=666)

# 由于create_dataset函数会消耗seq_len=5个样本，所以需要对应调整
y_test_actual = y_test.values[5:5 + min_len]

print("\n" + "=" * 80)
print("Diebold-Mariano 检验结果")
print("=" * 80)
print(f"样本数量: {len(y_test_actual)}")
print(f"比较对象: 集成模型 vs GRU模型")
print("-" * 80)

# 2. 执行DM检验（MSE损失）
print("\n正在执行DM检验（MSE损失）...")
dm_stat_mse, p_value_mse = dm_test(y_test_actual, stacked_preds, gru_preds, h=1, loss='MSE')

print("\n【基于MSE损失函数】")
print(f"DM统计量: {dm_stat_mse:.4f}")
print(f"P值: {p_value_mse:.4f}")

if p_value_mse < 0.01:
    significance = "***"
    print(f"结论: 在1%显著性水平下，两个模型的预测精度存在显著差异 {significance}")
elif p_value_mse < 0.05:
    significance = "**"
    print(f"结论: 在5%显著性水平下，两个模型的预测精度存在显著差异 {significance}")
elif p_value_mse < 0.1:
    significance = "*"
    print(f"结论: 在10%显著性水平下，两个模型的预测精度存在显著差异 {significance}")
else:
    significance = ""
    print("结论: 两个模型的预测精度无显著差异")

if dm_stat_mse < 0:
    print("解释: 集成模型的MSE显著低于GRU模型（集成模型更优）")
elif dm_stat_mse > 0:
    print("解释: GRU模型的MSE显著低于集成模型（GRU模型更优）")
else:
    print("解释: 两个模型的MSE相近")

# 3. 执行DM检验（MAE损失）
print("\n正在执行DM检验（MAE损失）...")
dm_stat_mae, p_value_mae = dm_test(y_test_actual, stacked_preds, gru_preds, h=1, loss='MAE')

print("\n【基于MAE损失函数】")
print(f"DM统计量: {dm_stat_mae:.4f}")
print(f"P值: {p_value_mae:.4f}")

if p_value_mae < 0.01:
    significance = "***"
    print(f"结论: 在1%显著性水平下，两个模型的预测精度存在显著差异 {significance}")
elif p_value_mae < 0.05:
    significance = "**"
    print(f"结论: 在5%显著性水平下，两个模型的预测精度存在显著差异 {significance}")
elif p_value_mae < 0.1:
    significance = "*"
    print(f"结论: 在10%显著性水平下，两个模型的预测精度存在显著差异 {significance}")
else:
    significance = ""
    print("结论: 两个模型的预测精度无显著差异")

if dm_stat_mae < 0:
    print("解释: 集成模型的MAE显著低于GRU模型（集成模型更优）")
elif dm_stat_mae > 0:
    print("解释: GRU模型的MAE显著低于集成模型（GRU模型更优）")
else:
    print("解释: 两个模型的MAE相近")

# 4. 计算实际的误差统计
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse_stacked = mean_squared_error(y_test_actual, stacked_preds)
mse_gru = mean_squared_error(y_test_actual, gru_preds)
mae_stacked = mean_absolute_error(y_test_actual, stacked_preds)
mae_gru = mean_absolute_error(y_test_actual, gru_preds)
r2_stacked = r2_score(y_test_actual, stacked_preds)
r2_gru = r2_score(y_test_actual, gru_preds)

print("\n" + "=" * 80)
print("模型性能对比")
print("=" * 80)
print(f"{'指标':<15} {'集成模型':<15} {'GRU模型':<15} {'改进幅度(%)':<15}")
print("-" * 80)
print(f"{'R²':<15} {r2_stacked:<15.4f} {r2_gru:<15.4f} {(r2_stacked - r2_gru) / abs(r2_gru) * 100:<15.2f}")
print(f"{'MSE':<15} {mse_stacked:<15.4f} {mse_gru:<15.4f} {(mse_gru - mse_stacked) / mse_gru * 100:<15.2f}")
print(f"{'RMSE':<15} {sqrt(mse_stacked):<15.4f} {sqrt(mse_gru):<15.4f} "
      f"{(sqrt(mse_gru) - sqrt(mse_stacked)) / sqrt(mse_gru) * 100:<15.2f}")
print(f"{'MAE':<15} {mae_stacked:<15.4f} {mae_gru:<15.4f} "
      f"{(mae_gru - mae_stacked) / mae_gru * 100:<15.2f}")
print("=" * 80)
print("注: 改进幅度为正值表示集成模型优于GRU模型")

# 5. 可视化损失差异
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("\n正在生成可视化图表...")

# 计算损失差异序列
e1 = y_test_actual - stacked_preds
e2 = y_test_actual - gru_preds
loss_diff_mse = e1 ** 2 - e2 ** 2
loss_diff_mae = np.abs(e1) - np.abs(e2)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# MSE损失差异
axes[0].plot(loss_diff_mse, linewidth=1.5, color='steelblue', alpha=0.7)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='零线')
axes[0].axhline(y=np.mean(loss_diff_mse), color='green', linestyle='--', linewidth=2,
                label=f'均值={np.mean(loss_diff_mse):.4f}')
axes[0].fill_between(range(len(loss_diff_mse)), loss_diff_mse, 0,
                     where=(loss_diff_mse < 0), alpha=0.3, color='green', label='集成模型更优')
axes[0].fill_between(range(len(loss_diff_mse)), loss_diff_mse, 0,
                     where=(loss_diff_mse > 0), alpha=0.3, color='red', label='GRU模型更优')
axes[0].set_title(f'MSE损失差异 (集成模型 - GRU模型) | DM统计量={dm_stat_mse:.4f}, p={p_value_mse:.4f}',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('样本点', fontsize=11)
axes[0].set_ylabel('MSE损失差异', fontsize=11)
axes[0].legend(fontsize=9, loc='best')
axes[0].grid(True, alpha=0.3)

# MAE损失差异
axes[1].plot(loss_diff_mae, linewidth=1.5, color='coral', alpha=0.7)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='零线')
axes[1].axhline(y=np.mean(loss_diff_mae), color='green', linestyle='--', linewidth=2,
                label=f'均值={np.mean(loss_diff_mae):.4f}')
axes[1].fill_between(range(len(loss_diff_mae)), loss_diff_mae, 0,
                     where=(loss_diff_mae < 0), alpha=0.3, color='green', label='集成模型更优')
axes[1].fill_between(range(len(loss_diff_mae)), loss_diff_mae, 0,
                     where=(loss_diff_mae > 0), alpha=0.3, color='red', label='GRU模型更优')
axes[1].set_title(f'MAE损失差异 (集成模型 - GRU模型) | DM统计量={dm_stat_mae:.4f}, p={p_value_mae:.4f}',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('样本点', fontsize=11)
axes[1].set_ylabel('MAE损失差异', fontsize=11)
axes[1].legend(fontsize=9, loc='best')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'dm_test_loss_difference.png', dpi=300, bbox_inches='tight')
print(f"图表已保存至: {results_directory}dm_test_loss_difference.png")
plt.show(block=True)

# 6. 保存结果到CSV
results_df = pd.DataFrame({
    '检验类型': ['MSE损失', 'MAE损失'],
    'DM统计量': [dm_stat_mse, dm_stat_mae],
    'P值': [p_value_mse, p_value_mae],
    '显著性': [
        '***' if p_value_mse < 0.01 else '**' if p_value_mse < 0.05 else '*' if p_value_mse < 0.1 else 'NS',
        '***' if p_value_mae < 0.01 else '**' if p_value_mae < 0.05 else '*' if p_value_mae < 0.1 else 'NS'
    ]
})
results_df.to_csv(results_directory + 'dm_test_results.csv', index=False, encoding='utf-8-sig')
print(f"检验结果已保存至: {results_directory}dm_test_results.csv")

print("\n" + "=" * 80)
print("DM检验完成！")
print("=" * 80)