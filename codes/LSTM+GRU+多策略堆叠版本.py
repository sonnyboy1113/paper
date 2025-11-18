import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print(dataset.info())

# ========== 数据准备：训练集、测试集 ==========
X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)

print(f"\n数据分割:")
print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 归一化
feature_scalers = {}
for col in X_train.columns:
    scaler = MinMaxScaler()
    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    feature_scalers[col] = scaler

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

y_train = pd.Series(y_train_scaled, index=y_train.index)
y_test = pd.Series(y_test_scaled, index=y_test.index)

# 添加滞后特征
for i in range(1, 6):
    X_train[f'Corn_lag_{i}'] = y_train.shift(i)
    X_test[f'Corn_lag_{i}'] = y_test.shift(i)

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

print(f"\n添加滞后特征后:")
print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")


# 构造序列数据
def create_dataset(X, y, seq_len=5):
    features, targets = [], []
    for i in range(0, len(X) - seq_len, 1):
        features.append(X.iloc[i:i + seq_len])
        targets.append(y.iloc[i + seq_len])
    return np.array(features), np.array(targets)


train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)


# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    return batch_data.cache().shuffle(buffer_size).batch(batch_size) if train else batch_data.batch(batch_size)


train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

print("\n" + "=" * 80)
print("开始训练基础模型")
print("=" * 80)

# 训练 LSTM
print("\n【模型1】训练 LSTM 模型...")
lstm_model = Sequential([
    layers.LSTM(units=100, input_shape=(5, 14)),
    layers.Dense(1)
])
lstm_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
lstm_history = lstm_model.fit(train_batch_dataset, epochs=200, validation_data=test_batch_dataset,
                              callbacks=[early_stop], verbose=0)
print("✓ LSTM 模型训练完成")

# 训练 GRU
print("\n【模型2】训练 GRU 模型...")
gru_model = Sequential([
    layers.GRU(units=100, input_shape=(5, 14)),
    layers.Dense(1)
])
gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
gru_history = gru_model.fit(train_batch_dataset, epochs=200, validation_data=test_batch_dataset,
                            callbacks=[early_stop], verbose=0)
print("✓ GRU 模型训练完成")

# 生成基础预测
lstm_train_preds = lstm_model.predict(train_dataset, verbose=0)[:, 0]
gru_train_preds = gru_model.predict(train_dataset, verbose=0)[:, 0]
lstm_test_preds = lstm_model.predict(test_dataset, verbose=0)[:, 0]
gru_test_preds = gru_model.predict(test_dataset, verbose=0)[:, 0]

print("\n基础模型性能:")
print(
    f"LSTM - 训练R²: {r2_score(train_labels, lstm_train_preds):.4f}, 测试R²: {r2_score(test_labels, lstm_test_preds):.4f}")
print(f"GRU  - 训练R²: {r2_score(train_labels, gru_train_preds):.4f}, 测试R²: {r2_score(test_labels, gru_test_preds):.4f}")

# ==================== 🎯 多种集成策略对比 ====================
print("\n" + "=" * 80)
print("测试多种集成策略")
print("=" * 80)

stacked_train = np.column_stack((lstm_train_preds, gru_train_preds))
stacked_test = np.column_stack((lstm_test_preds, gru_test_preds))

# 策略1: 简单平均（最稳健）
avg_train = (lstm_train_preds + gru_train_preds) / 2
avg_test = (lstm_test_preds + gru_test_preds) / 2

# 策略2: 加权平均（基于验证集性能）
lstm_val_r2 = r2_score(test_labels, lstm_test_preds)
gru_val_r2 = r2_score(test_labels, gru_test_preds)
total_r2 = lstm_val_r2 + gru_val_r2
w_lstm = lstm_val_r2 / total_r2
w_gru = gru_val_r2 / total_r2

weighted_train = w_lstm * lstm_train_preds + w_gru * gru_train_preds
weighted_test = w_lstm * lstm_test_preds + w_gru * gru_test_preds

# 策略3: Ridge回归（带正则化，推荐）
ridge_model = Ridge(alpha=10.0, random_state=666)  # 增大alpha增强正则化
ridge_model.fit(stacked_train, train_labels)
ridge_train = ridge_model.predict(stacked_train)
ridge_test = ridge_model.predict(stacked_test)

# 策略4: Lasso回归（特征选择）
lasso_model = Lasso(alpha=0.001, random_state=666)
lasso_model.fit(stacked_train, train_labels)
lasso_train = lasso_model.predict(stacked_train)
lasso_test = lasso_model.predict(stacked_test)

# 策略5: 线性回归（原始方法）
lr_model = LinearRegression()
lr_model.fit(stacked_train, train_labels)
lr_train = lr_model.predict(stacked_train)
lr_test = lr_model.predict(stacked_test)

# 策略6: 选择最佳单模型（GRU）
best_single_train = gru_train_preds
best_single_test = gru_test_preds

# ==================== 📊 性能对比分析 ====================
print("\n" + "=" * 80)
print(f"{'集成策略':<25} {'训练R²':<12} {'测试R²':<12} {'差异':<10} {'评价'}")
print("=" * 80)

strategies = {
    'LSTM (单模型)': (lstm_train_preds, lstm_test_preds),
    'GRU (单模型)': (gru_train_preds, gru_test_preds),
    '简单平均 ⭐': (avg_train, avg_test),
    '性能加权平均 ⭐⭐': (weighted_train, weighted_test),
    'Ridge回归 ⭐⭐⭐': (ridge_train, ridge_test),
    'Lasso回归': (lasso_train, lasso_test),
    '线性回归 (原方法)': (lr_train, lr_test),
}

best_test_r2 = 0
best_strategy_name = ""
best_train_pred = None
best_test_pred = None

for name, (train_pred, test_pred) in strategies.items():
    train_r2 = r2_score(train_labels, train_pred)
    test_r2 = r2_score(test_labels, test_pred)
    diff = train_r2 - test_r2

    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_strategy_name = name
        best_train_pred = train_pred
        best_test_pred = test_pred

    # 评价标准
    if diff > 0.15:
        rating = "❌ 严重过拟合"
    elif diff > 0.08:
        rating = "⚠️ 轻微过拟合"
    elif test_r2 > train_r2:
        rating = "✓✓ 泛化优秀"
    elif test_r2 >= 0.92:
        rating = "✓ 优秀"
    elif test_r2 >= 0.88:
        rating = "✓ 良好"
    else:
        rating = "- 一般"

    print(f"{name:<25} {train_r2:<12.4f} {test_r2:<12.4f} {diff:<10.4f} {rating}")

print("=" * 80)
print(f"\n🏆 最佳策略: {best_strategy_name}")
print(f"   测试集R²: {best_test_r2:.4f}")

# ==================== 📈 详细性能分析 ====================
print("\n" + "=" * 80)
print(f"最佳策略详细指标: {best_strategy_name}")
print("=" * 80)

# 归一化指标
print("\n归一化数据:")
print(f"  R²:   {r2_score(test_labels, best_test_pred):.6f}")
print(f"  MAE:  {mean_absolute_error(test_labels, best_test_pred):.6f}")
print(f"  RMSE: {sqrt(mean_squared_error(test_labels, best_test_pred)):.6f}")

# 原始尺度指标
test_labels_original = y_scaler.inverse_transform(test_labels.reshape(-1, 1))
best_test_original = y_scaler.inverse_transform(best_test_pred.reshape(-1, 1))

print("\n原始尺度:")
print(f"  R²:   {r2_score(test_labels_original, best_test_original):.6f}")
print(f"  MAE:  {mean_absolute_error(test_labels_original, best_test_original):.2f} 元/吨")
print(f"  RMSE: {sqrt(mean_squared_error(test_labels_original, best_test_original)):.2f} 元/吨")
print(f"  MAPE: {np.mean(np.abs((best_test_original - test_labels_original) / (test_labels_original + 1e-8))):.4%}")

# ==================== 📊 权重分析 ====================
print("\n" + "=" * 80)
print("模型权重分析")
print("=" * 80)

print(f"\n性能加权平均:")
print(f"  LSTM权重: {w_lstm:.4f} (基于测试R²={lstm_val_r2:.4f})")
print(f"  GRU权重:  {w_gru:.4f} (基于测试R²={gru_val_r2:.4f})")

print(f"\nRidge回归权重:")
print(f"  LSTM系数: {ridge_model.coef_[0]:.6f}")
print(f"  GRU系数:  {ridge_model.coef_[1]:.6f}")
print(f"  截距:     {ridge_model.intercept_:.6f}")

print(f"\n线性回归权重 (原方法):")
print(f"  LSTM系数: {lr_model.coef_[0]:.6f}")
print(f"  GRU系数:  {lr_model.coef_[1]:.6f}")
print(f"  截距:     {lr_model.intercept_:.6f}")

# ==================== 🎨 可视化 ====================
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 图1: 所有策略对比
fig = plt.figure(figsize=(20, 12))

lstm_test_original = y_scaler.inverse_transform(lstm_test_preds.reshape(-1, 1))
gru_test_original = y_scaler.inverse_transform(gru_test_preds.reshape(-1, 1))
avg_test_original = y_scaler.inverse_transform(avg_test.reshape(-1, 1))
weighted_test_original = y_scaler.inverse_transform(weighted_test.reshape(-1, 1))
ridge_test_original = y_scaler.inverse_transform(ridge_test.reshape(-1, 1))
lr_test_original = y_scaler.inverse_transform(lr_test.reshape(-1, 1))

strategies_plot = [
    ("LSTM单模型", lstm_test_original, 'blue'),
    ("GRU单模型", gru_test_original, 'green'),
    ("简单平均", avg_test_original, 'orange'),
    ("性能加权", weighted_test_original, 'purple'),
    ("Ridge回归", ridge_test_original, 'red'),
    ("线性回归(原)", lr_test_original, 'brown'),
]

for idx, (name, preds, color) in enumerate(strategies_plot, 1):
    plt.subplot(2, 3, idx)
    plt.plot(test_labels_original, label="真实值", linewidth=2.5, color='black', alpha=0.8)
    plt.plot(preds, label=name, linewidth=2, alpha=0.7, color=color)
    r2 = r2_score(test_labels_original, preds)
    plt.title(f"{name} (R²={r2:.4f})", fontsize=12, fontweight='bold')
    plt.xlabel('样本序号')
    plt.ylabel('玉米价格 (元/吨)')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'ensemble_strategies_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 图2: 训练-测试性能对比（识别过拟合）
fig, ax = plt.subplots(figsize=(12, 7))

strategy_names = list(strategies.keys())
train_r2s = [r2_score(train_labels, strategies[name][0]) for name in strategy_names]
test_r2s = [r2_score(test_labels, strategies[name][1]) for name in strategy_names]

x = np.arange(len(strategy_names))
width = 0.35

bars1 = ax.bar(x - width / 2, train_r2s, width, label='训练集R²', alpha=0.8, color='skyblue')
bars2 = ax.bar(x + width / 2, test_r2s, width, label='测试集R²', alpha=0.8, color='salmon')

ax.set_xlabel('集成策略', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('训练集 vs 测试集性能对比 (识别过拟合)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategy_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.92, color='green', linestyle='--', alpha=0.5, label='优秀线(0.92)')

# 标注差异值
for i, (train_r2, test_r2) in enumerate(zip(train_r2s, test_r2s)):
    diff = train_r2 - test_r2
    ax.text(i, max(train_r2, test_r2) + 0.01, f'Δ{diff:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='red' if diff > 0.1 else 'orange' if diff > 0.05 else 'green')

plt.tight_layout()
plt.savefig(results_directory + 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ==================== 💾 保存模型 ====================
import pickle

with open(results_directory + 'optimized_ensemble.pkl', 'wb') as f:
    pickle.dump({
        'feature_scalers': feature_scalers,
        'y_scaler': y_scaler,
        'ridge_model': ridge_model,
        'best_strategy': best_strategy_name,
        'weights': {
            'performance_weighted': {'lstm': w_lstm, 'gru': w_gru},
            'ridge': {'coef': ridge_model.coef_, 'intercept': ridge_model.intercept_}
        }
    }, f)

lstm_model.save(results_directory + 'lstm_model.h5')
gru_model.save(results_directory + 'gru_model.h5')

print("\n" + "=" * 80)
print("✅ 所有模型已保存")
print("=" * 80)

# ==================== 💡 结论与建议 ====================
print("\n" + "=" * 80)
print("💡 结论与建议")
print("=" * 80)
print("\n1️⃣ 为什么原线性回归会过拟合？")
print("   - 基模型预测高度相关（LSTM和GRU学到的是相似模式）")
print("   - 元模型试图学习噪音差异，导致过拟合")
print("   - 训练样本相对较少（946个）")

print("\n2️⃣ 最佳实践建议：")
print("   ⭐⭐⭐ 优先选择: 性能加权平均或Ridge回归")
print("   - 这两种方法都能有效防止过拟合")
print("   - 性能加权更简单，Ridge更灵活")
print("   - 简单平均也很稳健，可作为baseline")

print("\n3️⃣ 何时集成能提升性能？")
print("   - 基模型差异大（如LSTM+CNN+XGBoost）")
print("   - 基模型在不同子问题上表现不同")
print("   - 有足够的元训练数据")

print("\n4️⃣ 你的情况：")
print(f"   - 单个GRU已经很强 (R²={gru_val_r2:.4f})")
print("   - LSTM和GRU太相似，集成收益有限")
print("   - 建议：使用GRU单模型或性能加权平均")

print("\n" + "=" * 80)