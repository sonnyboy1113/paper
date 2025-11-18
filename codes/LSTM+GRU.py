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
from sklearn.linear_model import LinearRegression
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

# 特征数据集和标签数据集
X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

# 数据集分离（训练集80%，测试集20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)

print(f"\n数据分割:")
print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 只在训练集上fit归一化器
feature_scalers = {}
for col in X_train.columns:
    scaler = MinMaxScaler()
    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    feature_scalers[col] = scaler

# 对标签进行归一化
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# 转换回Series并保持索引
y_train = pd.Series(y_train_scaled, index=y_train.index)
y_test = pd.Series(y_test_scaled, index=y_test.index)

# 添加滞后特征
for i in range(1, 6):
    X_train[f'Corn_lag_{i}'] = y_train.shift(i)
    X_test[f'Corn_lag_{i}'] = y_test.shift(i)

# 删除因滞后特征产生的缺失值
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

print(f"\n添加滞后特征后:")
print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

# 构造特征数据集
def create_dataset(X, y, seq_len=5):
    features = []
    targets = []
    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i + seq_len]
        label = y.iloc[i + seq_len]
        features.append(data)
        targets.append(label)
    return np.array(features), np.array(targets)

train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)

print(f"\n序列数据形状:")
print(f"train_dataset={train_dataset.shape}")
print(f"test_dataset={test_dataset.shape}")

# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)

train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# 早停回调（使用训练集的一部分作为验证集）
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

print("\n" + "="*80)
print("开始训练所有模型")
print("="*80)

# ======================= 1. 训练 LSTM 模型 =======================
print("\n【模型1】训练 LSTM 模型...")
lstm_model = Sequential([
    layers.LSTM(units=100, input_shape=(5, 14)),
    layers.Dense(1)
])
lstm_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
lstm_history = lstm_model.fit(
    train_dataset, train_labels,  # 使用numpy数组而不是tf.data.Dataset
    epochs=200,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)
print("✓ LSTM 模型训练完成")

# ======================= 2. 训练 GRU 模型 =======================
print("\n【模型2】训练 GRU 模型...")
gru_model = Sequential([
    layers.GRU(units=100, input_shape=(5, 14)),
    layers.Dense(1)
])
gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
gru_history = gru_model.fit(
    train_dataset, train_labels,
    epochs=200,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)
print("✓ GRU 模型训练完成")

# ======================= 3. 训练堆叠集成模型 =======================
print("\n【模型3】训练堆叠集成模型...")
print("  使用训练集的预测结果训练线性回归元模型")

# 在训练集上获取所有基模型的预测（用于训练元模型）
lstm_train_preds = lstm_model.predict(train_dataset, verbose=0)[:, 0]
gru_train_preds = gru_model.predict(train_dataset, verbose=0)[:, 0]

# 堆叠特征
train_stacked_features = np.column_stack((lstm_train_preds, gru_train_preds))

# 训练元模型
meta_model = LinearRegression()
meta_model.fit(train_stacked_features, train_labels)

print(f"  线性回归系数 [LSTM, GRU]: {meta_model.coef_}")
print(f"  截距: {meta_model.intercept_:.6f}")
print("✓ 堆叠集成模型训练完成")

# ======================= 生成所有模型的预测 =======================
print("\n" + "="*80)
print("生成所有模型的预测结果")
print("="*80)

# 训练集最终预测
ensemble_train_preds = meta_model.predict(train_stacked_features)

# 测试集预测
lstm_test_preds = lstm_model.predict(test_dataset, verbose=0)[:, 0]
gru_test_preds = gru_model.predict(test_dataset, verbose=0)[:, 0]

test_stacked = np.column_stack((lstm_test_preds, gru_test_preds))
ensemble_test_preds = meta_model.predict(test_stacked)

# ======================= 性能评估对比 =======================
print("\n" + "="*80)
print("模型性能对比（归一化数据）")
print("="*80)

models_info = {
    'LSTM': (lstm_train_preds, lstm_test_preds),
    'GRU': (gru_train_preds, gru_test_preds),
    '堆叠集成': (ensemble_train_preds, ensemble_test_preds)
}

print("\n{:<15} {:<15} {:<15}".format("模型", "训练集R²", "测试集R²"))
print("-" * 50)
for name, (train_pred, test_pred) in models_info.items():
    train_r2 = r2_score(train_labels, train_pred)
    test_r2 = r2_score(test_labels, test_pred)
    print(f"{name:<15} {train_r2:<15.4f} {test_r2:<15.4f}")

# ======================= 测试集详细指标 =======================
print("\n" + "="*80)
print("测试集详细性能指标（归一化数据）")
print("="*80)

print("\n{:<15} {:<10} {:<10} {:<10} {:<10}".format("模型", "R²", "MAE", "RMSE", "MAPE"))
print("-" * 65)

for name, (_, test_pred) in models_info.items():
    r2 = r2_score(test_labels, test_pred)
    mae = mean_absolute_error(test_labels, test_pred)
    rmse = sqrt(mean_squared_error(test_labels, test_pred))
    mape = np.mean(np.abs((test_pred - test_labels) / (test_labels + 1e-8)))
    print(f"{name:<15} {r2:<10.4f} {mae:<10.6f} {rmse:<10.6f} {mape:<10.6f}")

# ========== 反归一化 ==========
test_labels_original = y_scaler.inverse_transform(test_labels.reshape(-1, 1))
lstm_preds_original = y_scaler.inverse_transform(lstm_test_preds.reshape(-1, 1))
gru_preds_original = y_scaler.inverse_transform(gru_test_preds.reshape(-1, 1))
ensemble_preds_original = y_scaler.inverse_transform(ensemble_test_preds.reshape(-1, 1))

# 原始尺度详细指标
print("\n" + "="*80)
print("测试集详细性能指标（原始尺度）")
print("="*80)

models_original = {
    'LSTM': lstm_preds_original,
    'GRU': gru_preds_original,
    '堆叠集成': ensemble_preds_original
}

print("\n{:<15} {:<10} {:<10} {:<10} {:<10}".format("模型", "R²", "MAE", "RMSE", "MAPE"))
print("-" * 65)

for name, preds in models_original.items():
    r2 = r2_score(test_labels_original, preds)
    mae = mean_absolute_error(test_labels_original, preds)
    rmse = sqrt(mean_squared_error(test_labels_original, preds))
    mape = np.mean(np.abs((preds - test_labels_original) / (test_labels_original + 1e-8)))
    print(f"{name:<15} {r2:<10.4f} {mae:<10.4f} {rmse:<10.4f} {mape:<10.6f}")

print("="*80)

# ======================= 可视化结果 =======================
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 1. 训练过程对比（只显示LSTM和GRU）
fig_training = plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='训练损失', linewidth=2)
plt.plot(lstm_history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('LSTM 模型训练过程', fontsize=13, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(gru_history.history['loss'], label='训练损失', linewidth=2)
plt.plot(gru_history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('GRU 模型训练过程', fontsize=13, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'models_training_loss.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 2. 所有模型预测对比（1x3布局）
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(test_labels_original, label="真实值", linewidth=2.5, color='black')
plt.plot(lstm_preds_original, label="LSTM预测", linewidth=2, alpha=0.8, color='blue')
plt.title("LSTM 模型", fontsize=14, fontweight='bold')
plt.xlabel('样本序号', fontsize=11)
plt.ylabel('玉米价格', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(test_labels_original, label="真实值", linewidth=2.5, color='black')
plt.plot(gru_preds_original, label="GRU预测", linewidth=2, alpha=0.8, color='green')
plt.title("GRU 模型", fontsize=14, fontweight='bold')
plt.xlabel('样本序号', fontsize=11)
plt.ylabel('玉米价格', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(test_labels_original, label="真实值", linewidth=2.8, color='black', alpha=0.9)
plt.plot(lstm_preds_original, label="LSTM", linewidth=1.5, alpha=0.6, linestyle='--')
plt.plot(gru_preds_original, label="GRU", linewidth=1.5, alpha=0.6, linestyle='--')
plt.plot(ensemble_preds_original, label="堆叠集成", linewidth=2.5, alpha=0.9, color='red')
plt.title("所有模型综合对比", fontsize=14, fontweight='bold')
plt.xlabel('样本序号', fontsize=11)
plt.ylabel('玉米价格', fontsize=11)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 3. 性能指标对比柱状图
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

metrics_data = {}
for name, preds in models_original.items():
    metrics_data[name] = {
        'R²': r2_score(test_labels_original, preds),
        'MAE': mean_absolute_error(test_labels_original, preds),
        'RMSE': sqrt(mean_squared_error(test_labels_original, preds)),
        'MAPE': np.mean(np.abs((preds - test_labels_original) / (test_labels_original + 1e-8)))
    }

model_names = list(metrics_data.keys())
colors = ['blue', 'green', 'red']

# R² 对比
axes[0, 0].bar(model_names, [metrics_data[m]['R²'] for m in model_names], color=colors, alpha=0.7)
axes[0, 0].set_title('R² 分数对比', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].tick_params(axis='x', rotation=15)

# MAE 对比
axes[0, 1].bar(model_names, [metrics_data[m]['MAE'] for m in model_names], color=colors, alpha=0.7)
axes[0, 1].set_title('MAE 对比', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].tick_params(axis='x', rotation=15)

# RMSE 对比
axes[1, 0].bar(model_names, [metrics_data[m]['RMSE'] for m in model_names], color=colors, alpha=0.7)
axes[1, 0].set_title('RMSE 对比', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].tick_params(axis='x', rotation=15)

# MAPE 对比
axes[1, 1].bar(model_names, [metrics_data[m]['MAPE'] for m in model_names], color=colors, alpha=0.7)
axes[1, 1].set_title('MAPE 对比', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('MAPE')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(results_directory + 'metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ======================= 保存所有模型 =======================
import pickle

with open(results_directory + 'stacked_scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

with open(results_directory + 'linear_meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)

lstm_model.save(results_directory + 'lstm_model.h5')
gru_model.save(results_directory + 'gru_model.h5')

print('\n' + '='*80)
print('所有模型保存完成')
print('='*80)
print('保存位置:', results_directory)
print('  - lstm_model.h5             (LSTM模型)')
print('  - gru_model.h5              (GRU模型)')
print('  - linear_meta_model.pkl     (线性回归元模型)')
print('  - stacked_scalers.pkl       (归一化器)')
print('='*80 + '\n')

# ======================= 最终总结 =======================
print("="*80)
print("🎉 模型训练与评估完成！")
print("="*80)
print(f"\n📊 使用的两个基础模型: LSTM, GRU")
print(f"\n🏆 最佳单模型: GRU (测试集R²: {r2_score(test_labels_original, gru_preds_original):.4f})")
print(f"🎯 堆叠集成模型: 测试集R²: {r2_score(test_labels_original, ensemble_preds_original):.4f}")
print(f"\n💾 所有结果已保存到 {results_directory} 目录")
print("="*80)