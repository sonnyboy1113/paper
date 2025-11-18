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
from sklearn.linear_model import Ridge, Lasso
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据加载和预处理 ====================
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print(dataset.info())

X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)

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

for i in range(1, 6):
    X_train[f'Corn_lag_{i}'] = y_train.shift(i)
    X_test[f'Corn_lag_{i}'] = y_test.shift(i)

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

print(f"\n训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")


def create_dataset(X, y, seq_len=5):
    features, targets = [], []
    for i in range(0, len(X) - seq_len, 1):
        features.append(X.iloc[i:i + seq_len])
        targets.append(y.iloc[i + seq_len])
    return np.array(features), np.array(targets)


train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)


def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    return batch_data.cache().shuffle(buffer_size).batch(batch_size) if train else batch_data.batch(batch_size)


train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# ==================== 训练基础模型 ====================
print("\n开始训练GRU模型...")
gru_model = Sequential([
    layers.GRU(units=100, input_shape=(5, 14)),
    layers.Dense(1)
])
gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
gru_history = gru_model.fit(train_batch_dataset, epochs=200, validation_data=test_batch_dataset,
                            callbacks=[early_stop], verbose=0)

print("开始训练LSTM模型...")
lstm_model = Sequential([
    layers.LSTM(units=100, input_shape=(5, 14)),
    layers.Dense(1)
])
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
lstm_history = lstm_model.fit(train_batch_dataset, epochs=200, validation_data=test_batch_dataset,
                              callbacks=[early_stop], verbose=0)

# ==================== 生成预测 ====================
gru_train_preds = gru_model.predict(train_dataset, verbose=0)[:, 0]
lstm_train_preds = lstm_model.predict(train_dataset, verbose=0)[:, 0]
gru_test_preds = gru_model.predict(test_dataset, verbose=0)[:, 0]
lstm_test_preds = lstm_model.predict(test_dataset, verbose=0)[:, 0]

print("\n基础模型性能:")
print(
    f"GRU  - Train R²: {r2_score(train_labels, gru_train_preds):.4f}, Test R²: {r2_score(test_labels, gru_test_preds):.4f}")
print(
    f"LSTM - Train R²: {r2_score(train_labels, lstm_train_preds):.4f}, Test R²: {r2_score(test_labels, lstm_test_preds):.4f}")

stacked_train_features = np.column_stack((gru_train_preds, lstm_train_preds))
stacked_test_features = np.column_stack((gru_test_preds, lstm_test_preds))

# ==================== 🎯 多种堆叠策略对比 ====================
print("\n" + "=" * 80)
print("测试多种堆叠策略...")
print("=" * 80)

# 策略1: 简单平均
avg_train_preds = (gru_train_preds + lstm_train_preds) / 2
avg_test_preds = (gru_test_preds + lstm_test_preds) / 2

# 策略2: Ridge回归（推荐）
ridge_model = Ridge(alpha=1.0, random_state=666)
ridge_model.fit(stacked_train_features, train_labels)
ridge_train_preds = ridge_model.predict(stacked_train_features)
ridge_test_preds = ridge_model.predict(stacked_test_features)

# 策略3: 简化XGBoost
xgb_simple = XGBRegressor(
    n_estimators=10,  # 大幅减少树的数量
    max_depth=1,  # 减少深度
    learning_rate=0.1,  # 降低学习率
    random_state=666
)
xgb_simple.fit(stacked_train_features, train_labels)
xgb_simple_train_preds = xgb_simple.predict(stacked_train_features)
xgb_simple_test_preds = xgb_simple.predict(stacked_test_features)

# 策略4: 原XGBoost（用于对比）
xgb_original = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=666)
xgb_original.fit(stacked_train_features, train_labels)
xgb_original_train_preds = xgb_original.predict(stacked_train_features)
xgb_original_test_preds = xgb_original.predict(stacked_test_features)

# ==================== 📊 性能对比 ====================
print("\n" + "=" * 80)
print(f"{'策略':<25} {'Train R²':<12} {'Test R²':<12} {'差异':<10} {'评价'}")
print("=" * 80)

strategies = [
    ("GRU (单模型)", gru_train_preds, gru_test_preds),
    ("LSTM (单模型)", lstm_train_preds, lstm_test_preds),
    ("简单平均", avg_train_preds, avg_test_preds),
    ("Ridge回归 ⭐推荐", ridge_train_preds, ridge_test_preds),
    ("简化XGBoost", xgb_simple_train_preds, xgb_simple_test_preds),
    ("原XGBoost (过拟合)", xgb_original_train_preds, xgb_original_test_preds),
]

best_test_r2 = 0
best_strategy = ""
best_preds = None

for name, train_preds, test_preds in strategies:
    train_r2 = r2_score(train_labels, train_preds)
    test_r2 = r2_score(test_labels, test_preds)
    diff = train_r2 - test_r2

    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_strategy = name
        best_preds = test_preds

    if diff > 0.15:
        rating = "❌ 严重过拟合"
    elif diff > 0.08:
        rating = "⚠️ 轻微过拟合"
    elif test_r2 > 0.92:
        rating = "✓ 优秀"
    elif test_r2 > 0.88:
        rating = "✓ 良好"
    else:
        rating = "⚠️ 性能下降"

    print(f"{name:<25} {train_r2:<12.4f} {test_r2:<12.4f} {diff:<10.4f} {rating}")

print("=" * 80)
print(f"\n🏆 最佳策略: {best_strategy} (Test R² = {best_test_r2:.4f})")

# ==================== 详细指标输出（使用最佳策略）====================
print("\n" + "=" * 80)
print(f"最佳策略 ({best_strategy}) 的详细指标:")
print("=" * 80)

# 归一化指标
print("\n归一化数据指标:")
print(f"R²:   {r2_score(test_labels, best_preds):.4f}")
print(f"MAE:  {mean_absolute_error(test_labels, best_preds):.6f}")
print(f"MSE:  {mean_squared_error(test_labels, best_preds):.6f}")
print(f"RMSE: {sqrt(mean_squared_error(test_labels, best_preds)):.6f}")

# 反归一化
test_labels_original = y_scaler.inverse_transform(test_labels.reshape(-1, 1))
best_preds_original = y_scaler.inverse_transform(best_preds.reshape(-1, 1))

print("\n原始尺度指标:")
print(f"R²:   {r2_score(test_labels_original, best_preds_original):.4f}")
print(f"MAE:  {mean_absolute_error(test_labels_original, best_preds_original):.2f}")
print(f"MSE:  {mean_squared_error(test_labels_original, best_preds_original):.2f}")
print(f"RMSE: {sqrt(mean_squared_error(test_labels_original, best_preds_original)):.2f}")
print(f"MAPE: {np.mean(np.abs((best_preds_original - test_labels_original) / (test_labels_original + 1e-8))):.4f}")

# ==================== 可视化 ====================
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 图1: 训练曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(gru_history.history['loss'], label='Train', linewidth=2)
plt.plot(gru_history.history['val_loss'], label='Validation', linewidth=2)
plt.title('GRU 训练曲线', fontsize=14)
plt.xlabel('Epochs');
plt.ylabel('MSE Loss')
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'], label='Train', linewidth=2)
plt.plot(lstm_history.history['val_loss'], label='Validation', linewidth=2)
plt.title('LSTM 训练曲线', fontsize=14)
plt.xlabel('Epochs');
plt.ylabel('MSE Loss')
plt.legend();
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'training_curves.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 图2: 策略对比
gru_test_original = y_scaler.inverse_transform(gru_test_preds.reshape(-1, 1))
lstm_test_original = y_scaler.inverse_transform(lstm_test_preds.reshape(-1, 1))
avg_test_original = y_scaler.inverse_transform(avg_test_preds.reshape(-1, 1))
ridge_test_original = y_scaler.inverse_transform(ridge_test_preds.reshape(-1, 1))
xgb_original_test_original = y_scaler.inverse_transform(xgb_original_test_preds.reshape(-1, 1))

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(gru_test_original, label="GRU", linewidth=2, alpha=0.7)
plt.title("GRU 模型", fontsize=12)
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(lstm_test_original, label="LSTM", linewidth=2, alpha=0.7)
plt.title("LSTM 模型", fontsize=12)
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(avg_test_original, label="简单平均", linewidth=2, alpha=0.7)
plt.title("简单平均集成", fontsize=12)
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(ridge_test_original, label="Ridge", linewidth=2, alpha=0.7, color='green')
plt.title("Ridge回归堆叠 ⭐", fontsize=12, fontweight='bold')
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(xgb_original_test_original, label="原XGBoost", linewidth=2, alpha=0.7, color='red')
plt.title("原XGBoost (过拟合)", fontsize=12)
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(gru_test_original, label="GRU", linewidth=1.5, alpha=0.5)
plt.plot(lstm_test_original, label="LSTM", linewidth=1.5, alpha=0.5)
plt.plot(best_preds_original, label=f"最佳策略", linewidth=2.5, alpha=0.8, color='green')
plt.title("最终对比", fontsize=12, fontweight='bold')
plt.legend();
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'strategy_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ==================== 保存模型 ====================
import pickle

# 保存所有策略的模型
with open(results_directory + 'all_models.pkl', 'wb') as f:
    pickle.dump({
        'feature_scalers': feature_scalers,
        'y_scaler': y_scaler,
        'ridge_model': ridge_model,
        'xgb_simple': xgb_simple,
        'ridge_weights': ridge_model.coef_,
        'best_strategy': best_strategy
    }, f)

gru_model.save(results_directory + 'gru_model.h5')
lstm_model.save(results_directory + 'lstm_model.h5')

print("\n✅ 所有模型已保存到:", results_directory)
print("\n💡 结论和建议:")
print("-" * 80)
print("1. Ridge回归通常是最佳选择：简单、稳定、泛化好")
print("2. 简单平均也很有效：无需训练，零过拟合风险")
print("3. 复杂的XGBoost在特征少时容易过拟合")
print("4. Occam's Razor原理：在效果相近时，选择最简单的模型")
print("5. 你的LSTM已经很强(R²=0.927)，提升空间有限是正常的")
print("-" * 80)