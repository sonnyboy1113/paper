import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import GRU, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print(dataset.info())

# ========== 关键修改：先分割，再归一化，然后添加滞后特征 ==========

# 特征数据集和标签数据集
X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

# 第一步：数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)

# 第二步：只在训练集上fit归一化器
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

# 第三步：添加滞后特征
for i in range(1, 6):
    X_train[f'Corn_lag_{i}'] = y_train.shift(i)
    X_test[f'Corn_lag_{i}'] = y_test.shift(i)

# 删除因滞后特征产生的缺失值
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

print(f"\n训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
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

print(f"\n序列数据形状: train_dataset={train_dataset.shape}, test_dataset={test_dataset.shape}")

# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)

train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# 早停回调
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# ======================= 训练 GRU 模型 =======================
print("\n开始训练GRU模型...")
gru_model = Sequential([
    layers.GRU(units=100, input_shape=(5, 14)),
    layers.Dense(1)
])
gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
gru_history = gru_model.fit(
    train_batch_dataset,
    epochs=200,
    validation_data=test_batch_dataset,
    callbacks=[early_stop],
    verbose=0
)

# ======================= 训练 LSTM 模型 =======================
print("开始训练LSTM模型...")
lstm_model = Sequential([
    layers.LSTM(units=100, input_shape=(5, 14)),
    layers.Dense(1)
])
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
lstm_history = lstm_model.fit(
    train_batch_dataset,
    epochs=200,
    validation_data=test_batch_dataset,
    callbacks=[early_stop],
    verbose=0
)

# ======================= 可视化过拟合验证 =======================
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(gru_history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(gru_history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('GRU 模型训练与验证损失', fontsize=14)
plt.xlabel('Epochs'); plt.ylabel('MSE Loss')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(lstm_history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('LSTM 模型训练与验证损失', fontsize=14)
plt.xlabel('Epochs'); plt.ylabel('MSE Loss')
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'model_overfitting_check.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ======================= 模型预测 =======================
print("\n生成基础模型预测...")
gru_preds = gru_model.predict(test_dataset, verbose=0)[:, 0]
lstm_preds = lstm_model.predict(test_dataset, verbose=0)[:, 0]

# ======================= 训练集 vs 测试集 性能对比 =======================
gru_train_preds = gru_model.predict(train_dataset, verbose=0)[:, 0]
lstm_train_preds = lstm_model.predict(train_dataset, verbose=0)[:, 0]

print("\n训练集与测试集性能对比:")
print("GRU - Train R²:", r2_score(train_labels, gru_train_preds))
print("GRU - Test  R²:", r2_score(test_labels, gru_preds))
print("LSTM - Train R²:", r2_score(train_labels, lstm_train_preds))
print("LSTM - Test  R²:", r2_score(test_labels, lstm_preds))

# ======================= XGBoost 堆叠 =======================
print("\n训练XGBoost元模型...")
stacked_features = np.column_stack((gru_preds, lstm_preds))
meta_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=666
)
meta_model.fit(stacked_features, test_labels)
final_preds = meta_model.predict(stacked_features)

# ======================= 归一化指标 =======================
print('\n' + '----' * 180)
print("归一化数据的指标：")
print("r^2 值为：", r2_score(test_labels, final_preds))
print("MAE:", mean_absolute_error(test_labels, final_preds))
print("MSE:", mean_squared_error(test_labels, final_preds))
print("RMSE:", sqrt(mean_squared_error(test_labels, final_preds)))
print("MAPE:", np.mean(np.abs((final_preds - test_labels) / (test_labels + 1e-8))))

# ========== 反归一化 ==========
test_labels_original = y_scaler.inverse_transform(test_labels.reshape(-1, 1))
final_preds_original = y_scaler.inverse_transform(final_preds.reshape(-1, 1))
gru_preds_original = y_scaler.inverse_transform(gru_preds.reshape(-1, 1))
lstm_preds_original = y_scaler.inverse_transform(lstm_preds.reshape(-1, 1))

# 原始尺度指标
print('\n原始尺度的指标:')
print("Stacked Model - r^2 值为：", r2_score(test_labels_original, final_preds_original))
print("Stacked Model - MAE:", mean_absolute_error(test_labels_original, final_preds_original))
print("Stacked Model - MSE:", mean_squared_error(test_labels_original, final_preds_original))
print("Stacked Model - RMSE:", sqrt(mean_squared_error(test_labels_original, final_preds_original)))
print("Stacked Model - MAPE:", np.mean(np.abs((final_preds_original - test_labels_original) / (test_labels_original + 1e-8))))

# ======================= 绘制预测结果 =======================
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(test_labels_original, label="True value", linewidth=2)
plt.plot(final_preds_original, label="Stacked Pred", linewidth=2, alpha=0.8)
plt.title("Stacked Model (LSTM + GRU + XGBoost)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(test_labels_original, label="True value", linewidth=2)
plt.plot(gru_preds_original, label="GRU Pred", linewidth=2, alpha=0.8)
plt.title("GRU Model")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(test_labels_original, label="True value", linewidth=2)
plt.plot(lstm_preds_original, label="LSTM Pred", linewidth=2, alpha=0.8)
plt.title("LSTM Model")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(test_labels_original, label="True value", linewidth=2)
plt.plot(gru_preds_original, label="GRU", linewidth=1.5, alpha=0.6)
plt.plot(lstm_preds_original, label="LSTM", linewidth=1.5, alpha=0.6)
plt.plot(final_preds_original, label="Stacked", linewidth=2, alpha=0.8)
plt.title("Model Comparison")
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'stacked_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ======================= 保存模型与归一化器 =======================
import pickle
with open(results_directory + 'stacked_scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

gru_model.save(results_directory + 'gru_model.h5')
lstm_model.save(results_directory + 'lstm_model.h5')
meta_model.save_model(results_directory + 'xgboost_meta_model.json')

print('\n所有模型和归一化器已保存到:', results_directory)
print('- gru_model.h5')
print('- lstm_model.h5')
print('- xgboost_meta_model.json')
print('- stacked_scalers.pkl')
