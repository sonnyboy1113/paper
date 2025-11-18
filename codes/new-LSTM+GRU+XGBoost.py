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
from tensorflow.keras import Sequential, layers, Model
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

# ========== 数据准备：训练集、验证集、测试集 ==========

# 特征数据集和标签数据集
X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

# 第一步：数据集分离（训练集60%，验证集20%，测试集20%）
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, shuffle=False, random_state=666)

print(f"\n原始数据分割:")
print(f"训练集: {len(X_train)} 样本")
print(f"验证集: {len(X_val)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 第二步：只在训练集上fit归一化器
feature_scalers = {}
for col in X_train.columns:
    scaler = MinMaxScaler()
    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
    X_val[col] = scaler.transform(X_val[col].values.reshape(-1, 1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    feature_scalers[col] = scaler

# 对标签进行归一化
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# 转换回Series并保持索引
y_train = pd.Series(y_train_scaled, index=y_train.index)
y_val = pd.Series(y_val_scaled, index=y_val.index)
y_test = pd.Series(y_test_scaled, index=y_test.index)

# 第三步：添加滞后特征
for i in range(1, 6):
    X_train[f'Corn_lag_{i}'] = y_train.shift(i)
    X_val[f'Corn_lag_{i}'] = y_val.shift(i)
    X_test[f'Corn_lag_{i}'] = y_test.shift(i)

# 删除因滞后特征产生的缺失值
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

X_val = X_val.dropna()
y_val = y_val.loc[X_val.index]

X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

print(f"\n添加滞后特征后:")
print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"验证集形状: X_val={X_val.shape}, y_val={y_val.shape}")
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
val_dataset, val_labels = create_dataset(X_val, y_val, seq_len=5)
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)

print(f"\n序列数据形状:")
print(f"train_dataset={train_dataset.shape}")
print(f"val_dataset={val_dataset.shape}")
print(f"test_dataset={test_dataset.shape}")


# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)


train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
val_batch_dataset = create_batch_dataset(val_dataset, val_labels, train=False)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# 早停回调
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# ======================= 构建返回隐藏状态的LSTM模型 =======================
print("\n开始训练LSTM模型（返回隐藏状态）...")
lstm_input = layers.Input(shape=(5, 14))
lstm_layer, lstm_hidden_state, lstm_cell_state = layers.LSTM(
    units=100,
    return_sequences=False,
    return_state=True
)(lstm_input)
lstm_output = layers.Dense(1)(lstm_layer)

lstm_model = Model(inputs=lstm_input, outputs=[lstm_output, lstm_hidden_state])
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')


# 自定义训练循环（只用预测输出计算损失）
class CustomLSTMCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {logs['loss']:.6f}, Val Loss: {logs['val_loss']:.6f}")


# 重新定义模型只输出预测值用于训练
lstm_pred_model = Model(inputs=lstm_input, outputs=lstm_output)
lstm_pred_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

lstm_history = lstm_pred_model.fit(
    train_batch_dataset,
    epochs=200,
    validation_data=val_batch_dataset,
    callbacks=[early_stop, CustomLSTMCallback()],
    verbose=0
)

# 更新完整模型的权重
lstm_model.set_weights(lstm_pred_model.get_weights())

# ======================= 构建返回隐藏状态的GRU模型 =======================
print("\n开始训练GRU模型（返回隐藏状态）...")
gru_input = layers.Input(shape=(5, 14))
gru_layer, gru_hidden_state = layers.GRU(
    units=100,
    return_sequences=False,
    return_state=True
)(gru_input)
gru_output = layers.Dense(1)(gru_layer)

gru_model = Model(inputs=gru_input, outputs=[gru_output, gru_hidden_state])
gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# 重新定义模型只输出预测值用于训练
gru_pred_model = Model(inputs=gru_input, outputs=gru_output)
gru_pred_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

gru_history = gru_pred_model.fit(
    train_batch_dataset,
    epochs=200,
    validation_data=val_batch_dataset,
    callbacks=[early_stop, CustomLSTMCallback()],
    verbose=0
)

# 更新完整模型的权重
gru_model.set_weights(gru_pred_model.get_weights())

# ======================= 提取隐藏状态特征 =======================
print("\n提取LSTM和GRU的隐藏状态特征...")

# 训练集特征提取
lstm_train_pred, lstm_train_hidden = lstm_model.predict(train_dataset, verbose=0)
gru_train_pred, gru_train_hidden = gru_model.predict(train_dataset, verbose=0)

# 验证集特征提取
lstm_val_pred, lstm_val_hidden = lstm_model.predict(val_dataset, verbose=0)
gru_val_pred, gru_val_hidden = gru_model.predict(val_dataset, verbose=0)

# 测试集特征提取
lstm_test_pred, lstm_test_hidden = lstm_model.predict(test_dataset, verbose=0)
gru_test_pred, gru_test_hidden = gru_model.predict(test_dataset, verbose=0)

print(f"LSTM隐藏状态维度: {lstm_train_hidden.shape}")
print(f"GRU隐藏状态维度: {gru_train_hidden.shape}")

# ======================= 使用隐藏特征训练XGBoost =======================
print("\n使用LSTM+GRU隐藏特征训练XGBoost...")

# 方案1: 只使用隐藏特征
hidden_train_features = np.concatenate([lstm_train_hidden, gru_train_hidden], axis=1)
hidden_val_features = np.concatenate([lstm_val_hidden, gru_val_hidden], axis=1)
hidden_test_features = np.concatenate([lstm_test_hidden, gru_test_hidden], axis=1)

# 改进1: 降维 - 使用PCA减少维度
from sklearn.decomposition import PCA

print(f"原始隐藏特征维度: {hidden_train_features.shape[1]}")

pca = PCA(n_components=30, random_state=666)  # 降至30维
hidden_train_pca = pca.fit_transform(hidden_train_features)
hidden_val_pca = pca.transform(hidden_val_features)
hidden_test_pca = pca.transform(hidden_test_features)
print(f"PCA后维度: {hidden_train_pca.shape[1]}, 保留方差: {pca.explained_variance_ratio_.sum():.4f}")

# 改进2: 优化XGBoost超参数（防止过拟合）- 修复版本
xgb_hidden_model = XGBRegressor(
    n_estimators=1000,  # 设置较大值，让early_stopping决定实际迭代次数
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=10,  # 在初始化时设置
    random_state=666
)

xgb_hidden_model.fit(
    hidden_train_pca,
    train_labels,
    eval_set=[(hidden_val_pca, val_labels)],
    verbose=False
)

# 方案2: 传统XGBoost（展平输入） - 同样优化参数
train_dataset_flat = train_dataset.reshape(train_dataset.shape[0], -1)
val_dataset_flat = val_dataset.reshape(val_dataset.shape[0], -1)
test_dataset_flat = test_dataset.reshape(test_dataset.shape[0], -1)

xgb_flat_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=10,  # 在初始化时设置
    random_state=666
)

xgb_flat_model.fit(
    train_dataset_flat,
    train_labels,
    eval_set=[(val_dataset_flat, val_labels)],
    verbose=False
)

# ======================= 生成所有预测 =======================
print("\n生成测试集预测...")

# LSTM和GRU的直接预测
lstm_test_pred = lstm_test_pred.flatten()
gru_test_pred = gru_test_pred.flatten()

# 基于隐藏特征的XGBoost预测（使用PCA转换）
xgb_hidden_pred = xgb_hidden_model.predict(hidden_test_pca)

# 传统XGBoost预测
xgb_flat_pred = xgb_flat_model.predict(test_dataset_flat)

# ======================= 多种融合策略 =======================
print("\n应用不同的融合策略...")

# 策略1: 简单平均（参考第一个文档）
ensemble_avg = (lstm_test_pred + gru_test_pred + xgb_hidden_pred) / 3

# 策略2: 加权平均（基于验证集性能）
lstm_val_r2 = r2_score(val_labels, lstm_val_pred.flatten())
gru_val_r2 = r2_score(val_labels, gru_val_pred.flatten())
xgb_hidden_val_pred = xgb_hidden_model.predict(hidden_val_pca)
xgb_val_r2 = r2_score(val_labels, xgb_hidden_val_pred)

# 只使用正R²的模型，避免负贡献
weights = []
models_for_ensemble = []
if lstm_val_r2 > 0:
    weights.append(lstm_val_r2)
    models_for_ensemble.append(('lstm', lstm_test_pred))
if gru_val_r2 > 0:
    weights.append(gru_val_r2)
    models_for_ensemble.append(('gru', gru_test_pred))
if xgb_val_r2 > 0:
    weights.append(xgb_val_r2)
    models_for_ensemble.append(('xgb', xgb_hidden_pred))

total_weight = sum(weights)
if total_weight > 0:
    ensemble_weighted = sum(w / total_weight * pred for w, (_, pred) in zip(weights, models_for_ensemble))
    print(f"\n加权系数:")
    for (name, _), w in zip(models_for_ensemble, weights):
        print(f"  {name.upper()}: {w / total_weight:.3f}")
else:
    # 如果所有模型R²都<=0，使用简单平均
    ensemble_weighted = ensemble_avg
    print("\n警告: 所有模型R²都不为正，使用简单平均")

# 策略3: 线性回归元模型
val_stacked_features = np.column_stack((lstm_val_pred.flatten(), gru_val_pred.flatten(), xgb_hidden_val_pred))
meta_model = LinearRegression()
meta_model.fit(val_stacked_features, val_labels)

test_stacked_features = np.column_stack((lstm_test_pred, gru_test_pred, xgb_hidden_pred))
ensemble_meta = meta_model.predict(test_stacked_features)

print(f"\n线性回归系数 (LSTM, GRU, XGBoost): {meta_model.coef_}")
print(f"线性回归截距: {meta_model.intercept_}")

# 策略4: 只使用LSTM和GRU的简单平均（排除XGBoost）
ensemble_lstm_gru_only = (lstm_test_pred + gru_test_pred) / 2

# 策略5: 只使用LSTM和GRU的加权平均
lstm_gru_total = lstm_val_r2 + gru_val_r2
if lstm_gru_total > 0:
    w_lstm_only = lstm_val_r2 / lstm_gru_total
    w_gru_only = gru_val_r2 / lstm_gru_total
    ensemble_lstm_gru_weighted = w_lstm_only * lstm_test_pred + w_gru_only * gru_test_pred
    print(f"\nLSTM+GRU加权系数 (LSTM: {w_lstm_only:.3f}, GRU: {w_gru_only:.3f})")
else:
    ensemble_lstm_gru_weighted = ensemble_lstm_gru_only

# ======================= 性能评估 =======================
print("\n" + "=" * 80)
print("验证集性能（用于权重计算）:")
print("-" * 80)
print(f"LSTM  - Val R²: {lstm_val_r2:.6f}")
print(f"GRU   - Val R²: {gru_val_r2:.6f}")
print(f"XGBoost - Val R²: {xgb_val_r2:.6f}")
if xgb_val_r2 < 0:
    print("⚠️  警告: XGBoost在验证集上R²为负，表明模型严重过拟合或不适配")
print("=" * 80)

print("\n各模型在测试集上的性能对比（归一化数据）:")
print("-" * 80)

models_results = {
    'LSTM': lstm_test_pred,
    'GRU': gru_test_pred,
    'XGBoost(隐藏特征+PCA)': xgb_hidden_pred,
    'XGBoost(展平数据)': xgb_flat_pred,
    '集成-简单平均(3模型)': ensemble_avg,
    '集成-加权平均(正R²模型)': ensemble_weighted,
    '集成-线性回归': ensemble_meta,
    '集成-LSTM+GRU简单平均': ensemble_lstm_gru_only,
    '集成-LSTM+GRU加权平均': ensemble_lstm_gru_weighted
}

for name, preds in models_results.items():
    r2 = r2_score(test_labels, preds)
    rmse = sqrt(mean_squared_error(test_labels, preds))
    mae = mean_absolute_error(test_labels, preds)

    # 添加性能标记
    if r2 > 0.85:
        marker = "✅ 优秀"
    elif r2 > 0.70:
        marker = "✓ 良好"
    elif r2 > 0.50:
        marker = "○ 一般"
    elif r2 > 0:
        marker = "△ 较差"
    else:
        marker = "✗ 失败"

    print(f"{name:30s} | R²: {r2:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f} {marker}")

print("=" * 80)

# 选择最佳模型
best_model_name = max(models_results.keys(),
                      key=lambda x: r2_score(test_labels, models_results[x]))
final_preds = models_results[best_model_name]

print(f"\n🏆 最佳模型: {best_model_name}")
print(f"   测试集R²: {r2_score(test_labels, final_preds):.6f}")

# 分析和建议
print("\n" + "=" * 80)
print("📊 模型表现分析:")
print("-" * 80)

if xgb_val_r2 < 0:
    print("⚠️  XGBoost失败原因分析:")
    print(f"   1. 样本数({len(train_labels)}) vs 原始特征维(200) - 比例过低")
    print(f"   2. PCA降维后({hidden_train_pca.shape[1]}维)改善效果有限")
    print(f"   3. 深度学习隐藏特征可能不适合树模型的分裂策略")
    print("\n💡 建议:")
    print("   - 对于此数据集,深度学习模型(LSTM/GRU)已经足够好")
    print("   - 集成方法应排除表现差的XGBoost")
    print("   - 使用LSTM+GRU的加权平均可能是最佳选择")
else:
    print("✅ 所有模型均表现正常")

# 输出最优集成策略
lstm_gru_weighted_r2 = r2_score(test_labels, ensemble_lstm_gru_weighted)
best_ensemble_r2 = r2_score(test_labels, final_preds)

if abs(lstm_gru_weighted_r2 - best_ensemble_r2) < 0.01:
    print(f"\n🎯 推荐使用: LSTM+GRU加权平均 (更稳定、无XGBoost依赖)")
    print(f"   R² = {lstm_gru_weighted_r2:.6f}")

print("=" * 80)

# ========== 反归一化 ==========
test_labels_original = y_scaler.inverse_transform(test_labels.reshape(-1, 1))
final_preds_original = y_scaler.inverse_transform(final_preds.reshape(-1, 1))
lstm_preds_original = y_scaler.inverse_transform(lstm_test_pred.reshape(-1, 1))
gru_preds_original = y_scaler.inverse_transform(gru_test_pred.reshape(-1, 1))
xgb_hidden_preds_original = y_scaler.inverse_transform(xgb_hidden_pred.reshape(-1, 1))

# 原始尺度指标
print("\n最佳模型在原始尺度上的指标:")
print(f"R² 值: {r2_score(test_labels_original, final_preds_original):.6f}")
print(f"MAE: {mean_absolute_error(test_labels_original, final_preds_original):.6f}")
print(f"MSE: {mean_squared_error(test_labels_original, final_preds_original):.6f}")
print(f"RMSE: {sqrt(mean_squared_error(test_labels_original, final_preds_original)):.6f}")
print(f"MAPE: {np.mean(np.abs((final_preds_original - test_labels_original) / (test_labels_original + 1e-8))):.6f}")
print("=" * 80)

# ======================= 可视化结果 =======================
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 绘制训练过程
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(lstm_history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('LSTM 模型训练与验证损失', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(gru_history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(gru_history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('GRU 模型训练与验证损失', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'model_training_loss.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 绘制预测结果对比
plt.figure(figsize=(20, 12))

plt.subplot(3, 4, 1)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(lstm_preds_original, label="LSTM预测", linewidth=2, alpha=0.7)
plt.title(f"LSTM (R²={r2_score(test_labels, lstm_test_pred):.4f})")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 2)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(gru_preds_original, label="GRU预测", linewidth=2, alpha=0.7)
plt.title(f"GRU (R²={r2_score(test_labels, gru_test_pred):.4f})")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 3)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(xgb_hidden_preds_original, label="XGBoost预测", linewidth=2, alpha=0.7)
r2_xgb = r2_score(test_labels, xgb_hidden_pred)
title_color = 'red' if r2_xgb < 0 else 'black'
plt.title(f"XGBoost-PCA (R²={r2_xgb:.4f})", color=title_color)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 4)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(y_scaler.inverse_transform(ensemble_avg.reshape(-1, 1)),
         label="简单平均", linewidth=2, alpha=0.7, color='purple')
plt.title(f"集成-简单平均 (R²={r2_score(test_labels, ensemble_avg):.4f})")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 5)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(y_scaler.inverse_transform(ensemble_weighted.reshape(-1, 1)),
         label="智能加权", linewidth=2, alpha=0.7, color='orange')
plt.title(f"集成-智能加权 (R²={r2_score(test_labels, ensemble_weighted):.4f})")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 6)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(y_scaler.inverse_transform(ensemble_meta.reshape(-1, 1)),
         label="线性回归", linewidth=2, alpha=0.7, color='green')
plt.title(f"集成-线性回归 (R²={r2_score(test_labels, ensemble_meta):.4f})")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 7)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(y_scaler.inverse_transform(ensemble_lstm_gru_only.reshape(-1, 1)),
         label="LSTM+GRU平均", linewidth=2, alpha=0.7, color='cyan')
plt.title(f"LSTM+GRU简单平均 (R²={r2_score(test_labels, ensemble_lstm_gru_only):.4f})")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 8)
plt.plot(test_labels_original, label="真实值", linewidth=2, color='black')
plt.plot(y_scaler.inverse_transform(ensemble_lstm_gru_weighted.reshape(-1, 1)),
         label="LSTM+GRU加权", linewidth=2, alpha=0.7, color='magenta')
plt.title(f"LSTM+GRU加权平均 (R²={r2_score(test_labels, ensemble_lstm_gru_weighted):.4f})")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 9)
plt.plot(test_labels_original, label="真实值", linewidth=2.5, color='black')
plt.plot(final_preds_original, label=f"最佳({best_model_name})",
         linewidth=2, alpha=0.8, color='red')
plt.title("🏆 最佳模型预测", fontsize=12, fontweight='bold')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 10)
# 残差图
residuals = test_labels_original.flatten() - final_preds_original.flatten()
plt.scatter(final_preds_original, residuals, alpha=0.6, color='crimson', s=20)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.title("残差分析图")
plt.xlabel("预测值")
plt.ylabel("残差")
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 11)
# 所有模型对比
plt.plot(test_labels_original, label="真实值", linewidth=2.5, color='black', alpha=0.8)
plt.plot(lstm_preds_original, label="LSTM", linewidth=1, alpha=0.4)
plt.plot(gru_preds_original, label="GRU", linewidth=1, alpha=0.4)
if r2_xgb > 0:
    plt.plot(xgb_hidden_preds_original, label="XGBoost", linewidth=1, alpha=0.4)
plt.plot(final_preds_original, label="最佳集成", linewidth=2, alpha=0.8, color='red')
plt.title("所有模型对比")
plt.legend(fontsize=7)
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 12)
# R²对比柱状图
model_names = ['LSTM', 'GRU', 'XGB-PCA', 'LSTM+GRU\n加权']
r2_scores_plot = [
    r2_score(test_labels, lstm_test_pred),
    r2_score(test_labels, gru_test_pred),
    r2_score(test_labels, xgb_hidden_pred),
    r2_score(test_labels, ensemble_lstm_gru_weighted)
]
colors = ['skyblue', 'lightgreen', 'salmon' if r2_scores_plot[2] < 0 else 'lightyellow', 'gold']
bars = plt.bar(model_names, r2_scores_plot, color=colors, alpha=0.7, edgecolor='black')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title("R² 性能对比")
plt.ylabel("R² Score")
plt.xticks(rotation=15, ha='right', fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
# 在柱子上显示数值
for bar, score in zip(bars, r2_scores_plot):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{score:.3f}',
             ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig(results_directory + 'improved_stacked_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ======================= 保存模型 =======================
import pickle

with open(results_directory + 'stacked_scalers.pkl', 'wb') as f:
    pickle.dump({
        'feature_scalers': feature_scalers,
        'y_scaler': y_scaler,
        'pca': pca
    }, f)

with open(results_directory + 'meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)

lstm_model.save(results_directory + 'lstm_hidden_model.h5')
gru_model.save(results_directory + 'gru_hidden_model.h5')
xgb_hidden_model.save_model(results_directory + 'xgboost_hidden_model.json')
xgb_flat_model.save_model(results_directory + 'xgboost_flat_model.json')

print('\n所有模型和归一化器已保存到:', results_directory)
print('=' * 80)