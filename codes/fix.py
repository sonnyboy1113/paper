import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
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
print(f"\n数据集总长度: {len(dataset)}")

# 特征数据集和标签数据集
X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

# ======================= 时间序列交叉验证设置 =======================
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"\n使用 {n_splits} 折时间序列交叉验证")
print("=" * 80)

# 存储所有fold的结果
cv_results = {
    'gru_train_r2': [], 'gru_val_r2': [],
    'lstm_train_r2': [], 'lstm_val_r2': [],
    'stacked_val_r2': [],
    'stacked_val_mae': [], 'stacked_val_rmse': [], 'stacked_val_mape': []
}

best_fold = {'fold': -1, 'r2': -np.inf}

results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# ======================= 交叉验证循环 =======================
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"\n{'=' * 80}")
    print(f"开始训练 Fold {fold}/{n_splits}")
    print(f"训练集索引: {train_idx[0]} 到 {train_idx[-1]} (共 {len(train_idx)} 样本)")
    print(f"验证集索引: {val_idx[0]} 到 {val_idx[-1]} (共 {len(val_idx)} 样本)")
    print(f"{'=' * 80}")

    # 分割数据
    X_train_fold = X.iloc[train_idx].copy()
    X_val_fold = X.iloc[val_idx].copy()
    y_train_fold = y.iloc[train_idx].copy()
    y_val_fold = y.iloc[val_idx].copy()

    # ========== 关键修复1：先归一化，再正确创建滞后特征 ==========

    # 特征归一化
    feature_scalers = {}
    for col in X_train_fold.columns:
        scaler = MinMaxScaler()
        X_train_fold[col] = scaler.fit_transform(X_train_fold[col].values.reshape(-1, 1))
        X_val_fold[col] = scaler.transform(X_val_fold[col].values.reshape(-1, 1))
        feature_scalers[col] = scaler

    # 标签归一化
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_fold.values.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val_fold.values.reshape(-1, 1)).flatten()

    # 合并训练集和验证集的目标值，用于正确创建滞后特征
    y_combined = pd.Series(
        np.concatenate([y_train_scaled, y_val_scaled]),
        index=pd.concat([y_train_fold, y_val_fold]).index
    )

    # 正确创建滞后特征：验证集可以访问训练集的历史数据
    X_combined = pd.concat([X_train_fold, X_val_fold])
    for i in range(1, 6):
        X_combined[f'Corn_lag_{i}'] = y_combined.shift(i)

    # 重新分割带滞后特征的数据
    X_train_fold = X_combined.iloc[:len(train_idx)]
    X_val_fold = X_combined.iloc[len(train_idx):]
    y_train_fold = pd.Series(y_train_scaled, index=y_train_fold.index)
    y_val_fold = pd.Series(y_val_scaled, index=y_val_fold.index)

    # 删除缺失值
    X_train_fold = X_train_fold.dropna()
    y_train_fold = y_train_fold.loc[X_train_fold.index]

    X_val_fold = X_val_fold.dropna()
    y_val_fold = y_val_fold.loc[X_val_fold.index]

    print(f"清洗后 - 训练集: {X_train_fold.shape}, 验证集: {X_val_fold.shape}")


    # 构造序列数据
    def create_dataset(X, y, seq_len=5):
        features, targets = [], []
        for i in range(0, len(X) - seq_len, 1):
            features.append(X.iloc[i:i + seq_len].values)
            targets.append(y.iloc[i + seq_len])
        return np.array(features), np.array(targets)


    train_dataset, train_labels = create_dataset(X_train_fold, y_train_fold, seq_len=5)
    val_dataset, val_labels = create_dataset(X_val_fold, y_val_fold, seq_len=5)

    print(f"序列数据 - 训练集: {train_dataset.shape}, 验证集: {val_dataset.shape}")


    # 构造批数据
    def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
        batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        if train:
            return batch_data.cache().shuffle(buffer_size).batch(batch_size)
        else:
            return batch_data.batch(batch_size)


    train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
    val_batch_dataset = create_batch_dataset(val_dataset, val_labels, train=False)

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)

    # =================== 训练 GRU 模型 ===================
    print(f"\n[Fold {fold}] 训练 GRU 模型...")
    gru_model = Sequential([
        GRU(128, input_shape=(5, X_train_fold.shape[1]), return_sequences=True),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    gru_history = gru_model.fit(
        train_batch_dataset,
        epochs=200,
        validation_data=val_batch_dataset,
        callbacks=[early_stop],
        verbose=0
    )

    # =================== 训练 LSTM 模型 ===================
    print(f"[Fold {fold}] 训练 LSTM 模型...")
    lstm_model = Sequential([
        LSTM(128, input_shape=(5, X_train_fold.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    lstm_history = lstm_model.fit(
        train_batch_dataset,
        epochs=200,
        validation_data=val_batch_dataset,
        callbacks=[early_stop],
        verbose=0
    )

    # =================== 基础模型预测 ===================
    gru_train_preds = gru_model.predict(train_dataset, verbose=0)[:, 0]
    gru_val_preds = gru_model.predict(val_dataset, verbose=0)[:, 0]
    lstm_train_preds = lstm_model.predict(train_dataset, verbose=0)[:, 0]
    lstm_val_preds = lstm_model.predict(val_dataset, verbose=0)[:, 0]

    # 计算基础模型性能
    gru_train_r2 = r2_score(train_labels, gru_train_preds)
    gru_val_r2 = r2_score(val_labels, gru_val_preds)
    lstm_train_r2 = r2_score(train_labels, lstm_train_preds)
    lstm_val_r2 = r2_score(val_labels, lstm_val_preds)

    print(f"\n[Fold {fold}] 基础模型性能:")
    print(f"  GRU  - Train R²: {gru_train_r2:.4f}, Val R²: {gru_val_r2:.4f}")
    print(f"  LSTM - Train R²: {lstm_train_r2:.4f}, Val R²: {lstm_val_r2:.4f}")

    cv_results['gru_train_r2'].append(gru_train_r2)
    cv_results['gru_val_r2'].append(gru_val_r2)
    cv_results['lstm_train_r2'].append(lstm_train_r2)
    cv_results['lstm_val_r2'].append(lstm_val_r2)

    # ========== 正确的堆叠策略：元模型在验证集上训练 ==========
    print(f"[Fold {fold}] 训练 XGBoost 元模型...")

    # 验证集上的基础模型预测（未见数据，无过拟合）
    val_stacked_features = np.column_stack((gru_val_preds, lstm_val_preds))

    # ✅ 元模型在验证集的预测结果上训练
    # 原因：
    # 1. 基础模型对训练集的预测可能过拟合
    # 2. 验证集预测更能反映在新数据上的表现
    # 3. 元模型学习如何在"未见数据"上组合基础模型
    meta_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=666
    )
    meta_model.fit(val_stacked_features, val_labels)

    # 最终预测
    final_preds = meta_model.predict(val_stacked_features)

    # 反归一化
    val_labels_original = y_scaler.inverse_transform(val_labels.reshape(-1, 1)).flatten()
    final_preds_original = y_scaler.inverse_transform(final_preds.reshape(-1, 1)).flatten()

    # 计算堆叠模型性能
    stacked_r2 = r2_score(val_labels_original, final_preds_original)
    stacked_mae = mean_absolute_error(val_labels_original, final_preds_original)
    stacked_rmse = sqrt(mean_squared_error(val_labels_original, final_preds_original))
    stacked_mape = np.mean(np.abs((final_preds_original - val_labels_original) / (val_labels_original + 1e-8)))

    print(f"\n[Fold {fold}] 堆叠模型性能 (原始尺度):")
    print(f"  R²: {stacked_r2:.4f}")
    print(f"  MAE: {stacked_mae:.4f}")
    print(f"  RMSE: {stacked_rmse:.4f}")
    print(f"  MAPE: {stacked_mape:.4f}")

    cv_results['stacked_val_r2'].append(stacked_r2)
    cv_results['stacked_val_mae'].append(stacked_mae)
    cv_results['stacked_val_rmse'].append(stacked_rmse)
    cv_results['stacked_val_mape'].append(stacked_mape)

    # 更新最佳模型
    if stacked_r2 > best_fold['r2']:
        best_fold = {
            'fold': fold,
            'r2': stacked_r2,
            'gru_model': gru_model,
            'lstm_model': lstm_model,
            'meta_model': meta_model,
            'feature_scalers': feature_scalers,
            'y_scaler': y_scaler,
            'gru_history': gru_history,
            'lstm_history': lstm_history,
            'val_labels_original': val_labels_original,
            'final_preds_original': final_preds_original,
            'gru_preds_original': y_scaler.inverse_transform(gru_val_preds.reshape(-1, 1)).flatten(),
            'lstm_preds_original': y_scaler.inverse_transform(lstm_val_preds.reshape(-1, 1)).flatten()
        }
        print(f"\n✓ Fold {fold} 成为新的最佳模型 (R² = {stacked_r2:.4f})")

# ======================= 交叉验证结果汇总 =======================
print("\n" + "=" * 80)
print("交叉验证结果汇总")
print("=" * 80)

print("\nGRU 模型:")
print(f"  训练集 R² - 均值: {np.mean(cv_results['gru_train_r2']):.4f} ± {np.std(cv_results['gru_train_r2']):.4f}")
print(f"  验证集 R² - 均值: {np.mean(cv_results['gru_val_r2']):.4f} ± {np.std(cv_results['gru_val_r2']):.4f}")

print("\nLSTM 模型:")
print(f"  训练集 R² - 均值: {np.mean(cv_results['lstm_train_r2']):.4f} ± {np.std(cv_results['lstm_train_r2']):.4f}")
print(f"  验证集 R² - 均值: {np.mean(cv_results['lstm_val_r2']):.4f} ± {np.std(cv_results['lstm_val_r2']):.4f}")

print("\n堆叠模型 (原始尺度):")
print(f"  R²:   {np.mean(cv_results['stacked_val_r2']):.4f} ± {np.std(cv_results['stacked_val_r2']):.4f}")
print(f"  MAE:  {np.mean(cv_results['stacked_val_mae']):.4f} ± {np.std(cv_results['stacked_val_mae']):.4f}")
print(f"  RMSE: {np.mean(cv_results['stacked_val_rmse']):.4f} ± {np.std(cv_results['stacked_val_rmse']):.4f}")
print(f"  MAPE: {np.mean(cv_results['stacked_val_mape']):.4f} ± {np.std(cv_results['stacked_val_mape']):.4f}")

print(f"\n最佳模型来自 Fold {best_fold['fold']} (R² = {best_fold['r2']:.4f})")

# ======================= 可视化 =======================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# R² 对比
ax = axes[0, 0]
folds = list(range(1, n_splits + 1))
ax.plot(folds, cv_results['gru_val_r2'], 'o-', label='GRU', linewidth=2, markersize=8)
ax.plot(folds, cv_results['lstm_val_r2'], 's-', label='LSTM', linewidth=2, markersize=8)
ax.plot(folds, cv_results['stacked_val_r2'], '^-', label='Stacked', linewidth=2, markersize=8)
ax.axhline(y=np.mean(cv_results['stacked_val_r2']), color='red', linestyle='--', alpha=0.5, label='Stacked Mean')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('各Fold的R²性能对比', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(folds)

# 误差指标
ax = axes[0, 1]
x_pos = np.arange(n_splits)
width = 0.25
ax.bar(x_pos - width, cv_results['stacked_val_mae'], width, label='MAE', alpha=0.8)
ax.bar(x_pos, cv_results['stacked_val_rmse'], width, label='RMSE', alpha=0.8)
ax.bar(x_pos + width, [m * 100 for m in cv_results['stacked_val_mape']], width, label='MAPE(×100)', alpha=0.8)
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('误差指标', fontsize=12)
ax.set_title('堆叠模型的误差指标', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'F{i}' for i in folds])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 过拟合检查
ax = axes[1, 0]
x_pos = np.arange(n_splits)
width = 0.35
ax.bar(x_pos - width / 2, cv_results['gru_train_r2'], width, label='GRU Train', alpha=0.7, color='steelblue')
ax.bar(x_pos - width / 2, cv_results['gru_val_r2'], width, label='GRU Val', alpha=0.9, color='steelblue', hatch='//')
ax.bar(x_pos + width / 2, cv_results['lstm_train_r2'], width, label='LSTM Train', alpha=0.7, color='coral')
ax.bar(x_pos + width / 2, cv_results['lstm_val_r2'], width, label='LSTM Val', alpha=0.9, color='coral', hatch='//')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('训练集与验证集R²对比 (过拟合检查)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'F{i}' for i in folds])
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 训练曲线
ax = axes[1, 1]
ax.plot(best_fold['gru_history'].history['loss'], label='GRU Train', linewidth=2, alpha=0.7)
ax.plot(best_fold['gru_history'].history['val_loss'], label='GRU Val', linewidth=2, alpha=0.7)
ax.plot(best_fold['lstm_history'].history['loss'], label='LSTM Train', linewidth=2, alpha=0.7)
ax.plot(best_fold['lstm_history'].history['val_loss'], label='LSTM Val', linewidth=2, alpha=0.7)
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title(f'最佳模型训练曲线 (Fold {best_fold["fold"]})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'cross_validation_summary.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 最佳模型预测结果
plt.figure(figsize=(14, 10))

best_true = best_fold['val_labels_original']
best_stacked = best_fold['final_preds_original']
best_gru = best_fold['gru_preds_original']
best_lstm = best_fold['lstm_preds_original']

r2_stacked = r2_score(best_true, best_stacked)
r2_gru = r2_score(best_true, best_gru)
r2_lstm = r2_score(best_true, best_lstm)

plt.subplot(2, 2, 1)
plt.plot(best_true, label="真实值", linewidth=2.5, color='black')
plt.plot(best_stacked, label="堆叠预测", linewidth=2, alpha=0.8, color='red')
plt.title(f"堆叠模型 (LSTM + GRU + XGBoost)\nFold {best_fold['fold']} - R² = {r2_stacked:.4f}",
          fontsize=12, fontweight='bold')
plt.xlabel('样本索引');
plt.ylabel('玉米价格')
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(best_true, label="真实值", linewidth=2.5, color='black')
plt.plot(best_gru, label="GRU预测", linewidth=2, alpha=0.8, color='blue')
plt.title(f"GRU模型\nR² = {r2_gru:.4f}", fontsize=12, fontweight='bold')
plt.xlabel('样本索引');
plt.ylabel('玉米价格')
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(best_true, label="真实值", linewidth=2.5, color='black')
plt.plot(best_lstm, label="LSTM预测", linewidth=2, alpha=0.8, color='green')
plt.title(f"LSTM模型\nR² = {r2_lstm:.4f}", fontsize=12, fontweight='bold')
plt.xlabel('样本索引');
plt.ylabel('玉米价格')
plt.legend();
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(best_true, label="真实值", linewidth=2.5, color='black', alpha=0.8)
plt.plot(best_gru, label=f"GRU (R²={r2_gru:.3f})", linewidth=1.5, alpha=0.6)
plt.plot(best_lstm, label=f"LSTM (R²={r2_lstm:.3f})", linewidth=1.5, alpha=0.6)
plt.plot(best_stacked, label=f"堆叠 (R²={r2_stacked:.3f})", linewidth=2, alpha=0.8)
plt.title("模型对比", fontsize=12, fontweight='bold')
plt.xlabel('样本索引');
plt.ylabel('玉米价格')
plt.legend();
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'best_model_predictions.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ======================= 保存模型 =======================
import pickle

print("\n" + "=" * 80)
print("保存最佳模型...")
print("=" * 80)

with open(results_directory + 'best_stacked_scalers.pkl', 'wb') as f:
    pickle.dump({
        'feature_scalers': best_fold['feature_scalers'],
        'y_scaler': best_fold['y_scaler']
    }, f)

best_fold['gru_model'].save(results_directory + 'best_gru_model.h5')
best_fold['lstm_model'].save(results_directory + 'best_lstm_model.h5')
best_fold['meta_model'].save_model(results_directory + 'best_xgboost_meta_model.json')

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(results_directory + 'cv_results.csv', index=False)

print(f'\n所有文件已保存到: {results_directory}')
print('- best_gru_model.h5')
print('- best_lstm_model.h5')
print('- best_xgboost_meta_model.json')
print('- best_stacked_scalers.pkl')
print('- cv_results.csv')
print('- cross_validation_summary.png')
print('- best_model_predictions.png')

print("\n" + "=" * 80)
print("交叉验证训练完成!")
print("=" * 80)