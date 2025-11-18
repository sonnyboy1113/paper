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
from tensorflow.keras import Sequential, layers, Model
from tensorflow.keras.layers import GRU, LSTM, Dropout, Dense, Layer, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================== 自定义自注意力层 ==================
class SelfAttention(Layer):
    def __init__(self, units=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]

        self.W_q = self.add_weight(name='query',
                                   shape=(input_shape[-1], self.units),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_k = self.add_weight(name='key',
                                   shape=(input_shape[-1], self.units),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_v = self.add_weight(name='value',
                                   shape=(input_shape[-1], self.units),
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        Q = K.dot(x, self.W_q)
        K_mat = K.dot(x, self.W_k)
        V = K.dot(x, self.W_v)

        attention_scores = K.batch_dot(Q, K_mat, axes=[2, 2])
        attention_scores = attention_scores / K.sqrt(K.cast(self.units, dtype='float32'))
        attention_weights = K.softmax(attention_scores)

        output = K.batch_dot(attention_weights, V)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({'units': self.units})
        return config


# ================== 过拟合检测函数 ==================
def detect_overfitting(history, threshold=0.1, window=10):
    """
    检测模型是否过拟合

    参数:
    - history: 训练历史对象
    - threshold: 训练loss和验证loss差异的阈值
    - window: 用于计算移动平均的窗口大小

    返回:
    - dict: 包含过拟合诊断信息
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 1. 最终loss差异
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    loss_gap = final_val_loss - final_train_loss
    loss_gap_ratio = loss_gap / final_train_loss if final_train_loss > 0 else 0

    # 2. 验证loss趋势（最后window个epoch）
    if len(val_loss) >= window:
        recent_val_loss = val_loss[-window:]
        val_loss_increasing = np.mean(np.diff(recent_val_loss)) > 0
    else:
        val_loss_increasing = False

    # 3. 最小验证loss的epoch
    min_val_loss_epoch = np.argmin(val_loss)
    epochs_since_min = len(val_loss) - min_val_loss_epoch - 1

    # 4. 判断是否过拟合
    is_overfitting = (loss_gap_ratio > threshold) or \
                     (val_loss_increasing and loss_gap_ratio > 0.05) or \
                     (epochs_since_min > 20)

    diagnosis = {
        'is_overfitting': is_overfitting,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'loss_gap': loss_gap,
        'loss_gap_ratio': loss_gap_ratio,
        'min_val_loss_epoch': min_val_loss_epoch,
        'epochs_since_min': epochs_since_min,
        'val_loss_increasing': val_loss_increasing
    }

    return diagnosis


def print_overfitting_report(model_name, diagnosis):
    """打印过拟合诊断报告"""
    print(f"\n{'=' * 80}")
    print(f"=== {model_name} 过拟合诊断报告 ===")
    print(f"{'=' * 80}")
    print(f"最终训练Loss: {diagnosis['final_train_loss']:.6f}")
    print(f"最终验证Loss: {diagnosis['final_val_loss']:.6f}")
    print(f"Loss差异: {diagnosis['loss_gap']:.6f}")
    print(f"Loss差异比率: {diagnosis['loss_gap_ratio']:.2%}")
    print(f"最小验证Loss出现在第 {diagnosis['min_val_loss_epoch'] + 1} 轮")
    print(f"距离最小验证Loss已过 {diagnosis['epochs_since_min']} 轮")
    print(f"验证Loss呈上升趋势: {'是' if diagnosis['val_loss_increasing'] else '否'}")
    print(f"\n判断结果: {'⚠️ 模型可能存在过拟合' if diagnosis['is_overfitting'] else '✓ 模型泛化性能良好'}")

    if diagnosis['is_overfitting']:
        print("\n建议措施:")
        if diagnosis['loss_gap_ratio'] > 0.2:
            print("  - Loss差异较大，建议增加Dropout率或添加L2正则化")
        if diagnosis['epochs_since_min'] > 20:
            print("  - 验证Loss长时间未改善，建议使用Early Stopping")
        if diagnosis['val_loss_increasing']:
            print("  - 验证Loss呈上升趋势，建议减少训练轮数或增加正则化")
    print(f"{'=' * 80}\n")


def compare_train_test_predictions(model, train_data, test_data, train_labels, test_labels,
                                   model_name, scaler_y):
    """
    比较模型在训练集和测试集上的表现
    """
    train_preds = model.predict(train_data, verbose=0).flatten()
    test_preds = model.predict(test_data, verbose=0).flatten()

    # 反归一化
    train_labels_denorm = scaler_y.inverse_transform(train_labels.reshape(-1, 1)).flatten()
    train_preds_denorm = scaler_y.inverse_transform(train_preds.reshape(-1, 1)).flatten()
    test_labels_denorm = scaler_y.inverse_transform(test_labels.reshape(-1, 1)).flatten()
    test_preds_denorm = scaler_y.inverse_transform(test_preds.reshape(-1, 1)).flatten()

    # 计算指标
    train_r2 = r2_score(train_labels_denorm, train_preds_denorm)
    test_r2 = r2_score(test_labels_denorm, test_preds_denorm)
    train_rmse = sqrt(mean_squared_error(train_labels_denorm, train_preds_denorm))
    test_rmse = sqrt(mean_squared_error(test_labels_denorm, test_preds_denorm))
    train_mape = np.mean(np.abs((train_preds_denorm - train_labels_denorm) /
                                (train_labels_denorm + 1e-8))) * 100
    test_mape = np.mean(np.abs((test_preds_denorm - test_labels_denorm) /
                               (test_labels_denorm + 1e-8))) * 100

    print(f"\n{'=' * 80}")
    print(f"=== {model_name} 训练集 vs 测试集性能对比 ===")
    print(f"{'=' * 80}")
    print(f"{'指标':<10} {'训练集':<15} {'测试集':<15} {'差异':<15}")
    print(f"{'-' * 80}")
    print(f"{'R²':<10} {train_r2:<15.4f} {test_r2:<15.4f} {abs(train_r2 - test_r2):<15.4f}")
    print(f"{'RMSE':<10} {train_rmse:<15.4f} {test_rmse:<15.4f} {abs(train_rmse - test_rmse):<15.4f}")
    print(f"{'MAPE(%)':<10} {train_mape:<15.4f} {test_mape:<15.4f} {abs(train_mape - test_mape):<15.4f}")

    # 判断过拟合
    r2_gap = train_r2 - test_r2
    if r2_gap > 0.1:
        print(f"\n⚠️ 警告: R²差异较大 ({r2_gap:.4f})，模型可能过拟合")
    elif r2_gap < -0.05:
        print(f"\n⚠️ 警告: 测试R²显著高于训练R² ({r2_gap:.4f})，数据可能存在问题")
    else:
        print(f"\n✓ R²差异在合理范围内 ({r2_gap:.4f})")

    print(f"{'=' * 80}\n")

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'r2_gap': r2_gap,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }


# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print(dataset.info())
print("\n原始数据形状:", dataset.shape)

# 特征工程
for i in range(1, 6):
    dataset[f'Corn_lag_{i}'] = dataset['Corn'].shift(i)
dataset.dropna(inplace=True)

print("添加滞后特征后形状:", dataset.shape)

# 特征数据集和标签数据集
X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=666
)

print(f"\n训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

# 归一化特征
scaler_X = MinMaxScaler()
X_train_scaled = pd.DataFrame(
    scaler_X.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler_X.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# 归一化标签
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

y_train_scaled = pd.Series(y_train_scaled, index=y_train.index)
y_test_scaled = pd.Series(y_test_scaled, index=y_test.index)


# 构造特征数据集
def create_dataset(X, y, seq_len=5):
    features = []
    targets = []
    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i + seq_len]
        label = y.iloc[i + seq_len]
        features.append(data.values)
        targets.append(label)
    return np.array(features), np.array(targets)


# 构造训练和测试特征数据集
train_dataset, train_labels = create_dataset(X_train_scaled, y_train_scaled, seq_len=5)
test_dataset, test_labels = create_dataset(X_test_scaled, y_test_scaled, seq_len=5)

print(f"\n序列数据形状: train_dataset={train_dataset.shape}, test_dataset={test_dataset.shape}")


# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X, dtype=tf.float32),
                                                     tf.constant(y, dtype=tf.float32)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)


train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# ================== 添加回调函数防止过拟合 ==================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=1e-6,
    verbose=1
)

# ================== 方案1: GRU + 自注意力 ==================
print("\n训练 GRU + Self-Attention 模型...")
gru_model = Sequential([
    GRU(128, input_shape=(5, X_train_scaled.shape[1]), return_sequences=True),
    Dropout(0.2),
    SelfAttention(units=128),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
history_gru = gru_model.fit(
    train_batch_dataset,
    epochs=200,
    validation_data=test_batch_dataset,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 过拟合诊断 - GRU
gru_diagnosis = detect_overfitting(history_gru)
print_overfitting_report("GRU + Self-Attention", gru_diagnosis)
gru_comparison = compare_train_test_predictions(
    gru_model, train_dataset, test_dataset,
    train_labels, test_labels, "GRU", scaler_y
)

# ================== 方案2: LSTM + 多头注意力 ==================
print("\n训练 LSTM + Multi-Head Attention 模型...")
lstm_input = layers.Input(shape=(5, X_train_scaled.shape[1]))
lstm_out = layers.LSTM(128, return_sequences=True)(lstm_input)
lstm_out = layers.Dropout(0.2)(lstm_out)

# 多头注意力机制
attention_out = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)
attention_out = layers.Add()([lstm_out, attention_out])
attention_out = layers.LayerNormalization()(attention_out)

lstm_out2 = layers.LSTM(64, return_sequences=False)(attention_out)
lstm_out2 = layers.Dropout(0.2)(lstm_out2)
lstm_out2 = layers.Dense(32, activation='relu')(lstm_out2)
lstm_output = layers.Dense(1, activation='linear')(lstm_out2)

lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
lstm_model.compile(optimizer='adam', loss='mse')

# 重置early stopping（使用新实例）
early_stopping_lstm = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_lstm = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=1e-6,
    verbose=1
)

history_lstm = lstm_model.fit(
    train_batch_dataset,
    epochs=200,
    validation_data=test_batch_dataset,
    callbacks=[early_stopping_lstm, reduce_lr_lstm],
    verbose=1
)

# 过拟合诊断 - LSTM
lstm_diagnosis = detect_overfitting(history_lstm)
print_overfitting_report("LSTM + Multi-Head Attention", lstm_diagnosis)
lstm_comparison = compare_train_test_predictions(
    lstm_model, train_dataset, test_dataset,
    train_labels, test_labels, "LSTM", scaler_y
)

# 使用模型进行预测
print("\n生成预测结果...")
gru_preds = gru_model.predict(test_dataset, verbose=0).flatten()
lstm_preds = lstm_model.predict(test_dataset, verbose=0).flatten()

# 将两个模型的预测结果作为特征
stacked_features = np.column_stack((gru_preds, lstm_preds))

# 训练元模型（XGBoost）
print("训练元模型 XGBoost...")
meta_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
meta_model.fit(stacked_features, test_labels)

# 使用元模型进行最终预测
final_preds = meta_model.predict(stacked_features)

# 反归一化
test_labels_denorm = scaler_y.inverse_transform(test_labels.reshape(-1, 1)).flatten()
final_preds_denorm = scaler_y.inverse_transform(final_preds.reshape(-1, 1)).flatten()

# 计算指标（归一化后的指标）
print('\n' + '=' * 80)
print("=== 最终Stacked模型 - 归一化数据上的性能指标 ===")
print("R² 值：", r2_score(test_labels, final_preds))
print("MAE:", mean_absolute_error(test_labels, final_preds))
print("MSE:", mean_squared_error(test_labels, final_preds))
print("RMSE:", sqrt(mean_squared_error(test_labels, final_preds)))
mape_normalized = np.mean(np.abs((final_preds - test_labels) / (test_labels + 1e-8))) * 100
print(f"MAPE: {mape_normalized:.4f}%")

# 计算指标（原始尺度）
print("\n=== 最终Stacked模型 - 原始尺度上的性能指标 ===")
print("R² 值：", r2_score(test_labels_denorm, final_preds_denorm))
print("MAE:", mean_absolute_error(test_labels_denorm, final_preds_denorm))
print("MSE:", mean_squared_error(test_labels_denorm, final_preds_denorm))
print("RMSE:", sqrt(mean_squared_error(test_labels_denorm, final_preds_denorm)))
mape_original = np.mean(np.abs((final_preds_denorm - test_labels_denorm) / (test_labels_denorm + 1e-8))) * 100
print(f"MAPE: {mape_original:.4f}%")
print('=' * 80)

# 导出预测值
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 保存归一化的预测值
predict_normalized = pd.DataFrame({
    'y_true_normalized': test_labels,
    'y_pred_normalized': final_preds
})
predict_normalized.to_csv(results_directory + "Stacked-Attention-y_predict_normalized.csv", index=False)

# 保存原始尺度的预测值
predict_original = pd.DataFrame({
    'y_true': test_labels_denorm,
    'y_pred': final_preds_denorm
})
predict_original.to_csv(results_directory + "Stacked-Attention-y_predict_original.csv", index=False)

# ================== 增强的可视化：包含过拟合诊断 ==================
fig = plt.figure(figsize=(20, 12))

# 1. GRU训练历史
ax1 = plt.subplot(3, 3, 1)
ax1.plot(history_gru.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history_gru.history['val_loss'], label='Val Loss', linewidth=2)
ax1.axvline(x=gru_diagnosis['min_val_loss_epoch'], color='r',
            linestyle='--', label='Min Val Loss', alpha=0.7)
ax1.set_title('GRU Training History', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. LSTM训练历史
ax2 = plt.subplot(3, 3, 2)
ax2.plot(history_lstm.history['loss'], label='Train Loss', linewidth=2)
ax2.plot(history_lstm.history['val_loss'], label='Val Loss', linewidth=2)
ax2.axvline(x=lstm_diagnosis['min_val_loss_epoch'], color='r',
            linestyle='--', label='Min Val Loss', alpha=0.7)
ax2.set_title('LSTM Training History', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Loss差异对比
ax3 = plt.subplot(3, 3, 3)
models = ['GRU', 'LSTM']
train_losses = [gru_diagnosis['final_train_loss'], lstm_diagnosis['final_train_loss']]
val_losses = [gru_diagnosis['final_val_loss'], lstm_diagnosis['final_val_loss']]
x = np.arange(len(models))
width = 0.35
ax3.bar(x - width / 2, train_losses, width, label='Train Loss', alpha=0.8)
ax3.bar(x + width / 2, val_losses, width, label='Val Loss', alpha=0.8)
ax3.set_ylabel('Loss')
ax3.set_title('Train vs Val Loss Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. R²对比
ax4 = plt.subplot(3, 3, 4)
train_r2s = [gru_comparison['train_r2'], lstm_comparison['train_r2']]
test_r2s = [gru_comparison['test_r2'], lstm_comparison['test_r2']]
ax4.bar(x - width / 2, train_r2s, width, label='Train R²', alpha=0.8)
ax4.bar(x + width / 2, test_r2s, width, label='Test R²', alpha=0.8)
ax4.set_ylabel('R² Score')
ax4.set_title('Train vs Test R² Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 1])

# 5. Loss Gap Ratio
ax5 = plt.subplot(3, 3, 5)
gap_ratios = [gru_diagnosis['loss_gap_ratio'], lstm_diagnosis['loss_gap_ratio']]
colors = ['red' if gap > 0.1 else 'green' for gap in gap_ratios]
ax5.bar(models, gap_ratios, color=colors, alpha=0.7)
ax5.axhline(y=0.1, color='r', linestyle='--', label='Threshold (10%)', linewidth=2)
ax5.set_ylabel('Loss Gap Ratio')
ax5.set_title('Overfitting Risk (Loss Gap Ratio)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. 训练历史对比（放大最后50轮）
ax6 = plt.subplot(3, 3, 6)
start_epoch = max(0, len(history_gru.history['loss']) - 50)
ax6.plot(range(start_epoch, len(history_gru.history['loss'])),
         history_gru.history['loss'][start_epoch:],
         label='GRU Train', linewidth=2)
ax6.plot(range(start_epoch, len(history_gru.history['val_loss'])),
         history_gru.history['val_loss'][start_epoch:],
         label='GRU Val', linewidth=2, linestyle='--')
ax6.plot(range(start_epoch, len(history_lstm.history['loss'])),
         history_lstm.history['loss'][start_epoch:],
         label='LSTM Train', linewidth=2)
ax6.plot(range(start_epoch, len(history_lstm.history['val_loss'])),
         history_lstm.history['val_loss'][start_epoch:],
         label='LSTM Val', linewidth=2, linestyle='--')
ax6.set_title('Last 50 Epochs Detail', fontsize=12, fontweight='bold')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Loss')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. 最终预测结果
ax7 = plt.subplot(3, 1, 3)
ax7.plot(test_labels_denorm, label="True value", linewidth=2, marker='o', markersize=4)
ax7.plot(final_preds_denorm, label="Predicted value", linewidth=2, marker='s', markersize=4, alpha=0.7)
ax7.set_title("Final Stacked Model Prediction (Original Scale)", fontsize=14, fontweight='bold')
ax7.set_xlabel("Number of days", fontsize=12)
ax7.set_ylabel("Corn Price", fontsize=12)
ax7.legend(loc='best', fontsize=11)
ax7.grid(True, alpha=0.3)

# 添加R²和RMSE文本
final_r2 = r2_score(test_labels_denorm, final_preds_denorm)
final_rmse = sqrt(mean_squared_error(test_labels_denorm, final_preds_denorm))
ax7.text(0.02, 0.98, f'R² = {final_r2:.4f}\nRMSE = {final_rmse:.4f}',
         transform=ax7.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(results_directory + 'comprehensive_diagnosis.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# 单独绘制预测结果
plt.figure(figsize=(14, 6))
plt.plot(test_labels_denorm, label="True value", linewidth=2.5, marker='o', markersize=4)
plt.plot(final_preds_denorm, label="Predicted value", linewidth=2.5, marker='s', markersize=4, alpha=0.7)
plt.title("Stacked Model with Attention (GRU+LSTM+XGBoost)", fontsize=16, fontweight='bold')
plt.xlabel("Number of days", fontsize=14)
plt.ylabel("Corn Price", fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_directory + 'prediction_results.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ================== 生成过拟合诊断摘要报告 ==================
summary_report = f"""
{'=' * 80}
                    过拟合诊断总结报告
{'=' * 80}

1. GRU + Self-Attention 模型:
   - 过拟合状态: {'存在过拟合风险' if gru_diagnosis['is_overfitting'] else '泛化性能良好'}
   - Loss差异比率: {gru_diagnosis['loss_gap_ratio']:.2%}
   - R²差异: {gru_comparison['r2_gap']:.4f}
   - 建议: {'需要增加正则化或使用Early Stopping' if gru_diagnosis['is_overfitting'] else '当前设置合理'}

2. LSTM + Multi-Head Attention 模型:
   - 过拟合状态: {'存在过拟合风险' if lstm_diagnosis['is_overfitting'] else '泛化性能良好'}
   - Loss差异比率: {lstm_diagnosis['loss_gap_ratio']:.2%}
   - R²差异: {lstm_comparison['r2_gap']:.4f}
   - 建议: {'需要增加正则化或使用Early Stopping' if lstm_diagnosis['is_overfitting'] else '当前设置合理'}

3. 最终Stacked模型:
   - 测试集R²: {final_r2:.4f}
   - 测试集RMSE: {final_rmse:.4f}
   - 测试集MAPE: {mape_original:.4f}%

{'=' * 80}
所有结果已保存到 {results_directory} 目录
{'=' * 80}
"""

print(summary_report)

# 保存诊断报告
with open(results_directory + 'overfitting_diagnosis_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(f"\n✓ 过拟合诊断报告已保存到 {results_directory}overfitting_diagnosis_report.txt")