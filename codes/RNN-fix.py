import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.cluster import k_means
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
dataset.info()

# 特征数据集
X = dataset.drop(columns=['Corn'], axis=1)
# 标签数据集
y = dataset['Corn']

# 数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)

# 只在训练集上fit归一化器,然后transform训练集和测试集
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

y_train = pd.Series(y_train_scaled, index=y_train.index)
y_test = pd.Series(y_test_scaled, index=y_test.index)

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

# 构造训练和测试特征数据集
train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)

# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=100, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)

train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# 创建结果目录
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# ========== 添加回调函数来防止过拟合 ==========
# Early Stopping: 当验证损失不再改善时停止训练
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# ModelCheckpoint: 保存最佳模型
checkpoint = ModelCheckpoint(
    results_directory + 'best_rnn_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# 模型 (保持原有RNN结构不变)
model = Sequential([
    SimpleRNN(100, input_shape=(5, 9), return_sequences=True),
    Dropout(0.01),
    SimpleRNN(100),
    Dropout(0.01),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# ========== 训练模型并保存历史记录 ==========
history = model.fit(
    train_batch_dataset,
    epochs=200,
    validation_data=test_batch_dataset,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# ========== 绘制训练历史以检测过拟合 ==========
plt.figure(figsize=(14, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失', linewidth=2)
plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('模型损失曲线', fontsize=16)
plt.xlabel('训练轮次 (Epoch)', fontsize=14)
plt.ylabel('损失 (MSE)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 绘制MAE曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='训练MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='验证MAE', linewidth=2)
plt.title('模型MAE曲线', fontsize=16)
plt.xlabel('训练轮次 (Epoch)', fontsize=14)
plt.ylabel('平均绝对误差 (MAE)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'rnn_training_history.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# 保存训练历史
history_df = pd.DataFrame(history.history)
history_df.to_csv(results_directory + 'rnn_training_history.csv', index=False)

# ========== 在训练集和测试集上分别进行预测 ==========
train_preds = model.predict(train_dataset, verbose=0)
train_preds = train_preds[:, 0]

test_preds = model.predict(test_dataset, verbose=0)
test_preds = test_preds[:, 0]

# ========== 对比训练集和测试集性能(归一化数据) ==========
print('=' * 100)
print("【归一化数据的指标对比 - 检测过拟合】")
print('=' * 100)

print("\n训练集指标:")
print(f"  R² 值：{r2_score(train_labels, train_preds):.6f}")
print(f"  MAE：{mean_absolute_error(train_labels, train_preds):.6f}")
print(f"  MSE：{mean_squared_error(train_labels, train_preds):.6f}")
print(f"  RMSE：{sqrt(mean_squared_error(train_labels, train_preds)):.6f}")
print(f"  MAPE：{np.mean(np.abs((train_preds - train_labels) / (train_labels + 1e-8))):.6f}")

print("\n测试集指标:")
print(f"  R² 值：{r2_score(test_labels, test_preds):.6f}")
print(f"  MAE：{mean_absolute_error(test_labels, test_preds):.6f}")
print(f"  MSE：{mean_squared_error(test_labels, test_preds):.6f}")
print(f"  RMSE：{sqrt(mean_squared_error(test_labels, test_preds)):.6f}")
print(f"  MAPE：{np.mean(np.abs((test_preds - test_labels) / (test_labels + 1e-8))):.6f}")

# 计算过拟合程度
train_r2 = r2_score(train_labels, train_preds)
test_r2 = r2_score(test_labels, test_preds)
train_mae = mean_absolute_error(train_labels, train_preds)
test_mae = mean_absolute_error(test_labels, test_preds)

print("\n【过拟合分析】")
print(f"  R² 差异：{abs(train_r2 - test_r2):.6f} (差异越小越好)")
print(f"  MAE 差异：{abs(train_mae - test_mae):.6f} (差异越小越好)")
if train_mae < test_mae * 0.7:
    print("  ⚠️  警告：训练误差远小于测试误差，可能存在过拟合！")
elif train_mae < test_mae * 0.85:
    print("  ⚠️  注意：训练误差明显小于测试误差，存在轻微过拟合")
else:
    print("  ✓  模型泛化性能良好")

# ========== 反归一化并计算原始尺度指标 ==========
train_labels_original = y_scaler.inverse_transform(train_labels.reshape(-1, 1))
train_preds_original = y_scaler.inverse_transform(train_preds.reshape(-1, 1))
test_labels_original = y_scaler.inverse_transform(test_labels.reshape(-1, 1))
test_preds_original = y_scaler.inverse_transform(test_preds.reshape(-1, 1))

print('\n' + '=' * 100)
print("【原始尺度的指标对比】")
print('=' * 100)

print("\n训练集指标:")
print(f"  R² 值：{r2_score(train_labels_original, train_preds_original):.6f}")
print(f"  MAE：{mean_absolute_error(train_labels_original, train_preds_original):.4f}")
print(f"  MSE：{mean_squared_error(train_labels_original, train_preds_original):.4f}")
print(f"  RMSE：{sqrt(mean_squared_error(train_labels_original, train_preds_original)):.4f}")
print(f"  MAPE：{np.mean(np.abs((train_preds_original - train_labels_original) / (train_labels_original + 1e-8))):.4f}")

print("\n测试集指标:")
print(f"  R² 值：{r2_score(test_labels_original, test_preds_original):.6f}")
print(f"  MAE：{mean_absolute_error(test_labels_original, test_preds_original):.4f}")
print(f"  MSE：{mean_squared_error(test_labels_original, test_preds_original):.4f}")
print(f"  RMSE：{sqrt(mean_squared_error(test_labels_original, test_preds_original)):.4f}")
print(f"  MAPE：{np.mean(np.abs((test_preds_original - test_labels_original) / (test_labels_original + 1e-8))):.4f}")

# 导出预测结果
predict_original = pd.DataFrame(test_preds_original)
predict_original.to_csv(results_directory + "RNN-y_predict_original.csv", header=["y_predict"])

# ========== 绘制训练集和测试集的预测对比图 ==========
plt.figure(figsize=(14, 5))

# 训练集预测
plt.subplot(1, 2, 1)
plt.plot(train_labels_original, label="真实值", alpha=0.7, linewidth=2)
plt.plot(train_preds_original, label="预测值", alpha=0.7, linewidth=2)
plt.title("训练集预测结果 (RNN)", fontsize=16)
plt.xlabel("样本序号", fontsize=14)
plt.ylabel("价格", fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)

# 测试集预测
plt.subplot(1, 2, 2)
plt.plot(test_labels_original, label="真实值", alpha=0.7, linewidth=2)
plt.plot(test_preds_original, label="预测值", alpha=0.7, linewidth=2)
plt.title("测试集预测结果 (RNN)", fontsize=16)
plt.xlabel("样本序号", fontsize=14)
plt.ylabel("价格", fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'rnn_train_test_predictions.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 保存归一化器
import pickle
with open(results_directory + 'rnn_scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

print('\n归一化器已保存到:', results_directory + 'rnn_scalers.pkl')
print('最佳模型已保存到:', results_directory + 'best_rnn_model.h5')
print('训练历史已保存到:', results_directory + 'rnn_training_history.csv')
print('训练曲线图已保存到:', results_directory + 'rnn_training_history.png')
print('预测对比图已保存到:', results_directory + 'rnn_train_test_predictions.png')