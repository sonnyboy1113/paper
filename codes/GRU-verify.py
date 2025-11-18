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
from tensorflow.keras.layers import Input, Dropout, Dense, GRU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.cluster import k_means
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential, layers, utils, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
dataset.info()

# 进行归一化
columns = ['Corn', 'TR', 'IR', 'ER', 'ine', 'FP.CFI', 'CFD', 'GPR', 'EPU', 'corn']
for col in columns:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))

# 特征数据集
X = dataset.drop(columns=['Corn'], axis=1)
# 标签数据集
y = dataset['Corn']
# 数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)


# 构造特征数据集
def create_dataset(X, y, seq_len=5):
    features = []
    targets = []
    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i + seq_len]  # 序列数据
        label = y.iloc[i + seq_len]  # 标签数据
        features.append(data)
        targets.append(label)
    return np.array(features), np.array(targets)


# 构造训练特征数据集
train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
# 构造测试特征数据集
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)


# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)


# 训练批数据
train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
# 测试批数据
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# ============== 添加过拟合验证部分 ==============
# 1. 添加回调函数记录训练过程
history = History()
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 模型
model = Sequential()
model.add(GRU(64, input_shape=(5, 9), return_sequences=True))
model.add(Dropout(0.01))
model.add(GRU(16, input_shape=(5, 9), return_sequences=False))
model.add(Dropout(0.01))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# 训练模型并记录历史
history = model.fit(
    train_batch_dataset,
    epochs=200,
    validation_data=test_batch_dataset,
    callbacks=[early_stopping, history],
    verbose=1
)

# 2. 绘制训练和验证损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# 3. 绘制训练和验证准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()


# 4. 计算训练集和测试集的性能差异
def evaluate_performance(model, train_data, test_data, train_labels, test_labels):
    train_pred = model.predict(train_data, verbose=0)[:, 0]
    test_pred = model.predict(test_data, verbose=0)[:, 0]

    metrics = {
        'Train R2': r2_score(train_labels, train_pred),
        'Test R2': r2_score(test_labels, test_pred),
        'R2 Gap': r2_score(train_labels, train_pred) - r2_score(test_labels, test_pred),
        'Train MAE': mean_absolute_error(train_labels, train_pred),
        'Test MAE': mean_absolute_error(test_labels, test_pred),
        'MAE Gap': mean_absolute_error(test_labels, test_pred) - mean_absolute_error(train_labels, train_pred),
        'Train MSE': mean_squared_error(train_labels, train_pred),
        'Test MSE': mean_squared_error(test_labels, test_pred),
        'MSE Gap': mean_squared_error(test_labels, test_pred) - mean_squared_error(train_labels, train_pred)
    }
    return metrics


# 评估模型性能
performance_metrics = evaluate_performance(model, train_dataset, test_dataset, train_labels, test_labels)
print("\nModel Performance Metrics:")
print(pd.DataFrame([performance_metrics]))


# 5. 添加学习曲线分析
def plot_learning_curve(model, X_train, y_train, X_val, y_val, max_samples=100, step=10):
    train_errors, val_errors = [], []
    sample_sizes = range(10, min(max_samples, len(X_train)), step)

    for m in sample_sizes:
        # 注意：这里需要重新创建小批量的数据集
        small_train_dataset, small_train_labels = create_dataset(
            X_train.iloc[:m + 5],  # +5是因为create_dataset会消耗前5个样本
            y_train.iloc[:m + 5],
            seq_len=5
        )

        # 重新训练模型
        temp_model = Sequential()
        temp_model.add(GRU(64, input_shape=(5, 9), return_sequences=True))
        temp_model.add(Dropout(0.01))
        temp_model.add(GRU(16, input_shape=(5, 9), return_sequences=False))
        temp_model.add(Dropout(0.01))
        temp_model.add(Dense(1, activation='relu'))
        temp_model.compile(loss='mse', optimizer='adam')

        temp_model.fit(
            create_batch_dataset(small_train_dataset, small_train_labels),
            epochs=20,
            verbose=0
        )

        # 评估
        train_pred = temp_model.predict(small_train_dataset, verbose=0)[:, 0]
        val_pred = temp_model.predict(X_val, verbose=0)[:, 0]

        train_errors.append(mean_squared_error(small_train_labels, train_pred))
        val_errors.append(mean_squared_error(y_val, val_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(sample_sizes, np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend()
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.title("Learning Curve")
    plt.show()


# 绘制学习曲线
plot_learning_curve(model, X_train, y_train, test_dataset, test_labels)

# ============== 原模型后续部分保持不变 ==============
# 预测
test_preds = model.predict(test_dataset, verbose=2)
test_preds = test_preds[:, 0]

# 计算指标
print('----' * 20)
print("Final Model Performance:")
print("r^2 值为：", r2_score(test_labels, test_preds))
print("MAE:", mean_absolute_error(test_labels, test_preds))
print("MSE:", mean_squared_error(test_labels, test_preds))
print("RMSE:", sqrt(mean_squared_error(test_labels, test_preds)))
print('MAPE:', np.mean(np.abs((test_preds - test_labels) / test_labels)))

# 导出预测值
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
    predict = pd.DataFrame(test_preds)
    predict.to_csv(results_directory + f"GRU-y_predict.csv", header=["y_predict"])

# 反归一化处理
test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1))
test_preds = scaler.inverse_transform(test_preds.reshape(-1, 1))

# 绘制预测与真值结果
plt.figure(figsize=(7, 4))
plt.plot(test_labels, label="True value")
plt.plot(test_preds, label="Pred value")
plt.title("GRU", fontsize=16)
plt.xlabel("number of days", fontsize=16)
plt.ylabel("Price", fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.tick_params(labelsize=16)
plt.show(block=True)
