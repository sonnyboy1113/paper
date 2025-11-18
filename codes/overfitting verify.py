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
try:
    from tensorflow.keras import Sequential, layers
    from tensorflow.keras.layers import GRU, Dropout, Dense
    from tensorflow.keras.callbacks import History
except ImportError:
    from keras import Sequential, layers
    from keras.layers import GRU, Dropout, Dense
    from keras.callbacks import History
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print(dataset.info())

# 数据预处理
columns = ['Corn', 'TR', 'IR', 'ER', 'ine', 'FP.CFI', 'CFD', 'GPR', 'EPU', 'corn']
for col in columns:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))

# 特征工程：添加滞后特征
for i in range(1, 6):
    dataset[f'Corn_lag_{i}'] = dataset['Corn'].shift(i)
dataset.dropna(inplace=True)  # 删除因滞后特征产生的缺失值

# 特征数据集和标签数据集
X = dataset.drop(columns=['Corn'], axis=1)
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


# 构造训练和测试特征数据集
train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)


# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)


# 训练和测试批数据
train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# ============== 添加过拟合验证部分 ==============
# 1. 添加回调函数记录训练过程

# 创建历史记录对象
history = History()

# GRU 模型（保持原参数不变）
gru_model = Sequential()
gru_model.add(GRU(64, input_shape=(5, X_train.shape[1]), return_sequences=True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(32, return_sequences=False))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(1, activation='relu'))
gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
gru_history = gru_model.fit(train_batch_dataset,
                            epochs=200,
                            validation_data=test_batch_dataset,
                            callbacks=[history],
                            verbose=0)

# 绘制GRU训练和验证损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(gru_history.history['loss'], label='GRU Train Loss')
plt.plot(gru_history.history['val_loss'], label='GRU Val Loss')
plt.title('GRU Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# LSTM 模型（保持原参数不变）
lstm_model = Sequential([
    layers.LSTM(64, input_shape=(5, X_train.shape[1]), return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(1, activation='relu'),
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_history = lstm_model.fit(train_batch_dataset,
                              epochs=200,
                              validation_data=test_batch_dataset,
                              callbacks=[history],
                              verbose=0)

# 绘制LSTM训练和验证损失曲线
plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'], label='LSTM Train Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Val Loss')
plt.title('LSTM Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show(block=True)


# 2. 计算训练集和测试集的性能差异
def evaluate_model(model, train_data, test_data, train_labels, test_labels):
    train_pred = model.predict(train_data, verbose=0)[:, 0]
    test_pred = model.predict(test_data, verbose=0)[:, 0]

    train_r2 = r2_score(train_labels, train_pred)
    test_r2 = r2_score(test_labels, test_pred)

    train_mae = mean_absolute_error(train_labels, train_pred)
    test_mae = mean_absolute_error(test_labels, test_pred)

    return {
        'Train R2': train_r2,
        'Test R2': test_r2,
        'R2 Gap': train_r2 - test_r2,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'MAE Gap': test_mae - train_mae
    }


# 评估GRU模型
gru_metrics = evaluate_model(gru_model, train_dataset, test_dataset, train_labels, test_labels)
print("\nGRU Model Performance:")
print(pd.DataFrame([gru_metrics]))

# 评估LSTM模型
lstm_metrics = evaluate_model(lstm_model, train_dataset, test_dataset, train_labels, test_labels)
print("\nLSTM Model Performance:")
print(pd.DataFrame([lstm_metrics]))

# ============== 原模型后续部分保持不变 ==============
# 使用GRU和LSTM模型进行预测
gru_preds = gru_model.predict(test_dataset, verbose=0)
gru_preds = gru_preds[:, 0]

lstm_preds = lstm_model.predict(test_dataset, verbose=0)
lstm_preds = lstm_preds[:, 0]

# 将GRU和LSTM的预测结果作为特征
stacked_features = np.column_stack((gru_preds, lstm_preds))

# 训练元模型（XGBoost）
meta_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
meta_model.fit(stacked_features, test_labels)

# 3. 评估元模型的过拟合情况
# 使用训练集预测结果来评估元模型
train_gru_preds = gru_model.predict(train_dataset, verbose=0)[:, 0]
train_lstm_preds = lstm_model.predict(train_dataset, verbose=0)[:, 0]
train_stacked_features = np.column_stack((train_gru_preds, train_lstm_preds))

meta_train_preds = meta_model.predict(train_stacked_features)
meta_test_preds = meta_model.predict(stacked_features)

meta_metrics = {
    'Train R2': r2_score(train_labels, meta_train_preds),
    'Test R2': r2_score(test_labels, meta_test_preds),
    'R2 Gap': r2_score(train_labels, meta_train_preds) - r2_score(test_labels, meta_test_preds),
    'Train MAE': mean_absolute_error(train_labels, meta_train_preds),
    'Test MAE': mean_absolute_error(test_labels, meta_test_preds),
    'MAE Gap': mean_absolute_error(test_labels, meta_test_preds) - mean_absolute_error(train_labels, meta_train_preds)
}

print("\nMeta Model (XGBoost) Performance:")
print(pd.DataFrame([meta_metrics]))

# 使用元模型进行最终预测
final_preds = meta_model.predict(stacked_features)

# 计算指标
print('----' * 20)
print("Final Stacked Model Performance:")
print("r^2 值为：", r2_score(test_labels, final_preds))
print("MAE:", mean_absolute_error(test_labels, final_preds))
print("MSE:", mean_squared_error(test_labels, final_preds))
print("RMSE:", sqrt(mean_squared_error(test_labels, final_preds)))
print('MAPE:', np.mean(np.abs((final_preds - test_labels) / test_labels)))

# 导出预测值
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
predict = pd.DataFrame(final_preds)
predict.to_csv(results_directory + "Stacked-y_predict.csv", header=["y_predict"])

# 反归一化处理
test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1))
final_preds = scaler.inverse_transform(final_preds.reshape(-1, 1))

# 绘制预测与真值结果
plt.figure(figsize=(7, 4))
plt.plot(test_labels, label="True value")
plt.plot(final_preds, label="Pred value")
plt.title("Stacked Model (LSTM + GRU +XGBoost)", fontsize=16)
plt.xlabel("number of days", fontsize=16)
plt.ylabel("Price", fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.tick_params(labelsize=16)
plt.show(block=True)  # 直接显示图像
