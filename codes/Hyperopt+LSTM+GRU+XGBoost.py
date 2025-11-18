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
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import GRU, Dropout, Dense
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
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
dataset.dropna(inplace=True)

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
        data = X.iloc[i:i + seq_len]
        label = y.iloc[i + seq_len]
        features.append(data)
        targets.append(label)
    return np.array(features), np.array(targets)


train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=5)
test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=5)


# 构造批数据
def create_batch_dataset(X, y, train=True, buffer_size=200, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)


train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)

# ==================== Hyperopt优化部分 ====================

# 定义GRU模型的搜索空间
gru_space = {
    'units1': hp.quniform('units1', 64, 256, 32),
    'units2': hp.quniform('units2', 32, 128, 32),
    'dropout1': hp.uniform('dropout1', 0.1, 0.5),
    'dropout2': hp.uniform('dropout2', 0.1, 0.5),
    'dense_units': hp.quniform('dense_units', 16, 64, 16),
    'learning_rate': hp.loguniform('learning_rate', -5, -2)
}

# 定义LSTM模型的搜索空间
lstm_space = {
    'units1': hp.quniform('units1', 64, 256, 32),
    'units2': hp.quniform('units2', 32, 128, 32),
    'dropout1': hp.uniform('dropout1', 0.1, 0.5),
    'dropout2': hp.uniform('dropout2', 0.1, 0.5),
    'dense_units': hp.quniform('dense_units', 16, 64, 16),
    'learning_rate': hp.loguniform('learning_rate', -5, -2)
}

# 定义XGBoost的搜索空间
xgb_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
}


# GRU模型优化目标函数
def gru_objective(params):
    params = {
        'units1': int(params['units1']),
        'units2': int(params['units2']),
        'dropout1': params['dropout1'],
        'dropout2': params['dropout2'],
        'dense_units': int(params['dense_units']),
        'learning_rate': params['learning_rate']
    }

    model = Sequential()
    model.add(GRU(params['units1'], input_shape=(5, X_train.shape[1]), return_sequences=True))
    model.add(Dropout(params['dropout1']))
    model.add(GRU(params['units2'], return_sequences=False))
    model.add(Dropout(params['dropout2']))
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    )

    history = model.fit(
        train_batch_dataset,
        epochs=100,
        validation_data=test_batch_dataset,
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model}


# LSTM模型优化目标函数
def lstm_objective(params):
    params = {
        'units1': int(params['units1']),
        'units2': int(params['units2']),
        'dropout1': params['dropout1'],
        'dropout2': params['dropout2'],
        'dense_units': int(params['dense_units']),
        'learning_rate': params['learning_rate']
    }

    model = Sequential()
    model.add(layers.LSTM(params['units1'], input_shape=(5, X_train.shape[1]), return_sequences=True))
    model.add(Dropout(params['dropout1']))
    model.add(layers.LSTM(params['units2'], return_sequences=False))
    model.add(Dropout(params['dropout2']))
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    )

    history = model.fit(
        train_batch_dataset,
        epochs=100,
        validation_data=test_batch_dataset,
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model}


# XGBoost模型优化目标函数
def xgb_objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'learning_rate': params['learning_rate'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree']
    }

    model = XGBRegressor(**params)

    # 使用GRU和LSTM的预测作为特征
    gru_preds = gru_best_model.predict(test_dataset, verbose=0)[:, 0]
    lstm_preds = lstm_best_model.predict(test_dataset, verbose=0)[:, 0]
    stacked_features = np.column_stack((gru_preds, lstm_preds))

    model.fit(stacked_features, test_labels)
    preds = model.predict(stacked_features)
    mse = mean_squared_error(test_labels, preds)

    return {'loss': mse, 'status': STATUS_OK, 'model': model}


# 运行Hyperopt优化
print("开始优化GRU模型...")
gru_trials = Trials()
gru_best = fmin(
    fn=gru_objective,
    space=gru_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=gru_trials
)
gru_best_params = {
    'units1': int(gru_best['units1']),
    'units2': int(gru_best['units2']),
    'dropout1': gru_best['dropout1'],
    'dropout2': gru_best['dropout2'],
    'dense_units': int(gru_best['dense_units']),
    'learning_rate': gru_best['learning_rate']
}
gru_best_model = gru_trials.best_trial['result']['model']

print("开始优化LSTM模型...")
lstm_trials = Trials()
lstm_best = fmin(
    fn=lstm_objective,
    space=lstm_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=lstm_trials
)
lstm_best_params = {
    'units1': int(lstm_best['units1']),
    'units2': int(lstm_best['units2']),
    'dropout1': lstm_best['dropout1'],
    'dropout2': lstm_best['dropout2'],
    'dense_units': int(lstm_best['dense_units']),
    'learning_rate': lstm_best['learning_rate']
}
lstm_best_model = lstm_trials.best_trial['result']['model']

print("开始优化XGBoost模型...")
xgb_trials = Trials()
xgb_best = fmin(
    fn=xgb_objective,
    space=xgb_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=xgb_trials
)
xgb_best_params = {
    'n_estimators': int(xgb_best['n_estimators']),
    'max_depth': int(xgb_best['max_depth']),
    'learning_rate': xgb_best['learning_rate'],
    'subsample': xgb_best['subsample'],
    'colsample_bytree': xgb_best['colsample_bytree']
}
xgb_best_model = xgb_trials.best_trial['result']['model']

# ==================== 使用优化后的模型进行预测 ====================

# 使用最佳GRU和LSTM模型进行预测
gru_preds = gru_best_model.predict(test_dataset, verbose=0)[:, 0]
lstm_preds = lstm_best_model.predict(test_dataset, verbose=0)[:, 0]

# 使用最佳XGBoost模型进行最终预测
stacked_features = np.column_stack((gru_preds, lstm_preds))
final_preds = xgb_best_model.predict(stacked_features)

# 计算指标
print('----' * 20)
print("优化后的模型性能:")
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
predict.to_csv(results_directory + "Optimized_Stacked-y_predict.csv", header=["y_predict"])

# 反归一化处理
test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1))
final_preds = scaler.inverse_transform(final_preds.reshape(-1, 1))

# 绘制预测与真值结果
plt.figure(figsize=(7, 4))
plt.plot(test_labels, label="True value")
plt.plot(final_preds, label="Pred value")
plt.title("Optimized Stacked Model (LSTM + GRU + XGBoost)", fontsize=16)
plt.xlabel("number of days", fontsize=16)
plt.ylabel("Price", fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.tick_params(labelsize=16)
plt.show(block=True)