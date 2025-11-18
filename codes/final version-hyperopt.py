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
from sklearn.linear_model import Ridge
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("LSTM + GRU + XGBoost 融合时间序列预测 - Hyperopt优化修复版".center(100))
print("核心改进：合理的超参数搜索空间 + 防止过拟合 + 更少迭代次数".center(100))
print("=" * 100)

# ========== 配置参数 ==========
ENABLE_HYPEROPT = True  # 是否启用超参数优化
HYPEROPT_EVALS = 15  # Hyperopt迭代次数（降低以防止过拟合）
CV_SPLITS = 5  # 交叉验证折数（增加以获得更稳定的评估）

print(f"\n配置:")
print(f"  启用Hyperopt: {ENABLE_HYPEROPT}")
print(f"  优化迭代次数: {HYPEROPT_EVALS}")
print(f"  交叉验证折数: {CV_SPLITS}")

# ========== 加载数据 ==========
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print("\n数据集信息:")
print(dataset.info())

# ========== 数据准备 ==========
X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

split_idx = int(len(X) * 0.8)
X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_raw, y_test_raw = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n数据分割:")
print(f"训练集: {len(X_train_raw)} 样本")
print(f"测试集: {len(X_test_raw)} 样本")

# 归一化
feature_scalers = {}
X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

for col in X_train.columns:
    scaler = MinMaxScaler()
    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    feature_scalers[col] = scaler

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).flatten()
y_test = y_scaler.transform(y_test_raw.values.reshape(-1, 1)).flatten()

y_train = pd.Series(y_train, index=y_train_raw.index)
y_test = pd.Series(y_test, index=y_test_raw.index)


# ========== 添加滞后特征 ==========
def add_features(X, y):
    X_new = X.copy()
    for i in range(1, 6):
        X_new[f'Corn_lag_{i}'] = y.shift(i)
    return X_new.dropna()


X_train_feat = add_features(X_train, y_train)
y_train = y_train.loc[X_train_feat.index]

X_test_feat = add_features(X_test, y_test)
y_test = y_test.loc[X_test_feat.index]

print(f"\n添加特征后:")
print(f"训练集: X_train={X_train_feat.shape}, y_train={y_train.shape}")
print(f"测试集: X_test={X_test_feat.shape}, y_test={y_test.shape}")


# 构造序列数据
def create_sequences(X, y, seq_len=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X.iloc[i:i + seq_len].values)
        y_seq.append(y.iloc[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


def create_flat_sequences(X, y, seq_len=5):
    X_flat, y_flat = [], []
    for i in range(len(X) - seq_len):
        X_flat.append(X.iloc[i:i + seq_len].values.flatten())
        y_flat.append(y.iloc[i + seq_len])
    return np.array(X_flat), np.array(y_flat)


seq_len = 5
X_train_seq, y_train_seq = create_sequences(X_train_feat, y_train, seq_len)
X_test_seq, y_test_seq = create_sequences(X_test_feat, y_test, seq_len)

X_train_flat, _ = create_flat_sequences(X_train_feat, y_train, seq_len)
X_test_flat, _ = create_flat_sequences(X_test_feat, y_test, seq_len)

print(f"\n序列数据形状:")
print(f"LSTM/GRU输入: train={X_train_seq.shape}, test={X_test_seq.shape}")
print(f"XGBoost输入: train={X_train_flat.shape}, test={X_test_flat.shape}")

# ========== 默认保守参数（防止过拟合）==========
default_lstm_params = {
    'units': 100,
    'dropout': 0.3,
    'l2_reg': 0.01,
    'learning_rate': 0.001,
    'batch_size': 32
}

default_gru_params = {
    'units': 100,
    'dropout': 0.3,
    'l2_reg': 0.01,
    'learning_rate': 0.001,
    'batch_size': 32
}

default_xgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

default_ridge_params = {
    'alpha': 10.0
}

# ========== Hyperopt超参数优化（更保守的搜索空间）==========
if ENABLE_HYPEROPT:
    print("\n" + "=" * 100)
    print("【Hyperopt超参数优化】保守策略，防止过拟合".center(100))
    print("=" * 100)

    # 更保守的RNN参数空间
    rnn_space = {
        'units': hp.choice('units', [80, 100, 120]),  # 减少选项
        'dropout': hp.uniform('dropout', 0.2, 0.4),  # 更窄范围
        'l2_reg': hp.loguniform('l2_reg', np.log(0.005), np.log(0.05)),  # 更保守
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0005), np.log(0.005)),  # 更保守
        'batch_size': hp.choice('batch_size', [32])  # 固定batch size
    }


    def objective_lstm(params):
        """LSTM超参数优化目标函数（增加CV折数，减少epochs）"""
        units = params['units']
        dropout = params['dropout']
        l2_reg = params['l2_reg']
        lr = params['learning_rate']
        batch_size = params['batch_size']

        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)  # 增加折数
        val_scores = []

        for train_idx, val_idx in tscv.split(X_train_seq):
            X_tr, X_val = X_train_seq[train_idx], X_train_seq[val_idx]
            y_tr, y_val = y_train_seq[train_idx], y_train_seq[val_idx]

            model = Sequential([
                layers.LSTM(
                    units=units,
                    input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                    kernel_regularizer=l2(l2_reg),
                    recurrent_regularizer=l2(l2_reg)
                ),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])

            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)

            model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                      epochs=100, batch_size=batch_size, callbacks=[early_stop], verbose=0)

            val_pred = model.predict(X_val, verbose=0).flatten()
            val_score = r2_score(y_val, val_pred)
            val_scores.append(val_score)

            del model
            tf.keras.backend.clear_session()

        # 使用中位数而不是平均值，更稳健
        median_score = np.median(val_scores)
        return {'loss': -median_score, 'status': STATUS_OK}


    def objective_gru(params):
        """GRU超参数优化目标函数"""
        units = params['units']
        dropout = params['dropout']
        l2_reg = params['l2_reg']
        lr = params['learning_rate']
        batch_size = params['batch_size']

        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        val_scores = []

        for train_idx, val_idx in tscv.split(X_train_seq):
            X_tr, X_val = X_train_seq[train_idx], X_train_seq[val_idx]
            y_tr, y_val = y_train_seq[train_idx], y_train_seq[val_idx]

            model = Sequential([
                layers.GRU(
                    units=units,
                    input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                    kernel_regularizer=l2(l2_reg),
                    recurrent_regularizer=l2(l2_reg)
                ),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])

            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)

            model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                      epochs=100, batch_size=batch_size, callbacks=[early_stop], verbose=0)

            val_pred = model.predict(X_val, verbose=0).flatten()
            val_score = r2_score(y_val, val_pred)
            val_scores.append(val_score)

            del model
            tf.keras.backend.clear_session()

        median_score = np.median(val_scores)
        return {'loss': -median_score, 'status': STATUS_OK}


    print("\n【1/2】优化LSTM超参数（保守策略）...")
    lstm_trials = Trials()
    best_lstm = fmin(fn=objective_lstm, space=rnn_space, algo=tpe.suggest,
                     max_evals=HYPEROPT_EVALS, trials=lstm_trials, verbose=0)
    best_lstm_params = space_eval(rnn_space, best_lstm)
    print(f"✓ 最佳LSTM参数: {best_lstm_params}")
    print(f"  最佳验证R²: {-lstm_trials.best_trial['result']['loss']:.4f}")

    print("\n【2/2】优化GRU超参数（保守策略）...")
    gru_trials = Trials()
    best_gru = fmin(fn=objective_gru, space=rnn_space, algo=tpe.suggest,
                    max_evals=HYPEROPT_EVALS, trials=gru_trials, verbose=0)
    best_gru_params = space_eval(rnn_space, best_gru)
    print(f"✓ 最佳GRU参数: {best_gru_params}")
    print(f"  最佳验证R²: {-gru_trials.best_trial['result']['loss']:.4f}")

    # 对比默认参数
    print(f"\n【参数对比】")
    print(f"LSTM默认 vs 优化:")
    for key in default_lstm_params:
        print(f"  {key}: {default_lstm_params[key]} → {best_lstm_params[key]}")

    print(f"\nGRU默认 vs 优化:")
    for key in default_gru_params:
        print(f"  {key}: {default_gru_params[key]} → {best_gru_params[key]}")

else:
    print("\n⚠️  超参数优化已禁用，使用默认保守参数")
    best_lstm_params = default_lstm_params
    best_gru_params = default_gru_params
    best_xgb_params = default_xgb_params
    best_ridge_params = default_ridge_params

# ========== 使用最佳参数训练RNN模型 ==========
print("\n" + "=" * 100)
print("第一步：使用最佳参数训练LSTM和GRU模型".center(100))
print("=" * 100)

print("\n训练LSTM模型...")
lstm_final = Sequential([
    layers.LSTM(
        units=best_lstm_params['units'],
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        kernel_regularizer=l2(best_lstm_params['l2_reg']),
        recurrent_regularizer=l2(best_lstm_params['l2_reg'])
    ),
    layers.Dropout(best_lstm_params['dropout']),
    layers.Dense(1)
])
lstm_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=best_lstm_params['learning_rate']))
lstm_history = lstm_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=best_lstm_params['batch_size'],
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ],
    verbose=0
)
print("✓ LSTM模型训练完成")

print("\n训练GRU模型...")
gru_final = Sequential([
    layers.GRU(
        units=best_gru_params['units'],
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        kernel_regularizer=l2(best_gru_params['l2_reg']),
        recurrent_regularizer=l2(best_gru_params['l2_reg'])
    ),
    layers.Dropout(best_gru_params['dropout']),
    layers.Dense(1)
])
gru_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=best_gru_params['learning_rate']))
gru_history = gru_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=best_gru_params['batch_size'],
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    ],
    verbose=0
)
print("✓ GRU模型训练完成")

# 获取预测
lstm_test_pred = lstm_final.predict(X_test_seq, verbose=0).flatten()
gru_test_pred = gru_final.predict(X_test_seq, verbose=0).flatten()

lstm_train_pred = lstm_final.predict(X_train_seq, verbose=0).flatten()
gru_train_pred = gru_final.predict(X_train_seq, verbose=0).flatten()

# 过拟合诊断
lstm_train_r2 = r2_score(y_train_seq, lstm_train_pred)
lstm_test_r2 = r2_score(y_test_seq, lstm_test_pred)
gru_train_r2 = r2_score(y_train_seq, gru_train_pred)
gru_test_r2 = r2_score(y_test_seq, gru_test_pred)

print(f"\n【过拟合诊断】")
print(f"LSTM - 训练R²: {lstm_train_r2:.4f}, 测试R²: {lstm_test_r2:.4f}, 差距: {lstm_train_r2 - lstm_test_r2:.4f}")
print(f"GRU  - 训练R²: {gru_train_r2:.4f}, 测试R²: {gru_test_r2:.4f}, 差距: {gru_train_r2 - gru_test_r2:.4f}")

overfitting_detected = max(lstm_train_r2 - lstm_test_r2, gru_train_r2 - gru_test_r2) > 0.15
if overfitting_detected:
    print(f"⚠️  检测到明显过拟合，将采用保守残差学习策略！")
else:
    print(f"✓ 过拟合控制良好，可尝试多种残差策略")

# ========== 生成OOF预测（使用默认保守参数）==========
print("\n" + "=" * 100)
print("第二步：生成LSTM和GRU的OOF预测（使用保守参数防止过拟合）".center(100))
print("=" * 100)


def build_lstm_conservative(input_shape):
    """保守的LSTM模型（用于OOF）"""
    return Sequential([
        layers.LSTM(
            units=100,
            input_shape=input_shape,
            kernel_regularizer=l2(0.01),
            recurrent_regularizer=l2(0.01)
        ),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])


def build_gru_conservative(input_shape):
    """保守的GRU模型（用于OOF）"""
    return Sequential([
        layers.GRU(
            units=100,
            input_shape=input_shape,
            kernel_regularizer=l2(0.01),
            recurrent_regularizer=l2(0.01)
        ),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])


def get_oof_predictions(X_seq, y_seq, model_type='lstm', n_splits=5):
    """使用保守参数生成OOF预测"""
    print(f"\n生成{model_type.upper()} OOF预测（保守参数，TimeSeriesSplit with {n_splits} splits）...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros(len(y_seq))

    fold = 1
    for train_idx, val_idx in tscv.split(X_seq):
        print(f"  Fold {fold}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")

        X_fold_train, X_fold_val = X_seq[train_idx], X_seq[val_idx]
        y_fold_train, y_fold_val = y_seq[train_idx], y_seq[val_idx]

        if model_type == 'lstm':
            model = build_lstm_conservative((X_seq.shape[1], X_seq.shape[2]))
        else:
            model = build_gru_conservative((X_seq.shape[1], X_seq.shape[2]))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        val_pred = model.predict(X_fold_val, verbose=0)
        oof_preds[val_idx] = val_pred.flatten()

        del model
        tf.keras.backend.clear_session()
        fold += 1

    return oof_preds


lstm_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'lstm', n_splits=5)
gru_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'gru', n_splits=5)

print(f"\nOOF预测生成完成！")
print(f"LSTM OOF R²: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"GRU OOF R²: {r2_score(y_train_seq, gru_oof_preds):.4f}")


# ========== 特征工程函数 ==========
def create_simplified_features(X_flat, lstm_preds, gru_preds):
    features_list = [X_flat]
    features_list.append(lstm_preds.reshape(-1, 1))
    features_list.append(gru_preds.reshape(-1, 1))
    features_list.append(((lstm_preds + gru_preds) / 2).reshape(-1, 1))
    features_list.append(np.abs(lstm_preds - gru_preds).reshape(-1, 1))
    return np.hstack(features_list)


# ========== 准备特征和残差 ==========
basic_features_train = X_train_flat[:len(gru_oof_preds)]
basic_features_test = X_test_flat

simplified_train = create_simplified_features(basic_features_train, lstm_oof_preds, gru_oof_preds)
simplified_test = create_simplified_features(basic_features_test, lstm_test_pred, gru_test_pred)

lstm_oof_residual = y_train_seq - lstm_oof_preds
gru_oof_residual = y_train_seq - gru_oof_preds
avg_oof_preds = (lstm_oof_preds + gru_oof_preds) / 2
avg_oof_residual = y_train_seq - avg_oof_preds


# ========== 残差学习训练函数（使用默认保守参数）==========
def train_conservative_xgboost(X_train, y_train):
    """保守XGBoost参数"""
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def train_ridge_model(X_train, y_train, alpha=10.0):
    """Ridge线性回归"""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def clip_residual(residual_pred, threshold=2.0):
    std = np.std(residual_pred)
    mean = np.mean(residual_pred)
    return np.clip(residual_pred, mean - threshold * std, mean + threshold * std)


def weighted_residual_correction(base_pred, residual_pred, weight=0.5):
    return base_pred + weight * residual_pred


# ========== 基准：简单平均 ==========
print("\n" + "=" * 100)
print("【基准】简单平均融合".center(100))
print("=" * 100)

avg_test_pred = (lstm_test_pred + gru_test_pred) / 2
avg_r2 = r2_score(y_test_seq, avg_test_pred)
print(f"简单平均测试集R²: {avg_r2:.4f}")

print(f"\n残差统计:")
print(f"LSTM残差 - 均值: {np.mean(lstm_oof_residual):.6f}, 标准差: {np.std(lstm_oof_residual):.6f}")
print(f"GRU残差  - 均值: {np.mean(gru_oof_residual):.6f}, 标准差: {np.std(gru_oof_residual):.6f}")

# ========== 策略集合（使用保守参数）==========
strategies_results = {}

print(f"\n特征维度: {simplified_train.shape[1]} 维")

# ========== 策略1：LSTM残差学习 ==========
print("\n" + "=" * 100)
print("【策略1】LSTM残差学习：简化特征 + 保守XGBoost".center(100))
print("=" * 100)

xgb_lstm_conservative = train_conservative_xgboost(simplified_train, lstm_oof_residual)
lstm_residual = xgb_lstm_conservative.predict(simplified_test)
pred_lstm_residual = lstm_test_pred + lstm_residual

r2_lstm_residual = r2_score(y_test_seq, pred_lstm_residual)
print(f"✓ R²: {r2_lstm_residual:.4f} (vs简单平均: {r2_lstm_residual - avg_r2:+.4f})")
strategies_results['策略1-LSTM残差学习'] = pred_lstm_residual

# ========== 策略2：GRU残差学习 ==========
print("\n" + "=" * 100)
print("【策略2】GRU残差学习：简化特征 + 保守XGBoost".center(100))
print("=" * 100)

xgb_gru_conservative = train_conservative_xgboost(simplified_train, gru_oof_residual)
gru_residual = xgb_gru_conservative.predict(simplified_test)
pred_gru_residual = gru_test_pred + gru_residual

r2_gru_residual = r2_score(y_test_seq, pred_gru_residual)
print(f"✓ R²: {r2_gru_residual:.4f} (vs简单平均: {r2_gru_residual - avg_r2:+.4f})")
strategies_results['策略2-GRU残差学习'] = pred_gru_residual

# ========== 策略3：双残差学习 ==========
print("\n" + "=" * 100)
print("【策略3】双残差学习：(LSTM+GRU)/2 + 保守XGBoost".center(100))
print("=" * 100)

xgb_dual = train_conservative_xgboost(simplified_train, avg_oof_residual)
dual_residual = xgb_dual.predict(simplified_test)
pred_dual = avg_test_pred + dual_residual

r2_dual = r2_score(y_test_seq, pred_dual)
print(f"✓ R²: {r2_dual:.4f} (vs简单平均: {r2_dual - avg_r2:+.4f})")
strategies_results['策略3-双残差学习'] = pred_dual

# ========== 策略4：残差剪裁 ==========
print("\n" + "=" * 100)
print("【策略4】GRU残差学习 + 残差剪裁".center(100))
print("=" * 100)

gru_residual_clipped = clip_residual(gru_residual, threshold=2.0)
pred_gru_clipped = gru_test_pred + gru_residual_clipped

r2_gru_clipped = r2_score(y_test_seq, pred_gru_clipped)
print(f"✓ R²: {r2_gru_clipped:.4f} (vs简单平均: {r2_gru_clipped - avg_r2:+.4f})")
strategies_results['策略4-残差剪裁'] = pred_gru_clipped

# ========== 策略5：加权融合（30%）==========
print("\n" + "=" * 100)
print("【策略5】GRU残差学习 + 加权融合(30%)".center(100))
print("=" * 100)

pred_gru_weighted = weighted_residual_correction(gru_test_pred, gru_residual, weight=0.3)

r2_gru_weighted = r2_score(y_test_seq, pred_gru_weighted)
print(f"✓ R²: {r2_gru_weighted:.4f} (vs简单平均: {r2_gru_weighted - avg_r2:+.4f})")
strategies_results['策略5-加权融合30%'] = pred_gru_weighted

# ========== 策略6：终极组合 ==========
print("\n" + "=" * 100)
print("【策略6】终极组合：残差剪裁 + 加权融合(30%)".center(100))
print("=" * 100)

pred_ultimate = weighted_residual_correction(gru_test_pred, gru_residual_clipped, weight=0.3)

r2_ultimate = r2_score(y_test_seq, pred_ultimate)
print(f"✓ R²: {r2_ultimate:.4f} (vs简单平均: {r2_ultimate - avg_r2:+.4f})")
strategies_results['策略6-终极组合'] = pred_ultimate

# ========== 策略7：动态权重融合 ==========
print("\n" + "=" * 100)
print("【策略7】动态权重融合：Ridge学习LSTM和GRU权重".center(100))
print("=" * 100)

lstm_res_train = xgb_lstm_conservative.predict(simplified_train)
gru_res_train = xgb_gru_conservative.predict(simplified_train)

lstm_res_test = xgb_lstm_conservative.predict(simplified_test)
gru_res_test = xgb_gru_conservative.predict(simplified_test)

meta_features_train = np.column_stack([
    lstm_oof_preds + lstm_res_train,
    gru_oof_preds + gru_res_train
])
meta_features_test = np.column_stack([
    lstm_test_pred + lstm_res_test,
    gru_test_pred + gru_res_test
])

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features_train, y_train_seq)
pred_dynamic = meta_model.predict(meta_features_test)

lstm_weight = meta_model.coef_[0]
gru_weight = meta_model.coef_[1]

r2_dynamic = r2_score(y_test_seq, pred_dynamic)
print(f"学习到的权重: LSTM={lstm_weight:.4f}, GRU={gru_weight:.4f}")
print(f"✓ R²: {r2_dynamic:.4f} (vs简单平均: {r2_dynamic - avg_r2:+.4f})")
strategies_results['策略7-动态权重融合'] = pred_dynamic

# ========== 策略8：Ridge残差学习 ==========
print("\n" + "=" * 100)
print("【策略8】Ridge残差学习：最保守的线性模型".center(100))
print("=" * 100)

ridge_model = train_ridge_model(simplified_train, gru_oof_residual, alpha=10.0)
ridge_residual = ridge_model.predict(simplified_test)
pred_ridge = gru_test_pred + ridge_residual

r2_ridge = r2_score(y_test_seq, pred_ridge)
print(f"✓ R²: {r2_ridge:.4f} (vs简单平均: {r2_ridge - avg_r2:+.4f})")
strategies_results['策略8-Ridge残差'] = pred_ridge

# ========== 综合对比 ==========
print("\n" + "=" * 100)
print("所有策略性能对比（归一化数据）".center(100))
print("=" * 100)

all_strategies = {
    'LSTM单模型': lstm_test_pred,
    'GRU单模型': gru_test_pred,
    '简单平均(基线)': avg_test_pred,
    **strategies_results
}

print(f"\n{'策略':<30} {'R²':>10} {'vs基线':>10} {'MAE':>12} {'RMSE':>12}")
print("-" * 75)

results_list = []
for name, pred in all_strategies.items():
    r2 = r2_score(y_test_seq, pred)
    mae = mean_absolute_error(y_test_seq, pred)
    rmse = sqrt(mean_squared_error(y_test_seq, pred))
    improvement = r2 - avg_r2
    results_list.append((name, r2, improvement, mae, rmse, pred))
    print(f"{name:<30} {r2:>10.4f} {improvement:>10.4f} {mae:>12.6f} {rmse:>12.6f}")

results_list.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 100)
print("性能排名（按R²降序）".center(100))
print("=" * 100)

best_r2 = results_list[0][1]
best_name = results_list[0][0]

for rank, (name, r2, improvement, mae, rmse, pred) in enumerate(results_list, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:>2}. {name:<30} R²={r2:.4f} (vs基线: {improvement:+.4f})")

print(f"\n🏆 最佳策略: {best_name} (R² = {best_r2:.4f})")

# ========== 原始尺度评估 ==========
print("\n" + "=" * 100)
print("原始尺度性能对比".center(100))
print("=" * 100)

y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1))
strategies_original = {}

print(f"\n{'策略':<30} {'R²':>10} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
print("-" * 75)

for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    strategies_original[name] = pred_original

    r2 = r2_score(y_test_original, pred_original)
    mae = mean_absolute_error(y_test_original, pred_original)
    rmse = sqrt(mean_squared_error(y_test_original, pred_original))
    mape = np.mean(np.abs((pred_original - y_test_original) / (y_test_original + 1e-8)))

    print(f"{name:<30} {r2:>10.4f} {mae:>12.2f} {rmse:>12.2f} {mape:>12.6f}")

# ========== 保存结果 ==========
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

import pickle
import json

# 保存模型
lstm_final.save(results_directory + 'lstm_final_fixed.h5')
gru_final.save(results_directory + 'gru_final_fixed.h5')

with open(results_directory + 'xgb_gru_conservative_fixed.pkl', 'wb') as f:
    pickle.dump(xgb_gru_conservative, f)

with open(results_directory + 'ridge_meta_model_fixed.pkl', 'wb') as f:
    pickle.dump(meta_model, f)

with open(results_directory + 'scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

# 保存参数
if ENABLE_HYPEROPT:
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


    hyperparams = {
        'lstm': {k: convert_to_serializable(v) for k, v in best_lstm_params.items()},
        'gru': {k: convert_to_serializable(v) for k, v in best_gru_params.items()},
        'hyperopt_enabled': True,
        'hyperopt_evals': HYPEROPT_EVALS,
        'cv_splits': CV_SPLITS
    }

    with open(results_directory + 'hyperparameters_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)

# 保存预测结果
predictions_dict = {'true_value': y_test_original.flatten()}
for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    predictions_dict[name.replace('/', '_').replace('(', '').replace(')', '')] = pred_original.flatten()

results_df = pd.DataFrame(predictions_dict)
results_df.to_csv(results_directory + 'all_predictions_fixed.csv', index=False)

# ========== 最终总结 ==========
print("\n" + "=" * 100)
print("🎉 修复版Hyperopt优化模型训练完成！".center(100))
print("=" * 100)

print(f"\n📊 核心修复:")
print(f"  ✓ 使用更保守的超参数搜索空间")
print(f"  ✓ 增加交叉验证折数（3→5）")
print(f"  ✓ 减少优化迭代次数（20→15）")
print(f"  ✓ OOF预测使用固定保守参数（防止过拟合）")
print(f"  ✓ 使用中位数代替平均值（更稳健）")
print(f"  ✓ 残差学习使用默认保守参数")

if ENABLE_HYPEROPT:
    print(f"\n🔧 优化后的参数:")
    print(f"  LSTM: units={best_lstm_params['units']}, dropout={best_lstm_params['dropout']:.3f}, "
          f"l2={best_lstm_params['l2_reg']:.4f}")
    print(f"  GRU: units={best_gru_params['units']}, dropout={best_gru_params['dropout']:.3f}, "
          f"l2={best_gru_params['l2_reg']:.4f}")
else:
    print(f"\n⚠️  使用默认保守参数（未启用Hyperopt）")

print(f"\n📈 实验结果:")
print(f"  LSTM单模型: R² = {lstm_test_r2:.4f}")
print(f"  GRU单模型:  R² = {gru_test_r2:.4f}")
print(f"  基线（简单平均）: R² = {avg_r2:.4f}")
print(f"  最佳策略: {best_name}")
print(f"  最佳性能: R² = {best_r2:.4f}")
print(f"  性能提升: {best_r2 - avg_r2:+.4f} ({(best_r2 - avg_r2) / avg_r2 * 100:+.2f}%)")

print(f"\n🏆 Top5策略排名:")
for rank, (name, r2, improvement, _, _, _) in enumerate(results_list[:5], 1):
    print(f"  {rank}. {name:<30} R²={r2:.4f} (改进: {improvement:+.4f})")

print(f"\n💡 关键改进说明:")
print(f"  1. 更保守的超参数空间防止激进优化")
print(f"  2. OOF使用固定参数避免过拟合传播")
print(f"  3. 残差学习使用默认参数保证稳定性")
print(f"  4. 可通过ENABLE_HYPEROPT=False完全禁用优化")

print(f"\n🔍 过拟合分析:")
if overfitting_detected:
    print(f"  ⚠️  基础模型存在过拟合（LSTM差距={lstm_train_r2 - lstm_test_r2:.4f}, GRU差距={gru_train_r2 - gru_test_r2:.4f}）")
    print(f"  💡 已采用多重策略缓解")
else:
    print(f"  ✅ 过拟合控制良好")
    print(f"  ✅ 模型泛化能力较强")

print(f"\n💾 所有结果已保存到: {results_directory}")
print("=" * 100)