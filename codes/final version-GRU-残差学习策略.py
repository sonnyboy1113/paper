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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("LSTM + GRU + XGBoost 融合时间序列预测 - 改进版残差学习".center(100))
print("核心改进：防止过拟合 + 简化特征 + 保守参数 + 多种残差策略".center(100))
print("=" * 100)

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


# 添加滞后特征
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


# 定义模型
def build_simple_lstm(input_shape):
    return Sequential([
        layers.LSTM(units=100, input_shape=input_shape),
        layers.Dense(1)
    ])


def build_simple_gru(input_shape):
    return Sequential([
        layers.GRU(units=100, input_shape=input_shape),
        layers.Dense(1)
    ])


# OOF预测生成
def get_oof_predictions(X_seq, y_seq, model_type='lstm', n_splits=5):
    print(f"\n生成{model_type.upper()} OOF预测（TimeSeriesSplit with {n_splits} splits）...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros(len(y_seq))

    fold = 1
    for train_idx, val_idx in tscv.split(X_seq):
        print(f"  Fold {fold}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")

        X_fold_train, X_fold_val = X_seq[train_idx], X_seq[val_idx]
        y_fold_train, y_fold_val = y_seq[train_idx], y_seq[val_idx]

        if model_type == 'lstm':
            model = build_simple_lstm((X_seq.shape[1], X_seq.shape[2]))
        else:
            model = build_simple_gru((X_seq.shape[1], X_seq.shape[2]))

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
        fold += 1

    return oof_preds


print("\n" + "=" * 100)
print("第一步：生成LSTM和GRU的OOF预测".center(100))
print("=" * 100)

lstm_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'lstm', n_splits=5)
gru_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'gru', n_splits=5)

print(f"\nOOF预测生成完成！")
print(f"LSTM OOF R^2: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"GRU OOF R^2: {r2_score(y_train_seq, gru_oof_preds):.4f}")

# 训练最终模型
print("\n" + "=" * 100)
print("第二步：训练最终的LSTM和GRU模型".center(100))
print("=" * 100)

print("\n训练最终LSTM模型...")
lstm_final = Sequential([
    layers.LSTM(
        units=80,
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        kernel_regularizer=l2(0.01),
        recurrent_regularizer=l2(0.01)
    ),
    layers.Dropout(0.3),
    layers.Dense(1)
])
lstm_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
lstm_history = lstm_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ],
    verbose=0
)
print("✓ LSTM模型训练完成")

print("\n训练最终GRU模型...")
gru_final = Sequential([
    layers.GRU(units=100, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    layers.Dropout(0.3),
    layers.Dense(1)
])
gru_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
gru_history = gru_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)
print("✓ GRU模型训练完成")

lstm_test_pred = lstm_final.predict(X_test_seq, verbose=0).flatten()
gru_test_pred = gru_final.predict(X_test_seq, verbose=0).flatten()

# 诊断过拟合
lstm_train_pred = lstm_final.predict(X_train_seq, verbose=0).flatten()
gru_train_pred = gru_final.predict(X_train_seq, verbose=0).flatten()

lstm_train_r2 = r2_score(y_train_seq, lstm_train_pred)
lstm_test_r2 = r2_score(y_test_seq, lstm_test_pred)
gru_train_r2 = r2_score(y_train_seq, gru_train_pred)
gru_test_r2 = r2_score(y_test_seq, gru_test_pred)

print(f"\n过拟合诊断:")
print(f"LSTM - 训练R^2: {lstm_train_r2:.4f}, 测试R^2: {lstm_test_r2:.4f}, 差距: {lstm_train_r2 - lstm_test_r2:.4f}")
print(f"GRU  - 训练R^2: {gru_train_r2:.4f}, 测试R^2: {gru_test_r2:.4f}, 差距: {gru_train_r2 - gru_test_r2:.4f}")

if max(lstm_train_r2 - lstm_test_r2, gru_train_r2 - gru_test_r2) > 0.15:
    print(f"⚠️  检测到明显过拟合，残差学习需要使用保守策略！")


# ========== 【核心改进】特征工程函数 ==========

def create_original_features(X_flat, lstm_preds, gru_preds):
    """原始增强特征（80维）- 容易过拟合"""
    features_list = [X_flat]
    features_list.append(lstm_preds.reshape(-1, 1))
    features_list.append(gru_preds.reshape(-1, 1))
    features_list.append((lstm_preds + gru_preds).reshape(-1, 1))
    features_list.append((lstm_preds - gru_preds).reshape(-1, 1))
    features_list.append(np.abs(lstm_preds - gru_preds).reshape(-1, 1))
    features_list.append((lstm_preds * gru_preds).reshape(-1, 1))
    features_list.append(np.maximum(lstm_preds, gru_preds).reshape(-1, 1))
    features_list.append(np.minimum(lstm_preds, gru_preds).reshape(-1, 1))
    disagreement = np.abs(lstm_preds - gru_preds)
    confidence = 1 / (1 + disagreement)
    features_list.append(confidence.reshape(-1, 1))
    weighted_avg = 0.5 * lstm_preds + 0.5 * gru_preds
    features_list.append(weighted_avg.reshape(-1, 1))
    return np.hstack(features_list)


def create_simplified_features(X_flat, lstm_preds, gru_preds):
    """简化特征（74维）- 只保留4个最关键的增强特征"""
    features_list = [X_flat]
    features_list.append(lstm_preds.reshape(-1, 1))
    features_list.append(gru_preds.reshape(-1, 1))
    features_list.append(((lstm_preds + gru_preds) / 2).reshape(-1, 1))
    features_list.append(np.abs(lstm_preds - gru_preds).reshape(-1, 1))
    return np.hstack(features_list)


def create_minimal_features(X_flat, lstm_preds, gru_preds):
    """最小化特征（72维）- 只添加预测值"""
    features_list = [X_flat]
    features_list.append(lstm_preds.reshape(-1, 1))
    features_list.append(gru_preds.reshape(-1, 1))
    return np.hstack(features_list)


# ========== 【核心改进】残差学习训练函数 ==========

def train_original_xgboost(X_train, y_train):
    """原始XGBoost参数（容易过拟合）"""
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def train_conservative_xgboost(X_train, y_train):
    """保守XGBoost参数（防止过拟合）"""
    model = XGBRegressor(
        n_estimators=100,  # 500 → 100
        learning_rate=0.01,  # 0.03 → 0.01
        max_depth=3,  # 4 → 3
        min_child_weight=5,  # 新增
        subsample=0.7,  # 0.8 → 0.7
        colsample_bytree=0.7,  # 0.8 → 0.7
        reg_alpha=0.1,  # 新增 L1
        reg_lambda=1.0,  # 新增 L2
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_with_early_stopping(X_train, y_train):
    """带早停的XGBoost（兼容新旧版本）"""
    split_point = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:split_point], X_train[split_point:]
    y_tr, y_val = y_train[:split_point], y_train[split_point:]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=20  # 新版本：作为初始化参数
    )

    try:
        # 新版本XGBoost (>=2.0.0)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except TypeError:
        # 旧版本XGBoost (<2.0.0) - 如果新版本失败，尝试旧版本方式
        model = XGBRegressor(
            n_estimators=500,
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
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
    return model


def train_ridge_model(X_train, y_train, alpha=10.0):
    """Ridge线性回归（最保守）"""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# ========== 【核心改进】残差处理函数 ==========

def clip_residual(residual_pred, threshold=2.0):
    """剪裁极端残差值"""
    std = np.std(residual_pred)
    mean = np.mean(residual_pred)
    return np.clip(residual_pred, mean - threshold * std, mean + threshold * std)


def weighted_residual_correction(base_pred, residual_pred, weight=0.5):
    """加权残差修正"""
    return base_pred + weight * residual_pred


# ========== 基准策略 ==========
print("\n" + "=" * 100)
print("基准策略：简单平均".center(100))
print("=" * 100)

avg_test_pred = (lstm_test_pred + gru_test_pred) / 2
avg_r2 = r2_score(y_test_seq, avg_test_pred)
print(f"简单平均测试R^2: {avg_r2:.4f}")

# 计算残差
lstm_oof_residual = y_train_seq - lstm_oof_preds
gru_oof_residual = y_train_seq - gru_oof_preds

print(f"\n残差统计:")
print(f"LSTM残差 - 均值: {np.mean(lstm_oof_residual):.6f}, 标准差: {np.std(lstm_oof_residual):.6f}")
print(f"GRU残差  - 均值: {np.mean(gru_oof_residual):.6f}, 标准差: {np.std(gru_oof_residual):.6f}")

# ========== 策略对比实验 ==========
print("\n" + "=" * 100)
print("改进版残差学习策略对比实验".center(100))
print("=" * 100)

strategies_results = {}

# 准备基础特征
basic_features_train = X_train_flat[:len(gru_oof_preds)]
basic_features_test = X_test_flat

# 原始特征
original_train = create_original_features(basic_features_train, lstm_oof_preds, gru_oof_preds)
original_test = create_original_features(basic_features_test, lstm_test_pred, gru_test_pred)

simplified_train = create_simplified_features(basic_features_train, lstm_oof_preds, gru_oof_preds)
simplified_test = create_simplified_features(basic_features_test, lstm_test_pred, gru_test_pred)

minimal_train = create_minimal_features(basic_features_train, lstm_oof_preds, gru_oof_preds)
minimal_test = create_minimal_features(basic_features_test, lstm_test_pred, gru_test_pred)

print(f"\n特征维度对比:")
print(f"  原始增强特征: {original_train.shape[1]} 维")
print(f"  简化特征: {simplified_train.shape[1]} 维")
print(f"  最小化特征: {minimal_train.shape[1]} 维")

# ========== 实验1：原始方法（重现问题）==========
print("\n" + "-" * 100)
print("【实验1】原始方法：原始特征 + 激进XGBoost（重现过拟合问题）")
print("-" * 100)

model_exp1 = train_original_xgboost(original_train, gru_oof_residual)
residual_exp1 = model_exp1.predict(original_test)
pred_exp1 = gru_test_pred + residual_exp1
r2_exp1 = r2_score(y_test_seq, pred_exp1)

print(f"✓ R^2: {r2_exp1:.4f} (vs简单平均: {r2_exp1 - avg_r2:+.4f})")
strategies_results['实验1-原始方法'] = pred_exp1

# ========== 实验2：简化特征 + 激进XGBoost ==========
print("\n" + "-" * 100)
print("【实验2】改进A：简化特征 + 激进XGBoost")
print("-" * 100)

model_exp2 = train_original_xgboost(simplified_train, gru_oof_residual)
residual_exp2 = model_exp2.predict(simplified_test)
pred_exp2 = gru_test_pred + residual_exp2
r2_exp2 = r2_score(y_test_seq, pred_exp2)

print(f"✓ R^2: {r2_exp2:.4f} (vs简单平均: {r2_exp2 - avg_r2:+.4f}, vs实验1: {r2_exp2 - r2_exp1:+.4f})")
strategies_results['实验2-简化特征'] = pred_exp2

# ========== 实验3：简化特征 + 保守XGBoost ==========
print("\n" + "-" * 100)
print("【实验3】改进B：简化特征 + 保守XGBoost")
print("-" * 100)

model_exp3 = train_conservative_xgboost(simplified_train, gru_oof_residual)
residual_exp3 = model_exp3.predict(simplified_test)
pred_exp3 = gru_test_pred + residual_exp3
r2_exp3 = r2_score(y_test_seq, pred_exp3)

print(f"✓ R^2: {r2_exp3:.4f} (vs简单平均: {r2_exp3 - avg_r2:+.4f}, vs实验1: {r2_exp3 - r2_exp1:+.4f})")
strategies_results['实验3-保守参数'] = pred_exp3

# ========== 实验4：简化特征 + 早停XGBoost ==========
print("\n" + "-" * 100)
print("【实验4】改进C：简化特征 + 早停XGBoost")
print("-" * 100)

model_exp4 = train_xgboost_with_early_stopping(simplified_train, gru_oof_residual)
residual_exp4 = model_exp4.predict(simplified_test)
pred_exp4 = gru_test_pred + residual_exp4
r2_exp4 = r2_score(y_test_seq, pred_exp4)

print(f"✓ R^2: {r2_exp4:.4f} (vs简单平均: {r2_exp4 - avg_r2:+.4f}, vs实验1: {r2_exp4 - r2_exp1:+.4f})")
strategies_results['实验4-早停机制'] = pred_exp4

# ========== 实验5：简化特征 + 保守XGBoost + 残差剪裁 ==========
print("\n" + "-" * 100)
print("【实验5】改进D：简化特征 + 保守XGBoost + 残差剪裁")
print("-" * 100)

model_exp5 = train_conservative_xgboost(simplified_train, gru_oof_residual)
residual_exp5 = model_exp5.predict(simplified_test)
residual_exp5_clipped = clip_residual(residual_exp5, threshold=2.0)
pred_exp5 = gru_test_pred + residual_exp5_clipped
r2_exp5 = r2_score(y_test_seq, pred_exp5)

print(f"✓ R^2: {r2_exp5:.4f} (vs简单平均: {r2_exp5 - avg_r2:+.4f}, vs实验1: {r2_exp5 - r2_exp1:+.4f})")
print(f"  残差剪裁前std: {np.std(residual_exp5):.6f}, 剪裁后std: {np.std(residual_exp5_clipped):.6f}")
strategies_results['实验5-残差剪裁'] = pred_exp5

# ========== 实验6：简化特征 + 保守XGBoost + 加权融合 ==========
print("\n" + "-" * 100)
print("【实验6】改进E：简化特征 + 保守XGBoost + 加权融合(50%)")
print("-" * 100)

model_exp6 = train_conservative_xgboost(simplified_train, gru_oof_residual)
residual_exp6 = model_exp6.predict(simplified_test)
pred_exp6 = weighted_residual_correction(gru_test_pred, residual_exp6, weight=0.5)
r2_exp6 = r2_score(y_test_seq, pred_exp6)

print(f"✓ R^2: {r2_exp6:.4f} (vs简单平均: {r2_exp6 - avg_r2:+.4f}, vs实验1: {r2_exp6 - r2_exp1:+.4f})")
strategies_results['实验6-加权融合50%'] = pred_exp6

# ========== 实验7：最小化特征 + 保守XGBoost ==========
print("\n" + "-" * 100)
print("【实验7】改进F：最小化特征(仅2个) + 保守XGBoost")
print("-" * 100)

model_exp7 = train_conservative_xgboost(minimal_train, gru_oof_residual)
residual_exp7 = model_exp7.predict(minimal_test)
pred_exp7 = gru_test_pred + residual_exp7
r2_exp7 = r2_score(y_test_seq, pred_exp7)

print(f"✓ R^2: {r2_exp7:.4f} (vs简单平均: {r2_exp7 - avg_r2:+.4f}, vs实验1: {r2_exp7 - r2_exp1:+.4f})")
strategies_results['实验7-最小特征'] = pred_exp7

# ========== 实验8：简化特征 + Ridge线性模型 ==========
print("\n" + "-" * 100)
print("【实验8】改进G：简化特征 + Ridge线性回归(alpha=10)")
print("-" * 100)

model_exp8 = train_ridge_model(simplified_train, gru_oof_residual, alpha=10.0)
residual_exp8 = model_exp8.predict(simplified_test)
pred_exp8 = gru_test_pred + residual_exp8
r2_exp8 = r2_score(y_test_seq, pred_exp8)

print(f"✓ R^2: {r2_exp8:.4f} (vs简单平均: {r2_exp8 - avg_r2:+.4f}, vs实验1: {r2_exp8 - r2_exp1:+.4f})")
strategies_results['实验8-Ridge回归'] = pred_exp8

# ========== 实验9：组合最优策略 ==========
print("\n" + "-" * 100)
print("【实验9】终极组合：简化特征 + 保守XGBoost + 残差剪裁 + 加权融合(30%)")
print("-" * 100)

model_exp9 = train_conservative_xgboost(simplified_train, gru_oof_residual)
residual_exp9 = model_exp9.predict(simplified_test)
residual_exp9_clipped = clip_residual(residual_exp9, threshold=2.0)
pred_exp9 = weighted_residual_correction(gru_test_pred, residual_exp9_clipped, weight=0.3)
r2_exp9 = r2_score(y_test_seq, pred_exp9)

print(f"✓ R^2: {r2_exp9:.4f} (vs简单平均: {r2_exp9 - avg_r2:+.4f}, vs实验1: {r2_exp9 - r2_exp1:+.4f})")
strategies_results['实验9-终极组合'] = pred_exp9

# ========== 动态权重融合（作为对比）==========
print("\n" + "-" * 100)
print("【对比】动态权重融合策略")
print("-" * 100)

# LSTM残差模型
xgb_lstm_dynamic = train_conservative_xgboost(simplified_train, lstm_oof_residual)
lstm_res_train = xgb_lstm_dynamic.predict(simplified_train)
lstm_res_test = xgb_lstm_dynamic.predict(simplified_test)

# GRU残差模型
xgb_gru_dynamic = train_conservative_xgboost(simplified_train, gru_oof_residual)
gru_res_train = xgb_gru_dynamic.predict(simplified_train)
gru_res_test = xgb_gru_dynamic.predict(simplified_test)

# Ridge学习权重
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
r2_dynamic = r2_score(y_test_seq, pred_dynamic)

lstm_weight = meta_model.coef_[0]
gru_weight = meta_model.coef_[1]

print(f"学习到的权重: LSTM={lstm_weight:.4f}, GRU={gru_weight:.4f}")
print(f"✓ R^2: {r2_dynamic:.4f} (vs简单平均: {r2_dynamic - avg_r2:+.4f}, vs实验1: {r2_dynamic - r2_exp1:+.4f})")
strategies_results['动态权重融合'] = pred_dynamic

# ========== 综合结果对比 ==========
print("\n" + "=" * 100)
print("所有策略综合对比（归一化数据）".center(100))
print("=" * 100)

all_strategies = {
    'GRU单模型': gru_test_pred,
    '简单平均(基线)': avg_test_pred,
    **strategies_results
}

print(f"\n{'策略':<35} {'R^2':>10} {'vs基线':>10} {'MAE':>12} {'RMSE':>12}")
print("-" * 85)

results_list = []
for name, pred in all_strategies.items():
    r2 = r2_score(y_test_seq, pred)
    mae = mean_absolute_error(y_test_seq, pred)
    rmse = sqrt(mean_squared_error(y_test_seq, pred))
    improvement = r2 - avg_r2
    results_list.append((name, r2, improvement, mae, rmse, pred))
    print(f"{name:<35} {r2:>10.4f} {improvement:>10.4f} {mae:>12.6f} {rmse:>12.6f}")

# 排序
results_list.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 100)
print("性能排名（按R^2降序）".center(100))
print("=" * 100)

best_r2 = results_list[0][1]
best_name = results_list[0][0]

for rank, (name, r2, improvement, mae, rmse, pred) in enumerate(results_list, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:>2}. {name:<35} R^2={r2:.4f} (vs基线: {improvement:+.4f})")

print(f"\n🏆 最佳策略: {best_name} (R^2 = {best_r2:.4f})")

# ========== 原始尺度评估 ==========
print("\n" + "=" * 100)
print("原始尺度性能对比".center(100))
print("=" * 100)

y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1))
strategies_original = {}

print(f"\n{'策略':<35} {'R^2':>10} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
print("-" * 85)

for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    strategies_original[name] = pred_original

    r2 = r2_score(y_test_original, pred_original)
    mae = mean_absolute_error(y_test_original, pred_original)
    rmse = sqrt(mean_squared_error(y_test_original, pred_original))
    mape = np.mean(np.abs((pred_original - y_test_original) / (y_test_original + 1e-8)))

    print(f"{name:<35} {r2:>10.4f} {mae:>12.2f} {rmse:>12.2f} {mape:>12.6f}")

# ========== 可视化 ==========
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 1. 改进效果对比柱状图
fig, ax = plt.subplots(figsize=(16, 8))

strategy_names = [name for name, _, _, _, _, _ in results_list]
r2_scores = [r2 for _, r2, _, _, _, _ in results_list]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(strategy_names)))

bars = ax.barh(range(len(strategy_names)), r2_scores, color=colors, alpha=0.8)

# 添加基线虚线
ax.axvline(x=avg_r2, color='red', linestyle='--', linewidth=2, label='简单平均基线', alpha=0.7)

# 添加数值标签
for i, (bar, r2, improvement) in enumerate(zip(bars, r2_scores, [imp for _, _, imp, _, _, _ in results_list])):
    label = f'{r2:.4f}'
    if improvement > 0:
        label += f' (+{improvement:.4f})'
        color = 'green'
    elif improvement < 0:
        label += f' ({improvement:.4f})'
        color = 'red'
    else:
        color = 'black'

    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            label, ha='left', va='center', fontweight='bold', fontsize=9, color=color)

ax.set_yticks(range(len(strategy_names)))
ax.set_yticklabels(strategy_names, fontsize=10)
ax.set_xlabel('R^2 Score', fontsize=12, fontweight='bold')
ax.set_title('改进版残差学习策略效果对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(results_directory + 'improved_residual_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 2. 关键改进策略对比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

key_strategies = [
    ('实验1-原始方法', strategies_results['实验1-原始方法'], 'red'),
    ('实验3-保守参数', strategies_results['实验3-保守参数'], 'orange'),
    ('实验4-早停机制', strategies_results['实验4-早停机制'], 'yellow'),
    ('实验6-加权融合50%', strategies_results['实验6-加权融合50%'], 'cyan'),
    ('实验9-终极组合', strategies_results['实验9-终极组合'], 'green'),
    ('动态权重融合', strategies_results['动态权重融合'], 'purple'),
]

for idx, (name, pred, color) in enumerate(key_strategies):
    ax = axes[idx // 3, idx % 3]

    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    r2 = r2_score(y_test_original, pred_original)
    improvement = r2 - r2_score(y_test_original, y_scaler.inverse_transform(avg_test_pred.reshape(-1, 1)))

    ax.plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
    ax.plot(pred_original, label=name, linewidth=2, alpha=0.8, color=color)
    ax.set_title(f'{name}\nR^2={r2:.4f} (vs基线: {improvement:+.4f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('样本序号', fontsize=9)
    ax.set_ylabel('玉米价格', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'key_strategies_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 3. 残差分析对比
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 原始方法残差
residual_original = y_test_seq - strategies_results['实验1-原始方法']
axes[0, 0].hist(residual_original, bins=30, color='red', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0, 0].set_title('实验1-原始方法 残差分布', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('残差')
axes[0, 0].set_ylabel('频数')
axes[0, 0].text(0.05, 0.95, f'std={np.std(residual_original):.5f}',
                transform=axes[0, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 保守参数残差
residual_conservative = y_test_seq - strategies_results['实验3-保守参数']
axes[0, 1].hist(residual_conservative, bins=30, color='orange', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_title('实验3-保守参数 残差分布', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('残差')
axes[0, 1].set_ylabel('频数')
axes[0, 1].text(0.05, 0.95, f'std={np.std(residual_conservative):.5f}',
                transform=axes[0, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 终极组合残差
residual_ultimate = y_test_seq - strategies_results['实验9-终极组合']
axes[1, 0].hist(residual_ultimate, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=2)
axes[1, 0].set_title('实验9-终极组合 残差分布', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('残差')
axes[1, 0].set_ylabel('频数')
axes[1, 0].text(0.05, 0.95, f'std={np.std(residual_ultimate):.5f}',
                transform=axes[1, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 残差对比（时间序列）
axes[1, 1].plot(residual_original, label='原始方法', linewidth=2, alpha=0.7, color='red')
axes[1, 1].plot(residual_conservative, label='保守参数', linewidth=2, alpha=0.7, color='orange')
axes[1, 1].plot(residual_ultimate, label='终极组合', linewidth=2, alpha=0.7, color='green')
axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 1].set_title('残差时间序列对比', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('样本序号')
axes[1, 1].set_ylabel('残差')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'residual_analysis_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 4. 改进效果雷达图
from math import pi

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 选择几个关键策略
radar_strategies = [
    '简单平均(基线)',
    '实验1-原始方法',
    '实验3-保守参数',
    '实验9-终极组合',
    '动态权重融合'
]

# 计算多个评价指标（归一化到0-1）
metrics = ['R^2', '1-MAE', '1-RMSE', '稳定性', '复杂度']
n_metrics = len(metrics)

angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
angles += angles[:1]

for strategy_name in radar_strategies:
    pred = all_strategies[strategy_name]

    r2 = r2_score(y_test_seq, pred)
    mae = mean_absolute_error(y_test_seq, pred)
    rmse = sqrt(mean_squared_error(y_test_seq, pred))
    stability = 1 - np.std(y_test_seq - pred) / np.std(y_test_seq)

    # 复杂度评分（简单=1，复杂=0）
    if '简单平均' in strategy_name:
        complexity = 1.0
    elif '原始方法' in strategy_name:
        complexity = 0.3
    elif '终极组合' in strategy_name:
        complexity = 0.5
    else:
        complexity = 0.7

    # 归一化
    values = [
        r2,
        1 - (mae / 0.1),  # 假设最大MAE=0.1
        1 - (rmse / 0.1),  # 假设最大RMSE=0.1
        stability,
        complexity
    ]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=strategy_name, alpha=0.7)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title('残差学习策略多维评估', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax.grid(True)

plt.tight_layout()
plt.savefig(results_directory + 'strategies_radar_chart.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ========== 保存模型和结果 ==========
import pickle

print("\n" + "=" * 100)
print("保存模型和结果".center(100))
print("=" * 100)

# 保存Keras模型
lstm_final.save(results_directory + 'lstm_final_model.h5')
gru_final.save(results_directory + 'gru_final_model.h5')

# 保存最佳XGBoost模型（终极组合使用的）
with open(results_directory + 'best_xgboost_model.pkl', 'wb') as f:
    pickle.dump(model_exp9, f)

# 保存动态权重模型
with open(results_directory + 'xgb_lstm_dynamic.pkl', 'wb') as f:
    pickle.dump(xgb_lstm_dynamic, f)

with open(results_directory + 'xgb_gru_dynamic.pkl', 'wb') as f:
    pickle.dump(xgb_gru_dynamic, f)

with open(results_directory + 'ridge_meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)

# 保存归一化器
with open(results_directory + 'scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

# 保存所有预测结果
predictions_dict = {'true_value': y_test_original.flatten()}
for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    predictions_dict[name.replace('/', '_')] = pred_original.flatten()

results_df = pd.DataFrame(predictions_dict)
results_df.to_csv(results_directory + 'all_predictions_improved.csv', index=False)

# 保存性能指标
metrics_data = []
for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    metrics_data.append({
        'strategy': name,
        'r2': r2_score(y_test_original, pred_original),
        'mae': mean_absolute_error(y_test_original, pred_original),
        'rmse': sqrt(mean_squared_error(y_test_original, pred_original)),
        'mape': np.mean(np.abs((pred_original - y_test_original) / (y_test_original + 1e-8)))
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.sort_values('r2', ascending=False)
metrics_df.to_csv(results_directory + 'performance_metrics.csv', index=False)

print("\n✓ 保存完成！")
print(f"  - lstm_final_model.h5")
print(f"  - gru_final_model.h5")
print(f"  - best_xgboost_model.pkl (终极组合模型)")
print(f"  - xgb_lstm_dynamic.pkl")
print(f"  - xgb_gru_dynamic.pkl")
print(f"  - ridge_meta_model.pkl")
print(f"  - scalers.pkl")
print(f"  - all_predictions_improved.csv")
print(f"  - performance_metrics.csv")

# ========== 最终总结报告 ==========
print("\n" + "=" * 100)
print("🎉 改进版残差学习训练完成！".center(100))
print("=" * 100)

print(f"\n📊 实验结果总结:")
print(f"  基线方法（简单平均）: R^2 = {avg_r2:.4f}")
print(f"  原始残差学习（过拟合）: R^2 = {r2_exp1:.4f} ({r2_exp1 - avg_r2:+.4f})")
print(f"  最佳改进方法: {best_name}")
print(f"  最佳性能: R^2 = {best_r2:.4f} ({best_r2 - avg_r2:+.4f})")

print(f"\n💡 关键发现:")

improvements = {
    '简化特征': r2_exp2 - r2_exp1,
    '保守参数': r2_exp3 - r2_exp1,
    '早停机制': r2_exp4 - r2_exp1,
    '残差剪裁': r2_exp5 - r2_exp3,
    '加权融合': r2_exp6 - r2_exp3,
    '终极组合': r2_exp9 - r2_exp1,
}

for improvement_name, improvement_value in improvements.items():
    status = "✓ 有效" if improvement_value > 0 else "✗ 无效"
    print(f"  {status} {improvement_name}: {improvement_value:+.4f}")

print(f"\n🔍 技术分析:")
print(f"  1. 特征工程:")
print(f"     - 原始特征(80维) → 简化特征(74维): 改进 {r2_exp2 - r2_exp1:+.4f}")
print(f"     - 最小化特征(72维) 效果: {r2_exp7:.4f}")
print(f"  2. 模型参数:")
print(f"     - 激进参数 → 保守参数: 改进 {r2_exp3 - r2_exp2:+.4f}")
print(f"     - 加入早停机制: {r2_exp4:.4f}")
print(f"  3. 残差处理:")
print(f"     - 残差剪裁效果: {r2_exp5:.4f}")
print(f"     - 加权融合(50%)效果: {r2_exp6:.4f}")
print(f"     - 加权融合(30%)效果: {r2_exp9:.4f}")
print(f"  4. 模型选择:")
print(f"     - XGBoost vs Ridge: Ridge R^2={r2_exp8:.4f}")
print(f"     - 动态权重融合: R^2={r2_dynamic:.4f}")

print(f"\n📈 最佳实践建议:")
if best_r2 > avg_r2:
    print(f"  ✅ 残差学习在本数据集上有效！")
    print(f"  ✅ 推荐使用: {best_name}")
    print(f"  ✅ 相比简单平均提升: {best_r2 - avg_r2:.4f} ({(best_r2 - avg_r2) / avg_r2 * 100:.2f}%)")
else:
    print(f"  ⚠️  残差学习未超过简单平均基线")
    print(f"  ⚠️  建议:")
    print(f"     1. 继续使用简单平均作为最终方案")
    print(f"     2. 收集更多训练数据")
    print(f"     3. 改进基础模型（减少过拟合）")
    print(f"     4. 尝试其他集成方法（如Voting）")

print(f"\n🎯 核心结论:")
if lstm_train_r2 - lstm_test_r2 > 0.15 or gru_train_r2 - gru_test_r2 > 0.15:
    print(f"  ⚠️  基础模型存在严重过拟合（训练测试差距>{0.15:.2f}）")
    print(f"  💡 过拟合是残差学习失效的主要原因")
    print(f"  📌 建议优先解决基础模型的过拟合问题:")
    print(f"     - 增加训练数据")
    print(f"     - 增强正则化")
    print(f"     - 简化模型结构")
    print(f"     - 使用数据增强")

print(f"\n💾 所有结果已保存到: {results_directory}")
print("=" * 100)