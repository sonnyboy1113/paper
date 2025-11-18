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
from sklearn.linear_model import Ridge  # Ridge回归
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
print("LSTM + GRU + XGBoost 融合时间序列预测 - 增强版".center(100))
print("新增：Ridge残差学习 + 保守XGBoost参数 + 简化特征".center(100))
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
def add_features(X, y, is_train=True):
    X_new = X.copy()
    for i in range(1, 6):
        X_new[f'Corn_lag_{i}'] = y.shift(i)
    return X_new.dropna()


X_train_feat = add_features(X_train, y_train, is_train=True)
y_train = y_train.loc[X_train_feat.index]

X_test_feat = add_features(X_test, y_test, is_train=False)
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
print(f"LSTM OOF R²: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"GRU OOF R²: {r2_score(y_train_seq, gru_oof_preds):.4f}")

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

print(f"\nLSTM测试集R²: {r2_score(y_test_seq, lstm_test_pred):.4f}")
print(f"GRU测试集R²: {r2_score(y_test_seq, gru_test_pred):.4f}")


# ========== 特征工程函数 ==========
def create_enhanced_xgboost_features(X_flat, lstm_preds, gru_preds):
    """增强特征（80维）"""
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


print("\n" + "=" * 100)
print("构造特征集".center(100))
print("=" * 100)

# 构造增强特征
enhanced_train_features = create_enhanced_xgboost_features(
    X_train_flat[:len(lstm_oof_preds)],
    lstm_oof_preds,
    gru_oof_preds
)

enhanced_test_features = create_enhanced_xgboost_features(
    X_test_flat,
    lstm_test_pred,
    gru_test_pred
)

# 构造简化特征
simplified_train_features = create_simplified_features(
    X_train_flat[:len(lstm_oof_preds)],
    lstm_oof_preds,
    gru_oof_preds
)

simplified_test_features = create_simplified_features(
    X_test_flat,
    lstm_test_pred,
    gru_test_pred
)

print(f"\n✓ 特征维度:")
print(f"  原始特征: {X_train_flat.shape[1]} 维")
print(f"  增强特征: {enhanced_train_features.shape[1]} 维")
print(f"  简化特征: {simplified_train_features.shape[1]} 维")


# ========== 定义训练函数 ==========
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


def train_ridge_model(X_train, y_train, alpha=10.0):
    """Ridge线性回归（最保守，参考实验8）"""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# ========== 策略1：简单平均融合 ==========
print("\n" + "=" * 100)
print("【策略1】简单平均融合：(LSTM + GRU) / 2".center(100))
print("=" * 100)

avg_test_pred = (lstm_test_pred + gru_test_pred) / 2
avg_r2 = r2_score(y_test_seq, avg_test_pred)
print(f"简单平均测试集R²: {avg_r2:.4f}")

# 计算残差
lstm_oof_residual = y_train_seq - lstm_oof_preds
gru_oof_residual = y_train_seq - gru_oof_preds

print(f"\n残差统计:")
print(f"LSTM残差 - 均值: {np.mean(lstm_oof_residual):.6f}, 标准差: {np.std(lstm_oof_residual):.6f}")
print(f"GRU残差  - 均值: {np.mean(gru_oof_residual):.6f}, 标准差: {np.std(gru_oof_residual):.6f}")

# ========== 策略2：LSTM残差学习（保守XGBoost + 增强特征）==========
print("\n" + "=" * 100)
print("【策略2】LSTM残差学习：LSTM基础 + 保守XGBoost残差修正【增强特征】".center(100))
print("=" * 100)

xgb_lstm_residual_model = train_conservative_xgboost(enhanced_train_features, lstm_oof_residual)
lstm_residual_pred = xgb_lstm_residual_model.predict(enhanced_test_features)
lstm_residual_strategy_pred = lstm_test_pred + lstm_residual_pred

lstm_residual_r2 = r2_score(y_test_seq, lstm_residual_strategy_pred)
print(f"✓ LSTM残差学习策略测试集R²: {lstm_residual_r2:.4f}")
print(f"  改进: {lstm_residual_r2 - avg_r2:.4f} vs 简单平均")

# ========== 策略3：GRU残差学习（保守XGBoost + 增强特征）==========
print("\n" + "=" * 100)
print("【策略3】GRU残差学习：GRU基础 + 保守XGBoost残差修正【增强特征】".center(100))
print("=" * 100)

xgb_gru_residual_model = train_conservative_xgboost(enhanced_train_features, gru_oof_residual)
gru_residual_pred = xgb_gru_residual_model.predict(enhanced_test_features)
gru_residual_strategy_pred = gru_test_pred + gru_residual_pred

gru_residual_r2 = r2_score(y_test_seq, gru_residual_strategy_pred)
print(f"✓ GRU残差学习策略测试集R²: {gru_residual_r2:.4f}")
print(f"  改进: {gru_residual_r2 - avg_r2:.4f} vs 简单平均")

# ========== 策略4：加权平均Stacking（基于验证集表现）==========
print("\n" + "=" * 100)
print("【策略4】加权平均Stacking：基于OOF表现学习权重".center(100))
print("=" * 100)

# 计算LSTM和GRU在OOF上的R²
lstm_oof_r2 = r2_score(y_train_seq, lstm_oof_preds)
gru_oof_r2 = r2_score(y_train_seq, gru_oof_preds)

# 基于R²计算权重（表现好的权重更高）
total_r2 = lstm_oof_r2 + gru_oof_r2
lstm_weight_r2 = lstm_oof_r2 / total_r2
gru_weight_r2 = gru_oof_r2 / total_r2

print(f"\n基于OOF R²的权重:")
print(f"  LSTM OOF R²: {lstm_oof_r2:.4f} → 权重: {lstm_weight_r2:.4f}")
print(f"  GRU OOF R²:  {gru_oof_r2:.4f} → 权重: {gru_weight_r2:.4f}")

weighted_stacking_pred = lstm_weight_r2 * lstm_test_pred + gru_weight_r2 * gru_test_pred
weighted_stacking_r2 = r2_score(y_test_seq, weighted_stacking_pred)

print(f"\n✓ 加权平均Stacking测试集R²: {weighted_stacking_r2:.4f}")
print(f"  改进: {weighted_stacking_r2 - avg_r2:.4f} vs 简单平均")
print(f"  【说明】根据训练集表现自动分配权重，无需额外训练")

# ========== 策略5：双残差学习（保守XGBoost + 增强特征）==========
print("\n" + "=" * 100)
print("【策略5】双残差学习：(LSTM+GRU)/2基础 + 保守XGBoost残差【增强特征】".center(100))
print("=" * 100)

avg_oof_preds = (lstm_oof_preds + gru_oof_preds) / 2
avg_oof_residual = y_train_seq - avg_oof_preds

xgb_dual_model = train_conservative_xgboost(enhanced_train_features, avg_oof_residual)
avg_test_pred_for_residual = (lstm_test_pred + gru_test_pred) / 2
dual_residual_pred = xgb_dual_model.predict(enhanced_test_features)
dual_strategy_pred = avg_test_pred_for_residual + dual_residual_pred

dual_r2 = r2_score(y_test_seq, dual_strategy_pred)
print(f"✓ 双残差学习策略测试集R²: {dual_r2:.4f}")
print(f"  改进: {dual_r2 - avg_r2:.4f} vs 简单平均")

# ========== 策略6：动态权重残差融合（保守XGBoost + 简化特征）==========
print("\n" + "=" * 100)
print("【策略6】动态权重残差融合：Ridge回归学习最优权重【简化特征】".center(100))
print("=" * 100)

print("\n第一步：训练LSTM残差修正模型（保守XGBoost）...")
xgb_lstm_dynamic = train_conservative_xgboost(simplified_train_features, lstm_oof_residual)
lstm_res_train = xgb_lstm_dynamic.predict(simplified_train_features)
lstm_res_test = xgb_lstm_dynamic.predict(simplified_test_features)

print("\n第二步：训练GRU残差修正模型（保守XGBoost）...")
xgb_gru_dynamic = train_conservative_xgboost(simplified_train_features, gru_oof_residual)
gru_res_train = xgb_gru_dynamic.predict(simplified_train_features)
gru_res_test = xgb_gru_dynamic.predict(simplified_test_features)

print("\n第三步：使用Ridge回归学习最优融合权重...")
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

lstm_weight = meta_model.coef_[0]
gru_weight = meta_model.coef_[1]
intercept = meta_model.intercept_

print(f"\n✓ 学习到的动态权重:")
print(f"  LSTM权重: {lstm_weight:.4f}")
print(f"  GRU权重:  {gru_weight:.4f}")
print(f"  截距项:   {intercept:.4f}")

dynamic_weight_pred = meta_model.predict(meta_features_test)

dynamic_weight_r2 = r2_score(y_test_seq, dynamic_weight_pred)
print(f"\n✓ 动态权重残差融合测试集R²: {dynamic_weight_r2:.4f}")
print(f"  改进: {dynamic_weight_r2 - avg_r2:.4f} vs 简单平均")

# ========== 【新增】策略7：LSTM+GRU Ridge残差学习（参考实验8）==========
print("\n" + "=" * 100)
print("【策略7】LSTM Ridge残差学习：LSTM基础 + Ridge残差修正【简化特征, alpha=10】".center(100))
print("=" * 100)

print("\n使用Ridge回归学习LSTM残差（alpha=10）...")
ridge_lstm_model = train_ridge_model(simplified_train_features, lstm_oof_residual, alpha=10.0)
ridge_lstm_residual_pred = ridge_lstm_model.predict(simplified_test_features)
ridge_lstm_strategy_pred = lstm_test_pred + ridge_lstm_residual_pred

ridge_lstm_r2 = r2_score(y_test_seq, ridge_lstm_strategy_pred)
print(f"✓ LSTM Ridge残差学习测试集R²: {ridge_lstm_r2:.4f}")
print(f"  改进: {ridge_lstm_r2 - avg_r2:.4f} vs 简单平均")
print(f"  vs XGBoost残差: {ridge_lstm_r2 - lstm_residual_r2:.4f}")

# ========== 【新增】策略8：GRU Ridge残差学习（参考实验8）==========
print("\n" + "=" * 100)
print("【策略8】GRU Ridge残差学习：GRU基础 + Ridge残差修正【简化特征, alpha=10】".center(100))
print("=" * 100)

print("\n使用Ridge回归学习GRU残差（alpha=10）...")
ridge_gru_model = train_ridge_model(simplified_train_features, gru_oof_residual, alpha=10.0)
ridge_gru_residual_pred = ridge_gru_model.predict(simplified_test_features)
ridge_gru_strategy_pred = gru_test_pred + ridge_gru_residual_pred

ridge_gru_r2 = r2_score(y_test_seq, ridge_gru_strategy_pred)
print(f"✓ GRU Ridge残差学习测试集R²: {ridge_gru_r2:.4f}")
print(f"  改进: {ridge_gru_r2 - avg_r2:.4f} vs 简单平均")
print(f"  vs XGBoost残差: {ridge_gru_r2 - gru_residual_r2:.4f}")

# ========== 【新增】策略9：双残差Ridge学习（参考实验8）==========
print("\n" + "=" * 100)
print("【策略9】双残差Ridge学习：(LSTM+GRU)/2基础 + Ridge残差【简化特征, alpha=10】".center(100))
print("=" * 100)

print("\n使用Ridge回归学习平均残差（alpha=10）...")
ridge_dual_model = train_ridge_model(simplified_train_features, avg_oof_residual, alpha=10.0)
ridge_dual_residual_pred = ridge_dual_model.predict(simplified_test_features)
ridge_dual_strategy_pred = avg_test_pred_for_residual + ridge_dual_residual_pred

ridge_dual_r2 = r2_score(y_test_seq, ridge_dual_strategy_pred)
print(f"✓ 双残差Ridge学习测试集R²: {ridge_dual_r2:.4f}")
print(f"  改进: {ridge_dual_r2 - avg_r2:.4f} vs 简单平均")
print(f"  vs XGBoost残差: {ridge_dual_r2 - dual_r2:.4f}")

# ========== 【新增】策略10：Ridge Stacking（最简单的融合）==========
print("\n" + "=" * 100)
print("【策略10】Ridge Stacking：Ridge直接融合LSTM和GRU预测【alpha=1.0】".center(100))
print("=" * 100)

# 最简单的元特征：只用LSTM和GRU的预测
ridge_stacking_train = np.column_stack([lstm_oof_preds, gru_oof_preds])
ridge_stacking_test = np.column_stack([lstm_test_pred, gru_test_pred])

print(f"Ridge Stacking元特征维度: {ridge_stacking_train.shape[1]} 维")

ridge_stacking_model = Ridge(alpha=1.0)
ridge_stacking_model.fit(ridge_stacking_train, y_train_seq)

ridge_stacking_pred = ridge_stacking_model.predict(ridge_stacking_test)
ridge_stacking_r2 = r2_score(y_test_seq, ridge_stacking_pred)

lstm_stacking_weight = ridge_stacking_model.coef_[0]
gru_stacking_weight = ridge_stacking_model.coef_[1]
stacking_intercept = ridge_stacking_model.intercept_

print(f"\n✓ 学习到的Stacking权重:")
print(f"  LSTM权重: {lstm_stacking_weight:.4f}")
print(f"  GRU权重:  {gru_stacking_weight:.4f}")
print(f"  截距项:   {stacking_intercept:.4f}")
print(f"  权重和:   {lstm_stacking_weight + gru_stacking_weight:.4f}")

print(f"\n✓ Ridge Stacking测试集R²: {ridge_stacking_r2:.4f}")
print(f"  改进: {ridge_stacking_r2 - avg_r2:.4f} vs 简单平均")

# ========== 性能对比 ==========
print("\n" + "=" * 100)
print("所有策略性能对比（归一化数据）".center(100))
print("=" * 100)

strategies = {
    'LSTM单模型': lstm_test_pred,
    'GRU单模型': gru_test_pred,
    '策略1-简单平均': avg_test_pred,
    '策略2-LSTM残差(XGB)': lstm_residual_strategy_pred,
    '策略3-GRU残差(XGB)': gru_residual_strategy_pred,
    '策略4-加权平均Stacking': weighted_stacking_pred,
    '策略5-双残差(XGB)': dual_strategy_pred,
    '策略6-动态权重融合': dynamic_weight_pred,
    '策略7-LSTM残差(Ridge)': ridge_lstm_strategy_pred,
    '策略8-GRU残差(Ridge)': ridge_gru_strategy_pred,
    '策略9-双残差(Ridge)': ridge_dual_strategy_pred,
    '策略10-Ridge Stacking': ridge_stacking_pred,
}

print("\n{:<30} {:<12} {:<12} {:<12} {:<12}".format("策略", "R²", "MAE", "RMSE", "MAPE"))
print("-" * 80)

for name, pred in strategies.items():
    r2 = r2_score(y_test_seq, pred)
    mae = mean_absolute_error(y_test_seq, pred)
    rmse = sqrt(mean_squared_error(y_test_seq, pred))
    mape = np.mean(np.abs((pred - y_test_seq) / (y_test_seq + 1e-8)))
    print(f"{name:<30} {r2:<12.4f} {mae:<12.6f} {rmse:<12.6f} {mape:<12.6f}")

# ========== 反归一化并评估原始尺度 ==========
print("\n" + "=" * 100)
print("所有策略性能对比（原始尺度）".center(100))
print("=" * 100)

strategies_original = {}
y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1))

for name, pred in strategies.items():
    strategies_original[name] = y_scaler.inverse_transform(pred.reshape(-1, 1))

print("\n{:<30} {:<12} {:<12} {:<12} {:<12}".format("策略", "R²", "MAE", "RMSE", "MAPE"))
print("-" * 80)

best_r2 = -np.inf
best_strategy = None

for name, pred in strategies_original.items():
    r2 = r2_score(y_test_original, pred)
    mae = mean_absolute_error(y_test_original, pred)
    rmse = sqrt(mean_squared_error(y_test_original, pred))
    mape = np.mean(np.abs((pred - y_test_original) / (y_test_original + 1e-8)))
    print(f"{name:<30} {r2:<12.4f} {mae:<12.4f} {rmse:<12.4f} {mape:<12.6f}")

    if r2 > best_r2:
        best_r2 = r2
        best_strategy = name

print(f"\n🏆 最佳策略: {best_strategy} (R² = {best_r2:.4f})")

# ========== XGBoost vs Ridge 残差学习对比分析 ==========
print("\n" + "=" * 100)
print("XGBoost vs Ridge 残差学习效果对比".center(100))
print("=" * 100)

comparison_data = {
    'LSTM残差': {
        'XGBoost': lstm_residual_r2,
        'Ridge': ridge_lstm_r2,
        'diff': ridge_lstm_r2 - lstm_residual_r2
    },
    'GRU残差': {
        'XGBoost': gru_residual_r2,
        'Ridge': ridge_gru_r2,
        'diff': ridge_gru_r2 - gru_residual_r2
    },
    '双残差': {
        'XGBoost': dual_r2,
        'Ridge': ridge_dual_r2,
        'diff': ridge_dual_r2 - dual_r2
    }
}

print("\n{:<15} {:<15} {:<15} {:<15}".format("残差类型", "XGBoost R²", "Ridge R²", "差异"))
print("-" * 60)
for residual_type, scores in comparison_data.items():
    diff_sign = "+" if scores['diff'] > 0 else ""
    print(f"{residual_type:<15} {scores['XGBoost']:<15.4f} {scores['Ridge']:<15.4f} {diff_sign}{scores['diff']:<15.4f}")

avg_diff = np.mean([v['diff'] for v in comparison_data.values()])
print(f"\n平均差异: {avg_diff:+.4f}")
if avg_diff > 0:
    print("✓ Ridge回归在残差学习上平均表现更好（更稳定）")
else:
    print("✓ XGBoost在残差学习上平均表现更好（更灵活）")

# ========== 可视化 ==========
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 1. 所有策略预测对比（6x2布局）
fig = plt.figure(figsize=(20, 24))

plot_data = [
    ('LSTM单模型', strategies_original['LSTM单模型'], 'blue'),
    ('GRU单模型', strategies_original['GRU单模型'], 'green'),
    ('策略1-简单平均', strategies_original['策略1-简单平均'], 'purple'),
    ('策略2-LSTM残差(XGB)', strategies_original['策略2-LSTM残差(XGB)'], 'orange'),
    ('策略3-GRU残差(XGB)', strategies_original['策略3-GRU残差(XGB)'], 'cyan'),
    ('策略4-加权平均Stacking', strategies_original['策略4-加权平均Stacking'], 'red'),
    ('策略5-双残差(XGB)', strategies_original['策略5-双残差(XGB)'], 'brown'),
    ('策略6-动态权重融合', strategies_original['策略6-动态权重融合'], 'magenta'),
    ('策略7-LSTM残差(Ridge)', strategies_original['策略7-LSTM残差(Ridge)'], 'coral'),
    ('策略8-GRU残差(Ridge)', strategies_original['策略8-GRU残差(Ridge)'], 'teal'),
    ('策略9-双残差(Ridge)', strategies_original['策略9-双残差(Ridge)'], 'gold'),
    ('策略10-Ridge Stacking', strategies_original['策略10-Ridge Stacking'], 'lime'),
]

for idx, (name, pred, color) in enumerate(plot_data, 1):
    plt.subplot(6, 2, idx)
    plt.plot(y_test_original, label="真实值", linewidth=2.5, color='black', alpha=0.7)
    plt.plot(pred, label=name, linewidth=2, alpha=0.8, color=color)
    r2 = r2_score(y_test_original, pred)
    plt.title(f"{name} (R²={r2:.4f})", fontsize=12, fontweight='bold')
    plt.xlabel('样本序号', fontsize=9)
    plt.ylabel('玉米价格', fontsize=9)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'all_strategies_with_ridge.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 2. XGBoost vs Ridge 残差学习对比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 2.1 LSTM残差对比
axes[0, 0].plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
axes[0, 0].plot(strategies_original['策略2-LSTM残差(XGB)'],
                label='XGBoost残差', linewidth=2, alpha=0.8, color='orange')
axes[0, 0].plot(strategies_original['策略7-LSTM残差(Ridge)'],
                label='Ridge残差', linewidth=2, alpha=0.8, color='coral')
r2_xgb = r2_score(y_test_original, strategies_original['策略2-LSTM残差(XGB)'])
r2_ridge = r2_score(y_test_original, strategies_original['策略7-LSTM残差(Ridge)'])
axes[0, 0].set_title(f'LSTM残差学习对比\nXGB: {r2_xgb:.4f} | Ridge: {r2_ridge:.4f}',
                     fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('样本序号')
axes[0, 0].set_ylabel('玉米价格')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 2.2 GRU残差对比
axes[0, 1].plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
axes[0, 1].plot(strategies_original['策略3-GRU残差(XGB)'],
                label='XGBoost残差', linewidth=2, alpha=0.8, color='cyan')
axes[0, 1].plot(strategies_original['策略8-GRU残差(Ridge)'],
                label='Ridge残差', linewidth=2, alpha=0.8, color='teal')
r2_xgb = r2_score(y_test_original, strategies_original['策略3-GRU残差(XGB)'])
r2_ridge = r2_score(y_test_original, strategies_original['策略8-GRU残差(Ridge)'])
axes[0, 1].set_title(f'GRU残差学习对比\nXGB: {r2_xgb:.4f} | Ridge: {r2_ridge:.4f}',
                     fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('样本序号')
axes[0, 1].set_ylabel('玉米价格')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# 2.3 双残差对比
axes[0, 2].plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
axes[0, 2].plot(strategies_original['策略5-双残差(XGB)'],
                label='XGBoost残差', linewidth=2, alpha=0.8, color='brown')
axes[0, 2].plot(strategies_original['策略9-双残差(Ridge)'],
                label='Ridge残差', linewidth=2, alpha=0.8, color='gold')
r2_xgb = r2_score(y_test_original, strategies_original['策略5-双残差(XGB)'])
r2_ridge = r2_score(y_test_original, strategies_original['策略9-双残差(Ridge)'])
axes[0, 2].set_title(f'双残差学习对比\nXGB: {r2_xgb:.4f} | Ridge: {r2_ridge:.4f}',
                     fontsize=11, fontweight='bold')
axes[0, 2].set_xlabel('样本序号')
axes[0, 2].set_ylabel('玉米价格')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# 2.4 残差分布对比 - LSTM
residual_xgb_lstm = y_test_seq - lstm_residual_strategy_pred
residual_ridge_lstm = y_test_seq - ridge_lstm_strategy_pred
axes[1, 0].hist(residual_xgb_lstm, bins=30, color='orange', alpha=0.6,
                edgecolor='black', label='XGBoost')
axes[1, 0].hist(residual_ridge_lstm, bins=30, color='coral', alpha=0.6,
                edgecolor='black', label='Ridge')
axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=2)
axes[1, 0].set_title(f'LSTM残差分布\nXGB std={np.std(residual_xgb_lstm):.5f} | Ridge std={np.std(residual_ridge_lstm):.5f}',
                     fontsize=10, fontweight='bold')
axes[1, 0].set_xlabel('残差')
axes[1, 0].set_ylabel('频数')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 2.5 残差分布对比 - GRU
residual_xgb_gru = y_test_seq - gru_residual_strategy_pred
residual_ridge_gru = y_test_seq - ridge_gru_strategy_pred
axes[1, 1].hist(residual_xgb_gru, bins=30, color='cyan', alpha=0.6,
                edgecolor='black', label='XGBoost')
axes[1, 1].hist(residual_ridge_gru, bins=30, color='teal', alpha=0.6,
                edgecolor='black', label='Ridge')
axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_title(f'GRU残差分布\nXGB std={np.std(residual_xgb_gru):.5f} | Ridge std={np.std(residual_ridge_gru):.5f}',
                     fontsize=10, fontweight='bold')
axes[1, 1].set_xlabel('残差')
axes[1, 1].set_ylabel('频数')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 2.6 残差标准差对比
residual_std_comparison = {
    'LSTM-XGB': np.std(residual_xgb_lstm),
    'LSTM-Ridge': np.std(residual_ridge_lstm),
    'GRU-XGB': np.std(residual_xgb_gru),
    'GRU-Ridge': np.std(residual_ridge_gru),
}
colors_std = ['orange', 'coral', 'cyan', 'teal']
axes[1, 2].bar(range(len(residual_std_comparison)),
               list(residual_std_comparison.values()),
               color=colors_std, alpha=0.7)
axes[1, 2].set_title('残差标准差对比（越小越好）', fontsize=11, fontweight='bold')
axes[1, 2].set_ylabel('标准差')
axes[1, 2].set_xticks(range(len(residual_std_comparison)))
axes[1, 2].set_xticklabels(residual_std_comparison.keys(), rotation=15, ha='right')
axes[1, 2].grid(True, alpha=0.3, axis='y')

for i, (name, std) in enumerate(residual_std_comparison.items()):
    axes[1, 2].text(i, std + 0.001, f'{std:.5f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig(results_directory + 'xgboost_vs_ridge_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 3. 性能排名图
fig, ax = plt.subplots(figsize=(14, 9))

improvement_data = {}
for name, pred in strategies_original.items():
    r2 = r2_score(y_test_original, pred)
    improvement_data[name] = r2

sorted_strategies = sorted(improvement_data.items(), key=lambda x: x[1], reverse=True)
names = [name for name, _ in sorted_strategies]
values = [value for _, value in sorted_strategies]

colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
bars = ax.barh(range(len(names)), values, color=colors_bar, alpha=0.8)

# 添加基线
baseline_r2 = improvement_data['策略1-简单平均']
ax.axvline(x=baseline_r2, color='red', linestyle='--', linewidth=2, label='简单平均基线')

# 添加数值标签
for i, (bar, value) in enumerate(zip(bars, values)):
    improvement = value - baseline_r2
    label = f'{value:.4f}'
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

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('所有策略R²性能排名（含XGBoost + Ridge残差学习）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(results_directory + 'performance_ranking_with_ridge.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 4. R²对比柱状图（分组）
fig, ax = plt.subplots(figsize=(16, 8))

categories = ['LSTM残差', 'GRU残差', '双残差']
xgb_scores = [
    r2_score(y_test_original, strategies_original['策略2-LSTM残差(XGB)']),
    r2_score(y_test_original, strategies_original['策略3-GRU残差(XGB)']),
    r2_score(y_test_original, strategies_original['策略5-双残差(XGB)'])
]
ridge_scores = [
    r2_score(y_test_original, strategies_original['策略7-LSTM残差(Ridge)']),
    r2_score(y_test_original, strategies_original['策略8-GRU残差(Ridge)']),
    r2_score(y_test_original, strategies_original['策略9-双残差(Ridge)'])
]

x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x_pos - width / 2, xgb_scores, width, label='保守XGBoost', color='steelblue', alpha=0.8)
bars2 = ax.bar(x_pos + width / 2, ridge_scores, width, label='Ridge(alpha=10)', color='coral', alpha=0.8)

# 添加基线
ax.axhline(y=baseline_r2, color='red', linestyle='--', linewidth=2, label='简单平均基线', alpha=0.7)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('XGBoost vs Ridge 残差学习效果对比', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_directory + 'xgb_vs_ridge_bar_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ========== 保存模型 ==========
import pickle

print("\n" + "=" * 100)
print("保存所有模型和结果".center(100))
print("=" * 100)

# 保存Keras模型
lstm_final.save(results_directory + 'lstm_final_model.h5')
gru_final.save(results_directory + 'gru_final_model.h5')

# 保存XGBoost模型（保守参数版本）
with open(results_directory + 'xgb_lstm_residual_conservative.pkl', 'wb') as f:
    pickle.dump(xgb_lstm_residual_model, f)

with open(results_directory + 'xgb_gru_residual_conservative.pkl', 'wb') as f:
    pickle.dump(xgb_gru_residual_model, f)

with open(results_directory + 'xgb_dual_conservative.pkl', 'wb') as f:
    pickle.dump(xgb_dual_model, f)

with open(results_directory + 'xgb_lstm_dynamic_conservative.pkl', 'wb') as f:
    pickle.dump(xgb_lstm_dynamic, f)

with open(results_directory + 'xgb_gru_dynamic_conservative.pkl', 'wb') as f:
    pickle.dump(xgb_gru_dynamic, f)

# 保存Ridge回归模型
with open(results_directory + 'ridge_dynamic_meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)

with open(results_directory + 'ridge_lstm_residual_model.pkl', 'wb') as f:
    pickle.dump(ridge_lstm_model, f)

with open(results_directory + 'ridge_gru_residual_model.pkl', 'wb') as f:
    pickle.dump(ridge_gru_model, f)

with open(results_directory + 'ridge_dual_residual_model.pkl', 'wb') as f:
    pickle.dump(ridge_dual_model, f)

with open(results_directory + 'ridge_stacking_model.pkl', 'wb') as f:
    pickle.dump(ridge_stacking_model, f)

# 保存归一化器
with open(results_directory + 'scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

# 保存加权Stacking权重信息
weighted_stacking_info = pd.DataFrame({
    'model': ['LSTM', 'GRU'],
    'oof_r2': [lstm_oof_r2, gru_oof_r2],
    'weight': [lstm_weight_r2, gru_weight_r2]
})
weighted_stacking_info.to_csv(results_directory + 'weighted_stacking_info.csv', index=False)

# 保存预测结果
results_df = pd.DataFrame({
    'true_value': y_test_original.flatten(),
    'lstm': strategies_original['LSTM单模型'].flatten(),
    'gru': strategies_original['GRU单模型'].flatten(),
    'simple_avg': strategies_original['策略1-简单平均'].flatten(),
    'lstm_residual_xgb': strategies_original['策略2-LSTM残差(XGB)'].flatten(),
    'gru_residual_xgb': strategies_original['策略3-GRU残差(XGB)'].flatten(),
    'stacking_xgb': strategies_original['策略4-加权平均Stacking'].flatten(),
    'dual_residual_xgb': strategies_original['策略5-双残差(XGB)'].flatten(),
    'dynamic_weight': strategies_original['策略6-动态权重融合'].flatten(),
    'lstm_residual_ridge': strategies_original['策略7-LSTM残差(Ridge)'].flatten(),
    'gru_residual_ridge': strategies_original['策略8-GRU残差(Ridge)'].flatten(),
    'dual_residual_ridge': strategies_original['策略9-双残差(Ridge)'].flatten(),
})
results_df.to_csv(results_directory + 'all_predictions_xgb_ridge.csv', index=False)

# 保存性能指标
metrics_data = {}
for name, pred in strategies_original.items():
    metrics_data[name] = {
        'R²': r2_score(y_test_original, pred),
        'MAE': mean_absolute_error(y_test_original, pred),
        'RMSE': sqrt(mean_squared_error(y_test_original, pred)),
        'MAPE': np.mean(np.abs((pred - y_test_original) / (y_test_original + 1e-8)))
    }

performance_df = pd.DataFrame([
    {'strategy': name, **metrics}
    for name, metrics in metrics_data.items()
])
performance_df = performance_df.sort_values('R²', ascending=False)
performance_df.to_csv(results_directory + 'performance_metrics_xgb_ridge.csv', index=False)

print("\n✓ 模型保存完成！")
print(f"  - LSTM/GRU最终模型")
print(f"  - 保守XGBoost模型 x 5个")
print(f"  - Ridge回归模型 x 4个 (alpha=10)")
print(f"  - 归一化器")
print(f"  - 所有预测结果")
print(f"  - 性能指标")

# ========== 最终总结 ==========
print("\n" + "=" * 100)
print("🎉 增强版融合模型训练完成（含Ridge残差学习）！".center(100))
print("=" * 100)

print(f"\n📊 实现的融合策略（共12个）:")
print(f"  基础模型:")
print(f"    - LSTM单模型")
print(f"    - GRU单模型")
print(f"  简单融合:")
print(f"    1. 简单平均：(LSTM + GRU) / 2")
print(f"  XGBoost残差学习（保守参数 + 增强特征）:")
print(f"    2. LSTM残差学习")
print(f"    3. GRU残差学习")
print(f"    4. 加权平均Stacking（基于OOF表现）")
print(f"    5. 双残差学习")
print(f"    6. 动态权重融合")
print(f"  Ridge残差学习（alpha=10 + 简化特征）⭐新增:")
print(f"    7. LSTM Ridge残差学习")
print(f"    8. GRU Ridge残差学习")
print(f"    9. 双残差Ridge学习")
print(f"  Ridge Stacking（alpha=1.0）⭐新增:")
print(f"    10. Ridge直接融合LSTM和GRU")

print(f"\n🏆 最佳策略: {best_strategy}")
print(f"   测试集R²: {best_r2:.4f}")

# 性能排名
print(f"\n📈 所有策略性能排名（按R²降序）:")
sorted_r2_scores = sorted(
    [(name, r2_score(y_test_original, pred)) for name, pred in strategies_original.items()],
    key=lambda x: x[1], reverse=True
)

for rank, (name, r2) in enumerate(sorted_r2_scores, 1):
    improvement = r2 - baseline_r2
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"   {marker} {rank:>2}. {name:<30} R² = {r2:.4f}  (vs简单平均: {improvement:+.4f})")

print(f"\n💡 XGBoost vs Ridge 残差学习对比:")
print(f"  【LSTM残差】")
print(f"    XGBoost: {lstm_residual_r2:.4f}")
print(f"    Ridge:   {ridge_lstm_r2:.4f}")
print(f"    差异:    {ridge_lstm_r2 - lstm_residual_r2:+.4f}")
print(f"  【GRU残差】")
print(f"    XGBoost: {gru_residual_r2:.4f}")
print(f"    Ridge:   {ridge_gru_r2:.4f}")
print(f"    差异:    {ridge_gru_r2 - gru_residual_r2:+.4f}")
print(f"  【双残差】")
print(f"    XGBoost: {dual_r2:.4f}")
print(f"    Ridge:   {ridge_dual_r2:.4f}")
print(f"    差异:    {ridge_dual_r2 - dual_r2:+.4f}")

print(f"\n🔍 关键发现:")
if avg_diff > 0:
    print(f"  ✓ Ridge回归在残差学习上平均表现更好 (平均差异: {avg_diff:+.4f})")
    print(f"  ✓ Ridge优势：更稳定、更保守、防止过拟合")
    print(f"  ✓ 适用场景：样本量较小、基础模型容易过拟合")
else:
    print(f"  ✓ XGBoost在残差学习上平均表现更好 (平均差异: {avg_diff:+.4f})")
    print(f"  ✓ XGBoost优势：更灵活、捕捉非线性残差模式")
    print(f"  ✓ 适用场景：样本量充足、残差存在复杂非线性关系")

print(f"\n⭐ 技术参数:")
print(f"  【保守XGBoost】")
print(f"    n_estimators: 100, learning_rate: 0.01, max_depth: 3")
print(f"    min_child_weight: 5, subsample: 0.7, colsample_bytree: 0.7")
print(f"    reg_alpha: 0.1 (L1), reg_lambda: 1.0 (L2)")
print(f"  【Ridge回归】")
print(f"    alpha: 10.0 (强正则化)")
print(f"    特征: 简化特征（74维）")

print(f"\n💾 所有结果保存在: {results_directory}")
print("=" * 100)