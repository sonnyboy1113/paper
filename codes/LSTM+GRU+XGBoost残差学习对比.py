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

print("=" * 120)
print("对称残差学习实验 - LSTM vs GRU 全面对比".center(120))
print("核心改进：为LSTM和GRU分别进行9种策略实验，自动选择最优基础模型".center(120))
print("=" * 120)

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


print("\n" + "=" * 120)
print("第一步：生成LSTM和GRU的OOF预测".center(120))
print("=" * 120)

lstm_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'lstm', n_splits=5)
gru_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'gru', n_splits=5)

print(f"\nOOF预测生成完成！")
print(f"LSTM OOF R²: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"GRU OOF R²: {r2_score(y_train_seq, gru_oof_preds):.4f}")

# 训练最终模型
print("\n" + "=" * 120)
print("第二步：训练最终的LSTM和GRU模型".center(120))
print("=" * 120)

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

# 诊断过拟合
lstm_train_pred = lstm_final.predict(X_train_seq, verbose=0).flatten()
gru_train_pred = gru_final.predict(X_train_seq, verbose=0).flatten()

lstm_train_r2 = r2_score(y_train_seq, lstm_train_pred)
lstm_test_r2 = r2_score(y_test_seq, lstm_test_pred)
gru_train_r2 = r2_score(y_train_seq, gru_train_pred)
gru_test_r2 = r2_score(y_test_seq, gru_test_pred)

print(f"\n过拟合诊断:")
print(f"LSTM - 训练R²: {lstm_train_r2:.4f}, 测试R²: {lstm_test_r2:.4f}, 差距: {lstm_train_r2 - lstm_test_r2:.4f}")
print(f"GRU  - 训练R²: {gru_train_r2:.4f}, 测试R²: {gru_test_r2:.4f}, 差距: {gru_train_r2 - gru_test_r2:.4f}")

# 计算残差
lstm_oof_residual = y_train_seq - lstm_oof_preds
gru_oof_residual = y_train_seq - gru_oof_preds

print(f"\n残差统计:")
print(f"LSTM残差 - 均值: {np.mean(lstm_oof_residual):.6f}, 标准差: {np.std(lstm_oof_residual):.6f}")
print(f"GRU残差  - 均值: {np.mean(gru_oof_residual):.6f}, 标准差: {np.std(gru_oof_residual):.6f}")


# ========== 特征工程函数 ==========

def create_original_features(X_flat, pred1, pred2):
    """原始增强特征（80维）"""
    features_list = [X_flat]
    features_list.append(pred1.reshape(-1, 1))
    features_list.append(pred2.reshape(-1, 1))
    features_list.append((pred1 + pred2).reshape(-1, 1))
    features_list.append((pred1 - pred2).reshape(-1, 1))
    features_list.append(np.abs(pred1 - pred2).reshape(-1, 1))
    features_list.append((pred1 * pred2).reshape(-1, 1))
    features_list.append(np.maximum(pred1, pred2).reshape(-1, 1))
    features_list.append(np.minimum(pred1, pred2).reshape(-1, 1))
    disagreement = np.abs(pred1 - pred2)
    confidence = 1 / (1 + disagreement)
    features_list.append(confidence.reshape(-1, 1))
    weighted_avg = 0.5 * pred1 + 0.5 * pred2
    features_list.append(weighted_avg.reshape(-1, 1))
    return np.hstack(features_list)


def create_simplified_features(X_flat, pred1, pred2):
    """简化特征（74维）"""
    features_list = [X_flat]
    features_list.append(pred1.reshape(-1, 1))
    features_list.append(pred2.reshape(-1, 1))
    features_list.append(((pred1 + pred2) / 2).reshape(-1, 1))
    features_list.append(np.abs(pred1 - pred2).reshape(-1, 1))
    return np.hstack(features_list)


def create_minimal_features(X_flat, pred1, pred2):
    """最小化特征（72维）"""
    features_list = [X_flat]
    features_list.append(pred1.reshape(-1, 1))
    features_list.append(pred2.reshape(-1, 1))
    return np.hstack(features_list)


# ========== 残差学习训练函数 ==========

def train_original_xgboost(X_train, y_train):
    """原始XGBoost参数"""
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


def train_xgboost_with_early_stopping(X_train, y_train):
    """带早停的XGBoost"""
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
        early_stopping_rounds=20
    )

    try:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
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
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    return model


def train_ridge_model(X_train, y_train, alpha=10.0):
    """Ridge线性回归"""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# ========== 残差处理函数 ==========

def clip_residual(residual_pred, threshold=2.0):
    """剪裁极端残差值"""
    std = np.std(residual_pred)
    mean = np.mean(residual_pred)
    return np.clip(residual_pred, mean - threshold * std, mean + threshold * std)


def weighted_residual_correction(base_pred, residual_pred, weight=0.5):
    """加权残差修正"""
    return base_pred + weight * residual_pred


# ========== 【核心改进】对称实验框架 ==========

def run_symmetric_experiments(base_model_name, base_oof_preds, base_test_pred,
                              base_residual, other_oof_preds, other_test_pred,
                              X_train_flat, X_test_flat, y_test_seq):
    """
    为单个基础模型运行所有9个实验

    参数:
        base_model_name: 基础模型名称 ('LSTM' or 'GRU')
        base_oof_preds: 基础模型的OOF预测
        base_test_pred: 基础模型的测试集预测
        base_residual: 基础模型的残差
        other_oof_preds: 另一个模型的OOF预测（用于特征）
        other_test_pred: 另一个模型的测试集预测（用于特征）
        X_train_flat, X_test_flat: 扁平化特征
        y_test_seq: 测试集真实值
    """

    print(f"\n{'=' * 120}")
    print(f"运行{base_model_name}的9个对称实验".center(120))
    print(f"{'=' * 120}")

    results = {}
    basic_features_train = X_train_flat[:len(base_oof_preds)]
    basic_features_test = X_test_flat

    # 准备特征
    original_train = create_original_features(basic_features_train, base_oof_preds, other_oof_preds)
    original_test = create_original_features(basic_features_test, base_test_pred, other_test_pred)

    simplified_train = create_simplified_features(basic_features_train, base_oof_preds, other_oof_preds)
    simplified_test = create_simplified_features(basic_features_test, base_test_pred, other_test_pred)

    minimal_train = create_minimal_features(basic_features_train, base_oof_preds, other_oof_preds)
    minimal_test = create_minimal_features(basic_features_test, base_test_pred, other_test_pred)

    # 实验1：原始方法
    print(f"\n{base_model_name}-实验1：原始特征 + 激进XGBoost")
    model1 = train_original_xgboost(original_train, base_residual)
    residual1 = model1.predict(original_test)
    pred1 = base_test_pred + residual1
    r2_1 = r2_score(y_test_seq, pred1)
    print(f"  R²: {r2_1:.4f}")
    results[f'{base_model_name}-实验1-原始方法'] = {'pred': pred1, 'r2': r2_1}

    # 实验2：简化特征 + 激进XGBoost
    print(f"\n{base_model_name}-实验2：简化特征 + 激进XGBoost")
    model2 = train_original_xgboost(simplified_train, base_residual)
    residual2 = model2.predict(simplified_test)
    pred2 = base_test_pred + residual2
    r2_2 = r2_score(y_test_seq, pred2)
    print(f"  R²: {r2_2:.4f}")
    results[f'{base_model_name}-实验2-简化特征'] = {'pred': pred2, 'r2': r2_2}

    # 实验3：简化特征 + 保守XGBoost
    print(f"\n{base_model_name}-实验3：简化特征 + 保守XGBoost")
    model3 = train_conservative_xgboost(simplified_train, base_residual)
    residual3 = model3.predict(simplified_test)
    pred3 = base_test_pred + residual3
    r2_3 = r2_score(y_test_seq, pred3)
    print(f"  R²: {r2_3:.4f}")
    results[f'{base_model_name}-实验3-保守参数'] = {'pred': pred3, 'r2': r2_3}

    # 实验4：简化特征 + 早停XGBoost
    print(f"\n{base_model_name}-实验4：简化特征 + 早停XGBoost")
    model4 = train_xgboost_with_early_stopping(simplified_train, base_residual)
    residual4 = model4.predict(simplified_test)
    pred4 = base_test_pred + residual4
    r2_4 = r2_score(y_test_seq, pred4)
    print(f"  R²: {r2_4:.4f}")
    results[f'{base_model_name}-实验4-早停机制'] = {'pred': pred4, 'r2': r2_4}

    # 实验5：简化特征 + 保守XGBoost + 残差剪裁
    print(f"\n{base_model_name}-实验5：简化特征 + 保守XGBoost + 残差剪裁")
    model5 = train_conservative_xgboost(simplified_train, base_residual)
    residual5 = model5.predict(simplified_test)
    residual5_clipped = clip_residual(residual5, threshold=2.0)
    pred5 = base_test_pred + residual5_clipped
    r2_5 = r2_score(y_test_seq, pred5)
    print(f"  R²: {r2_5:.4f}")
    results[f'{base_model_name}-实验5-残差剪裁'] = {'pred': pred5, 'r2': r2_5}

    # 实验6：简化特征 + 保守XGBoost + 加权融合(50%)
    print(f"\n{base_model_name}-实验6：简化特征 + 保守XGBoost + 加权融合(50%)")
    model6 = train_conservative_xgboost(simplified_train, base_residual)
    residual6 = model6.predict(simplified_test)
    pred6 = weighted_residual_correction(base_test_pred, residual6, weight=0.5)
    r2_6 = r2_score(y_test_seq, pred6)
    print(f"  R²: {r2_6:.4f}")
    results[f'{base_model_name}-实验6-加权融合50%'] = {'pred': pred6, 'r2': r2_6}

    # 实验7：最小化特征 + 保守XGBoost
    print(f"\n{base_model_name}-实验7：最小化特征 + 保守XGBoost")
    model7 = train_conservative_xgboost(minimal_train, base_residual)
    residual7 = model7.predict(minimal_test)
    pred7 = base_test_pred + residual7
    r2_7 = r2_score(y_test_seq, pred7)
    print(f"  R²: {r2_7:.4f}")
    results[f'{base_model_name}-实验7-最小特征'] = {'pred': pred7, 'r2': r2_7}

    # 实验8：简化特征 + Ridge回归
    print(f"\n{base_model_name}-实验8：简化特征 + Ridge回归")
    model8 = train_ridge_model(simplified_train, base_residual, alpha=10.0)
    residual8 = model8.predict(simplified_test)
    pred8 = base_test_pred + residual8
    r2_8 = r2_score(y_test_seq, pred8)
    print(f"  R²: {r2_8:.4f}")
    results[f'{base_model_name}-实验8-Ridge回归'] = {'pred': pred8, 'r2': r2_8}

    # 实验9：终极组合（简化特征 + 保守XGBoost + 残差剪裁 + 加权融合30%）
    print(f"\n{base_model_name}-实验9：终极组合")
    model9 = train_conservative_xgboost(simplified_train, base_residual)
    residual9 = model9.predict(simplified_test)
    residual9_clipped = clip_residual(residual9, threshold=2.0)
    pred9 = weighted_residual_correction(base_test_pred, residual9_clipped, weight=0.3)
    r2_9 = r2_score(y_test_seq, pred9)
    print(f"  R²: {r2_9:.4f}")
    results[f'{base_model_name}-实验9-终极组合'] = {'pred': pred9, 'r2': r2_9}

    return results


# ========== 运行对称实验 ==========

print("\n" + "=" * 120)
print("第三步：运行对称残差学习实验".center(120))
print("=" * 120)

# 基线
avg_test_pred = (lstm_test_pred + gru_test_pred) / 2
avg_r2 = r2_score(y_test_seq, avg_test_pred)
print(f"\n基线（简单平均）R²: {avg_r2:.4f}")

# 运行LSTM的9个实验
lstm_results = run_symmetric_experiments(
    'LSTM', lstm_oof_preds, lstm_test_pred, lstm_oof_residual,
    gru_oof_preds, gru_test_pred,
    X_train_flat, X_test_flat, y_test_seq
)

# 运行GRU的9个实验
gru_results = run_symmetric_experiments(
    'GRU', gru_oof_preds, gru_test_pred, gru_oof_residual,
    lstm_oof_preds, lstm_test_pred,
    X_train_flat, X_test_flat, y_test_seq
)

# 合并所有结果
all_results = {
    'LSTM单模型': {'pred': lstm_test_pred, 'r2': lstm_test_r2},
    'GRU单模型': {'pred': gru_test_pred, 'r2': gru_test_r2},
    '简单平均': {'pred': avg_test_pred, 'r2': avg_r2},
    **lstm_results,
    **gru_results
}

# ========== 结果分析 ==========

print("\n" + "=" * 120)
print("对称实验结果汇总".center(120))
print("=" * 120)

# 按R²排序
sorted_results = sorted(all_results.items(), key=lambda x: x[1]['r2'], reverse=True)

print(f"\n{'排名':<5} {'策略':<50} {'R²':>10} {'vs基线':>10} {'vs单模型':>12}")
print("-" * 120)

for rank, (name, data) in enumerate(sorted_results, 1):
    r2 = data['r2']
    vs_baseline = r2 - avg_r2

    # 判断是LSTM系还是GRU系
    if 'LSTM' in name:
        vs_single = r2 - lstm_test_r2
    elif 'GRU' in name:
        vs_single = r2 - gru_test_r2
    else:
        vs_single = 0.0

    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:<4} {name:<50} {r2:>10.4f} {vs_baseline:>+10.4f} {vs_single:>+12.4f}")

# ========== 分组对比分析 ==========

print("\n" + "=" * 120)
print("分组对比分析".center(120))
print("=" * 120)

# 分离LSTM和GRU的结果
lstm_experiment_results = {k: v for k, v in all_results.items() if 'LSTM-实验' in k}
gru_experiment_results = {k: v for k, v in all_results.items() if 'GRU-实验' in k}

print(f"\n{'实验编号':<30} {'LSTM R²':>12} {'GRU R²':>12} {'差距':>12} {'更优者':>10}")
print("-" * 120)

for i in range(1, 10):
    lstm_key = f'LSTM-实验{i}-' + ['原始方法', '简化特征', '保守参数', '早停机制',
                                 '残差剪裁', '加权融合50%', '最小特征', 'Ridge回归', '终极组合'][i - 1]
    gru_key = f'GRU-实验{i}-' + ['原始方法', '简化特征', '保守参数', '早停机制',
                               '残差剪裁', '加权融合50%', '最小特征', 'Ridge回归', '终极组合'][i - 1]

    lstm_r2 = lstm_experiment_results[lstm_key]['r2']
    gru_r2 = gru_experiment_results[gru_key]['r2']
    diff = lstm_r2 - gru_r2
    winner = 'LSTM' if diff > 0 else 'GRU' if diff < 0 else '平局'

    exp_name = ['原始方法', '简化特征', '保守参数', '早停机制', '残差剪裁',
                '加权融合50%', '最小特征', 'Ridge回归', '终极组合'][i - 1]

    print(f"实验{i}-{exp_name:<22} {lstm_r2:>12.4f} {gru_r2:>12.4f} {diff:>+12.4f} {winner:>10}")

# 统计胜负
lstm_wins = sum(1 for i in range(1, 10) if
                lstm_experiment_results[f'LSTM-实验{i}-' + ['原始方法', '简化特征', '保守参数', '早停机制',
                                                          '残差剪裁', '加权融合50%', '最小特征', 'Ridge回归', '终极组合'][i - 1]]['r2'] >
                gru_experiment_results[f'GRU-实验{i}-' + ['原始方法', '简化特征', '保守参数', '早停机制',
                                                        '残差剪裁', '加权融合50%', '最小特征', 'Ridge回归', '终极组合'][i - 1]]['r2'])

gru_wins = 9 - lstm_wins

print(f"\n对称实验胜负统计:")
print(f"  LSTM胜出: {lstm_wins}/9 场")
print(f"  GRU胜出: {gru_wins}/9 场")

# ========== 找出最佳策略 ==========

best_strategy_name = sorted_results[0][0]
best_r2 = sorted_results[0][1]['r2']
best_pred = sorted_results[0][1]['pred']

print(f"\n🏆 全局最佳策略: {best_strategy_name}")
print(f"   R² = {best_r2:.4f}")
print(f"   相比基线提升: {best_r2 - avg_r2:+.4f}")
print(f"   相比单模型提升: {best_r2 - max(lstm_test_r2, gru_test_r2):+.4f}")

# ========== 可视化 ==========

results_directory = "./Predict_Symmetric/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 1. 对称实验对比图
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

experiment_names = ['原始方法', '简化特征', '保守参数', '早停机制', '残差剪裁',
                    '加权融合50%', '最小特征', 'Ridge回归', '终极组合']

for i, exp_name in enumerate(experiment_names):
    ax = axes[i]

    lstm_key = f'LSTM-实验{i + 1}-{exp_name}'
    gru_key = f'GRU-实验{i + 1}-{exp_name}'

    lstm_r2 = lstm_experiment_results[lstm_key]['r2']
    gru_r2 = gru_experiment_results[gru_key]['r2']

    bars = ax.bar(['LSTM', 'GRU'], [lstm_r2, gru_r2],
                  color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black', linewidth=1.5)

    # 添加基线
    ax.axhline(y=avg_r2, color='orange', linestyle='--', linewidth=2, label='简单平均基线', alpha=0.7)
    ax.axhline(y=lstm_test_r2, color='red', linestyle=':', linewidth=1.5, label='LSTM单模型', alpha=0.5)
    ax.axhline(y=gru_test_r2, color='blue', linestyle=':', linewidth=1.5, label='GRU单模型', alpha=0.5)

    # 添加数值标签
    for bar, r2 in zip(bars, [lstm_r2, gru_r2]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{r2:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(f'实验{i + 1}: {exp_name}', fontsize=11, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=9)
    ax.set_ylim(min(lstm_r2, gru_r2) - 0.05, max(lstm_r2, gru_r2) + 0.05)
    ax.grid(True, alpha=0.3, axis='y')

    if i == 0:
        ax.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig(results_directory + 'symmetric_experiments_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# 2. Top 10策略对比
fig, ax = plt.subplots(figsize=(16, 10))

top_10 = sorted_results[:10]
names = [name for name, _ in top_10]
r2_scores = [data['r2'] for _, data in top_10]

# 根据模型类型设置颜色
colors = []
for name in names:
    if 'LSTM' in name:
        colors.append('#FF6B6B')
    elif 'GRU' in name:
        colors.append('#4ECDC4')
    else:
        colors.append('#95E1D3')

bars = ax.barh(range(len(names)), r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加基线
ax.axvline(x=avg_r2, color='orange', linestyle='--', linewidth=2.5, label='简单平均基线', alpha=0.8)

# 添加数值标签
for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
    improvement = r2 - avg_r2
    label = f'{r2:.4f} ({improvement:+.4f})'
    color = 'green' if improvement > 0 else 'red'

    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            label, ha='left', va='center', fontweight='bold', fontsize=9, color=color)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Top 10 策略性能排名（对称实验）', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

# 添加图例说明
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#FF6B6B', alpha=0.8, label='LSTM系列'),
    Patch(facecolor='#4ECDC4', alpha=0.8, label='GRU系列'),
    Patch(facecolor='#95E1D3', alpha=0.8, label='混合策略')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(results_directory + 'top10_strategies_ranking.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# 3. 实验类型对比（分组对比）
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(experiment_names))
width = 0.35

lstm_scores = [lstm_experiment_results[f'LSTM-实验{i + 1}-{name}']['r2'] for i, name in enumerate(experiment_names)]
gru_scores = [gru_experiment_results[f'GRU-实验{i + 1}-{name}']['r2'] for i, name in enumerate(experiment_names)]

bars1 = ax.bar(x - width / 2, lstm_scores, width, label='LSTM', color='#FF6B6B', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width / 2, gru_scores, width, label='GRU', color='#4ECDC4', alpha=0.8, edgecolor='black')

# 添加基线
ax.axhline(y=avg_r2, color='orange', linestyle='--', linewidth=2, label='简单平均基线', alpha=0.7)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, rotation=0)

ax.set_xlabel('实验类型', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('LSTM vs GRU 对称实验全面对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiment_names, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_directory + 'lstm_vs_gru_full_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# 4. 最佳策略预测曲线
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1))

# 找出LSTM和GRU各自最佳策略
best_lstm = max(lstm_experiment_results.items(), key=lambda x: x[1]['r2'])
best_gru = max(gru_experiment_results.items(), key=lambda x: x[1]['r2'])

strategies_to_plot = [
    ('全局最佳', best_pred, sorted_results[0][1]['r2']),
    ('LSTM最佳', best_lstm[1]['pred'], best_lstm[1]['r2']),
    ('GRU最佳', best_gru[1]['pred'], best_gru[1]['r2']),
    ('简单平均', avg_test_pred, avg_r2)
]

for idx, (name, pred, r2) in enumerate(strategies_to_plot):
    ax = axes[idx // 2, idx % 2]

    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))

    ax.plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
    ax.plot(pred_original, label=name, linewidth=2, alpha=0.8)

    mae = mean_absolute_error(y_test_original, pred_original)
    rmse = sqrt(mean_squared_error(y_test_original, pred_original))

    ax.set_title(f'{name}\nR²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('样本序号', fontsize=10)
    ax.set_ylabel('玉米价格', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'best_strategies_predictions.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# 5. 胜负矩阵热力图
fig, ax = plt.subplots(figsize=(12, 10))

# 创建对比矩阵
comparison_matrix = np.zeros((9, 2))  # 9个实验 x 2个模型

for i in range(9):
    exp_name = experiment_names[i]
    lstm_r2 = lstm_experiment_results[f'LSTM-实验{i + 1}-{exp_name}']['r2']
    gru_r2 = gru_experiment_results[f'GRU-实验{i + 1}-{exp_name}']['r2']

    comparison_matrix[i, 0] = lstm_r2
    comparison_matrix[i, 1] = gru_r2

im = ax.imshow(comparison_matrix, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=0.9)

# 设置刻度
ax.set_xticks([0, 1])
ax.set_xticklabels(['LSTM', 'GRU'], fontsize=12, fontweight='bold')
ax.set_yticks(range(9))
ax.set_yticklabels([f'实验{i + 1}: {name}' for i, name in enumerate(experiment_names)], fontsize=10)

# 添加数值标签
for i in range(9):
    for j in range(2):
        text = ax.text(j, i, f'{comparison_matrix[i, j]:.4f}',
                       ha="center", va="center", color="black", fontsize=10, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('R² Score', fontsize=12, fontweight='bold')

ax.set_title('LSTM vs GRU 对称实验热力图', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(results_directory + 'lstm_gru_heatmap.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# 6. 改进效果趋势图
fig, ax = plt.subplots(figsize=(14, 8))

improvements_lstm = [lstm_experiment_results[f'LSTM-实验{i + 1}-{name}']['r2'] - lstm_test_r2
                     for i, name in enumerate(experiment_names)]
improvements_gru = [gru_experiment_results[f'GRU-实验{i + 1}-{name}']['r2'] - gru_test_r2
                    for i, name in enumerate(experiment_names)]

x = np.arange(len(experiment_names))

ax.plot(x, improvements_lstm, marker='o', linewidth=2.5, markersize=10,
        label='LSTM改进', color='#FF6B6B', alpha=0.8)
ax.plot(x, improvements_gru, marker='s', linewidth=2.5, markersize=10,
        label='GRU改进', color='#4ECDC4', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

# 标注数值
for i, (imp_l, imp_g) in enumerate(zip(improvements_lstm, improvements_gru)):
    ax.text(i, imp_l + 0.002, f'{imp_l:+.3f}', ha='center', va='bottom', fontsize=8, color='#FF6B6B')
    ax.text(i, imp_g - 0.002, f'{imp_g:+.3f}', ha='center', va='top', fontsize=8, color='#4ECDC4')

ax.set_xlabel('实验类型', fontsize=12, fontweight='bold')
ax.set_ylabel('相比单模型的R²改进', fontsize=12, fontweight='bold')
ax.set_title('残差学习改进效果趋势对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiment_names, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'improvement_trends.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ========== 保存结果 ==========

print("\n" + "=" * 120)
print("保存对称实验结果".center(120))
print("=" * 120)

# 保存所有预测结果
predictions_dict = {'true_value': y_test_original.flatten()}
for name, data in all_results.items():
    pred_original = y_scaler.inverse_transform(data['pred'].reshape(-1, 1))
    predictions_dict[name.replace('/', '_').replace('-', '_')] = pred_original.flatten()

results_df = pd.DataFrame(predictions_dict)
results_df.to_csv(results_directory + 'symmetric_all_predictions.csv', index=False)

# 保存性能指标
metrics_data = []
for name, data in all_results.items():
    pred_original = y_scaler.inverse_transform(data['pred'].reshape(-1, 1))

    base_model = 'LSTM' if 'LSTM' in name else 'GRU' if 'GRU' in name else 'Mixed'
    vs_single = data['r2'] - (lstm_test_r2 if base_model == 'LSTM' else gru_test_r2 if base_model == 'GRU' else 0)

    metrics_data.append({
        'strategy': name,
        'base_model': base_model,
        'r2': data['r2'],
        'mae': mean_absolute_error(y_test_original, pred_original),
        'rmse': sqrt(mean_squared_error(y_test_original, pred_original)),
        'mape': np.mean(np.abs((pred_original - y_test_original) / (y_test_original + 1e-8))),
        'vs_baseline': data['r2'] - avg_r2,
        'vs_single_model': vs_single
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.sort_values('r2', ascending=False)
metrics_df.to_csv(results_directory + 'symmetric_performance_metrics.csv', index=False)

# 保存对比分析
comparison_data = []
for i in range(9):
    exp_name = experiment_names[i]
    lstm_key = f'LSTM-实验{i + 1}-{exp_name}'
    gru_key = f'GRU-实验{i + 1}-{exp_name}'

    lstm_r2 = lstm_experiment_results[lstm_key]['r2']
    gru_r2 = gru_experiment_results[gru_key]['r2']

    comparison_data.append({
        'experiment': f'实验{i + 1}',
        'experiment_name': exp_name,
        'lstm_r2': lstm_r2,
        'gru_r2': gru_r2,
        'difference': lstm_r2 - gru_r2,
        'winner': 'LSTM' if lstm_r2 > gru_r2 else 'GRU' if gru_r2 > lstm_r2 else 'Tie'
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(results_directory + 'lstm_gru_comparison.csv', index=False)

print("\n✓ 保存完成！")
print(f"  - symmetric_all_predictions.csv (所有策略预测)")
print(f"  - symmetric_performance_metrics.csv (性能指标)")
print(f"  - lstm_gru_comparison.csv (LSTM vs GRU对比)")
print(f"  - 6张可视化图表")

# ========== 最终总结报告 ==========

print("\n" + "=" * 120)
print("🎉 对称实验完成！最终总结报告".center(120))
print("=" * 120)

print(f"\n📊 基础模型性能:")
print(f"  LSTM单模型: R² = {lstm_test_r2:.4f}")
print(f"  GRU单模型:  R² = {gru_test_r2:.4f}")
print(f"  简单平均:   R² = {avg_r2:.4f}")
print(f"  更优基础模型: {'LSTM' if lstm_test_r2 > gru_test_r2 else 'GRU'}")

print(f"\n🏆 对称实验结果:")
print(f"  全局最佳策略: {best_strategy_name}")
print(f"  最佳R²: {best_r2:.4f}")
print(f"  相比基线提升: {best_r2 - avg_r2:+.4f} ({(best_r2 - avg_r2) / avg_r2 * 100:+.2f}%)")

print(f"\n  LSTM最佳策略: {best_lstm[0]}")
print(f"  LSTM最佳R²: {best_lstm[1]['r2']:.4f} (提升: {best_lstm[1]['r2'] - lstm_test_r2:+.4f})")

print(f"\n  GRU最佳策略: {best_gru[0]}")
print(f"  GRU最佳R²: {best_gru[1]['r2']:.4f} (提升: {best_gru[1]['r2'] - gru_test_r2:+.4f})")

print(f"\n📈 对称实验统计:")
print(f"  LSTM胜出场次: {lstm_wins}/9")
print(f"  GRU胜出场次: {gru_wins}/9")
print(f"  整体更优者: {'LSTM' if lstm_wins > gru_wins else 'GRU' if gru_wins > lstm_wins else '平手'}")

print(f"\n💡 核心发现:")
avg_lstm_improvement = np.mean(improvements_lstm)
avg_gru_improvement = np.mean(improvements_gru)
print(f"  LSTM平均改进: {avg_lstm_improvement:+.4f}")
print(f"  GRU平均改进: {avg_gru_improvement:+.4f}")

positive_lstm = sum(1 for x in improvements_lstm if x > 0)
positive_gru = sum(1 for x in improvements_gru if x > 0)
print(f"  LSTM正向改进次数: {positive_lstm}/9")
print(f"  GRU正向改进次数: {positive_gru}/9")

print(f"\n🎯 最佳实践建议:")
if best_r2 > avg_r2:
    print(f"  ✅ 残差学习在本数据集上有效")
    print(f"  ✅ 推荐使用: {best_strategy_name}")
    print(f"  ✅ 预期性能提升: {(best_r2 - avg_r2) / avg_r2 * 100:.2f}%")
else:
    print(f"  ⚠️ 残差学习未超过简单平均")
    print(f"  💡 建议优先改进基础模型")

print(f"\n📁 所有结果已保存到: {results_directory}")
print("=" * 120)