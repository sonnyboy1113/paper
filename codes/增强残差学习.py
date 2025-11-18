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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("LSTM + GRU + XGBoost 融合时间序列预测 - 增强残差学习版".center(100))
print("=" * 100)

# ========== 数据加载和预处理 ==========
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print("\n数据集信息:")
print(dataset.info())

X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

split_idx = int(len(X) * 0.8)
X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_raw, y_test_raw = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n数据分割: 训练集={len(X_train_raw)}, 测试集={len(X_test_raw)}")

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

print(f"序列数据: train={X_train_seq.shape}, test={X_test_seq.shape}")


# ========== 模型定义和OOF预测 ==========
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


def get_oof_predictions(X_seq, y_seq, model_type='lstm', n_splits=5):
    print(f"\n生成{model_type.upper()} OOF预测...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros(len(y_seq))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq), 1):
        print(f"  Fold {fold}/{n_splits}")
        X_fold_train, X_fold_val = X_seq[train_idx], X_seq[val_idx]
        y_fold_train, y_fold_val = y_seq[train_idx], y_seq[val_idx]

        if model_type == 'lstm':
            model = build_simple_lstm((X_seq.shape[1], X_seq.shape[2]))
        else:
            model = build_simple_gru((X_seq.shape[1], X_seq.shape[2]))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        early_stop = EarlyStopping(monitor='val_loss', patience=20,
                                   restore_best_weights=True, verbose=0)
        model.fit(X_fold_train, y_fold_train, validation_data=(X_fold_val, y_fold_val),
                  epochs=200, batch_size=32, callbacks=[early_stop], verbose=0)

        oof_preds[val_idx] = model.predict(X_fold_val, verbose=0).flatten()

    return oof_preds


print("\n" + "=" * 100)
print("第一步：生成OOF预测".center(100))
print("=" * 100)

lstm_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'lstm')
gru_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'gru')

print(f"\nLSTM OOF R²: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"GRU OOF R²: {r2_score(y_train_seq, gru_oof_preds):.4f}")

# ========== 训练最终模型 ==========
print("\n" + "=" * 100)
print("第二步：训练最终模型".center(100))
print("=" * 100)

# LSTM with regularization
print("\n训练LSTM...")
lstm_final = Sequential([
    layers.LSTM(units=80, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(1)
])
lstm_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
lstm_history = lstm_final.fit(
    X_train_seq, y_train_seq, validation_split=0.2, epochs=200, batch_size=32,
    callbacks=[EarlyStopping('val_loss', patience=15, restore_best_weights=True, verbose=1),
               ReduceLROnPlateau('val_loss', factor=0.5, patience=5, verbose=1)],
    verbose=0
)

# GRU (original config)
print("\n训练GRU...")
gru_final = Sequential([
    layers.GRU(units=100, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    layers.Dense(1)
])
gru_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
early_stop = EarlyStopping('val_loss', patience=20, restore_best_weights=True, verbose=1)
gru_history = gru_final.fit(
    X_train_seq, y_train_seq, validation_split=0.2, epochs=200,
    batch_size=32, callbacks=[early_stop], verbose=0
)

lstm_test_pred = lstm_final.predict(X_test_seq, verbose=0).flatten()
gru_test_pred = gru_final.predict(X_test_seq, verbose=0).flatten()

print(f"\nLSTM测试R²: {r2_score(y_test_seq, lstm_test_pred):.4f}")
print(f"GRU测试R²: {r2_score(y_test_seq, gru_test_pred):.4f}")


# ========== 【核心改进】增强残差学习函数 ==========
def create_advanced_residual_features(X_flat, base_pred):
    """构造增强残差特征"""
    features_list = [X_flat]

    # 1. 预测相关特征
    features_list.append(base_pred.reshape(-1, 1))
    features_list.append((base_pred ** 2).reshape(-1, 1))
    features_list.append(np.sqrt(np.abs(base_pred)).reshape(-1, 1))

    # 2. 时间序列统计特征
    seq_len = 5
    n_features = X_flat.shape[1] // seq_len

    for i in range(seq_len):
        step_features = X_flat[:, i * n_features:(i + 1) * n_features]
        features_list.append(np.mean(step_features, axis=1).reshape(-1, 1))
        features_list.append(np.std(step_features, axis=1).reshape(-1, 1))
        features_list.append(np.max(step_features, axis=1).reshape(-1, 1))
        features_list.append(np.min(step_features, axis=1).reshape(-1, 1))

    # 3. 趋势特征
    recent_mean = np.mean(X_flat[:, -n_features:], axis=1).reshape(-1, 1)
    historical_mean = np.mean(X_flat[:, :-n_features], axis=1).reshape(-1, 1)
    features_list.append(recent_mean)
    features_list.append(historical_mean)
    features_list.append(recent_mean - historical_mean)

    # 4. 波动率
    features_list.append(np.std(X_flat, axis=1).reshape(-1, 1))

    return np.hstack(features_list)


def train_hierarchical_residual(X_train, y_train, base_pred_train,
                                X_test, base_pred_test, n_layers=2):
    """多层级联残差学习"""
    print(f"\n训练{n_layers}层级联残差模型...")
    models = []
    current_pred_train = base_pred_train.copy()
    current_pred_test = base_pred_test.copy()

    for layer in range(n_layers):
        residual = y_train - current_pred_train
        print(f"  第{layer + 1}层: 残差std={np.std(residual):.6f}")

        model = XGBRegressor(
            n_estimators=300 - layer * 50,
            learning_rate=0.05,
            max_depth=5 - layer,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + layer,
            verbosity=0
        )
        model.fit(X_train, residual)
        models.append(model)

        residual_pred_train = model.predict(X_train)
        residual_pred_test = model.predict(X_test)

        current_pred_train += residual_pred_train
        current_pred_test += residual_pred_test

    return models, current_pred_test


def train_ensemble_residual(X_train, residual, X_test, n_models=3):
    """集成多个XGBoost模型"""
    print(f"\n训练{n_models}个集成残差模型...")
    configs = [
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05},
        {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.03},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.04},
    ]

    predictions = []
    for i, config in enumerate(configs[:n_models]):
        model = XGBRegressor(**config, subsample=0.8, colsample_bytree=0.8,
                             random_state=42 + i, verbosity=0)
        model.fit(X_train, residual)
        predictions.append(model.predict(X_test))

    return np.mean(predictions, axis=0)


def train_weighted_fusion(X_train, y_train, lstm_pred, gru_pred,
                          X_test, lstm_test_pred, gru_test_pred):
    """动态权重残差融合"""
    print("\n训练动态权重残差融合...")

    # LSTM残差模型
    lstm_residual = y_train - lstm_pred
    xgb_lstm = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                            random_state=42, verbosity=0)
    xgb_lstm.fit(X_train, lstm_residual)
    lstm_res_train = xgb_lstm.predict(X_train)
    lstm_res_test = xgb_lstm.predict(X_test)

    # GRU残差模型
    gru_residual = y_train - gru_pred
    xgb_gru = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                           random_state=43, verbosity=0)
    xgb_gru.fit(X_train, gru_residual)
    gru_res_train = xgb_gru.predict(X_train)
    gru_res_test = xgb_gru.predict(X_test)

    # 学习最优权重
    meta_features_train = np.column_stack([
        lstm_pred + lstm_res_train,
        gru_pred + gru_res_train
    ])
    meta_features_test = np.column_stack([
        lstm_test_pred + lstm_res_test,
        gru_test_pred + gru_res_test
    ])

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features_train, y_train)
    print(f"  权重: LSTM={meta_model.coef_[0]:.3f}, GRU={meta_model.coef_[1]:.3f}")

    return meta_model.predict(meta_features_test)


# ========== 增强残差学习策略 ==========
print("\n" + "=" * 100)
print("第三步：增强残差学习策略".center(100))
print("=" * 100)

# 基础策略
avg_test_pred = (lstm_test_pred + gru_test_pred) / 2
avg_r2 = r2_score(y_test_seq, avg_test_pred)

# 准备基础特征
basic_features_train = X_train_flat[:len(gru_oof_preds)]
basic_features_test = X_test_flat
gru_residual = y_train_seq - gru_oof_preds

# 策略1: 原始残差学习
print("\n【策略1】原始残差学习")
xgb_original = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                            random_state=42, verbosity=0)
xgb_original.fit(basic_features_train, gru_residual)
original_residual_pred = xgb_original.predict(basic_features_test)
strategy_1_pred = gru_test_pred + original_residual_pred
strategy_1_r2 = r2_score(y_test_seq, strategy_1_pred)
print(f"✓ R²: {strategy_1_r2:.4f}")

# 策略2: 增强特征残差学习
print("\n【策略2】增强特征残差学习")
enhanced_features_train = create_advanced_residual_features(basic_features_train, gru_oof_preds)
enhanced_features_test = create_advanced_residual_features(basic_features_test, gru_test_pred)
print(f"特征维度: {basic_features_train.shape[1]} → {enhanced_features_train.shape[1]}")

xgb_enhanced = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
xgb_enhanced.fit(enhanced_features_train, gru_residual)
enhanced_residual_pred = xgb_enhanced.predict(enhanced_features_test)
strategy_2_pred = gru_test_pred + enhanced_residual_pred
strategy_2_r2 = r2_score(y_test_seq, strategy_2_pred)
print(f"✓ R²: {strategy_2_r2:.4f} (改进: {strategy_2_r2 - strategy_1_r2:+.4f})")

# 策略3: 多层级联
print("\n【策略3】多层级联残差")
_, strategy_3_pred = train_hierarchical_residual(
    basic_features_train, y_train_seq, gru_oof_preds,
    basic_features_test, gru_test_pred, n_layers=2
)
strategy_3_r2 = r2_score(y_test_seq, strategy_3_pred)
print(f"✓ R^2: {strategy_3_r2:.4f} (改进: {strategy_3_r2 - strategy_1_r2:+.4f})")

# 策略4: 集成残差模型
print("\n【策略4】集成残差模型")
ensemble_residual_pred = train_ensemble_residual(
    basic_features_train, gru_residual, basic_features_test, n_models=3
)
strategy_4_pred = gru_test_pred + ensemble_residual_pred
strategy_4_r2 = r2_score(y_test_seq, strategy_4_pred)
print(f"✓ R^2: {strategy_4_r2:.4f} (改进: {strategy_4_r2 - strategy_1_r2:+.4f})")

# 策略5: 动态权重融合
print("\n【策略5】动态权重融合")
strategy_5_pred = train_weighted_fusion(
    basic_features_train, y_train_seq, lstm_oof_preds, gru_oof_preds,
    basic_features_test, lstm_test_pred, gru_test_pred
)
strategy_5_r2 = r2_score(y_test_seq, strategy_5_pred)
print(f"✓ R^2: {strategy_5_r2:.4f} (改进: {strategy_5_r2 - strategy_1_r2:+.4f})")

# ========== 性能对比 ==========
print("\n" + "=" * 100)
print("所有策略性能对比".center(100))
print("=" * 100)

strategies = {
    'LSTM单模型': lstm_test_pred,
    'GRU单模型': gru_test_pred,
    '简单平均': avg_test_pred,
    '原始残差学习': strategy_1_pred,
    '增强特征残差': strategy_2_pred,
    '多层级联残差': strategy_3_pred,
    '集成残差模型': strategy_4_pred,
    '动态权重融合': strategy_5_pred,
}

print("\n{:<20} {:<12} {:<12} {:<12}".format("策略", "R^2", "MAE", "RMSE"))
print("-" * 60)

best_r2 = -np.inf
best_strategy = None

for name, pred in strategies.items():
    r2 = r2_score(y_test_seq, pred)
    mae = mean_absolute_error(y_test_seq, pred)
    rmse = sqrt(mean_squared_error(y_test_seq, pred))
    print(f"{name:<20} {r2:<12.4f} {mae:<12.6f} {rmse:<12.6f}")
    if r2 > best_r2:
        best_r2 = r2
        best_strategy = name

print(f"\n🏆 最佳策略: {best_strategy} (R^2 = {best_r2:.4f})")

# 反归一化评估
print("\n原始尺度评估:")
print("-" * 60)
y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1))

strategies_original = {}
for name, pred in strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    strategies_original[name] = pred_original
    r2 = r2_score(y_test_original, pred_original)
    mae = mean_absolute_error(y_test_original, pred_original)
    rmse = sqrt(mean_squared_error(y_test_original, pred_original))
    print(f"{name:<20} R^2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

# ========== 可视化 ==========
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 1. 策略对比图
fig = plt.figure(figsize=(18, 10))

plot_configs = [
    ('LSTM单模型', strategies_original['LSTM单模型'], 'blue'),
    ('GRU单模型', strategies_original['GRU单模型'], 'green'),
    ('简单平均', strategies_original['简单平均'], 'purple'),
    ('原始残差学习', strategies_original['原始残差学习'], 'orange'),
    ('增强特征残差', strategies_original['增强特征残差'], 'cyan'),
    ('多层级联残差', strategies_original['多层级联残差'], 'red'),
    ('集成残差模型', strategies_original['集成残差模型'], 'brown'),
    ('动态权重融合', strategies_original['动态权重融合'], 'magenta'),
]

for idx, (name, pred, color) in enumerate(plot_configs, 1):
    plt.subplot(4, 2, idx)
    plt.plot(y_test_original, label="真实值", linewidth=2.5, color='black', alpha=0.7)
    plt.plot(pred, label=name, linewidth=2, alpha=0.8, color=color)
    r2 = r2_score(y_test_original, pred)
    plt.title(f"{name} (R^2={r2:.4f})", fontsize=12, fontweight='bold')
    plt.xlabel('样本序号', fontsize=9)
    plt.ylabel('玉米价格', fontsize=9)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'residual_strategies_comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 2. 性能改进柱状图
fig, ax = plt.subplots(figsize=(14, 6))

strategy_names = list(strategies.keys())
r2_scores = [r2_score(y_test_seq, pred) for pred in strategies.values()]
colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'red', 'brown', 'magenta']

bars = ax.bar(range(len(strategy_names)), r2_scores, color=colors, alpha=0.7)

for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_ylabel('R^2 Score', fontsize=12, fontweight='bold')
ax.set_title('残差学习策略性能对比', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(strategy_names)))
ax.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=9)
ax.axhline(y=avg_r2, color='red', linestyle='--', linewidth=2, label='简单平均基线')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_directory + 'performance_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# 3. 残差改进效果
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# GRU基础残差
gru_residuals = y_test_seq - gru_test_pred
axes[0, 0].plot(gru_residuals, label='GRU基础残差', linewidth=2, color='red', alpha=0.7)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0, 0].set_title('GRU基础模型残差', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('残差')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 增强特征残差改进
strategy_2_residuals = y_test_seq - strategy_2_pred
axes[0, 1].plot(gru_residuals, label='修正前', linewidth=2, color='red', alpha=0.7)
axes[0, 1].plot(strategy_2_residuals, label='增强特征修正后', linewidth=2, color='cyan', alpha=0.7)
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_title('增强特征残差修正效果', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 多层级联残差改进
strategy_3_residuals = y_test_seq - strategy_3_pred
axes[1, 0].plot(gru_residuals, label='修正前', linewidth=2, color='red', alpha=0.7)
axes[1, 0].plot(strategy_3_residuals, label='多层级联修正后', linewidth=2, color='red', alpha=0.7)
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_title('多层级联残差修正效果', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('样本序号')
axes[1, 0].set_ylabel('残差')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 动态权重残差改进
strategy_5_residuals = y_test_seq - strategy_5_pred
axes[1, 1].plot(gru_residuals, label='修正前', linewidth=2, color='red', alpha=0.7)
axes[1, 1].plot(strategy_5_residuals, label='动态权重修正后', linewidth=2, color='magenta', alpha=0.7)
axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 1].set_title('动态权重残差修正效果', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('样本序号')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'residual_improvement_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# ========== 保存结果 ==========
import pickle

print("\n" + "=" * 100)
print("保存模型和结果".center(100))
print("=" * 100)

# 保存模型
lstm_final.save(results_directory + 'lstm_final_model.h5')
gru_final.save(results_directory + 'gru_final_model.h5')

with open(results_directory + 'xgb_enhanced_model.pkl', 'wb') as f:
    pickle.dump(xgb_enhanced, f)

with open(results_directory + 'scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

# 保存预测结果
results_df = pd.DataFrame({
    'true_value': y_test_original.flatten(),
    'lstm': strategies_original['LSTM单模型'].flatten(),
    'gru': strategies_original['GRU单模型'].flatten(),
    'simple_avg': strategies_original['简单平均'].flatten(),
    'original_residual': strategies_original['原始残差学习'].flatten(),
    'enhanced_residual': strategies_original['增强特征残差'].flatten(),
    'hierarchical': strategies_original['多层级联残差'].flatten(),
    'ensemble': strategies_original['集成残差模型'].flatten(),
    'weighted_fusion': strategies_original['动态权重融合'].flatten(),
})
results_df.to_csv(results_directory + 'enhanced_residual_predictions.csv', index=False)

print("\n✓ 保存完成！")

# ========== 最终总结 ==========
print("\n" + "=" * 100)
print("🎉 增强残差学习训练完成！".center(100))
print("=" * 100)

print(f"\n📊 实现的增强残差策略:")
print(f"  1. 原始残差学习 (基线)")
print(f"  2. 增强特征残差 - 添加统计和趋势特征")
print(f"  3. 多层级联残差 - 逐步精细化修正")
print(f"  4. 集成残差模型 - 多个XGBoost取长补短")
print(f"  5. 动态权重融合 - 自适应学习最优权重")

print(f"\n🏆 最佳策略: {best_strategy}")
print(f"   测试集R^2: {best_r2:.4f}")

print(f"\n📈 性能排名:")
sorted_strategies = sorted(strategies.items(), key=lambda x: r2_score(y_test_seq, x[1]), reverse=True)
for rank, (name, pred) in enumerate(sorted_strategies, 1):
    r2 = r2_score(y_test_seq, pred)
    improvement = r2 - avg_r2
    print(f"   {rank}. {name:<20} R^2={r2:.4f} (vs简单平均: {improvement:+.4f})")

print(f"\n💡 关键发现:")
improvement_2 = strategy_2_r2 - strategy_1_r2
improvement_3 = strategy_3_r2 - strategy_1_r2
improvement_4 = strategy_4_r2 - strategy_1_r2
improvement_5 = strategy_5_r2 - strategy_1_r2

print(f"  ✓ 增强特征改进: {improvement_2:+.4f}")
print(f"  ✓ 多层级联改进: {improvement_3:+.4f}")
print(f"  ✓ 集成模型改进: {improvement_4:+.4f}")
print(f"  ✓ 动态权重改进: {improvement_5:+.4f}")

print(f"\n💾 所有结果保存在: {results_directory}")
print("=" * 100)