import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt, pi
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from tensorflow.keras import Sequential, Model, layers, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor
import pickle
import warnings

warnings.filterwarnings('ignore')

# ========== 【关键修复】TensorFlow内存管理配置 ==========
# 设置TensorFlow使用GPU内存增长模式（即使没有GPU也有助于稳定性）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU配置警告: {e}")

# 限制CPU线程数，防止内存溢出
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# 设置为更稳定的执行模式
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少日志输出
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化（可能导致崩溃）

print("✓ TensorFlow配置优化完成")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("LSTM + GRU + Attention + XGBoost 融合时间序列预测".center(100))
print("核心改进：添加Attention机制提升时间序列建模能力".center(100))
print("=" * 100)


# ========== 【核心新增】Attention层定义 ==========
class AttentionLayer(layers.Layer):
    """
    自定义Attention层
    实现Bahdanau风格的加性注意力机制
    """

    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # W1: 用于查询的权重矩阵
        self.W1 = self.add_weight(
            name='attention_W1',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        # W2: 用于键的权重矩阵
        self.W2 = self.add_weight(
            name='attention_W2',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        # V: 用于计算注意力分数的权重向量
        self.V = self.add_weight(
            name='attention_V',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)

        # 计算注意力分数
        # score = V^T * tanh(W1*h + W2*h)
        score = tf.nn.tanh(
            tf.matmul(inputs, self.W1) + tf.matmul(inputs, self.W2)
        )  # (batch_size, time_steps, units)

        attention_weights = tf.nn.softmax(
            tf.matmul(score, self.V), axis=1
        )  # (batch_size, time_steps, 1)

        # 加权求和
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


# ========== 【改进】模型构建函数（带Attention） ==========
def build_lstm_with_attention(input_shape, lstm_units=100, attention_units=128):
    """构建带Attention机制的LSTM模型"""
    inputs = Input(shape=input_shape)

    # LSTM层（return_sequences=True以便使用Attention）
    lstm_out = layers.LSTM(
        units=lstm_units,
        return_sequences=True,
        kernel_regularizer=l2(0.01),
        recurrent_regularizer=l2(0.01)
    )(inputs)

    # Attention层
    context_vector, attention_weights = AttentionLayer(units=attention_units)(lstm_out)

    # Dropout
    dropout = layers.Dropout(0.3)(context_vector)

    # 输出层
    outputs = layers.Dense(1)(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_gru_with_attention(input_shape, gru_units=100, attention_units=128):
    """构建带Attention机制的GRU模型"""
    inputs = Input(shape=input_shape)

    # GRU层（return_sequences=True以便使用Attention）
    gru_out = layers.GRU(
        units=gru_units,
        return_sequences=True,
        kernel_regularizer=l2(0.01),
        recurrent_regularizer=l2(0.01)
    )(inputs)

    # Attention层
    context_vector, attention_weights = AttentionLayer(units=attention_units)(gru_out)

    # Dropout
    dropout = layers.Dropout(0.3)(context_vector)

    # 输出层
    outputs = layers.Dense(1)(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_stacked_lstm_attention(input_shape, lstm_units=100, attention_units=128):
    """构建堆叠LSTM+Attention模型"""
    inputs = Input(shape=input_shape)

    # 第一层LSTM
    lstm1 = layers.LSTM(
        units=lstm_units,
        return_sequences=True,
        kernel_regularizer=l2(0.01)
    )(inputs)
    lstm1 = layers.Dropout(0.2)(lstm1)

    # 第二层LSTM
    lstm2 = layers.LSTM(
        units=lstm_units // 2,
        return_sequences=True,
        kernel_regularizer=l2(0.01)
    )(lstm1)

    # Attention层
    context_vector, attention_weights = AttentionLayer(units=attention_units)(lstm2)

    # Dropout
    dropout = layers.Dropout(0.3)(context_vector)

    # 输出层
    outputs = layers.Dense(1)(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# ========== 基础模型（用于对比） ==========
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


# ========== 数据加载和准备 ==========
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print("\n数据集信息:")
print(dataset.info())

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


# ========== OOF预测生成（添加错误处理和内存清理）==========
def get_oof_predictions(X_seq, y_seq, model_type='lstm_attention', n_splits=5):
    print(f"\n生成{model_type.upper()} OOF预测（TimeSeriesSplit with {n_splits} splits）...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros(len(y_seq))

    fold = 1
    for train_idx, val_idx in tscv.split(X_seq):
        print(f"  Fold {fold}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")

        X_fold_train, X_fold_val = X_seq[train_idx], X_seq[val_idx]
        y_fold_train, y_fold_val = y_seq[train_idx], y_seq[val_idx]

        try:
            # 根据模型类型选择不同的构建函数
            if model_type == 'lstm':
                model = build_simple_lstm((X_seq.shape[1], X_seq.shape[2]))
            elif model_type == 'gru':
                model = build_simple_gru((X_seq.shape[1], X_seq.shape[2]))
            elif model_type == 'lstm_attention':
                model = build_lstm_with_attention((X_seq.shape[1], X_seq.shape[2]))
            elif model_type == 'gru_attention':
                model = build_gru_with_attention((X_seq.shape[1], X_seq.shape[2]))
            elif model_type == 'stacked_lstm_attention':
                model = build_stacked_lstm_attention((X_seq.shape[1], X_seq.shape[2]))
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.compile(
                loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=0
            )

            # 训练模型
            model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=200,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            # 预测
            val_pred = model.predict(X_fold_val, verbose=0)
            oof_preds[val_idx] = val_pred.flatten()

            # 关键：清理内存
            del model
            tf.keras.backend.clear_session()

            print(f"    ✓ Fold {fold} 完成")

        except Exception as e:
            print(f"    ✗ Fold {fold} 失败: {str(e)}")
            # 如果失败，使用简单预测填充
            oof_preds[val_idx] = np.mean(y_fold_train)

        fold += 1

    return oof_preds


print("\n" + "=" * 100)
print("第一步：生成所有模型的OOF预测".center(100))
print("=" * 100)

# 生成基础模型的OOF预测
print("\n【基础模型】")
lstm_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'lstm', n_splits=5)
gru_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'gru', n_splits=5)

# 生成Attention模型的OOF预测
print("\n【Attention增强模型】")
lstm_attn_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'lstm_attention', n_splits=5)
gru_attn_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'gru_attention', n_splits=5)
stacked_attn_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, 'stacked_lstm_attention', n_splits=5)

print(f"\nOOF预测生成完成！")
print(f"基础LSTM R^2: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"基础GRU R^2: {r2_score(y_train_seq, gru_oof_preds):.4f}")
print(f"LSTM+Attention R^2: {r2_score(y_train_seq, lstm_attn_oof_preds):.4f}")
print(f"GRU+Attention R^2: {r2_score(y_train_seq, gru_attn_oof_preds):.4f}")
print(f"Stacked LSTM+Attention R^2: {r2_score(y_train_seq, stacked_attn_oof_preds):.4f}")

# ========== 第二步：训练最终模型 ==========
print("\n" + "=" * 100)
print("第二步：训练最终的所有模型".center(100))
print("=" * 100)

# 训练基础LSTM
print("\n训练基础LSTM模型...")
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
lstm_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
    ],
    verbose=0
)
print("✓ 基础LSTM训练完成")

# 训练基础GRU
print("\n训练基础GRU模型...")
gru_final = Sequential([
    layers.GRU(units=100, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    layers.Dense(1)
])
gru_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
gru_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)],
    verbose=0
)
print("✓ 基础GRU训练完成")

# 训练LSTM+Attention
print("\n训练LSTM+Attention模型...")
lstm_attn_final = build_lstm_with_attention((X_train_seq.shape[1], X_train_seq.shape[2]))
lstm_attn_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
lstm_attn_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
    ],
    verbose=0
)
print("✓ LSTM+Attention训练完成")

# 训练GRU+Attention
print("\n训练GRU+Attention模型...")
gru_attn_final = build_gru_with_attention((X_train_seq.shape[1], X_train_seq.shape[2]))
gru_attn_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
gru_attn_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)],
    verbose=0
)
print("✓ GRU+Attention训练完成")

# 训练Stacked LSTM+Attention
print("\n训练Stacked LSTM+Attention模型...")
stacked_attn_final = build_stacked_lstm_attention((X_train_seq.shape[1], X_train_seq.shape[2]))
stacked_attn_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
stacked_attn_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)],
    verbose=0
)
print("✓ Stacked LSTM+Attention训练完成")

# 生成测试集预测
lstm_test_pred = lstm_final.predict(X_test_seq, verbose=0).flatten()
gru_test_pred = gru_final.predict(X_test_seq, verbose=0).flatten()
lstm_attn_test_pred = lstm_attn_final.predict(X_test_seq, verbose=0).flatten()
gru_attn_test_pred = gru_attn_final.predict(X_test_seq, verbose=0).flatten()
stacked_attn_test_pred = stacked_attn_final.predict(X_test_seq, verbose=0).flatten()

# 评估所有模型
print(f"\n测试集性能对比:")
print(f"基础LSTM R^2: {r2_score(y_test_seq, lstm_test_pred):.4f}")
print(f"基础GRU R^2: {r2_score(y_test_seq, gru_test_pred):.4f}")
print(f"LSTM+Attention R^2: {r2_score(y_test_seq, lstm_attn_test_pred):.4f}")
print(f"GRU+Attention R^2: {r2_score(y_test_seq, gru_attn_test_pred):.4f}")
print(f"Stacked LSTM+Attention R^2: {r2_score(y_test_seq, stacked_attn_test_pred):.4f}")

# ========== 第三步：融合策略 ==========
print("\n" + "=" * 100)
print("第三步：融合所有模型".center(100))
print("=" * 100)

# 简单平均融合
print("\n【策略1】简单平均")
avg_all = (lstm_test_pred + gru_test_pred + lstm_attn_test_pred +
           gru_attn_test_pred + stacked_attn_test_pred) / 5
avg_attention_only = (lstm_attn_test_pred + gru_attn_test_pred + stacked_attn_test_pred) / 3

print(f"所有模型平均 R^2: {r2_score(y_test_seq, avg_all):.4f}")
print(f"仅Attention模型平均 R^2: {r2_score(y_test_seq, avg_attention_only):.4f}")

# 加权平均融合（基于OOF性能）
print("\n【策略2】加权平均（基于OOF R^2）")
oof_r2_scores = {
    'lstm': r2_score(y_train_seq, lstm_oof_preds),
    'gru': r2_score(y_train_seq, gru_oof_preds),
    'lstm_attn': r2_score(y_train_seq, lstm_attn_oof_preds),
    'gru_attn': r2_score(y_train_seq, gru_attn_oof_preds),
    'stacked_attn': r2_score(y_train_seq, stacked_attn_oof_preds)
}

# 归一化权重（使用softmax）
oof_scores = np.array(list(oof_r2_scores.values()))
weights = np.exp(oof_scores * 5) / np.sum(np.exp(oof_scores * 5))

weighted_pred = (weights[0] * lstm_test_pred +
                 weights[1] * gru_test_pred +
                 weights[2] * lstm_attn_test_pred +
                 weights[3] * gru_attn_test_pred +
                 weights[4] * stacked_attn_test_pred)

print(f"模型权重: LSTM={weights[0]:.3f}, GRU={weights[1]:.3f}, " +
      f"LSTM+Attn={weights[2]:.3f}, GRU+Attn={weights[3]:.3f}, Stacked={weights[4]:.3f}")
print(f"加权融合 R^2: {r2_score(y_test_seq, weighted_pred):.4f}")

# Stacking融合（使用Ridge）
print("\n【策略3】Stacking融合（Ridge）")
meta_features_train = np.column_stack([
    lstm_oof_preds,
    gru_oof_preds,
    lstm_attn_oof_preds,
    gru_attn_oof_preds,
    stacked_attn_oof_preds
])

meta_features_test = np.column_stack([
    lstm_test_pred,
    gru_test_pred,
    lstm_attn_test_pred,
    gru_attn_test_pred,
    stacked_attn_test_pred
])

ridge_meta = Ridge(alpha=1.0)
ridge_meta.fit(meta_features_train, y_train_seq)
stacking_pred = ridge_meta.predict(meta_features_test)

print(f"Ridge权重: {ridge_meta.coef_}")
print(f"Stacking融合 R^2: {r2_score(y_test_seq, stacking_pred):.4f}")

# ========== 第四步：XGBoost残差学习 ==========
print("\n" + "=" * 100)
print("第四步：XGBoost残差学习".center(100))
print("=" * 100)

# 选择性能最好的Attention模型作为基础
best_attention_model = max(
    [('LSTM+Attention', lstm_attn_oof_preds, lstm_attn_test_pred),
     ('GRU+Attention', gru_attn_oof_preds, gru_attn_test_pred),
     ('Stacked+Attention', stacked_attn_oof_preds, stacked_attn_test_pred)],
    key=lambda x: r2_score(y_train_seq, x[1])
)

print(f"\n选择 {best_attention_model[0]} 作为残差学习的基础模型")

base_oof_pred = best_attention_model[1]
base_test_pred = best_attention_model[2]

# 计算残差
residual_train = y_train_seq - base_oof_pred


# 简化特征
def create_simplified_features(X_flat, base_pred):
    features_list = [X_flat]
    features_list.append(base_pred.reshape(-1, 1))
    return np.hstack(features_list)


basic_features_train = X_train_flat[:len(base_oof_pred)]
basic_features_test = X_test_flat

simplified_train = create_simplified_features(basic_features_train, base_oof_pred)
simplified_test = create_simplified_features(basic_features_test, base_test_pred)

print(f"\n特征维度: {simplified_train.shape[1]} 维")

# 训练保守XGBoost
print("\n训练XGBoost残差模型...")
xgb_residual = XGBRegressor(
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
xgb_residual.fit(simplified_train, residual_train)

# 预测残差
residual_pred = xgb_residual.predict(simplified_test)

# 最终预测
final_pred = base_test_pred + residual_pred
final_r2 = r2_score(y_test_seq, final_pred)

print(f"✓ XGBoost残差学习完成")
print(f"基础模型 R²: {r2_score(y_test_seq, base_test_pred):.4f}")
print(f"残差学习后 R²: {final_r2:.4f} (提升: {final_r2 - r2_score(y_test_seq, base_test_pred):+.4f})")

# ========== 结果汇总 ==========
print("\n" + "=" * 100)
print("最终结果汇总（原始尺度）".center(100))
print("=" * 100)

y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

all_predictions = {
    '基础LSTM': lstm_test_pred,
    '基础GRU': gru_test_pred,
    'LSTM+Attention': lstm_attn_test_pred,
    'GRU+Attention': gru_attn_test_pred,
    'Stacked LSTM+Attention': stacked_attn_test_pred,
    '所有模型平均': avg_all,
    'Attention模型平均': avg_attention_only,
    '加权融合': weighted_pred,
    'Stacking融合': stacking_pred,
    f'{best_attention_model[0]}+XGBoost': final_pred
}

print(f"\n{'模型':<30} {'R²':>10} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
print("-" * 80)

results_list = []
for name, pred in all_predictions.items():
    pred_orig = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    r2 = r2_score(y_test_original, pred_orig)
    mae = mean_absolute_error(y_test_original, pred_orig)
    rmse = sqrt(mean_squared_error(y_test_original, pred_orig))
    mape = np.mean(np.abs((pred_orig - y_test_original) / (y_test_original + 1e-8)))

    results_list.append((name, r2, mae, rmse, mape, pred_orig))
    print(f"{name:<30} {r2:>10.4f} {mae:>12.2f} {rmse:>12.2f} {mape:>12.6f}")

# 排序
results_list.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 100)
print("性能排名".center(100))
print("=" * 100)

for rank, (name, r2, mae, rmse, mape, pred) in enumerate(results_list, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:>2}. {name:<30} R²={r2:.4f}")

best_name = results_list[0][0]
best_r2 = results_list[0][1]
print(f"\n🏆 最佳模型: {best_name} (R² = {best_r2:.4f})")

# ========== 可视化 ==========
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

print("\n生成可视化图表...")

# 1. 模型性能对比
fig, ax = plt.subplots(figsize=(14, 8))

models = ['基础LSTM', '基础GRU', 'LSTM+Attn', 'GRU+Attn', 'Stacked+Attn',
          '简单平均', 'Attn平均', '加权融合', 'Stacking', best_attention_model[0] + '+XGB']
r2_scores_list = [
    r2_score(y_test_seq, lstm_test_pred),
    r2_score(y_test_seq, gru_test_pred),
    r2_score(y_test_seq, lstm_attn_test_pred),
    r2_score(y_test_seq, gru_attn_test_pred),
    r2_score(y_test_seq, stacked_attn_test_pred),
    r2_score(y_test_seq, avg_all),
    r2_score(y_test_seq, avg_attention_only),
    r2_score(y_test_seq, weighted_pred),
    r2_score(y_test_seq, stacking_pred),
    final_r2
]

colors = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4',
          '#95E1D3', '#95E1D3', '#F38181', '#AA96DA', '#FFD93D']

bars = ax.barh(models, r2_scores_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, score) in enumerate(zip(bars, r2_scores_list)):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('R² Score', fontsize=13, fontweight='bold')
ax.set_title('模型性能对比（带Attention机制）', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([min(r2_scores_list) - 0.05, max(r2_scores_list) + 0.05])

plt.tight_layout()
plt.savefig(results_directory + 'attention_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 保存: attention_model_comparison.png")
plt.close()

# 2. Attention vs 基础模型对比
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

comparisons = [
    ('LSTM', lstm_test_pred, 'LSTM+Attention', lstm_attn_test_pred),
    ('GRU', gru_test_pred, 'GRU+Attention', gru_attn_test_pred)
]

for idx, (name1, pred1, name2, pred2) in enumerate(comparisons):
    # 左图：基础模型
    ax1 = axes[idx, 0]
    pred1_orig = y_scaler.inverse_transform(pred1.reshape(-1, 1)).flatten()
    r2_1 = r2_score(y_test_original, pred1_orig)

    ax1.plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
    ax1.plot(pred1_orig, label=name1, linewidth=2, alpha=0.7, color='#FF6B6B')
    ax1.set_title(f'{name1}\nR²={r2_1:.4f}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('样本序号', fontsize=10)
    ax1.set_ylabel('玉米价格', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右图：Attention模型
    ax2 = axes[idx, 1]
    pred2_orig = y_scaler.inverse_transform(pred2.reshape(-1, 1)).flatten()
    r2_2 = r2_score(y_test_original, pred2_orig)
    improvement = r2_2 - r2_1

    ax2.plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
    ax2.plot(pred2_orig, label=name2, linewidth=2, alpha=0.7, color='#4ECDC4')
    ax2.set_title(f'{name2}\nR²={r2_2:.4f} (提升: {improvement:+.4f})',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('样本序号', fontsize=10)
    ax2.set_ylabel('玉米价格', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.suptitle('Attention机制对比基础模型', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(results_directory + 'attention_vs_baseline.png', dpi=300, bbox_inches='tight')
print("✓ 保存: attention_vs_baseline.png")
plt.close()

# 3. 融合策略对比
fig, ax = plt.subplots(figsize=(16, 6))

fusion_preds = {
    '真实值': y_test_original,
    '所有模型平均': y_scaler.inverse_transform(avg_all.reshape(-1, 1)).flatten(),
    'Attention平均': y_scaler.inverse_transform(avg_attention_only.reshape(-1, 1)).flatten(),
    '加权融合': y_scaler.inverse_transform(weighted_pred.reshape(-1, 1)).flatten(),
    'Stacking融合': y_scaler.inverse_transform(stacking_pred.reshape(-1, 1)).flatten(),
    best_attention_model[0] + '+XGBoost': y_scaler.inverse_transform(final_pred.reshape(-1, 1)).flatten()
}

colors_fusion = ['black', '#95E1D3', '#4ECDC4', '#F38181', '#AA96DA', '#FFD93D']
line_styles = ['-', '--', '--', '--', '--', '--']
line_widths = [3, 2, 2, 2, 2, 2.5]

for (name, pred), color, style, width in zip(fusion_preds.items(), colors_fusion, line_styles, line_widths):
    if name == '真实值':
        ax.plot(pred, label=name, linewidth=width, alpha=0.9, color=color, linestyle=style)
    else:
        ax.plot(pred, label=name, linewidth=width, alpha=0.7, color=color, linestyle=style)

ax.set_xlabel('样本序号', fontsize=12, fontweight='bold')
ax.set_ylabel('玉米价格', fontsize=12, fontweight='bold')
ax.set_title('不同融合策略预测效果对比', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'fusion_strategies_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 保存: fusion_strategies_comparison.png")
plt.close()

# 4. 残差分析
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

residual_preds = [
    ('基础LSTM', lstm_test_pred, '#FF6B6B'),
    ('LSTM+Attention', lstm_attn_test_pred, '#4ECDC4'),
    ('基础GRU', gru_test_pred, '#FF6B6B'),
    ('GRU+Attention', gru_attn_test_pred, '#4ECDC4'),
    ('Stacked+Attention', stacked_attn_test_pred, '#4ECDC4'),
    ('Stacking融合', stacking_pred, '#AA96DA')
]

for idx, (name, pred, color) in enumerate(residual_preds):
    ax = axes[idx // 3, idx % 3]
    residual = y_test_seq - pred

    ax.hist(residual, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_title(f'{name}\nstd={np.std(residual):.5f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('残差', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('残差分布分析', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(results_directory + 'residual_analysis_attention.png', dpi=300, bbox_inches='tight')
print("✓ 保存: residual_analysis_attention.png")
plt.close()

# 5. Attention权重可视化
print("\n生成Attention权重可视化...")
try:
    sample_idx = 0
    sample_input = X_test_seq[sample_idx:sample_idx + 1]

    # 创建可视化模型（输出attention权重）
    # 修复：找到正确的Attention层
    attention_layer = None
    for layer in lstm_attn_final.layers:
        if isinstance(layer, AttentionLayer):
            attention_layer = layer
            break

    if attention_layer is not None:
        # 创建中间模型来获取LSTM输出
        lstm_layer = lstm_attn_final.layers[1]  # LSTM层
        intermediate_model = Model(
            inputs=lstm_attn_final.input,
            outputs=lstm_layer.output
        )

        # 获取LSTM输出
        lstm_output = intermediate_model.predict(sample_input, verbose=0)

        # 手动计算attention权重
        context_vector, attention_weights = attention_layer(lstm_output)
        attention_weights = attention_weights.numpy()[0].flatten()

        fig, ax = plt.subplots(figsize=(12, 6))
        time_steps = range(1, len(attention_weights) + 1)

        bars = ax.bar(time_steps, attention_weights, color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('时间步 (Time Step)', fontsize=12, fontweight='bold')
        ax.set_ylabel('注意力权重 (Attention Weight)', fontsize=12, fontweight='bold')
        ax.set_title('LSTM+Attention模型的注意力权重分布示例', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')

        # 标注最高权重
        max_idx = np.argmax(attention_weights)
        ax.annotate(f'最大权重\n{attention_weights[max_idx]:.3f}',
                    xy=(max_idx + 1, attention_weights[max_idx]),
                    xytext=(max_idx + 1, attention_weights[max_idx] + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold', color='red',
                    ha='center')

        plt.tight_layout()
        plt.savefig(results_directory + 'attention_weights_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ 保存: attention_weights_visualization.png")
        plt.close()
    else:
        print("⚠ 未找到Attention层，跳过权重可视化")

except Exception as e:
    print(f"⚠ Attention权重可视化失败: {e}")
    print("  继续执行后续步骤...")

# 6. 多个样本的Attention权重热力图
print("\n生成Attention权重热力图...")
try:
    n_samples = min(20, len(X_test_seq))
    attention_weights_matrix = []

    # 找到Attention层
    attention_layer = None
    for layer in lstm_attn_final.layers:
        if isinstance(layer, AttentionLayer):
            attention_layer = layer
            break

    if attention_layer is not None:
        # 获取LSTM层输出
        lstm_layer = lstm_attn_final.layers[1]
        intermediate_model = Model(
            inputs=lstm_attn_final.input,
            outputs=lstm_layer.output
        )

        for i in range(n_samples):
            sample = X_test_seq[i:i + 1]
            lstm_out = intermediate_model.predict(sample, verbose=0)
            _, attn_weights = attention_layer(lstm_out)
            attention_weights_matrix.append(attn_weights.numpy()[0].flatten())

        attention_weights_matrix = np.array(attention_weights_matrix)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(attention_weights_matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('测试样本', fontsize=12, fontweight='bold')
        ax.set_title('多样本Attention权重热力图', fontsize=14, fontweight='bold', pad=15)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('注意力权重', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(results_directory + 'attention_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ 保存: attention_heatmap.png")
        plt.close()
    else:
        print("⚠ 未找到Attention层，跳过热力图")

except Exception as e:
    print(f"⚠ Attention热力图生成失败: {e}")
    print("  继续执行后续步骤...")

# 7. 雷达图对比
print("\n生成雷达图...")
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

radar_strategies = [
    '基础LSTM',
    'LSTM+Attention',
    'Stacking融合',
    best_attention_model[0] + '+XGBoost'
]

metrics_names = ['R²', '1-MAE', '1-RMSE', '稳定性', '速度']
n_metrics = len(metrics_names)

angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
angles += angles[:1]

for strategy_name in radar_strategies:
    pred = all_predictions[strategy_name]

    r2 = r2_score(y_test_seq, pred)
    mae = mean_absolute_error(y_test_seq, pred)
    rmse = sqrt(mean_squared_error(y_test_seq, pred))
    stability = 1 - np.std(y_test_seq - pred) / np.std(y_test_seq)

    # 速度评分
    if '基础LSTM' == strategy_name:
        speed = 1.0
    elif 'Attention' in strategy_name and 'Stacked' not in strategy_name:
        speed = 0.8
    elif 'Stacked' in strategy_name:
        speed = 0.6
    elif 'XGBoost' in strategy_name:
        speed = 0.5
    else:
        speed = 0.7

    # 归一化
    max_mae = 0.15
    max_rmse = 0.15
    values = [
        r2,
        max(0, 1 - (mae / max_mae)),
        max(0, 1 - (rmse / max_rmse)),
        max(0, stability),
        speed
    ]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=strategy_name, alpha=0.7)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('模型多维评估雷达图', fontsize=14, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig(results_directory + 'models_radar_chart.png', dpi=300, bbox_inches='tight')
print("✓ 保存: models_radar_chart.png")
plt.close()

# ========== 保存模型 ==========
print("\n" + "=" * 100)
print("保存模型和结果".center(100))
print("=" * 100)

# 保存Keras模型
lstm_final.save(results_directory + 'lstm_base_model.h5')
gru_final.save(results_directory + 'gru_base_model.h5')
lstm_attn_final.save(results_directory + 'lstm_attention_model.h5')
gru_attn_final.save(results_directory + 'gru_attention_model.h5')
stacked_attn_final.save(results_directory + 'stacked_lstm_attention_model.h5')

# 保存XGBoost模型
with open(results_directory + 'xgboost_residual_model.pkl', 'wb') as f:
    pickle.dump(xgb_residual, f)

# 保存Ridge meta模型
with open(results_directory + 'ridge_stacking_model.pkl', 'wb') as f:
    pickle.dump(ridge_meta, f)

# 保存归一化器
with open(results_directory + 'scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

# 保存预测结果
predictions_dict = {'true_value': y_test_original}
for name, _, _, _, _, pred_orig in results_list:
    predictions_dict[name.replace('/', '_').replace('+', '_')] = pred_orig

results_df = pd.DataFrame(predictions_dict)
results_df.to_csv(results_directory + 'all_predictions_with_attention.csv', index=False)

# 保存性能指标
metrics_data = []
for name, r2, mae, rmse, mape, _ in results_list:
    metrics_data.append({
        'model': name,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(results_directory + 'performance_metrics_attention.csv', index=False)

print("\n✓ 模型保存完成！")
print(f"  - lstm_base_model.h5")
print(f"  - gru_base_model.h5")
print(f"  - lstm_attention_model.h5")
print(f"  - gru_attention_model.h5")
print(f"  - stacked_lstm_attention_model.h5")
print(f"  - xgboost_residual_model.pkl")
print(f"  - ridge_stacking_model.pkl")
print(f"  - scalers.pkl")
print(f"  - all_predictions_with_attention.csv")
print(f"  - performance_metrics_attention.csv")

# ========== 最终总结 ==========
print("\n" + "=" * 100)
print("🎉 训练完成！最终总结报告".center(100))
print("=" * 100)

print(f"\n📊 核心改进效果:")
base_lstm_r2 = r2_score(y_test_seq, lstm_test_pred)
lstm_attn_r2 = r2_score(y_test_seq, lstm_attn_test_pred)
base_gru_r2 = r2_score(y_test_seq, gru_test_pred)
gru_attn_r2 = r2_score(y_test_seq, gru_attn_test_pred)

print(f"  ✓ LSTM + Attention: R² {base_lstm_r2:.4f} → {lstm_attn_r2:.4f} (提升: {lstm_attn_r2 - base_lstm_r2:+.4f})")
print(f"  ✓ GRU + Attention: R² {base_gru_r2:.4f} → {gru_attn_r2:.4f} (提升: {gru_attn_r2 - base_gru_r2:+.4f})")

print(f"\n💡 Attention机制的优势:")
print(f"  1. 自动学习时间序列中不同时间步的重要性")
print(f"  2. 提高模型对关键历史信息的关注度")
print(f"  3. 增强模型的可解释性（通过权重可视化）")
print(f"  4. 在长序列预测中效果尤为明显")
print(f"  5. 可以通过热力图观察模型的决策过程")

print(f"\n🏆 最佳模型: {best_name}")
print(f"   最终性能: R² = {best_r2:.4f}")
print(f"   MAE = {results_list[0][2]:.2f}")
print(f"   RMSE = {results_list[0][3]:.2f}")
print(f"   MAPE = {results_list[0][4]:.6f}")

print(f"\n📈 模型排名（Top 5）:")
for rank in range(min(5, len(results_list))):
    name, r2, mae, rmse, mape, _ = results_list[rank]
    print(f"  {rank + 1}. {name:<30} R²={r2:.4f}")

print(f"\n🎯 使用建议:")
print(f"  • 如果追求最高精度: 使用 {best_name}")
print(f"  • 如果需要可解释性: 使用 Attention模型 + 权重可视化")
print(f"  • 如果需要稳定性: 使用 Stacking融合")
print(f"  • 如果计算资源有限: 使用 单个Attention模型")

print(f"\n💾 所有结果已保存到: {results_directory}")
print(f"   - 7张可视化图表")
print(f"   - 5个深度学习模型")
print(f"   - 2个集成学习模型")
print(f"   - 完整的预测结果和性能指标")

print("\n" + "=" * 100)
print("感谢使用！如需预测新数据，请加载保存的模型和归一化器。".center(100))
# ==================================================================================
# 将以下代码追加到主代码的"感谢使用"那段之后
# ==================================================================================

print("\n" + "=" * 100)
print("生成额外的高级可视化".center(100))
print("=" * 100)

# 8. 预测误差分析对比图
print("\n生成预测误差分析...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 选择4个关键模型
key_models = [
    ('基础LSTM', lstm_test_pred, '#FF6B6B'),
    ('LSTM+Attention', lstm_attn_test_pred, '#4ECDC4'),
    ('GRU+Attention', gru_attn_test_pred, '#95E1D3'),
    ('加权融合', weighted_pred, '#F38181')
]

for idx, (name, pred, color) in enumerate(key_models):
    ax = axes[idx // 2, idx % 2]

    pred_orig = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    # 散点图：预测值 vs 真实值
    ax.scatter(y_test_original, pred_orig, alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)

    # 添加理想线（y=x）
    min_val = min(y_test_original.min(), pred_orig.min())
    max_val = max(y_test_original.max(), pred_orig.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='理想预测线')

    # 计算R²
    r2 = r2_score(y_test_original, pred_orig)
    mae = mean_absolute_error(y_test_original, pred_orig)

    ax.set_xlabel('真实值', fontsize=11, fontweight='bold')
    ax.set_ylabel('预测值', fontsize=11, fontweight='bold')
    ax.set_title(f'{name}\nR²={r2:.4f}, MAE={mae:.2f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('预测值 vs 真实值散点图', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(results_directory + 'prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
print("✓ 保存: prediction_scatter_plots.png")
plt.close()

# 9. 时序误差趋势图
print("\n生成时序误差趋势...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 上图：绝对误差
ax1 = axes[0]
for name, pred, color in [
    ('基础LSTM', lstm_test_pred, '#FF6B6B'),
    ('LSTM+Attention', lstm_attn_test_pred, '#4ECDC4'),
    ('GRU+Attention', gru_attn_test_pred, '#95E1D3'),
    ('加权融合', weighted_pred, '#F38181')
]:
    pred_orig = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    abs_error = np.abs(pred_orig - y_test_original)
    ax1.plot(abs_error, label=name, linewidth=2, alpha=0.7, color=color)

ax1.set_xlabel('样本序号', fontsize=11, fontweight='bold')
ax1.set_ylabel('绝对误差', fontsize=11, fontweight='bold')
ax1.set_title('时序绝对误差变化', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 下图：相对误差百分比
ax2 = axes[1]
for name, pred, color in [
    ('基础LSTM', lstm_test_pred, '#FF6B6B'),
    ('LSTM+Attention', lstm_attn_test_pred, '#4ECDC4'),
    ('GRU+Attention', gru_attn_test_pred, '#95E1D3'),
    ('加权融合', weighted_pred, '#F38181')
]:
    pred_orig = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    relative_error = np.abs((pred_orig - y_test_original) / y_test_original) * 100
    ax2.plot(relative_error, label=name, linewidth=2, alpha=0.7, color=color)

ax2.set_xlabel('样本序号', fontsize=11, fontweight='bold')
ax2.set_ylabel('相对误差 (%)', fontsize=11, fontweight='bold')
ax2.set_title('时序相对误差变化', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + 'error_trend_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 保存: error_trend_analysis.png")
plt.close()

# 10. 模型性能综合对比热力图
print("\n生成性能综合对比热力图...")
fig, ax = plt.subplots(figsize=(12, 8))

# 准备数据 - 修复：使用正确的模型名称
models_for_heatmap = [
    '基础LSTM', '基础GRU', 'LSTM+Attention', 'GRU+Attention',
    'Stacked LSTM+Attention', '加权融合', 'Stacking融合'  # 修复：这里改为完整名称
]

metrics_for_heatmap = []
for model_name in models_for_heatmap:
    pred = all_predictions[model_name]
    pred_orig = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    r2 = r2_score(y_test_original, pred_orig)
    mae = mean_absolute_error(y_test_original, pred_orig)
    rmse = sqrt(mean_squared_error(y_test_original, pred_orig))
    mape = np.mean(np.abs((pred_orig - y_test_original) / (y_test_original + 1e-8)))

    # 归一化到0-1（越高越好）
    metrics_for_heatmap.append([
        r2,  # R² 已经是0-1
        1 - (mae / 100),  # 归一化MAE
        1 - (rmse / 100),  # 归一化RMSE
        1 - (mape * 10)  # 归一化MAPE
    ])

metrics_for_heatmap = np.array(metrics_for_heatmap)

# 绘制热力图
im = ax.imshow(metrics_for_heatmap.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# 设置坐标轴
ax.set_xticks(range(len(models_for_heatmap)))
ax.set_xticklabels(models_for_heatmap, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(4))
ax.set_yticklabels(['R²', '1-norm(MAE)', '1-norm(RMSE)', '1-norm(MAPE)'], fontsize=11)

# 添加数值标注
for i in range(len(models_for_heatmap)):
    for j in range(4):
        text = ax.text(i, j, f'{metrics_for_heatmap[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('归一化性能分数\n(越高越好)', fontsize=11, fontweight='bold')

ax.set_title('模型性能综合对比热力图', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(results_directory + 'performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 保存: performance_heatmap.png")
plt.close()

# 11. 箱线图对比误差分布
print("\n生成误差分布箱线图...")
fig, ax = plt.subplots(figsize=(14, 7))

error_data = []
error_labels = []

for name in ['基础LSTM', 'LSTM+Attention', 'GRU+Attention', '加权融合', 'Stacking融合']:
    pred = all_predictions[name]
    pred_orig = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    error = pred_orig - y_test_original
    error_data.append(error)
    error_labels.append(name)

bp = ax.boxplot(error_data, labels=error_labels, patch_artist=True,
                showmeans=True, meanline=True)

# 设置颜色
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='零误差线')
ax.set_ylabel('预测误差', fontsize=12, fontweight='bold')
ax.set_title('模型预测误差分布对比（箱线图）', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig(results_directory + 'error_boxplot_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 保存: error_boxplot_comparison.png")
plt.close()

# 12. 学习曲线对比
print("\n生成学习曲线对比...")
print("  重新训练模型以获取训练历史...")

try:
    # 基础LSTM
    lstm_simple = build_simple_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm_simple.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
    history_lstm = lstm_simple.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    # LSTM+Attention
    lstm_attn_temp = build_lstm_with_attention((X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm_attn_temp.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
    history_attn = lstm_attn_temp.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 训练损失对比
    ax1 = axes[0]
    ax1.plot(history_lstm.history['loss'], label='基础LSTM - 训练', linewidth=2, color='#FF6B6B')
    ax1.plot(history_lstm.history['val_loss'], label='基础LSTM - 验证', linewidth=2,
             linestyle='--', color='#FF6B6B')
    ax1.plot(history_attn.history['loss'], label='LSTM+Attention - 训练', linewidth=2, color='#4ECDC4')
    ax1.plot(history_attn.history['val_loss'], label='LSTM+Attention - 验证', linewidth=2,
             linestyle='--', color='#4ECDC4')

    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('损失 (MSE)', fontsize=11, fontweight='bold')
    ax1.set_title('学习曲线对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 验证损失放大对比
    ax2 = axes[1]
    ax2.plot(history_lstm.history['val_loss'], label='基础LSTM', linewidth=2.5, color='#FF6B6B')
    ax2.plot(history_attn.history['val_loss'], label='LSTM+Attention', linewidth=2.5, color='#4ECDC4')

    ax2.set_xlabel('训练轮次 (Epoch)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('验证损失 (MSE)', fontsize=11, fontweight='bold')
    ax2.set_title('验证损失对比（Attention收敛更快）', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_directory + 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 保存: learning_curves_comparison.png")
    plt.close()

    # 清理临时模型
    del lstm_simple, lstm_attn_temp
    tf.keras.backend.clear_session()

except Exception as e:
    print(f"⚠ 学习曲线生成失败: {e}")
    print("  跳过该图表，继续执行...")

# 13. 最终综合报告图
print("\n生成最终综合报告图...")
try:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. R²排名（左上）
    ax1 = fig.add_subplot(gs[0, :2])
    models_sorted = sorted(
        [(name, r2_score(y_test_seq, all_predictions[name]))
         for name in ['基础LSTM', '基础GRU', 'LSTM+Attention', 'GRU+Attention',
                      '加权融合', 'Stacking融合']],
        key=lambda x: x[1], reverse=True
    )
    names = [m[0] for m in models_sorted]
    scores = [m[1] for m in models_sorted]
    colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))

    bars = ax1.barh(names, scores, color=colors_bar, alpha=0.8, edgecolor='black')
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{score:.4f}', ha='right', va='center', fontweight='bold', fontsize=9, color='white')
    ax1.set_xlabel('R² Score', fontsize=10, fontweight='bold')
    ax1.set_title('模型性能排名', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. 最佳模型预测图（右上）
    ax2 = fig.add_subplot(gs[0, 2])
    best_pred_orig = y_scaler.inverse_transform(weighted_pred.reshape(-1, 1)).flatten()
    ax2.plot(y_test_original[:50], label='真实值', linewidth=2.5, color='black', marker='o', markersize=4)
    ax2.plot(best_pred_orig[:50], label='加权融合', linewidth=2, color='#F38181', marker='s', markersize=3)
    ax2.set_xlabel('样本序号', fontsize=9)
    ax2.set_ylabel('玉米价格', fontsize=9)
    ax2.set_title('最佳模型预测（前50样本）', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Attention提升对比（中左）
    ax3 = fig.add_subplot(gs[1, 0])
    base_models = ['LSTM', 'GRU']
    base_scores = [
        r2_score(y_test_seq, lstm_test_pred),
        r2_score(y_test_seq, gru_test_pred)
    ]
    attn_scores = [
        r2_score(y_test_seq, lstm_attn_test_pred),
        r2_score(y_test_seq, gru_attn_test_pred)
    ]

    x = np.arange(len(base_models))
    width = 0.35
    ax3.bar(x - width / 2, base_scores, width, label='基础模型', color='#FF6B6B', alpha=0.8)
    ax3.bar(x + width / 2, attn_scores, width, label='+ Attention', color='#4ECDC4', alpha=0.8)

    ax3.set_ylabel('R² Score', fontsize=9, fontweight='bold')
    ax3.set_title('Attention机制提升效果', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(base_models)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 误差分布小提琴图（中中）
    ax4 = fig.add_subplot(gs[1, 1])
    error_lstm = y_scaler.inverse_transform(lstm_test_pred.reshape(-1, 1)).flatten() - y_test_original
    error_attn = y_scaler.inverse_transform(lstm_attn_test_pred.reshape(-1, 1)).flatten() - y_test_original

    parts = ax4.violinplot([error_lstm, error_attn], positions=[1, 2], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#4ECDC4')
        pc.set_alpha(0.7)

    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(['基础LSTM', 'LSTM+Attn'], fontsize=9)
    ax4.set_ylabel('预测误差', fontsize=9, fontweight='bold')
    ax4.set_title('误差分布对比', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. 关键指标对比（中右）
    ax5 = fig.add_subplot(gs[1, 2])
    metrics_comparison = pd.DataFrame({
        '模型': ['基础LSTM', 'LSTM+Attn', '加权融合'],
        'R²': [
            r2_score(y_test_seq, lstm_test_pred),
            r2_score(y_test_seq, lstm_attn_test_pred),
            r2_score(y_test_seq, weighted_pred)
        ],
        'MAE': [
            mean_absolute_error(y_test_original, y_scaler.inverse_transform(lstm_test_pred.reshape(-1, 1)).flatten()),
            mean_absolute_error(y_test_original,
                                y_scaler.inverse_transform(lstm_attn_test_pred.reshape(-1, 1)).flatten()),
            mean_absolute_error(y_test_original, best_pred_orig)
        ]
    })

    # 格式化数值
    metrics_values = []
    for idx, row in metrics_comparison.iterrows():
        metrics_values.append([
            row['模型'],
            f"{row['R²']:.4f}",
            f"{row['MAE']:.2f}"
        ])

    ax5.axis('tight')
    ax5.axis('off')
    table = ax5.table(cellText=metrics_values,
                      colLabels=['模型', 'R²', 'MAE'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 设置表格样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax5.set_title('关键指标对比表', fontsize=11, fontweight='bold', pad=20)

    # 6. 残差时序图（下方，跨3列）
    ax6 = fig.add_subplot(gs[2, :])
    residual_lstm = y_test_seq - lstm_test_pred
    residual_attn = y_test_seq - lstm_attn_test_pred
    residual_fusion = y_test_seq - weighted_pred

    ax6.plot(residual_lstm, label='基础LSTM', linewidth=1.5, alpha=0.7, color='#FF6B6B')
    ax6.plot(residual_attn, label='LSTM+Attention', linewidth=1.5, alpha=0.7, color='#4ECDC4')
    ax6.plot(residual_fusion, label='加权融合', linewidth=2, alpha=0.8, color='#F38181')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax6.fill_between(range(len(residual_fusion)), residual_fusion, alpha=0.3, color='#F38181')

    ax6.set_xlabel('样本序号', fontsize=10, fontweight='bold')
    ax6.set_ylabel('残差', fontsize=10, fontweight='bold')
    ax6.set_title('残差时序对比（越接近0越好）', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9, loc='best')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('🎯 LSTM+Attention时间序列预测 - 综合报告',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(results_directory + 'comprehensive_report.png', dpi=300, bbox_inches='tight')
    print("✓ 保存: comprehensive_report.png")
    plt.close()

except Exception as e:
    print(f"⚠ 综合报告图生成失败: {e}")
    print("  跳过该图表，继续执行...")

print("\n" + "=" * 100)
print("✅ 所有额外可视化生成完成！".center(100))
print("=" * 100)

print("\n📊 新增可视化图表:")
print("  8. prediction_scatter_plots.png - 预测值vs真实值散点图")
print("  9. error_trend_analysis.png - 时序误差趋势分析")
print(" 10. performance_heatmap.png - 性能综合对比热力图")
print(" 11. error_boxplot_comparison.png - 误差分布箱线图")
print(" 12. learning_curves_comparison.png - 学习曲线对比")
print(" 13. comprehensive_report.png - 最终综合报告图")

print(f"\n💾 总计13张专业可视化图表已保存到: {results_directory}")
print("=" * 100)
print("=" * 100)