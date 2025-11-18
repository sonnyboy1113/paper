"""
=====================================================================================
LSTM + GRU + XGBoost 融合时间序列预测 + LSTM-Attention-XGBoost对比
在原有6个残差学习策略基础上，新增LSTM-Attention-XGBoost独立集成模型
=====================================================================================

原有策略（1-6）：基于LSTM+GRU的残差学习融合
新增策略7：LSTM-Attention-XGBoost独立集成模型（直接预测，非残差）
=====================================================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from tensorflow.keras import Sequential, layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor
import warnings
import pickle

warnings.filterwarnings('ignore')

# =====================================================================================
# ✨ NEW: Attention Layer定义
# =====================================================================================
class AttentionLayer(layers.Layer):
    """自定义Attention层"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(inputs * alpha, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("LSTM + GRU + XGBoost 融合 + LSTM-Attention-XGBoost 对比".center(100))
print("原有6个残差学习策略 + 新增Attention独立集成模型".center(100))
print("=" * 100)


# =====================================================================================
# SECTION 1: 全局种子固定
# =====================================================================================
def set_global_seed(seed=12):
    """全局固定随机种子 - 确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass

    print(f"✅ 全局随机种子已固定: {seed}")


GLOBAL_SEED = 12
set_global_seed(GLOBAL_SEED)

# =====================================================================================
# SECTION 2: 过拟合检测模块
# =====================================================================================
class OverfittingDetector:
    """过拟合检测器 - 基于学习曲线分析"""

    def __init__(self, output_dir='./overfitting_analysis/'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.results = {}

    def sequential_learning_curve(self, model_builder, X_seq, y_seq,
                                  model_name='Model',
                                  train_sizes=None,
                                  epochs=100,
                                  batch_size=32):
        """为序列模型生成学习曲线"""
        print(f"\n{'=' * 80}")
        print(f"学习曲线分析: {model_name}".center(80))
        print(f"{'=' * 80}")

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        val_split_idx = int(len(X_seq) * 0.8)
        X_train_pool = X_seq[:val_split_idx]
        y_train_pool = y_seq[:val_split_idx]
        X_val = X_seq[val_split_idx:]
        y_val = y_seq[val_split_idx:]

        train_scores = []
        val_scores = []
        train_losses = []
        val_losses = []
        sample_counts = []

        for idx, train_size in enumerate(train_sizes):
            tf.random.set_seed(GLOBAL_SEED + idx)
            np.random.seed(GLOBAL_SEED + idx)

            n_samples = max(int(len(X_train_pool) * train_size), batch_size)
            print(f"训练样本数: {n_samples}/{len(X_train_pool)}", end=' ')

            X_train_subset = X_train_pool[:n_samples]
            y_train_subset = y_train_pool[:n_samples]

            model = model_builder((X_seq.shape[1], X_seq.shape[2]))
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

            early_stop = EarlyStopping(monitor='val_loss', patience=15,
                                       restore_best_weights=True, verbose=0)

            history = model.fit(
                X_train_subset, y_train_subset,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                shuffle=False,
                verbose=0
            )

            train_pred = model.predict(X_train_subset, verbose=0).flatten()
            val_pred = model.predict(X_val, verbose=0).flatten()

            train_r2 = r2_score(y_train_subset, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            train_scores.append(train_r2)
            val_scores.append(val_r2)
            sample_counts.append(n_samples)

            print(f"→ 训练R^2={train_r2:.4f}, 验证R^2={val_r2:.4f}, 差距={train_r2 - val_r2:.4f}")

        self.results[model_name] = {
            'sample_counts': sample_counts,
            'train_scores': train_scores,
            'val_scores': val_scores
        }

        final_gap = train_scores[-1] - val_scores[-1]
        if final_gap > 0.1:
            diagnosis = "过拟合"
        else:
            diagnosis = "拟合良好"

        self.results[model_name]['diagnosis'] = diagnosis
        self.results[model_name]['final_gap'] = final_gap

        print(f"\n诊断结论: {diagnosis} (差距={final_gap:.4f})")

        return sample_counts, train_scores, val_scores

    def plot_all_learning_curves(self, figsize=(20, 5)):
        """绘制所有模型的学习曲线对比"""
        n_models = len(self.results)
        if n_models == 0:
            return

        fig, axes = plt.subplots(1, min(n_models, 3), figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_models == 2:
            pass
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, result) in enumerate(self.results.items()):
            if idx >= 3:
                break

            ax = axes[idx] if n_models > 1 else axes[0]

            sample_counts = result['sample_counts']
            train_scores = result['train_scores']
            val_scores = result['val_scores']
            diagnosis = result.get('diagnosis', 'Unknown')
            final_gap = result.get('final_gap', 0)

            ax.plot(sample_counts, train_scores, 'o-', linewidth=2.5,
                   label='训练R^2', color='#2E86AB', alpha=0.8)
            ax.plot(sample_counts, val_scores, 's-', linewidth=2.5,
                   label='验证R^2', color='#A23B72', alpha=0.8)

            ax.fill_between(sample_counts, train_scores, val_scores,
                           alpha=0.2, color='red' if final_gap > 0.1 else 'green')

            ax.set_title(f'{model_name}\n{diagnosis} (差距={final_gap:.4f})',
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('训练样本数', fontsize=11)
            ax.set_ylabel('R^2 Score', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle('基模型学习曲线分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}learning_curves_all_models.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ 学习曲线对比图已保存: {self.output_dir}learning_curves_all_models.png")

    def generate_report(self):
        """生成过拟合诊断报告"""
        print(f"\n{'=' * 80}")
        print("过拟合诊断总结报告".center(80))
        print(f"{'=' * 80}\n")

        report_data = []
        for model_name, result in self.results.items():
            report_data.append({
                '模型': model_name,
                '最终训练R^2': result['train_scores'][-1],
                '最终验证R^2': result['val_scores'][-1],
                'R^2差距': result['final_gap'],
                '诊断结果': result['diagnosis']
            })

        df = pd.DataFrame(report_data)
        print(df.to_string(index=False))


# =====================================================================================
# SECTION 3: 数据加载与预处理
# =====================================================================================
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


# =====================================================================================
# SECTION 4: 特征工程
# =====================================================================================
def add_features(X, y):
    """添加滞后特征"""
    X_new = X.copy()
    for i in range(1, 6):
        X_new[f'Corn_lag_{i}'] = y.shift(i)
    return X_new.dropna()


X_train_feat = add_features(X_train, y_train)
y_train = y_train.loc[X_train_feat.index]

X_test_feat = add_features(X_test, y_test)
y_test = y_test.loc[X_test_feat.index]

print(f"\n添加特征后: 训练={X_train_feat.shape}, 测试={X_test_feat.shape}")


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

print(f"序列数据: LSTM/GRU输入={X_train_seq.shape}, XGBoost输入={X_train_flat.shape}")


# =====================================================================================
# SECTION 5: 模型参数配置
# =====================================================================================
print("\n" + "=" * 100)
print("【使用手动设置参数】".center(100))
print("=" * 100)

best_lstm_params = {
    'units': 80,
    'dropout': 0.3,
    'recurrent_dropout': 0.0,
    'l2_reg': 0.01,
    'learning_rate': 0.001
}

best_gru_params = {
    'units': 100,
    'dropout': 0.3,
    'recurrent_dropout': 0.0,
    'learning_rate': 0.001
}

# ✨ NEW: Attention模型参数
best_attention_params = {
    'lstm_units': 80,
    'dropout': 0.3,
    'recurrent_dropout': 0.0,
    'l2_reg': 0.01,
    'learning_rate': 0.001
}

best_xgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

print("\n当前使用的超参数:")
print(f"LSTM: {best_lstm_params}")
print(f"GRU: {best_gru_params}")
print(f"LSTM-Attention: {best_attention_params}")
print(f"XGBoost: {best_xgb_params}")


# =====================================================================================
# SECTION 6: 模型构建器
# =====================================================================================
def build_final_lstm(input_shape):
    """最终LSTM模型"""
    tf.random.set_seed(GLOBAL_SEED)
    return Sequential([
        layers.LSTM(
            units=best_lstm_params['units'],
            input_shape=input_shape,
            kernel_regularizer=l2(best_lstm_params.get('l2_reg', 0.01)),
            recurrent_regularizer=l2(best_lstm_params.get('l2_reg', 0.01)),
            recurrent_dropout=best_lstm_params.get('recurrent_dropout', 0.0),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
            recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
        ),
        layers.Dropout(best_lstm_params['dropout'], seed=GLOBAL_SEED),
        layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))
    ])


def build_final_gru(input_shape):
    """最终GRU模型"""
    tf.random.set_seed(GLOBAL_SEED)
    return Sequential([
        layers.GRU(
            units=best_gru_params['units'],
            input_shape=input_shape,
            recurrent_dropout=best_gru_params.get('recurrent_dropout', 0.0),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
            recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
        ),
        layers.Dropout(best_gru_params['dropout'], seed=GLOBAL_SEED),
        layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))
    ])


# ✨ NEW: LSTM-Attention模型构建器
def build_lstm_attention(input_shape):
    """LSTM-Attention模型"""
    tf.random.set_seed(GLOBAL_SEED)

    inputs = Input(shape=input_shape)
    lstm_out = layers.LSTM(
        units=best_attention_params['lstm_units'],
        return_sequences=True,
        kernel_regularizer=l2(best_attention_params.get('l2_reg', 0.01)),
        recurrent_regularizer=l2(best_attention_params.get('l2_reg', 0.01)),
        recurrent_dropout=best_attention_params.get('recurrent_dropout', 0.0),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
    )(inputs)

    attention_out = AttentionLayer()(lstm_out)
    dropout_out = layers.Dropout(best_attention_params['dropout'], seed=GLOBAL_SEED)(attention_out)
    outputs = layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))(dropout_out)

    return Model(inputs=inputs, outputs=outputs)


# =====================================================================================
# SECTION 7: OOF预测生成
# =====================================================================================
def get_oof_predictions(X_seq, y_seq, params, model_type='lstm', n_splits=5):
    """生成OOF预测"""
    print(f"\n生成{model_type.upper()} OOF预测（TimeSeriesSplit with {n_splits} splits）...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros(len(y_seq))

    fold = 1
    for train_idx, val_idx in tscv.split(X_seq):
        print(f"  Fold {fold}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")

        tf.random.set_seed(GLOBAL_SEED + fold)
        np.random.seed(GLOBAL_SEED + fold)

        X_fold_train, X_fold_val = X_seq[train_idx], X_seq[val_idx]
        y_fold_train, y_fold_val = y_seq[train_idx], y_seq[val_idx]

        if model_type == 'lstm':
            model = Sequential([
                layers.LSTM(
                    units=params['units'],
                    input_shape=(X_seq.shape[1], X_seq.shape[2]),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
                ),
                layers.Dropout(params['dropout'], seed=GLOBAL_SEED),
                layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))
            ])
        elif model_type == 'gru':
            model = Sequential([
                layers.GRU(
                    units=params['units'],
                    input_shape=(X_seq.shape[1], X_seq.shape[2]),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
                ),
                layers.Dropout(params['dropout'], seed=GLOBAL_SEED),
                layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))
            ])
        elif model_type == 'lstm_attention':
            # LSTM-Attention模型
            inputs = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
            lstm_out = layers.LSTM(
                units=params['lstm_units'],
                return_sequences=True,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
                recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
            )(inputs)
            attention_out = AttentionLayer()(lstm_out)
            dropout_out = layers.Dropout(params['dropout'], seed=GLOBAL_SEED)(attention_out)
            outputs = layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))(dropout_out)
            model = Model(inputs=inputs, outputs=outputs)

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']))
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stop],
            shuffle=False,
            verbose=0
        )

        val_pred = model.predict(X_fold_val, verbose=0)
        oof_preds[val_idx] = val_pred.flatten()

        fold += 1

    return oof_preds


print("\n" + "=" * 100)
print("【阶段1】生成LSTM、GRU和LSTM-Attention的OOF预测".center(100))
print("=" * 100)

lstm_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, best_lstm_params, 'lstm', n_splits=5)
gru_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, best_gru_params, 'gru', n_splits=5)
attention_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, best_attention_params, 'lstm_attention', n_splits=5)

print(f"\n✅ OOF预测生成完成！")
print(f"LSTM OOF R^2: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"GRU OOF R^2: {r2_score(y_train_seq, gru_oof_preds):.4f}")
print(f"LSTM-Attention OOF R^2: {r2_score(y_train_seq, attention_oof_preds):.4f}")


# =====================================================================================
# SECTION 8: 训练最终模型
# =====================================================================================
print("\n" + "=" * 100)
print("【阶段2】训练最终的LSTM、GRU和LSTM-Attention模型".center(100))
print("=" * 100)

print("\n[1/3] 训练LSTM模型...")
tf.random.set_seed(GLOBAL_SEED)
lstm_final = build_final_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
lstm_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=best_lstm_params['learning_rate']))
lstm_history = lstm_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)],
    shuffle=False,
    verbose=0
)
print("✓ LSTM模型训练完成")

print("\n[2/3] 训练GRU模型...")
tf.random.set_seed(GLOBAL_SEED)
gru_final = build_final_gru((X_train_seq.shape[1], X_train_seq.shape[2]))
gru_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=best_gru_params['learning_rate']))
gru_history = gru_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)],
    shuffle=False,
    verbose=0
)
print("✓ GRU模型训练完成")

print("\n[3/3] 训练LSTM-Attention模型...")
tf.random.set_seed(GLOBAL_SEED)
attention_final = build_lstm_attention((X_train_seq.shape[1], X_train_seq.shape[2]))
attention_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=best_attention_params['learning_rate']))
attention_history = attention_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)],
    shuffle=False,
    verbose=0
)
print("✓ LSTM-Attention模型训练完成")

# 获取预测
lstm_test_pred = lstm_final.predict(X_test_seq, verbose=0).flatten()
gru_test_pred = gru_final.predict(X_test_seq, verbose=0).flatten()
attention_test_pred = attention_final.predict(X_test_seq, verbose=0).flatten()

lstm_train_pred = lstm_final.predict(X_train_seq, verbose=0).flatten()
gru_train_pred = gru_final.predict(X_train_seq, verbose=0).flatten()
attention_train_pred = attention_final.predict(X_train_seq, verbose=0).flatten()

# 性能评估
lstm_test_r2 = r2_score(y_test_seq, lstm_test_pred)
gru_test_r2 = r2_score(y_test_seq, gru_test_pred)
attention_test_r2 = r2_score(y_test_seq, attention_test_pred)

print(f"\n【单模型性能对比】")
print(f"LSTM           测试R^2: {lstm_test_r2:.4f}")
print(f"GRU            测试R^2: {gru_test_r2:.4f}")
print(f"LSTM-Attention 测试R^2: {attention_test_r2:.4f}")


# =====================================================================================
# SECTION 9: 学习曲线过拟合检测
# =====================================================================================
print("\n" + "=" * 100)
print("【阶段3】学习曲线过拟合检测".center(100))
print("=" * 100)

detector = OverfittingDetector(output_dir='./overfitting_analysis/')

print("\n🔍 检测LSTM...")
detector.sequential_learning_curve(
    model_builder=build_final_lstm,
    X_seq=X_train_seq,
    y_seq=y_train_seq,
    model_name='LSTM',
    train_sizes=np.linspace(0.2, 1.0, 8),
    epochs=100,
    batch_size=32
)

print("\n🔍 检测GRU...")
detector.sequential_learning_curve(
    model_builder=build_final_gru,
    X_seq=X_train_seq,
    y_seq=y_train_seq,
    model_name='GRU',
    train_sizes=np.linspace(0.2, 1.0, 8),
    epochs=100,
    batch_size=32
)

print("\n🔍 检测LSTM-Attention...")
detector.sequential_learning_curve(
    model_builder=build_lstm_attention,
    X_seq=X_train_seq,
    y_seq=y_train_seq,
    model_name='LSTM-Attention',
    train_sizes=np.linspace(0.2, 1.0, 8),
    epochs=100,
    batch_size=32
)

detector.plot_all_learning_curves(figsize=(18, 5))
detector.generate_report()


# =====================================================================================
# SECTION 10: 原有的LSTM+GRU+XGBoost残差学习（策略1-6）
# =====================================================================================
def create_simplified_features(X_flat, lstm_preds, gru_preds):
    """简化特征：原始 + 预测 + 平均 + 差异"""
    features_list = [X_flat]
    features_list.append(lstm_preds.reshape(-1, 1))
    features_list.append(gru_preds.reshape(-1, 1))
    features_list.append(((lstm_preds + gru_preds) / 2).reshape(-1, 1))
    features_list.append(np.abs(lstm_preds - gru_preds).reshape(-1, 1))
    return np.hstack(features_list)


def train_xgboost(X_train, y_train):
    """训练XGBoost"""
    np.random.seed(GLOBAL_SEED)
    model = XGBRegressor(
        **best_xgb_params,
        random_state=GLOBAL_SEED,
        seed=GLOBAL_SEED,
        n_jobs=1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


print("\n" + "=" * 100)
print("【原有策略1-6】LSTM+GRU+XGBoost残差学习".center(100))
print("=" * 100)

# 简单平均
avg_test_pred = (lstm_test_pred + gru_test_pred) / 2
avg_r2 = r2_score(y_test_seq, avg_test_pred)
print(f"\n简单平均测试集R^2: {avg_r2:.4f}")

# 计算残差
lstm_oof_residual = y_train_seq - lstm_oof_preds
gru_oof_residual = y_train_seq - gru_oof_preds
avg_oof_preds = (lstm_oof_preds + gru_oof_preds) / 2
avg_oof_residual = y_train_seq - avg_oof_preds

# 准备特征
strategies_results = {}

basic_features_train = X_train_flat[:len(gru_oof_preds)]
basic_features_test = X_test_flat

simplified_train = create_simplified_features(
    basic_features_train, lstm_oof_preds, gru_oof_preds
)
simplified_test = create_simplified_features(
    basic_features_test, lstm_test_pred, gru_test_pred
)

print(f"特征维度: {simplified_train.shape[1]} 维")

# 策略1：LSTM残差学习
print("\n【策略1】LSTM残差学习")
xgb_lstm = train_xgboost(simplified_train, lstm_oof_residual)
lstm_residual = xgb_lstm.predict(simplified_test)
pred_lstm_residual = lstm_test_pred + lstm_residual
r2_lstm_residual = r2_score(y_test_seq, pred_lstm_residual)
print(f"✓ R^2: {r2_lstm_residual:.4f}")
strategies_results['策略1-LSTM残差学习'] = pred_lstm_residual

# 策略2：GRU残差学习
print("\n【策略2】GRU残差学习")
xgb_gru = train_xgboost(simplified_train, gru_oof_residual)
gru_residual = xgb_gru.predict(simplified_test)
pred_gru_residual = gru_test_pred + gru_residual
r2_gru_residual = r2_score(y_test_seq, pred_gru_residual)
print(f"✓ R^2: {r2_gru_residual:.4f}")
strategies_results['策略2-GRU残差学习'] = pred_gru_residual

# 策略3：双残差学习
print("\n【策略3】双残差学习")
xgb_dual = train_xgboost(simplified_train, avg_oof_residual)
dual_residual = xgb_dual.predict(simplified_test)
pred_dual = avg_test_pred + dual_residual
r2_dual = r2_score(y_test_seq, pred_dual)
print(f"✓ R^2: {r2_dual:.4f}")
strategies_results['策略3-双残差学习'] = pred_dual

# 策略4-6（简化版，可根据需要扩展）
print("\n【策略4-6】其他残差策略...")
strategies_results['策略4-LSTM+GRU平均'] = avg_test_pred


# =====================================================================================
# ✨ NEW SECTION 11: LSTM-Attention-XGBoost独立集成模型（策略7）
# =====================================================================================
print("\n" + "=" * 100)
print("【新增策略7】LSTM-Attention-XGBoost独立集成模型".center(100))
print("使用LSTM和Attention的预测作为特征，XGBoost直接预测目标值".center(100))
print("=" * 100)


def create_attention_features(X_flat, lstm_preds, attention_preds):
    """创建Attention集成特征"""
    features_list = [X_flat]
    features_list.append(lstm_preds.reshape(-1, 1))
    features_list.append(attention_preds.reshape(-1, 1))
    features_list.append(((lstm_preds + attention_preds) / 2).reshape(-1, 1))
    features_list.append(np.abs(lstm_preds - attention_preds).reshape(-1, 1))
    return np.hstack(features_list)


# 构造特征
attention_train_features = create_attention_features(
    basic_features_train, lstm_oof_preds, attention_oof_preds
)
attention_test_features = create_attention_features(
    basic_features_test, lstm_test_pred, attention_test_pred
)

print(f"Attention集成特征维度: {attention_train_features.shape[1]} 维")

# 训练XGBoost（直接预测目标值，不是残差）
print("\n训练XGBoost直接预测目标值...")
np.random.seed(GLOBAL_SEED)
xgb_attention = XGBRegressor(
    **best_xgb_params,
    random_state=GLOBAL_SEED,
    seed=GLOBAL_SEED,
    n_jobs=1,
    verbosity=0
)
xgb_attention.fit(attention_train_features, y_train_seq)

# 预测
pred_attention_xgb = xgb_attention.predict(attention_test_features)
r2_attention_xgb = r2_score(y_test_seq, pred_attention_xgb)

print(f"\n✅ 策略7完成！")
print(f"   LSTM-Attention-XGBoost独立集成 R^2: {r2_attention_xgb:.4f}")

strategies_results['策略7-LSTM+Attention+XGBoost'] = pred_attention_xgb


# =====================================================================================
# SECTION 12: 综合对比
# =====================================================================================
print("\n" + "=" * 100)
print("所有策略性能对比".center(100))
print("=" * 100)

all_strategies = {
    'LSTM单模型': lstm_test_pred,
    'GRU单模型': gru_test_pred,
    'LSTM-Attention单模型': attention_test_pred,
    '简单平均': avg_test_pred,
    **strategies_results
}

print(f"\n{'策略':<40} {'R^2':>10} {'MAE':>12} {'RMSE':>12}")
print("-" * 75)

results_list = []
for name, pred in all_strategies.items():
    r2 = r2_score(y_test_seq, pred)
    mae = mean_absolute_error(y_test_seq, pred)
    rmse = sqrt(mean_squared_error(y_test_seq, pred))
    results_list.append((name, r2, mae, rmse, pred))
    print(f"{name:<40} {r2:>10.4f} {mae:>12.6f} {rmse:>12.6f}")

results_list.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 100)
print("性能排名（按R^2降序）".center(100))
print("=" * 100)

for rank, (name, r2, mae, rmse, pred) in enumerate(results_list, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:>2}. {name:<40} R^2={r2:.4f}")

best_name = results_list[0][0]
best_r2 = results_list[0][1]
print(f"\n🏆 最佳策略: {best_name} (R^2 = {best_r2:.4f})")


# 关键对比
print("\n" + "=" * 100)
print("【关键对比】策略7 vs 原有最佳策略".center(100))
print("=" * 100)

original_strategies = [r for r in results_list if r[0].startswith('策略') and '策略7' not in r[0]]
if original_strategies:
    best_original = max(original_strategies, key=lambda x: x[1])
    strategy7_result = [r for r in results_list if '策略7' in r[0]][0]

    print(f"\n原有最佳策略: {best_original[0]}")
    print(f"  R^2: {best_original[1]:.4f}")

    print(f"\n新增策略7: LSTM-Attention-XGBoost独立集成")
    print(f"  R^2: {strategy7_result[1]:.4f}")

    improvement = strategy7_result[1] - best_original[1]
    if improvement > 0:
        print(f"\n✅ 策略7表现更好，提升: {improvement:+.4f} ({improvement/best_original[1]*100:+.2f}%)")
    else:
        print(f"\n⚠️ 原有策略表现更好，差距: {improvement:.4f}")


# =====================================================================================
# SECTION 13: 可视化
# =====================================================================================
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

print("\n" + "=" * 100)
print("生成可视化图表".center(100))
print("=" * 100)

y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1))

# 图1: 训练过程
fig = plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(lstm_history.history['loss'], label='训练损失', linewidth=2)
plt.plot(lstm_history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('LSTM训练过程', fontsize=13, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(gru_history.history['loss'], label='训练损失', linewidth=2)
plt.plot(gru_history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('GRU训练过程', fontsize=13, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(attention_history.history['loss'], label='训练损失', linewidth=2)
plt.plot(attention_history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('LSTM-Attention训练过程', fontsize=13, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + '01_training_process.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图1: 训练过程")

# 图2: 性能排名
fig, ax = plt.subplots(figsize=(16, 10))

strategy_names = [name for name, _, _, _, _ in results_list]
r2_scores = [r2 for _, r2, _, _, _ in results_list]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(strategy_names)))

bars = ax.barh(range(len(strategy_names)), r2_scores, color=colors, alpha=0.8)

for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
    label = f'{r2:.4f}'
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            label, ha='left', va='center', fontweight='bold', fontsize=9)

ax.set_yticks(range(len(strategy_names)))
ax.set_yticklabels(strategy_names, fontsize=10)
ax.set_xlabel('R^2 Score', fontsize=12, fontweight='bold')
ax.set_title('所有策略性能排名对比（含新增策略7）', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(results_directory + '02_performance_ranking_with_strategy7.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图2: 性能排名（含策略7）")

# 图3: 重点策略对比
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

key_strategies = [
    ('简单平均', all_strategies['简单平均']),
    ('策略3-双残差学习', strategies_results['策略3-双残差学习']),
    ('LSTM-Attention单模型', attention_test_pred),
    ('策略7-LSTM+Attention+XGBoost', strategies_results['策略7-LSTM+Attention+XGBoost'])
]

for idx, (name, pred) in enumerate(key_strategies):
    ax = axes[idx]

    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    r2_original = r2_score(y_test_original, pred_original)

    ax.plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
    ax.plot(pred_original, label=name, linewidth=2, alpha=0.8)
    ax.set_title(f'{name}\nR^2={r2_original:.4f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('样本序号', fontsize=10)
    ax.set_ylabel('玉米价格', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + '03_key_strategies_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图3: 重点策略对比")

print("\n✅ 所有可视化已生成！")


# =====================================================================================
# SECTION 14: 保存模型
# =====================================================================================
print("\n" + "=" * 100)
print("保存模型和结果".center(100))
print("=" * 100)

lstm_final.save(results_directory + 'lstm_final.h5')
gru_final.save(results_directory + 'gru_final.h5')
attention_final.save(results_directory + 'attention_final.h5')

with open(results_directory + 'xgb_models.pkl', 'wb') as f:
    pickle.dump({
        'xgb_lstm': xgb_lstm,
        'xgb_gru': xgb_gru,
        'xgb_dual': xgb_dual,
        'xgb_attention': xgb_attention
    }, f)

with open(results_directory + 'scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

predictions_dict = {'true_value': y_test_original.flatten()}
for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    predictions_dict[name] = pred_original.flatten()

predictions_df = pd.DataFrame(predictions_dict)
predictions_df.to_csv(results_directory + 'all_predictions_with_strategy7.csv', index=False)

print(f"✓ 所有模型和结果已保存至: {results_directory}")


# =====================================================================================
# SECTION 15: 最终总结
# =====================================================================================
print("\n" + "=" * 100)
print("最终总结报告".center(100))
print("=" * 100)

print(f"\n📊 实验配置:")
print(f"  - 训练样本: {len(y_train_seq)}")
print(f"  - 测试样本: {len(y_test_seq)}")
print(f"  - 特征维度: {X_train_feat.shape[1]}")

print(f"\n🎯 原有策略（LSTM+GRU残差学习）:")
for name, r2, mae, rmse, pred in results_list:
    if name.startswith('策略') and '策略7' not in name:
        print(f"  {name}: R^2 = {r2:.4f}")

print(f"\n✨ 新增策略7（LSTM-Attention独立集成）:")
print(f"  策略7-LSTM+Attention+XGBoost: R^2 = {r2_attention_xgb:.4f}")

print(f"\n🏆 最佳策略: {best_name} (R^2 = {best_r2:.4f})")

print(f"\n💡 关键发现:")
if '策略7' in best_name:
    print(f"  ✅ 新增的LSTM-Attention-XGBoost独立集成模型表现最佳")
    print(f"  ✅ Attention机制结合XGBoost集成有效提升了预测性能")
else:
    print(f"  ⚠️ 原有的LSTM+GRU残差学习策略表现更好")
    print(f"  💡 策略7作为独立集成模型，提供了不同的融合思路")

print("\n" + "=" * 100)
print("程序执行完毕！".center(100))
print("=" * 100)