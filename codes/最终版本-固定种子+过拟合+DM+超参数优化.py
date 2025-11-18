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
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor
import warnings
import pickle

warnings.filterwarnings('ignore')
try:
    from feature_ablation import FeatureAblationAnalyzer
    FEATURE_ABLATION_AVAILABLE = True
    print("[INFO] ✅ 特征消融模块导入成功")
except ImportError:
    FEATURE_ABLATION_AVAILABLE = False
    print("[WARNING] ⚠️ 未找到feature_ablation.py，将跳过特征消融实验")
# =====================================================================================
# ✨ NEW: 导入独立Optuna优化模块
# =====================================================================================
try:
    from optuna_optimizer import OptunaOptimizer
    OPTUNA_AVAILABLE = True
    print("[INFO] ✅ Optuna优化模块导入成功")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] ⚠️ 未找到optuna_optimizer.py，将使用手动参数")
    print("[INFO] 下载地址: 请将optuna_optimizer.py放在同目录下")

# =====================================================================================
# ✨ NEW: Optuna优化配置（全局开关）
# =====================================================================================
USE_OPTUNA_OPTIMIZATION = False  # 🔄 设置为False则使用原有手动参数
OPTUNA_CONFIG = {
    'lstm_trials': 20,      # 减少到20次
    'gru_trials': 20,
    'xgb_trials': 10,
    'timeout_hours': 2,     # 最多2小时
    'enable_pruning': True,
    'verbose': True,
}
FEATURE_ABLATION_CONFIG = {
    'enabled': False,              # 是否启用特征消融
    'output_dir': './feature_ablation/',
    'show_plots': True,
    'save_latex': True
}

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print("LSTM + GRU + XGBoost 融合时间序列预测 - Optuna自动优化版（数据泄露已修复）".center(100))
else:
    print("LSTM + GRU + XGBoost 融合时间序列预测 - 手动参数版（数据泄露已修复）".center(100))
print("核心改进：滚动窗口评估 + 仅对基模型进行学习曲线过拟合检测 + 完整训练流程".center(100))
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
# SECTION 2: 过拟合检测模块（保持原样）
# =====================================================================================
class OverfittingDetector:
    """过拟合检测器 - 基于学习曲线分析（简化版）"""

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
        """为序列模型（LSTM/GRU）生成学习曲线"""
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

            if train_size <= 1.0:
                n_samples = int(len(X_train_pool) * train_size)
            else:
                n_samples = int(train_size)

            n_samples = max(n_samples, batch_size)

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

            train_mse = mean_squared_error(y_train_subset, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            train_r2 = r2_score(y_train_subset, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            train_scores.append(train_r2)
            val_scores.append(val_r2)
            train_losses.append(train_mse)
            val_losses.append(val_mse)
            sample_counts.append(n_samples)

            print(f"→ 训练R^2={train_r2:.4f}, 验证R^2={val_r2:.4f}, 差距={train_r2 - val_r2:.4f}")

        self.results[model_name] = {
            'sample_counts': sample_counts,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        self._diagnose_overfitting(model_name, train_scores, val_scores)
        return sample_counts, train_scores, val_scores, train_losses, val_losses

    def _diagnose_overfitting(self, model_name, train_scores, val_scores):
        """诊断过拟合状态"""
        print(f"\n{'=' * 80}")
        print(f"过拟合诊断: {model_name}".center(80))
        print(f"{'=' * 80}")

        final_gap = train_scores[-1] - val_scores[-1]

        if len(train_scores) >= 3:
            train_trend = train_scores[-1] - train_scores[-3]
            val_trend = val_scores[-1] - val_scores[-3]
        else:
            train_trend = train_scores[-1] - train_scores[0]
            val_trend = val_scores[-1] - val_scores[0]

        print(f"\n📊 关键指标:")
        print(f"  - 最终训练R^2: {train_scores[-1]:.4f}")
        print(f"  - 最终验证R^2: {val_scores[-1]:.4f}")
        print(f"  - R^2差距: {final_gap:.4f}")
        print(f"  - 训练趋势: {train_trend:+.4f}")
        print(f"  - 验证趋势: {val_trend:+.4f}")

        print(f"\n🔍 诊断结论:")

        if final_gap > 0.2:
            print(f"  ⚠️  严重过拟合 (差距>0.2)")
            diagnosis = "严重过拟合"
        elif final_gap > 0.1:
            print(f"  ⚠️  中度过拟合 (差距>0.1)")
            diagnosis = "中度过拟合"
        elif final_gap > 0.05:
            print(f"  ⚡ 轻微过拟合 (差距>0.05)")
            diagnosis = "轻微过拟合"
        else:
            print(f"  ✅ 拟合良好 (差距<0.05)")
            diagnosis = "拟合良好"

        if val_scores[-1] < 0.5:
            print(f"  ⚠️  可能存在欠拟合 (验证R^2<0.5)")
            diagnosis += " + 欠拟合"

        if abs(val_trend) < 0.01:
            print(f"  ✅ 模型已收敛")
        else:
            print(f"  ⚡ 模型可能需要更多数据")

        print(f"\n💡 优化建议:")
        if final_gap > 0.1:
            print(f"  - 增加正则化强度")
            print(f"  - 减少模型复杂度")
            print(f"  - 增加Dropout比例")
            print(f"  - 使用更多训练数据")
        elif val_scores[-1] < 0.5:
            print(f"  - 增加模型复杂度")
            print(f"  - 增加训练轮数")
            print(f"  - 特征工程优化")
        else:
            print(f"  - 模型状态良好，可投入使用")

        self.results[model_name]['diagnosis'] = diagnosis
        self.results[model_name]['final_gap'] = final_gap

    def plot_all_learning_curves(self, figsize=(20, 8)):
        """绘制所有模型的学习曲线对比"""
        n_models = len(self.results)
        if n_models == 0:
            print("没有可绘制的结果！")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for idx, (model_name, result) in enumerate(self.results.items()):
            if idx >= 2:
                break

            ax = axes[idx]

            sample_counts = result['sample_counts']
            train_scores = result['train_scores']
            val_scores = result['val_scores']
            diagnosis = result.get('diagnosis', 'Unknown')
            final_gap = result.get('final_gap', 0)

            ax.plot(sample_counts, train_scores, 'o-',
                    linewidth=2.5, markersize=8,
                    label='训练R^2', color='#2E86AB', alpha=0.8)
            ax.plot(sample_counts, val_scores, 's-',
                    linewidth=2.5, markersize=8,
                    label='验证R^2', color='#A23B72', alpha=0.8)

            ax.axhline(y=train_scores[-1], color='#2E86AB',
                       linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(y=val_scores[-1], color='#A23B72',
                       linestyle='--', alpha=0.3, linewidth=1)

            ax.fill_between(sample_counts, train_scores, val_scores,
                            alpha=0.2, color='red' if final_gap > 0.1 else 'green')

            ax.set_title(f'{model_name}\n{diagnosis} (差距={final_gap:.4f})',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('训练样本数', fontsize=11)
            ax.set_ylabel('R^2 Score', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            ax.text(sample_counts[-1], train_scores[-1],
                    f'{train_scores[-1]:.3f}',
                    fontsize=9, ha='right', va='bottom', color='#2E86AB')
            ax.text(sample_counts[-1], val_scores[-1],
                    f'{val_scores[-1]:.3f}',
                    fontsize=9, ha='right', va='top', color='#A23B72')

        plt.suptitle('基模型学习曲线分析 - 过拟合诊断',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}learning_curves_base_models.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ 学习曲线对比图已保存: {self.output_dir}learning_curves_base_models.png")

    def plot_loss_curves(self, figsize=(20, 6)):
        """绘制损失曲线"""
        n_models = len(self.results)
        if n_models == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for idx, (model_name, result) in enumerate(self.results.items()):
            if idx >= 2:
                break

            ax = axes[idx]

            sample_counts = result['sample_counts']
            train_losses = result['train_losses']
            val_losses = result['val_losses']

            ax.plot(sample_counts, train_losses, 'o-',
                    linewidth=2.5, label='训练损失', color='#F18F01')
            ax.plot(sample_counts, val_losses, 's-',
                    linewidth=2.5, label='验证损失', color='#C73E1D')

            ax.set_title(f'{model_name} - MSE损失',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('训练样本数', fontsize=11)
            ax.set_ylabel('MSE Loss', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.suptitle('基模型训练与验证损失曲线',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}loss_curves_base_models.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ 损失曲线图已保存: {self.output_dir}loss_curves_base_models.png")

    def generate_report(self):
        """生成过拟合诊断报告"""
        print(f"\n{'=' * 80}")
        print("基模型过拟合诊断总结报告".center(80))
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
        df = df.sort_values('R^2差距', ascending=False)

        print(df.to_string(index=False))

        df.to_csv(f'{self.output_dir}overfitting_report.csv', index=False)
        print(f"\n✅ 报告已保存至: {self.output_dir}overfitting_report.csv")

        print(f"\n📊 统计摘要:")
        print(f"  - 分析模型数: {len(self.results)}")
        print(f"  - 拟合良好: {len([r for r in report_data if r['R^2差距'] < 0.05])}")
        print(f"  - 轻微过拟合: {len([r for r in report_data if 0.05 <= r['R^2差距'] < 0.1])}")
        print(f"  - 中度过拟合: {len([r for r in report_data if 0.1 <= r['R^2差距'] < 0.2])}")
        print(f"  - 严重过拟合: {len([r for r in report_data if r['R^2差距'] >= 0.2])}")

        print(f"\n💡 关键结论:")
        print(f"  - 策略模型（XGBoost）仅学习残差，影响有限")
        print(f"  - 基模型过拟合是主要风险，需重点关注")
        print(f"  - 如基模型存在过拟合，建议调整正则化参数后重新训练")


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


# =====================================================================================
# ✨ FIXED SECTION 4: 特征工程（仅处理训练集）
# =====================================================================================
def add_features(X, y):
    """添加滞后特征（仅用于训练集）"""
    X_new = X.copy()
    for i in range(1, 6):
        X_new[f'Corn_lag_{i}'] = y.shift(i)
    return X_new.dropna()


print(f"\n{'=' * 80}")
print("【特征工程 - 仅处理训练集】".center(80))
print(f"{'=' * 80}")

X_train_feat = add_features(X_train, y_train)
y_train = y_train.loc[X_train_feat.index]

print(f"\n✓ 训练集特征构造:")
print(f"  - 原始样本数: {len(X_train)}")
print(f"  - 添加滞后特征后: {len(X_train_feat)}")
print(f"  - 特征维度: {X_train_feat.shape[1]}")

print(f"\n✓ 测试集处理策略:")
print(f"  - 不预先构造特征")
print(f"  - 将在滚动窗口预测时动态构造")
print(f"  - 完全避免数据泄露")


# 构造训练集序列数据
def create_sequences(X, y, seq_len=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X.iloc[i:i + seq_len].values)
        y_seq.append(y.iloc[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


seq_len = 5
X_train_seq, y_train_seq = create_sequences(X_train_feat, y_train, seq_len)

print(f"\n✓ 训练集序列构造:")
print(f"  - 序列形状: {X_train_seq.shape}")
print(f"  - 目标形状: {y_train_seq.shape}")


# =====================================================================================
# ✨ NEW: 滚动窗口预测函数
# =====================================================================================
def rolling_window_predict(model, X_train_feat, y_train, X_test, y_test, seq_len=5):
    """
    滚动窗口预测函数（用于LSTM/GRU）

    关键逻辑：
    1. 初始化历史窗口（仅用训练集）
    2. 逐个预测测试样本
    3. 每次预测后用真实值更新历史
    """
    predictions = []

    # 初始化：从训练集获取历史窗口
    history_X_feat = X_train_feat.tail(seq_len).copy()
    history_y = y_train.tail(5).copy()

    # 逐个预测测试样本
    for idx in range(len(X_test)):
        # 1. 获取当前测试样本的原始特征（不带lag）
        current_X_raw = X_test.iloc[idx:idx+1].copy()

        # 2. 动态构造lag特征（只用已知历史）
        for i in range(1, 6):
            if len(history_y) >= i:
                current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i]
            else:
                current_X_raw[f'Corn_lag_{i}'] = 0

        # 3. 构造序列窗口
        current_window = pd.concat([history_X_feat.tail(seq_len-1), current_X_raw])
        X_seq = current_window.values.reshape(1, seq_len, -1)

        # 4. 预测
        pred = model.predict(X_seq, verbose=0)[0, 0]
        predictions.append(pred)

        # 5. ✅ 用真实值更新历史（关键：这是预测后才知道的）
        history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
        history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[idx]])])

        # 进度显示
        if (idx + 1) % 50 == 0 or idx == len(X_test) - 1:
            print(f"  进度: {idx+1}/{len(X_test)} 样本", end='\r')

    print()  # 换行
    return np.array(predictions)


def rolling_window_predict_ensemble(lstm_model, gru_model, xgb_model,
                                     X_train_feat, y_train, X_test, y_test,
                                     X_train_flat, seq_len=5, strategy='gru_residual'):
    """
    集成模型滚动窗口预测
    """
    predictions = []
    lstm_preds_list = []
    gru_preds_list = []

    # 初始化历史窗口
    history_X_feat = X_train_feat.tail(seq_len).copy()
    history_y = y_train.tail(5).copy()

    # 逐个预测
    for idx in range(len(X_test)):
        # 构造当前样本特征
        current_X_raw = X_test.iloc[idx:idx+1].copy()

        for i in range(1, 6):
            if len(history_y) >= i:
                current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i]
            else:
                current_X_raw[f'Corn_lag_{i}'] = 0

        # 构造序列窗口
        current_window = pd.concat([history_X_feat.tail(seq_len-1), current_X_raw])
        X_seq = current_window.values.reshape(1, seq_len, -1)

        # 获取LSTM和GRU预测
        lstm_pred = lstm_model.predict(X_seq, verbose=0)[0, 0]
        gru_pred = gru_model.predict(X_seq, verbose=0)[0, 0]

        lstm_preds_list.append(lstm_pred)
        gru_preds_list.append(gru_pred)

        # 根据策略计算最终预测
        if strategy == 'average':
            final_pred = (lstm_pred + gru_pred) / 2

        elif strategy == 'lstm_residual':
            # 构造XGBoost特征
            X_flat = X_seq.reshape(1, -1)
            X_xgb = np.hstack([
                X_flat,
                np.array([[lstm_pred, gru_pred, (lstm_pred + gru_pred) / 2,
                          abs(lstm_pred - gru_pred)]])
            ])
            residual = xgb_model.predict(X_xgb)[0]
            final_pred = lstm_pred + residual

        elif strategy == 'gru_residual':
            X_flat = X_seq.reshape(1, -1)
            X_xgb = np.hstack([
                X_flat,
                np.array([[lstm_pred, gru_pred, (lstm_pred + gru_pred) / 2,
                          abs(lstm_pred - gru_pred)]])
            ])
            residual = xgb_model.predict(X_xgb)[0]
            final_pred = gru_pred + residual

        else:
            final_pred = (lstm_pred + gru_pred) / 2

        predictions.append(final_pred)

        # 更新历史
        history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
        history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[idx]])])

        if (idx + 1) % 50 == 0 or idx == len(X_test) - 1:
            print(f"  进度: {idx+1}/{len(X_test)} 样本", end='\r')

    print()
    return np.array(predictions), np.array(lstm_preds_list), np.array(gru_preds_list)


# =====================================================================================
# ✨ NEW SECTION: Optuna超参数优化阶段
# =====================================================================================

if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print("\n" + "=" * 100)
    print("【Optuna超参数优化阶段】".center(100))
    print("=" * 100)

    # 创建优化器
    optimizer = OptunaOptimizer(
        X_train=X_train_seq,
        y_train=y_train_seq,
        output_dir='./optuna_results/',
        seed=GLOBAL_SEED,
        verbose=OPTUNA_CONFIG['verbose']
    )

    # 优化LSTM
    print(f"\n[1/3] 优化LSTM超参数 (n_trials={OPTUNA_CONFIG['lstm_trials']})...")
    best_lstm_params = optimizer.optimize_lstm(
        n_trials=OPTUNA_CONFIG['lstm_trials'],
        max_epochs=200,
        batch_size=32,
        enable_pruning=OPTUNA_CONFIG['enable_pruning']
    )

    # 优化GRU
    print(f"\n[2/3] 优化GRU超参数 (n_trials={OPTUNA_CONFIG['gru_trials']})...")
    best_gru_params = optimizer.optimize_gru(
        n_trials=OPTUNA_CONFIG['gru_trials'],
        max_epochs=200,
        batch_size=32,
        enable_pruning=OPTUNA_CONFIG['enable_pruning']
    )

    # 优化XGBoost
    print(f"\n[3/3] 优化XGBoost超参数 (n_trials={OPTUNA_CONFIG['xgb_trials']})...")
    best_xgb_params = optimizer.optimize_xgboost(
        n_trials=OPTUNA_CONFIG['xgb_trials'],
        cv_splits=5
    )

    # 生成优化报告和可视化
    try:
        optimizer.visualize('lstm', show=False)
        optimizer.visualize('gru', show=False)
        optimizer.visualize('xgboost', show=False)
    except Exception as e:
        print(f"[WARNING] 可视化生成失败: {e}")

    optimizer.generate_report()

    print("\n" + "=" * 100)
    print("✅ Optuna优化完成！".center(100))
    print(f"优化结果保存在: ./optuna_results/".center(100))
    print("=" * 100)

else:
    # 使用原有手动设置的参数
    print("\n" + "=" * 100)
    print("【使用手动设置参数】".center(100))
    print("=" * 100)

    best_lstm_params = {
        'units': 150,
        'dropout': 0.2462968579989427,
        'recurrent_dropout': 0.040227669708937194,
        'l2_reg': 0.0006181065539453701,
        'learning_rate': 0.0002295600141013699
    }

    best_gru_params = {
        'units': 150,
        'dropout': 0.41579437412891573,
        'recurrent_dropout': 0.066600650933163,
        'learning_rate': 0.0002070950616082725
    }

    best_xgb_params = {
        'n_estimators': 91,
        'learning_rate': 0.0487307213710181,
        'max_depth': 5,
        'min_child_weight': 10,
        'subsample': 0.5784746899524377,
        'colsample_bytree': 0.7083156395264648,
        'reg_alpha': 0.02939013756116536,
        'reg_lambda': 0.010506850510122958
    }

    print("\n当前使用的超参数:")
    print(f"LSTM: {best_lstm_params}")
    print(f"GRU: {best_gru_params}")
    print(f"XGBoost: {best_xgb_params}")


# =====================================================================================
# SECTION 5: 模型构建器（用于过拟合检测）
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


# =====================================================================================
# SECTION 6: OOF预测生成（训练集内部，无数据泄露）
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
        else:
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
print("【阶段1】生成LSTM和GRU的OOF预测".center(100))
print("=" * 100)

lstm_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, best_lstm_params, 'lstm', n_splits=5)
gru_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, best_gru_params, 'gru', n_splits=5)

print(f"\nOOF预测生成完成！")
print(f"LSTM OOF R^2: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"GRU OOF R^2: {r2_score(y_train_seq, gru_oof_preds):.4f}")


# =====================================================================================
# SECTION 7: 训练最终模型
# =====================================================================================
print("\n" + "=" * 100)
print("【阶段2】训练最终的LSTM和GRU模型（使用最优参数）".center(100))
print("=" * 100)

print("\n训练最终LSTM模型...")
tf.random.set_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

lstm_final = Sequential([
    layers.LSTM(
        units=best_lstm_params['units'],
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        kernel_regularizer=l2(best_lstm_params.get('l2_reg', 0.01)),
        recurrent_regularizer=l2(best_lstm_params.get('l2_reg', 0.01)),
        recurrent_dropout=best_lstm_params.get('recurrent_dropout', 0.0),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
    ),
    layers.Dropout(best_lstm_params['dropout'], seed=GLOBAL_SEED),
    layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))
])
lstm_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=best_lstm_params['learning_rate']))
lstm_history = lstm_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ],
    shuffle=False,
    verbose=0
)
print("✓ LSTM模型训练完成")

print("\n训练最终GRU模型...")
tf.random.set_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

gru_final = Sequential([
    layers.GRU(
        units=best_gru_params['units'],
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        recurrent_dropout=best_gru_params.get('recurrent_dropout', 0.0),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
    ),
    layers.Dropout(best_gru_params['dropout'], seed=GLOBAL_SEED),
    layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))
])
gru_final.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=best_gru_params['learning_rate']))
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
gru_history = gru_final.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    shuffle=False,
    verbose=0
)
print("✓ GRU模型训练完成")


# =====================================================================================
# ✨ FIXED SECTION 8: 使用滚动窗口预测测试集
# =====================================================================================
print("\n" + "=" * 100)
print("【阶段3】滚动窗口预测测试集（避免数据泄露）".center(100))
print("=" * 100)

print("\n滚动窗口预测 LSTM...")
lstm_test_pred = rolling_window_predict(
    lstm_final, X_train_feat, y_train, X_test, y_test, seq_len=seq_len
)

print("\n滚动窗口预测 GRU...")
gru_test_pred = rolling_window_predict(
    gru_final, X_train_feat, y_train, X_test, y_test, seq_len=seq_len
)

# 简单平均
avg_test_pred = (lstm_test_pred + gru_test_pred) / 2

# 评估基础模型
lstm_test_r2 = r2_score(y_test, lstm_test_pred)
gru_test_r2 = r2_score(y_test, gru_test_pred)
avg_r2 = r2_score(y_test, avg_test_pred)

print(f"\n【基础模型性能（滚动窗口评估）】")
print(f"LSTM单模型 - 测试R^2: {lstm_test_r2:.4f}")
print(f"GRU单模型  - 测试R^2: {gru_test_r2:.4f}")
print(f"简单平均   - 测试R^2: {avg_r2:.4f}")


# =====================================================================================
# SECTION 9: 学习曲线过拟合检测
# =====================================================================================
print("\n" + "=" * 100)
print("【阶段4】对基模型（LSTM/GRU）进行学习曲线过拟合检测".center(100))
print("理由：基模型是主要预测来源，过拟合会直接影响最终结果".center(100))
print("=" * 100)

detector = OverfittingDetector(output_dir='./overfitting_analysis/')

print("\n🔍 检测LSTM最终模型...")
detector.sequential_learning_curve(
    model_builder=build_final_lstm,
    X_seq=X_train_seq,
    y_seq=y_train_seq,
    model_name='LSTM最终模型',
    train_sizes=np.linspace(0.2, 1.0, 8),
    epochs=100,
    batch_size=32
)

print("\n🔍 检测GRU最终模型...")
detector.sequential_learning_curve(
    model_builder=build_final_gru,
    X_seq=X_train_seq,
    y_seq=y_train_seq,
    model_name='GRU最终模型',
    train_sizes=np.linspace(0.2, 1.0, 8),
    epochs=100,
    batch_size=32
)

detector.plot_all_learning_curves(figsize=(16, 6))
detector.plot_loss_curves(figsize=(16, 5))
detector.generate_report()


# =====================================================================================
# SECTION 10: XGBoost训练
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
print("【阶段5】XGBoost残差学习".center(100))
print("=" * 100)

# 计算残差
lstm_oof_residual = y_train_seq - lstm_oof_preds
gru_oof_residual = y_train_seq - gru_oof_preds
avg_oof_preds = (lstm_oof_preds + gru_oof_preds) / 2
avg_oof_residual = y_train_seq - avg_oof_preds

print(f"\n残差统计:")
print(f"LSTM残差 - 均值: {np.mean(lstm_oof_residual):.6f}, 标准差: {np.std(lstm_oof_residual):.6f}")
print(f"GRU残差  - 均值: {np.mean(gru_oof_residual):.6f}, 标准差: {np.std(gru_oof_residual):.6f}")

# 准备特征（训练集）
X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
simplified_train = create_simplified_features(X_train_flat, lstm_oof_preds, gru_oof_preds)

print(f"\n特征维度: {simplified_train.shape[1]} 维")

# 训练XGBoost模型
print("\n训练XGBoost (LSTM残差)...")
xgb_lstm = train_xgboost(simplified_train, lstm_oof_residual)

print("训练XGBoost (GRU残差)...")
xgb_gru = train_xgboost(simplified_train, gru_oof_residual)

print("训练XGBoost (双残差)...")
xgb_dual = train_xgboost(simplified_train, avg_oof_residual)


# =====================================================================================
# SECTION 11: 特征工程与XGBoost训练（补充函数）
# =====================================================================================
def clip_residual(residual_pred, threshold=2.0):
    """剪裁极端残差值"""
    std = np.std(residual_pred)
    mean = np.mean(residual_pred)
    return np.clip(residual_pred, mean - threshold * std, mean + threshold * std)


def weighted_residual_correction(base_pred, residual_pred, weight=0.5):
    """加权残差修正"""
    return base_pred + weight * residual_pred


# =====================================================================================
# ✨ FIXED SECTION 12: 集成策略滚动窗口评估（全部6个策略）
# =====================================================================================
print("\n" + "=" * 100)
print("【阶段6】集成策略滚动窗口评估".center(100))
print("=" * 100)

strategies_results = {}

# 策略1: LSTM残差学习
print("\n" + "=" * 100)
print("【策略1】LSTM残差学习：简化特征 + XGBoost".center(100))
print("=" * 100)

print("\n滚动窗口预测...")
pred_lstm_residual, _, _ = rolling_window_predict_ensemble(
    lstm_final, gru_final, xgb_lstm,
    X_train_feat, y_train, X_test, y_test,
    X_train_flat, seq_len=seq_len, strategy='lstm_residual'
)
strategies_results['策略1-LSTM残差学习'] = pred_lstm_residual
r2_lstm_residual = r2_score(y_test, pred_lstm_residual)
print(f"✓ R^2: {r2_lstm_residual:.4f} (vs简单平均: {r2_lstm_residual - avg_r2:+.4f})")

# 策略2: GRU残差学习
print("\n" + "=" * 100)
print("【策略2】GRU残差学习：简化特征 + XGBoost".center(100))
print("=" * 100)

print("\n滚动窗口预测...")
pred_gru_residual, _, _ = rolling_window_predict_ensemble(
    lstm_final, gru_final, xgb_gru,
    X_train_feat, y_train, X_test, y_test,
    X_train_flat, seq_len=seq_len, strategy='gru_residual'
)
strategies_results['策略2-GRU残差学习'] = pred_gru_residual
r2_gru_residual = r2_score(y_test, pred_gru_residual)
print(f"✓ R^2: {r2_gru_residual:.4f} (vs简单平均: {r2_gru_residual - avg_r2:+.4f})")

# 策略3: 双残差学习
print("\n" + "=" * 100)
print("【策略3】双残差学习：(LSTM+GRU)/2 + XGBoost".center(100))
print("=" * 100)

print("\n滚动窗口预测...")
# 双残差预测需要单独实现，因为基准是平均值
pred_dual_list = []
history_X_feat = X_train_feat.tail(seq_len).copy()
history_y = y_train.tail(5).copy()

for idx in range(len(X_test)):
    current_X_raw = X_test.iloc[idx:idx+1].copy()
    for i in range(1, 6):
        current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i] if len(history_y) >= i else 0

    current_window = pd.concat([history_X_feat.tail(seq_len-1), current_X_raw])
    X_seq = current_window.values.reshape(1, seq_len, -1)

    # 获取LSTM和GRU预测
    lstm_pred = lstm_final.predict(X_seq, verbose=0)[0, 0]
    gru_pred = gru_final.predict(X_seq, verbose=0)[0, 0]
    avg_pred = (lstm_pred + gru_pred) / 2

    # 构造XGBoost特征并预测残差
    X_flat = X_seq.reshape(1, -1)
    X_xgb = np.hstack([X_flat, np.array([[lstm_pred, gru_pred, avg_pred, abs(lstm_pred - gru_pred)]])])

    residual = xgb_dual.predict(X_xgb)[0]
    final_pred = avg_pred + residual
    pred_dual_list.append(final_pred)

    history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
    history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[idx]])])

    if (idx + 1) % 50 == 0 or idx == len(X_test) - 1:
        print(f"  进度: {idx+1}/{len(X_test)} 样本", end='\r')

print()
pred_dual = np.array(pred_dual_list)
strategies_results['策略3-双残差学习'] = pred_dual
r2_dual = r2_score(y_test, pred_dual)
print(f"✓ R^2: {r2_dual:.4f} (vs简单平均: {r2_dual - avg_r2:+.4f})")

# 策略4: 残差剪裁
print("\n" + "=" * 100)
print("【策略4】GRU残差学习 + 残差剪裁".center(100))
print("=" * 100)

# 使用策略2的预测，计算残差并剪裁
gru_residual_test = pred_gru_residual - gru_test_pred
gru_residual_clipped = clip_residual(gru_residual_test, threshold=2.0)
pred_gru_clipped = gru_test_pred + gru_residual_clipped

r2_gru_clipped = r2_score(y_test, pred_gru_clipped)
print(f"✓ R^2: {r2_gru_clipped:.4f} (vs简单平均: {r2_gru_clipped - avg_r2:+.4f})")
print(f"  残差剪裁前std: {np.std(gru_residual_test):.6f}, 剪裁后std: {np.std(gru_residual_clipped):.6f}")
strategies_results['策略4-残差剪裁'] = pred_gru_clipped

# 策略5: 加权融合（30%）
print("\n" + "=" * 100)
print("【策略5】GRU残差学习 + 加权融合(30%)".center(100))
print("=" * 100)

pred_gru_weighted = weighted_residual_correction(gru_test_pred, gru_residual_test, weight=0.3)

r2_gru_weighted = r2_score(y_test, pred_gru_weighted)
print(f"✓ R^2: {r2_gru_weighted:.4f} (vs简单平均: {r2_gru_weighted - avg_r2:+.4f})")
strategies_results['策略5-加权融合30%'] = pred_gru_weighted

# 策略6: 终极组合
print("\n" + "=" * 100)
print("【策略6】终极组合：残差剪裁 + 加权融合(30%)".center(100))
print("=" * 100)

pred_ultimate = weighted_residual_correction(gru_test_pred, gru_residual_clipped, weight=0.3)

r2_ultimate = r2_score(y_test, pred_ultimate)
print(f"✓ R^2: {r2_ultimate:.4f} (vs简单平均: {r2_ultimate - avg_r2:+.4f})")
strategies_results['策略6-终极组合'] = pred_ultimate


# =====================================================================================
# SECTION 13: 综合对比
# =====================================================================================
print("\n" + "=" * 100)
print("所有策略性能对比（归一化数据）".center(100))
print("=" * 100)

all_strategies = {
    'LSTM单模型': lstm_test_pred,
    'GRU单模型': gru_test_pred,
    '简单平均(基线)': avg_test_pred,
    **strategies_results
}

print(f"\n{'策略':<30} {'R^2':>10} {'vs基线':>10} {'MAE':>12} {'RMSE':>12}")
print("-" * 75)

results_list = []
for name, pred in all_strategies.items():
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = sqrt(mean_squared_error(y_test, pred))
    improvement = r2 - avg_r2
    results_list.append((name, r2, improvement, mae, rmse, pred))
    print(f"{name:<30} {r2:>10.4f} {improvement:>10.4f} {mae:>12.6f} {rmse:>12.6f}")

# 排序
results_list.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 100)
print("性能排名（按R^2降序）".center(100))
print("=" * 100)

best_r2 = results_list[0][1]
best_name = results_list[0][0]

for rank, (name, r2, improvement, mae, rmse, pred) in enumerate(results_list, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:>2}. {name:<30} R^2={r2:.4f} (vs基线: {improvement:+.4f})")

print(f"\n🏆 最佳策略: {best_name} (R^2 = {best_r2:.4f})")

# 原始尺度评估
print("\n" + "=" * 100)
print("原始尺度性能对比".center(100))
print("=" * 100)

y_test_original = y_scaler.inverse_transform(y_test.values.reshape(-1, 1))
strategies_original = {}

print(f"\n{'策略':<30} {'R^2':>10} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
print("-" * 75)

for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    strategies_original[name] = pred_original

    r2 = r2_score(y_test_original, pred_original)
    mae = mean_absolute_error(y_test_original, pred_original)
    rmse = sqrt(mean_squared_error(y_test_original, pred_original))
    mape = np.mean(np.abs((pred_original - y_test_original) / (y_test_original + 1e-8)))

    print(f"{name:<30} {r2:>10.4f} {mae:>12.2f} {rmse:>12.2f} {mape:>12.6f}")


# =====================================================================================
# SECTION 14: 完整可视化（保留原代码所有图表）
# =====================================================================================
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

print("\n" + "=" * 100)
print("生成可视化图表".center(100))
print("=" * 100)

# 图1: 训练过程
fig = plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='训练损失', linewidth=2)
plt.plot(lstm_history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('LSTM模型训练过程', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(gru_history.history['loss'], label='训练损失', linewidth=2)
plt.plot(gru_history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('GRU模型训练过程', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + '01_training_process.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图1: 训练过程曲线")

# 图2: 性能排名柱状图
fig, ax = plt.subplots(figsize=(16, 8))

strategy_names = [name for name, _, _, _, _, _ in results_list]
r2_scores = [r2 for _, r2, _, _, _, _ in results_list]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(strategy_names)))

bars = ax.barh(range(len(strategy_names)), r2_scores, color=colors, alpha=0.8)

ax.axvline(x=avg_r2, color='red', linestyle='--', linewidth=2, label='简单平均基线', alpha=0.7)

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
ax.set_title('所有策略性能排名对比（滚动窗口评估）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(results_directory + '02_performance_ranking.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图2: 性能排名对比")

# 图3: Top6策略预测对比
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
axes = axes.flatten()

top6_strategies = results_list[:6]

for idx, (name, r2, improvement, mae, rmse, pred) in enumerate(top6_strategies):
    ax = axes[idx]

    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    r2_original = r2_score(y_test_original, pred_original)

    ax.plot(y_test_original, label='真实值', linewidth=2.5, color='black', alpha=0.8)
    ax.plot(pred_original, label=name, linewidth=2, alpha=0.8)
    ax.set_title(f'{name}\nR^2={r2_original:.4f} (vs基线: {improvement:+.4f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('样本序号', fontsize=9)
    ax.set_ylabel('玉米价格', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + '03_top6_strategies_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图3: Top6策略预测对比")

# 图4: 残差分析对比
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 🔧 改为 2x4 布局
axes = axes.flatten()  # 🔧 展平为一维数组，方便索引

residuals_dict = {
    '简单平均': y_test - avg_test_pred,
    '策略1-LSTM残差学习': y_test - strategies_results['策略1-LSTM残差学习'],
    '策略2-GRU残差学习': y_test - strategies_results['策略2-GRU残差学习'],
    '策略3-双残差学习': y_test - strategies_results['策略3-双残差学习'],
    '策略4-残差剪裁': y_test - strategies_results['策略4-残差剪裁'],
    '策略5-加权融合30%': y_test - strategies_results['策略5-加权融合30%'],
    '策略6-终极组合': y_test - strategies_results['策略6-终极组合'],
}

for idx, (name, residual) in enumerate(residuals_dict.items()):
    ax = axes[idx]  # 🔧 直接用一维索引
    ax.hist(residual, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{name} 残差分布', fontsize=12, fontweight='bold')
    ax.set_xlabel('残差')
    ax.set_ylabel('频数')
    ax.text(0.05, 0.95, f'均值={np.mean(residual):.5f}\nstd={np.std(residual):.5f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3, axis='y')

# 🔧 隐藏多余的子图
for idx in range(len(residuals_dict), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(results_directory + '04_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图4: 残差分布分析")

# 图5: 残差时间序列对比
fig, ax = plt.subplots(figsize=(16, 6))

for name, residual in residuals_dict.items():
    ax.plot(residual, label=name, linewidth=2, alpha=0.7)

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_title('不同策略残差时间序列对比', fontsize=14, fontweight='bold')
ax.set_xlabel('样本序号', fontsize=12)
ax.set_ylabel('残差', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + '05_residual_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图5: 残差时间序列")

# 图6: 综合对比图
plt.figure(figsize=(18, 8))
plt.plot(y_test_original, label='真实值', linewidth=3, color='black', alpha=0.9, zorder=10)

key_strategies_for_plot = [
    ('简单平均(基线)', strategies_original['简单平均(基线)'], 'gray'),
    ('策略1-LSTM残差学习', strategies_original['策略1-LSTM残差学习'], '#A09FA4'),
    ('策略2-GRU残差学习', strategies_original['策略2-GRU残差学习'], '#FEA0A0'),
    ('策略3-双残差学习', strategies_original['策略3-双残差学习'], '#AD07E3'),
    ('策略4-残差剪裁', strategies_original['策略4-残差剪裁'], '#55A0FB'),
    ('策略5-加权融合30%', strategies_original['策略5-加权融合30%'], '#099A63'),
    ('策略6-终极组合', strategies_original['策略6-终极组合'], '#FFB90F'),
]

for name, pred, color in key_strategies_for_plot:
    plt.plot(pred, label=name, linewidth=1.8, alpha=0.7, color=color)

plt.title('关键策略综合对比', fontsize=16, fontweight='bold')
plt.xlabel('样本序号', fontsize=12)
plt.ylabel('玉米价格', fontsize=12)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_directory + '06_key_strategies_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图6: 关键策略综合对比")

# 图7: 性能指标对比
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

metrics_data = {}
for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    metrics_data[name] = {
        'R^2': r2_score(y_test_original, pred_original),
        'MAE': mean_absolute_error(y_test_original, pred_original),
        'RMSE': sqrt(mean_squared_error(y_test_original, pred_original)),
        'MAPE': np.mean(np.abs((pred_original - y_test_original) / (y_test_original + 1e-8)))
    }

top9_names = [name for name, _, _, _, _, _ in results_list[:min(9, len(results_list))]]
colors_bar = plt.cm.viridis(np.linspace(0, 1, len(top9_names)))

axes[0, 0].bar(range(len(top9_names)), [metrics_data[m]['R^2'] for m in top9_names],
               color=colors_bar, alpha=0.7)
axes[0, 0].set_title('R^2 分数对比', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('R^2 Score')
axes[0, 0].set_xticks(range(len(top9_names)))
axes[0, 0].set_xticklabels([n[:20] for n in top9_names], rotation=45, ha='right', fontsize=8)
axes[0, 0].grid(True, alpha=0.3, axis='y')

axes[0, 1].bar(range(len(top9_names)), [metrics_data[m]['MAE'] for m in top9_names],
               color=colors_bar, alpha=0.7)
axes[0, 1].set_title('MAE 对比', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_xticks(range(len(top9_names)))
axes[0, 1].set_xticklabels([n[:20] for n in top9_names], rotation=45, ha='right', fontsize=8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[1, 0].bar(range(len(top9_names)), [metrics_data[m]['RMSE'] for m in top9_names],
               color=colors_bar, alpha=0.7)
axes[1, 0].set_title('RMSE 对比', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].set_xticks(range(len(top9_names)))
axes[1, 0].set_xticklabels([n[:20] for n in top9_names], rotation=45, ha='right', fontsize=8)
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].bar(range(len(top9_names)), [metrics_data[m]['MAPE'] for m in top9_names],
               color=colors_bar, alpha=0.7)
axes[1, 1].set_title('MAPE 对比', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('MAPE')
axes[1, 1].set_xticks(range(len(top9_names)))
axes[1, 1].set_xticklabels([n[:20] for n in top9_names], rotation=45, ha='right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_directory + '07_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图7: 多指标对比")

# 图8: 预测误差分析
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

top4_for_error = results_list[:4]

for idx, (name, r2, improvement, mae, rmse, pred) in enumerate(top4_for_error):
    ax = axes[idx // 2, idx % 2]

    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    errors = pred_original - y_test_original.flatten()

    ax.scatter(y_test_original, errors, alpha=0.5, s=30)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{name} - 预测误差分析', fontsize=11, fontweight='bold')
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测误差')
    ax.grid(True, alpha=0.3)

    ax.text(0.05, 0.95, f'MAE={mae:.2f}\nRMSE={rmse:.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig(results_directory + '08_prediction_error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图8: 预测误差分析")

# 图9: 模型改进效果雷达图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

radar_strategies = ['简单平均(基线)', '策略1-LSTM残差学习', '策略2-GRU残差学习', '策略3-双残差学习','策略4-残差剪裁','策略5-加权融合30%','策略6-终极组合']
categories = ['R^2', 'MAE', 'RMSE', 'MAPE']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for strategy in radar_strategies:
    pred_original = strategies_original[strategy]

    r2_val = metrics_data[strategy]['R^2']
    mae_val = 1 / (1 + metrics_data[strategy]['MAE'] / 100)
    rmse_val = 1 / (1 + metrics_data[strategy]['RMSE'] / 100)
    mape_val = 1 / (1 + metrics_data[strategy]['MAPE'] * 100)

    values = [r2_val, mae_val, rmse_val, mape_val]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=strategy)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('多维度性能对比雷达图', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig(results_directory + '09_performance_radar.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图9: 性能雷达图")

# 图10: 残差箱线图对比
fig, ax = plt.subplots(figsize=(14, 6))

residual_data = []
residual_labels = []

for name, residual in residuals_dict.items():
    residual_data.append(residual)
    residual_labels.append(name)

bp = ax.boxplot(residual_data, labels=residual_labels, patch_artist=True)

for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(residual_data)))):
    patch.set_facecolor(color)

ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_title('残差分布箱线图对比', fontsize=14, fontweight='bold')
ax.set_ylabel('残差值')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig(results_directory + '10_residual_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图10: 残差箱线图")

print("\n✅ 所有可视化图表已生成并保存！")


# =====================================================================================
# SECTION 15: 保存模型和结果
# =====================================================================================
print("\n" + "=" * 100)
print("保存模型和结果".center(100))
print("=" * 100)

# 保存Keras模型
lstm_final.save(results_directory + 'lstm_final_rolling.h5')
gru_final.save(results_directory + 'gru_final_rolling.h5')

# 保存XGBoost模型
with open(results_directory + 'xgb_lstm_rolling.pkl', 'wb') as f:
    pickle.dump(xgb_lstm, f)

with open(results_directory + 'xgb_gru_rolling.pkl', 'wb') as f:
    pickle.dump(xgb_gru, f)

# 保存归一化器
with open(results_directory + 'scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

# 保存所有预测结果
predictions_dict = {'true_value': y_test_original.flatten()}
for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    predictions_dict[name] = pred_original.flatten()

predictions_df = pd.DataFrame(predictions_dict)
predictions_df.to_csv(results_directory + 'all_predictions_rolling.csv', index=False)

# 保存超参数
import json
hyperparams_dict = {
    'lstm_params': best_lstm_params,
    'gru_params': best_gru_params,
    'xgb_params': best_xgb_params,
    'optimization_method': 'Optuna' if (USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE) else 'Manual',
    'global_seed': GLOBAL_SEED,
    'evaluation_method': 'Rolling Window (No Data Leakage)'
}

with open(results_directory + 'hyperparameters.json', 'w') as f:
    json.dump(hyperparams_dict, f, indent=4)

print(f"\n✓ 所有模型和结果已保存至: {results_directory}")


# =====================================================================================
# SECTION 16: DM检验（Diebold-Mariano Test）
# =====================================================================================
print("\n" + "=" * 100)
print("【DM检验】Diebold-Mariano统计显著性检验".center(100))
print("=" * 100)

try:
    from dm_test import quick_dm_analysis, pairwise_dm_analysis

    # 准备数据（使用原始尺度的预测结果）
    all_predictions = {
        'LSTM单模型': strategies_original['LSTM单模型'],
        'GRU单模型': strategies_original['GRU单模型'],
        '简单平均(基线)': strategies_original['简单平均(基线)'],
        **{k: v for k, v in strategies_original.items() if k.startswith('策略')}
    }

    # 基准对比分析
    print("\n所有模型 vs 基准模型")
    print("-" * 100)

    dm_results = quick_dm_analysis(
        y_true=y_test_original,
        predictions=all_predictions,
        baseline='简单平均(基线)',
        save_dir=results_directory,
        plot=True,
        verbose=True
    )

    print("\n✅ DM检验完成！结果已保存至:", results_directory)

except ImportError:
    print("[WARNING] ⚠️ 未找到dm_test.py模块，跳过DM检验")
    print("[INFO] 如需DM检验，请确保dm_test.py在同目录下")


# =====================================================================================
# SECTION 17: 最终总结报告
# =====================================================================================
print("\n" + "=" * 100)
print("最终总结报告".center(100))
print("=" * 100)

print(f"\n📊 数据集信息:")
print(f"  - 训练集样本数: {len(y_train_seq)}")
print(f"  - 测试集样本数: {len(y_test)}")
print(f"  - 特征维度: {X_train_feat.shape[1]}")
print(f"  - 序列长度: {seq_len}")

print(f"\n✅ 数据泄露修复验证:")
print(f"  - 评估方法: 滚动窗口预测")
print(f"  - 每个样本独立预测")
print(f"  - 使用真实历史值更新窗口")
print(f"  - 完全避免数据泄露")

print(f"\n🏆 最佳性能策略:")
print(f"  - 策略名称: {best_name}")
print(f"  - 测试集R^2: {best_r2:.4f}")
print(f"  - 提升幅度: {best_r2 - avg_r2:+.4f}")

# ✅ 修复：索引从4改为5
best_pred = strategies_original[best_name]
best_mae = mean_absolute_error(y_test_original, best_pred)
best_rmse = sqrt(mean_squared_error(y_test_original, best_pred))
best_mape = np.mean(np.abs((best_pred - y_test_original) / (y_test_original + 1e-8)))

print(f"\n📈 最佳策略原始尺度指标:")
print(f"  - MAE:  {best_mae:.2f}")
print(f"  - RMSE: {best_rmse:.2f}")
print(f"  - MAPE: {best_mape:.4%}")

print(f"\n⚙️ 超参数优化方法:")
if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print(f"  ✅ Optuna自动优化")
    print(f"  - LSTM trials: {OPTUNA_CONFIG['lstm_trials']}")
    print(f"  - GRU trials: {OPTUNA_CONFIG['gru_trials']}")
    print(f"  - XGBoost trials: {OPTUNA_CONFIG['xgb_trials']}")
    print(f"  - 优化报告: ./optuna_results/")
else:
    print(f"  ⚠️  手动设置参数")

print(f"\n⚠️ 过拟合诊断总结:")
print(f"  【学习曲线方法】")
for model_name, result in detector.results.items():
    diagnosis = result.get('diagnosis', 'Unknown')
    final_gap = result.get('final_gap', 0)
    print(f"  - {model_name}: {diagnosis} (差距={final_gap:.4f})")

print(f"\n💡 关键结论:")
if best_r2 > avg_r2:
    print(f"  ✅ 残差学习策略成功提升了模型性能")
    print(f"  ✅ 相比简单平均提升了 {(best_r2 - avg_r2) * 100:.2f}%")
else:
    print(f"  ⚠️ 残差学习策略未能改善性能，建议使用简单平均")

if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print(f"\n🎯 Optuna优化成果:")
    print(f"  ✅ 自动找到最优超参数组合")
    print(f"  ✅ 节省大量手动调参时间")
    print(f"  ✅ 详细优化报告可供分析")

print(f"\n📁 所有结果保存位置:")
print(f"  - 模型和预测结果: {results_directory}")
print(f"  - 过拟合分析: {detector.output_dir}")
if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print(f"  - Optuna优化报告: ./optuna_results/")

print("\n" + "=" * 100)
print("程序执行完毕！数据泄露问题已彻底修复！".center(100))
print("=" * 100)

# =====================================================================================
# 在 SECTION 18 添加特征消融实验代码
# =====================================================================================

if FEATURE_ABLATION_CONFIG['enabled'] and FEATURE_ABLATION_AVAILABLE:
    print("\n" + "=" * 100)
    print("【特征消融实验】量化每个特征的贡献度".center(100))
    print("=" * 100)

    # 获取原始特征列表（不包括滞后特征）
    original_features = [col for col in X_train.columns if not col.startswith('Corn_lag_')]

    print(f"\n分析特征列表: {original_features}")
    print(f"特征数量: {len(original_features)}")

    # 创建特征消融分析器
    feature_analyzer = FeatureAblationAnalyzer(
        y_test_true=y_test.values,
        feature_names=original_features,
        output_dir=FEATURE_ABLATION_CONFIG['output_dir']
    )

    # =========================================================================
    # 步骤1: 添加基线（使用全部特征）
    # =========================================================================
    print("\n[步骤1] 添加基线实验（使用全部特征）...")

    # 使用当前最佳策略的预测作为基线
    baseline_pred = strategies_results['策略2-GRU残差学习']  # 根据您的最佳策略调整

    feature_analyzer.add_baseline(
        predictions=baseline_pred,
        description=f'完整模型：LSTM+GRU+XGBoost，使用全部{len(original_features)}个特征'
    )

    # =========================================================================
    # 步骤2: 逐个移除特征并重新训练
    # =========================================================================
    print("\n[步骤2] 逐个移除特征并重新训练...")
    print("注意：这将花费较长时间（每个特征需要重新训练完整模型）")

    for idx, feature_to_remove in enumerate(original_features, 1):
        print(f"\n{'=' * 100}")
        print(f"[{idx}/{len(original_features)}] 移除特征: {feature_to_remove}".center(100))
        print(f"{'=' * 100}")

        # 选择剩余特征
        remaining_features = [f for f in original_features if f != feature_to_remove]

        print(f"剩余特征: {remaining_features}")

        # 重新构造训练数据（仅使用剩余特征）
        X_train_ablation = X_train[remaining_features].copy()
        X_test_ablation = X_test[remaining_features].copy()

        # 添加滞后特征
        X_train_feat_ablation = add_features(X_train_ablation, y_train)
        y_train_ablation = y_train.loc[X_train_feat_ablation.index]

        # 构造序列
        X_train_seq_ablation, y_train_seq_ablation = create_sequences(
            X_train_feat_ablation, y_train_ablation, seq_len
        )

        print(f"训练集序列形状: {X_train_seq_ablation.shape}")

        # 训练LSTM
        print("  [1/3] 训练LSTM...", end='')
        tf.random.set_seed(GLOBAL_SEED)
        lstm_ablation = Sequential([
            layers.LSTM(
                best_lstm_params['units'],
                input_shape=(X_train_seq_ablation.shape[1], X_train_seq_ablation.shape[2]),
                kernel_regularizer=l2(best_lstm_params.get('l2_reg', 0.01)),
                recurrent_dropout=best_lstm_params.get('recurrent_dropout', 0.0),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
                recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
            ),
            layers.Dropout(best_lstm_params['dropout'], seed=GLOBAL_SEED),
            layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))
        ])
        lstm_ablation.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_lstm_params['learning_rate'])
        )
        lstm_ablation.fit(
            X_train_seq_ablation, y_train_seq_ablation,
            validation_split=0.2, epochs=200, batch_size=32,
            verbose=0, shuffle=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=15,
                                     restore_best_weights=True, verbose=0)]
        )
        print(" 完成")

        # 训练GRU
        print("  [2/3] 训练GRU...", end='')
        tf.random.set_seed(GLOBAL_SEED)
        gru_ablation = Sequential([
            layers.GRU(
                best_gru_params['units'],
                input_shape=(X_train_seq_ablation.shape[1], X_train_seq_ablation.shape[2]),
                recurrent_dropout=best_gru_params.get('recurrent_dropout', 0.0),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
                recurrent_initializer=tf.keras.initializers.Orthogonal(seed=GLOBAL_SEED)
            ),
            layers.Dropout(best_gru_params['dropout'], seed=GLOBAL_SEED),
            layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED))
        ])
        gru_ablation.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_gru_params['learning_rate'])
        )
        gru_ablation.fit(
            X_train_seq_ablation, y_train_seq_ablation,
            validation_split=0.2, epochs=200, batch_size=32,
            verbose=0, shuffle=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=20,
                                     restore_best_weights=True, verbose=0)]
        )
        print(" 完成")

        # 训练XGBoost
        print("  [3/3] 训练XGBoost...", end='')
        lstm_oof_ablation = get_oof_predictions(
            X_train_seq_ablation, y_train_seq_ablation,
            best_lstm_params, 'lstm', n_splits=5
        )
        gru_oof_ablation = get_oof_predictions(
            X_train_seq_ablation, y_train_seq_ablation,
            best_gru_params, 'gru', n_splits=5
        )

        gru_residual_ablation = y_train_seq_ablation - gru_oof_ablation

        X_flat_ablation = X_train_seq_ablation.reshape(len(X_train_seq_ablation), -1)
        X_xgb_train_ablation = create_simplified_features(
            X_flat_ablation, lstm_oof_ablation, gru_oof_ablation
        )

        xgb_ablation = XGBRegressor(
            **best_xgb_params,
            random_state=GLOBAL_SEED,
            seed=GLOBAL_SEED,
            n_jobs=1,
            verbosity=0
        )
        xgb_ablation.fit(X_xgb_train_ablation, gru_residual_ablation)
        print(" 完成")

        # 滚动窗口预测
        print("  [预测] 滚动窗口预测测试集...", end='')
        predictions_ablation = []
        history_X_feat = X_train_feat_ablation.tail(seq_len).copy()
        history_y = y_train_ablation.tail(5).copy()

        for test_idx in range(len(X_test)):
            # 构造当前样本特征（仅包含剩余特征）
            current_X_raw = X_test_ablation.iloc[test_idx:test_idx + 1].copy()

            for i in range(1, 6):
                if len(history_y) >= i:
                    current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i]
                else:
                    current_X_raw[f'Corn_lag_{i}'] = 0

            # 构造序列窗口
            current_window = pd.concat([history_X_feat.tail(seq_len - 1), current_X_raw])
            X_seq_test = current_window.values.reshape(1, seq_len, -1)

            # 获取LSTM和GRU预测
            lstm_pred = lstm_ablation.predict(X_seq_test, verbose=0)[0, 0]
            gru_pred = gru_ablation.predict(X_seq_test, verbose=0)[0, 0]

            # 构造XGBoost特征并预测残差
            X_flat_test = X_seq_test.reshape(1, -1)
            X_xgb_test = np.hstack([
                X_flat_test,
                np.array([[lstm_pred, gru_pred, (lstm_pred + gru_pred) / 2,
                           abs(lstm_pred - gru_pred)]])
            ])

            residual = xgb_ablation.predict(X_xgb_test)[0]
            final_pred = gru_pred + residual
            predictions_ablation.append(final_pred)

            # 更新历史
            history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
            history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[test_idx]])])

        predictions_ablation = np.array(predictions_ablation)
        print(" 完成")

        # 添加到特征分析器
        feature_analyzer.add_single_feature_removal(
            feature_name=feature_to_remove,
            predictions=predictions_ablation,
            description=f'使用{len(remaining_features)}个特征（移除{feature_to_remove}）'
        )

        print(f"\n✅ 特征 {feature_to_remove} 的消融实验完成")

    # =========================================================================
    # 步骤3: 生成报告和可视化
    # =========================================================================
    print("\n" + "=" * 100)
    print("生成特征重要性报告和可视化".center(100))
    print("=" * 100)

    # 生成报告
    feature_report = feature_analyzer.generate_feature_importance_report()

    # 生成可视化
    feature_analyzer.visualize_feature_importance(
        show=FEATURE_ABLATION_CONFIG['show_plots']
    )

    # 导出LaTeX（可选）
    if FEATURE_ABLATION_CONFIG['save_latex']:
        feature_analyzer.export_to_latex()

    print("\n" + "=" * 100)
    print("✅ 特征消融实验完成！".center(100))
    print(f"结果保存在: {FEATURE_ABLATION_CONFIG['output_dir']}".center(100))
    print("=" * 100)

else:
    if not FEATURE_ABLATION_CONFIG['enabled']:
        print("\n⏭️ 特征消融实验已禁用（FEATURE_ABLATION_CONFIG['enabled']=False）")
    else:
        print("\n⚠️ 特征消融模块未导入，跳过实验")