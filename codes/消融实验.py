import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import warnings

warnings.filterwarnings('ignore')


# ========================================
# 固定所有随机种子 - 确保结果可重复
# ========================================
def set_seed(seed=42):
    """
    固定所有随机性来源，确保实验结果可重复

    参数:
        seed: 随机种子值，默认42
    """
    # Python内置random模块
    random.seed(seed)

    # Numpy随机数
    np.random.seed(seed)

    # TensorFlow随机数
    tf.random.set_seed(seed)

    # 环境变量设置
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 确保TensorFlow使用确定性算法
    tf.config.experimental.enable_op_determinism()

    print(f"✅ 已固定所有随机种子: {seed}")
    print(f"✅ 实验结果现在完全可重复\n")


# 在所有操作之前调用
set_seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("特征消融研究 - 量化每个因素的贡献度 (固定种子版本)".center(100))
print("=" * 100)


class AblationStudy:
    """特征消融研究类 - 固定种子版本"""

    def __init__(self, data_path='Corn-new.csv', output_dir='./ablation_results/',
                 model_type='lstm', random_seed=42):
        """
        初始化消融研究

        参数:
            data_path: 数据文件路径
            output_dir: 输出目录
                - 'gru': 单层GRU
            random_seed: 随机种子
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_type = model_type
        self.random_seed = random_seed

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 特征名称和描述
        self.feature_info = {
            'TR': '国债利率',
            'IR': '利率',
            'ER': '汇率',
            'ine': '原油期货价格',
            'FP.CFI': '农产品期货指数',
            'CFD': '美国玉米期货价格',
            'GPR': '地缘政治风险',
            'EPU': '经济政策不确定性'
        }

        self.results = {}
        self.baseline_performance = None

        print(f"\n📋 消融研究配置:")
        print(f"  使用模型: {model_type.upper()}")
        print(f"  随机种子: {random_seed}")
        if model_type == 'lstm':
            print(f"  说明: 单层LSTM，快速且稳定，适合消融研究")
        elif model_type == 'gru':
            print(f"  说明: 单层GRU，计算效率高")


    def load_and_preprocess_data(self, features_to_use=None):
        """加载和预处理数据"""
        # 加载数据
        dataset = pd.read_csv(self.data_path, parse_dates=['Date'], index_col=['Date'])

        X = dataset.drop(columns=['Corn'], axis=1)
        y = dataset['Corn']

        # 选择特征
        if features_to_use is not None:
            X = X[features_to_use]

        # 分割数据
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 归一化
        feature_scalers = {}
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        for col in X_train.columns:
            scaler = MinMaxScaler()
            X_train_scaled[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
            X_test_scaled[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
            feature_scalers[col] = scaler

        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler

    def add_lag_features(self, X, y, n_lags=5):
        """添加滞后特征"""
        X_new = X.copy()
        for i in range(1, n_lags + 1):
            X_new[f'Corn_lag_{i}'] = y.shift(i)
        return X_new.dropna()

    def create_sequences(self, X, y, seq_len=5):
        """创建序列数据"""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X.iloc[i:i + seq_len].values)
            y_seq.append(y.iloc[i + seq_len])
        return np.array(X_seq), np.array(y_seq)

    def build_lstm_model(self, input_shape):
        """构建LSTM模型（简化版 - 用于快速消融研究）"""
        model = Sequential([
            layers.LSTM(
                units=80,
                input_shape=input_shape,
                kernel_regularizer=l2(0.01),
                recurrent_regularizer=l2(0.01)
            ),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def build_gru_model(self, input_shape):
        """构建GRU模型"""
        model = Sequential([
            layers.GRU(units=100, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model


    def train_and_evaluate(self, features_to_use, experiment_name):
        """训练和评估模型"""
        print(f"\n{'=' * 80}")
        print(f"实验: {experiment_name}".center(80))
        print(f"使用特征: {features_to_use}".center(80))
        print(f"{'=' * 80}")

        # 在每次训练前重新设置种子，确保完全可重复
        set_seed(self.random_seed)

        # 加载数据
        X_train, X_test, y_train, y_test, y_scaler = self.load_and_preprocess_data(features_to_use)

        # 转换为Series以便添加滞后特征
        y_train_series = pd.Series(y_train, index=X_train.index)
        y_test_series = pd.Series(y_test, index=X_test.index)

        # 添加滞后特征
        X_train_feat = self.add_lag_features(X_train, y_train_series)
        y_train_aligned = y_train_series.loc[X_train_feat.index]

        X_test_feat = self.add_lag_features(X_test, y_test_series)
        y_test_aligned = y_test_series.loc[X_test_feat.index]

        # 创建序列
        seq_len = 5
        X_train_seq, y_train_seq = self.create_sequences(X_train_feat, y_train_aligned, seq_len)
        X_test_seq, y_test_seq = self.create_sequences(X_test_feat, y_test_aligned, seq_len)

        print(f"训练集: {X_train_seq.shape}, 测试集: {X_test_seq.shape}")

        # 根据model_type构建不同的模型
        if self.model_type == 'lstm':
            model = self.build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        elif self.model_type == 'gru':
            model = self.build_gru_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train_seq, y_train_seq,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # 预测
        y_train_pred = model.predict(X_train_seq, verbose=0).flatten()
        y_test_pred = model.predict(X_test_seq, verbose=0).flatten()

        # 计算指标
        metrics = {
            'train_r2': r2_score(y_train_seq, y_train_pred),
            'test_r2': r2_score(y_test_seq, y_test_pred),
            'test_mae': mean_absolute_error(y_test_seq, y_test_pred),
            'test_rmse': sqrt(mean_squared_error(y_test_seq, y_test_pred)),
            'test_mape': np.mean(np.abs((y_test_pred - y_test_seq) / (y_test_seq + 1e-8))),
            'n_features': X_train_seq.shape[2],
            'features': features_to_use
        }

        print(f"\n性能指标:")
        print(f"  训练R²: {metrics['train_r2']:.6f}")
        print(f"  测试R²: {metrics['test_r2']:.6f}")
        print(f"  测试MAE: {metrics['test_mae']:.6f}")
        print(f"  测试RMSE: {metrics['test_rmse']:.6f}")
        print(f"  测试MAPE: {metrics['test_mape']:.6f}")

        return metrics

    def run_full_ablation(self):
        """运行完整的消融研究"""
        all_features = list(self.feature_info.keys())

        # 1. 基线实验：使用所有特征
        print("\n" + "=" * 100)
        print("【基线实验】使用所有特征".center(100))
        print("=" * 100)

        baseline_metrics = self.train_and_evaluate(
            all_features,
            "基线模型（所有特征）"
        )
        self.baseline_performance = baseline_metrics
        self.results['Baseline'] = baseline_metrics

        # 2. 单特征消融：逐个移除特征
        print("\n" + "=" * 100)
        print("【单特征消融】逐个移除特征".center(100))
        print("=" * 100)

        for feature in all_features:
            remaining_features = [f for f in all_features if f != feature]
            experiment_name = f"移除 {feature} ({self.feature_info[feature]})"

            metrics = self.train_and_evaluate(
                remaining_features,
                experiment_name
            )
            self.results[f'Remove_{feature}'] = metrics

        # 3. 单特征实验：仅使用单个特征
        print("\n" + "=" * 100)
        print("【单特征实验】仅使用单个特征".center(100))
        print("=" * 100)

        for feature in all_features:
            experiment_name = f"仅使用 {feature} ({self.feature_info[feature]})"

            metrics = self.train_and_evaluate(
                [feature],
                experiment_name
            )
            self.results[f'Only_{feature}'] = metrics

    def calculate_feature_importance(self):
        """计算特征重要性"""
        baseline_r2 = self.baseline_performance['test_r2']

        importance_scores = {}

        for feature in self.feature_info.keys():
            # 移除该特征后的性能下降
            remove_key = f'Remove_{feature}'
            if remove_key in self.results:
                remove_r2 = self.results[remove_key]['test_r2']
                drop_score = baseline_r2 - remove_r2  # 性能下降 = 移除后下降的R²

            # 仅使用该特征的性能
            only_key = f'Only_{feature}'
            if only_key in self.results:
                only_r2 = self.results[only_key]['test_r2']
                standalone_score = only_r2  # 单独性能

            importance_scores[feature] = {
                'name': self.feature_info[feature],
                'drop_score': drop_score,  # 移除后性能下降（越大越重要）
                'standalone_score': standalone_score,  # 单独使用性能
                'combined_score': (drop_score + standalone_score) / 2  # 综合得分
            }

        return importance_scores

    def visualize_results(self):
        """可视化结果"""
        importance_scores = self.calculate_feature_importance()

        # 准备数据
        features = list(importance_scores.keys())
        feature_names = [importance_scores[f]['name'] for f in features]
        drop_scores = [importance_scores[f]['drop_score'] for f in features]
        standalone_scores = [importance_scores[f]['standalone_score'] for f in features]
        combined_scores = [importance_scores[f]['combined_score'] for f in features]

        # 排序
        sorted_indices = np.argsort(combined_scores)[::-1]
        features = [features[i] for i in sorted_indices]
        feature_names = [feature_names[i] for i in sorted_indices]
        drop_scores = [drop_scores[i] for i in sorted_indices]
        standalone_scores = [standalone_scores[i] for i in sorted_indices]
        combined_scores = [combined_scores[i] for i in sorted_indices]

        # 图1: 特征重要性综合对比
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # 1.1 移除后性能下降
        ax1 = axes[0, 0]
        colors1 = plt.cm.Reds(np.linspace(0.4, 0.9, len(features)))
        bars1 = ax1.barh(range(len(features)), drop_scores, color=colors1, alpha=0.8)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels([f"{features[i]}\n({feature_names[i]})" for i in range(len(features))], fontsize=9)
        ax1.set_xlabel('性能下降 (ΔR²)', fontsize=11, fontweight='bold')
        ax1.set_title('移除特征后的性能下降\n（越大表示该特征越重要）', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        for i, (bar, score) in enumerate(zip(bars1, drop_scores)):
            ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                     f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=9)

        # 1.2 单独使用性能
        ax2 = axes[0, 1]
        colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
        bars2 = ax2.barh(range(len(features)), standalone_scores, color=colors2, alpha=0.8)
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels([f"{features[i]}\n({feature_names[i]})" for i in range(len(features))], fontsize=9)
        ax2.set_xlabel('R² Score', fontsize=11, fontweight='bold')
        ax2.set_title('单独使用特征的性能\n（衡量特征的独立预测能力）', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        for i, (bar, score) in enumerate(zip(bars2, standalone_scores)):
            ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                     f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=9)

        # 1.3 综合重要性得分
        ax3 = axes[1, 0]
        colors3 = plt.cm.Greens(np.linspace(0.4, 0.9, len(features)))
        bars3 = ax3.barh(range(len(features)), combined_scores, color=colors3, alpha=0.8)
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels([f"{features[i]}\n({feature_names[i]})" for i in range(len(features))], fontsize=9)
        ax3.set_xlabel('综合得分', fontsize=11, fontweight='bold')
        ax3.set_title('特征综合重要性得分\n（移除影响 + 独立性能）/2', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        for i, (bar, score) in enumerate(zip(bars3, combined_scores)):
            ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                     f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=9)

        # 1.4 雷达图
        ax4 = axes[1, 1]
        ax4.remove()
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')

        top5_features = features[:5]
        top5_names = feature_names[:5]

        angles = np.linspace(0, 2 * np.pi, len(top5_features), endpoint=False).tolist()
        angles += angles[:1]

        for idx, (feature, name) in enumerate(zip(top5_features, top5_names)):
            values = [
                importance_scores[feature]['drop_score'] / max(drop_scores),
                importance_scores[feature]['standalone_score'] / max(standalone_scores)
            ]
            values = values + values[:1]

            plot_angles = [0, np.pi] + [0]
            ax4.plot(plot_angles, values, 'o-', linewidth=2, label=f"{feature} ({name})")
            ax4.fill(plot_angles, values, alpha=0.15)

        ax4.set_xticks([0, np.pi])
        ax4.set_xticklabels(['移除影响', '独立性能'])
        ax4.set_ylim(0, 1)
        ax4.set_title('Top5特征多维度对比', fontsize=12, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax4.grid(True)

        plt.suptitle(f'特征消融研究 - 固定种子={self.random_seed}',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}feature_importance_analysis_{self.model_type}_seed{self.random_seed}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✅ 特征重要性分析图已保存")

        # 图2: 所有实验对比
        self.plot_all_experiments()

    def plot_all_experiments(self):
        """绘制所有实验对比"""
        fig, ax = plt.subplots(figsize=(16, 10))

        experiment_names = []
        r2_scores = []
        colors_list = []

        # 基线
        experiment_names.append('基线（所有特征）')
        r2_scores.append(self.results['Baseline']['test_r2'])
        colors_list.append('gold')

        # 移除特征实验
        for feature in self.feature_info.keys():
            key = f'Remove_{feature}'
            if key in self.results:
                experiment_names.append(f'移除 {feature}')
                r2_scores.append(self.results[key]['test_r2'])
                colors_list.append('coral')

        # 单特征实验
        for feature in self.feature_info.keys():
            key = f'Only_{feature}'
            if key in self.results:
                experiment_names.append(f'仅 {feature}')
                r2_scores.append(self.results[key]['test_r2'])
                colors_list.append('skyblue')

        # 绘制
        bars = ax.barh(range(len(experiment_names)), r2_scores, color=colors_list, alpha=0.7)

        # 基线参考线
        baseline_r2 = self.results['Baseline']['test_r2']
        ax.axvline(x=baseline_r2, color='red', linestyle='--', linewidth=2,
                   label=f'基线 R²={baseline_r2:.6f}', alpha=0.7)

        # 标注
        for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
            diff = r2 - baseline_r2
            label = f'{r2:.6f}'
            if i > 0:  # 非基线实验
                if diff >= 0:
                    label += f' (+{diff:.6f})'
                    color = 'green'
                else:
                    label += f' ({diff:.6f})'
                    color = 'red'
            else:
                color = 'black'

            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    label, ha='left', va='center', fontweight='bold', fontsize=8, color=color)

        ax.set_yticks(range(len(experiment_names)))
        ax.set_yticklabels(experiment_names, fontsize=9)
        ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title(f'消融研究 - 所有实验性能对比\n(模型: {self.model_type.upper()}, 种子: {self.random_seed})',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}all_experiments_comparison_{self.model_type}_seed{self.random_seed}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 所有实验对比图已保存")

    def generate_report(self):
        """生成详细报告"""
        importance_scores = self.calculate_feature_importance()

        print("\n" + "=" * 100)
        print("消融研究详细报告 (固定种子版本)".center(100))
        print("=" * 100)

        print(f"\n【实验配置】")
        print(f"  使用模型: {self.model_type.upper()}")
        print(f"  随机种子: {self.random_seed}")
        print(f"  特征总数: {len(self.feature_info)}")
        print(f"  结果可重复性: ✅ 完全可重复")

        # 基线性能
        print(f"\n【基线性能】使用所有{len(self.feature_info)}个特征")
        print(f"  测试R²: {self.baseline_performance['test_r2']:.6f}")
        print(f"  测试MAE: {self.baseline_performance['test_mae']:.6f}")
        print(f"  测试RMSE: {self.baseline_performance['test_rmse']:.6f}")

        # 特征重要性排名
        print(f"\n【特征重要性排名】")
        print(f"{'排名':<6} {'特征':<10} {'中文名':<15} {'移除影响':<12} {'独立性能':<12} {'综合得分':<12}")
        print("-" * 85)

        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )

        for rank, (feature, scores) in enumerate(sorted_features, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{emoji} {rank:<4} {feature:<10} {scores['name']:<15} "
                  f"{scores['drop_score']:.6f}     {scores['standalone_score']:.6f}     "
                  f"{scores['combined_score']:.6f}")

        # 关键发现
        print(f"\n【关键发现】")
        top_feature = sorted_features[0]
        print(f"  🏆 最重要特征: {top_feature[0]} ({top_feature[1]['name']})")
        print(f"     - 移除后性能下降: {top_feature[1]['drop_score']:.6f}")
        print(f"     - 单独使用性能: {top_feature[1]['standalone_score']:.6f}")

        weakest_feature = sorted_features[-1]
        print(f"  ⚠️  最弱特征: {weakest_feature[0]} ({weakest_feature[1]['name']})")
        print(f"     - 移除后性能下降: {weakest_feature[1]['drop_score']:.6f}")
        print(f"     - 单独使用性能: {weakest_feature[1]['standalone_score']:.6f}")

        # 保存CSV报告
        report_data = []
        for feature, scores in sorted_features:
            report_data.append({
                '特征代码': feature,
                '特征名称': scores['name'],
                '移除后性能下降': scores['drop_score'],
                '单独使用性能': scores['standalone_score'],
                '综合重要性得分': scores['combined_score']
            })

        df_report = pd.DataFrame(report_data)
        df_report.to_csv(f'{self.output_dir}ablation_study_report_{self.model_type}_seed{self.random_seed}.csv',
                         index=False, encoding='utf-8-sig')
        print(f"\n✅ 详细报告已保存至: {self.output_dir}ablation_study_report_{self.model_type}_seed{self.random_seed}.csv")


# ========================================
# 主程序执行
# ========================================
if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("特征消融研究 - 模型选择 (固定种子版本)".center(100))
    print("=" * 100)
    print("\n可选模型类型:")
    print("  1. 'lstm'    - 单层LSTM（推荐：快速且稳定）⭐")
    print("  2. 'gru'     - 单层GRU（计算效率高）")
    print("\n" + "=" * 100)

    # 选择模型类型和随机种子
    # 方式1: 直接指定（推荐用于快速测试）
    #model_choice = 'gru'  # 改为 'gru'以使用其他模型
    #random_seed = 42  # 固定种子，确保结果可重复

    # 方式2: 交互式选择（取消注释以启用）
    model_choice = input("\n请选择模型类型 (lstm/gru，默认gru): ").strip().lower()
    if model_choice not in ['lstm', 'gru']:
         model_choice = 'gru'

    seed_input = input("请输入随机种子 (默认42): ").strip()
    random_seed = int(seed_input) if seed_input.isdigit() else 42

    print(f"\n🎯 开始消融研究...")
    print(f"  模型: {model_choice.upper()}")
    print(f"  种子: {random_seed}")
    print(f"  结果将完全可重复！")

    # 创建消融研究对象
    study = AblationStudy(
        data_path='Corn-new.csv',
        output_dir='./ablation_results/',
        model_type=model_choice,
        random_seed=random_seed
    )

    # 运行完整的消融研究
    study.run_full_ablation()

    # 可视化结果
    study.visualize_results()

    # 生成报告
    study.generate_report()

    print("\n" + "=" * 100)
    print("消融研究完成！".center(100))
    print(f"使用模型: {model_choice.upper()}, 随机种子: {random_seed}".center(100))
    print("✅ 所有结果完全可重复！".center(100))
    print("=" * 100)

    print("\n📌 使用提示:")
    print("  - 每次运行相同的种子，结果将完全一致")
    print("  - 如需测试结果稳定性，可尝试不同的种子值 (如 42, 123, 999)")
    print("  - 所有图表和报告已保存到 ./ablation_results/ 目录")
    print("  - 文件名包含模型类型和种子值，方便对比不同配置")