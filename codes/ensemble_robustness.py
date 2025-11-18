"""
支持融合模型的鲁棒性测试 - 集成代码
========================================

将此代码添加到主代码的末尾，或创建为独立脚本运行

核心改进：
1. ✅ 支持完整的 LSTM+GRU+XGBoost 融合模型
2. ✅ 支持所有6种策略的鲁棒性测试
3. ✅ 保持滚动窗口预测，避免数据泄露
4. ✅ 生成完整的鲁棒性报告

使用方法：
- 将此代码添加到主代码的 SECTION 17 之后
- 或单独运行（需要先运行主代码生成模型）
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from math import sqrt
import matplotlib.pyplot as plt

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class EnsembleRobustnessTester:
    """
    融合模型鲁棒性测试器

    支持测试：
    - LSTM单模型
    - GRU单模型
    - 简单平均
    - 策略1-6（残差学习等）
    """

    def __init__(self, data_path='Corn-new.csv', output_dir='./ensemble_robustness/'):
        self.data_path = data_path
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 加载数据
        self.dataset = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

        # 存储结果
        self.results = {}

        print(f"\n✅ 融合模型鲁棒性测试器初始化完成")
        print(f"   数据: {len(self.dataset)} 样本")
        print(f"   输出: {output_dir}")

    def train_ensemble_model(self, X_train_df, y_train_df, X_test_df, y_test_df,
                            strategy='gru_residual', verbose=False):
        """
        训练完整的融合模型（LSTM+GRU+XGBoost）

        Parameters:
            X_train_df: 训练特征 (DataFrame, 原始尺度)
            y_train_df: 训练标签 (Series, 原始尺度)
            X_test_df: 测试特征 (DataFrame, 原始尺度)
            y_test_df: 测试标签 (Series, 原始尺度)
            strategy: 融合策略
                - 'lstm': LSTM单模型
                - 'gru': GRU单模型
                - 'average': 简单平均
                - 'lstm_residual': 策略1-LSTM残差学习
                - 'gru_residual': 策略2-GRU残差学习
                - 'dual_residual': 策略3-双残差学习
                - 'gru_clipped': 策略4-残差剪裁
                - 'gru_weighted': 策略5-加权融合
                - 'ultimate': 策略6-终极组合
            verbose: 是否打印详细信息

        Returns:
            predictions: 原始尺度的预测值
        """

        # ========== 参数配置 ==========
        seq_len = 5
        lag_features = 5
        min_samples = 100

        # ========== 数据量检查 ==========
        if len(X_train_df) < min_samples:
            if verbose:
                print(f"  ⚠️ 训练样本过少 ({len(X_train_df)})，使用基线预测")
            return np.full(len(y_test_df), y_train_df.mean())

        # ========== 数据归一化 ==========
        X_train = X_train_df.copy()
        X_test = X_test_df.copy()

        feature_scalers = {}
        for col in X_train_df.columns:
            scaler = MinMaxScaler()
            X_train[col] = scaler.fit_transform(X_train_df[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test_df[col].values.reshape(-1, 1))
            feature_scalers[col] = scaler

        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train_df.values.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test_df.values.reshape(-1, 1)).flatten()

        y_train = pd.Series(y_train, index=y_train_df.index)
        y_test = pd.Series(y_test, index=y_test_df.index)

        # ========== 特征工程 ==========
        def add_lag_features(X, y, n_lags=5):
            X_new = X.copy()
            for i in range(1, n_lags + 1):
                X_new[f'Corn_lag_{i}'] = y.shift(i)
            return X_new.dropna()

        X_train_feat = add_lag_features(X_train, y_train, lag_features)
        y_train_aligned = y_train.loc[X_train_feat.index]

        if len(X_train_feat) < 50:
            if verbose:
                print(f"  ⚠️ 特征工程后样本过少，使用基线")
            return np.full(len(y_test_df), y_train_df.mean())

        # ========== 创建序列数据 ==========
        def create_sequences(X, y, seq_len=5):
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_len):
                X_seq.append(X.iloc[i:i + seq_len].values)
                y_seq.append(y.iloc[i + seq_len])
            return np.array(X_seq), np.array(y_seq)

        X_train_seq, y_train_seq = create_sequences(X_train_feat, y_train_aligned, seq_len)

        if len(X_train_seq) < 30:
            if verbose:
                print(f"  ⚠️ 序列数据过少，使用基线")
            return np.full(len(y_test_df), y_train_df.mean())

        # ========== 训练LSTM模型 ==========
        tf.random.set_seed(12)
        lstm_model = Sequential([
            layers.LSTM(80, input_shape=(seq_len, X_train_seq.shape[2]),
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=12)),
            layers.Dropout(0.3, seed=12),
            layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=12))
        ])
        lstm_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        lstm_model.fit(X_train_seq, y_train_seq, validation_split=0.2,
                      epochs=50, batch_size=32,
                      callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                      verbose=0)

        # ========== 训练GRU模型 ==========
        tf.random.set_seed(12)
        gru_model = Sequential([
            layers.GRU(100, input_shape=(seq_len, X_train_seq.shape[2]),
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=12)),
            layers.Dropout(0.3, seed=12),
            layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=12))
        ])
        gru_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        gru_model.fit(X_train_seq, y_train_seq, validation_split=0.2,
                     epochs=50, batch_size=32,
                     callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                     verbose=0)

        # ========== 生成OOF预测（用于训练XGBoost） ==========
        lstm_oof = lstm_model.predict(X_train_seq, verbose=0).flatten()
        gru_oof = gru_model.predict(X_train_seq, verbose=0).flatten()

        # ========== 训练XGBoost ==========
        if strategy in ['lstm_residual', 'gru_residual', 'dual_residual',
                       'gru_clipped', 'gru_weighted', 'ultimate']:
            X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
            X_xgb_train = np.hstack([
                X_train_flat,
                lstm_oof.reshape(-1, 1),
                gru_oof.reshape(-1, 1),
                ((lstm_oof + gru_oof) / 2).reshape(-1, 1),
                np.abs(lstm_oof - gru_oof).reshape(-1, 1)
            ])

            if strategy == 'lstm_residual':
                residual = y_train_seq - lstm_oof
            elif strategy in ['gru_residual', 'gru_clipped', 'gru_weighted', 'ultimate']:
                residual = y_train_seq - gru_oof
            else:  # dual_residual
                residual = y_train_seq - (lstm_oof + gru_oof) / 2

            xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3,
                                    random_state=42, verbosity=0)
            xgb_model.fit(X_xgb_train, residual)

        # ========== 滚动窗口预测 ==========
        predictions_scaled = []
        history_X_feat = X_train_feat.tail(seq_len).copy()
        history_y = y_train_aligned.tail(lag_features).copy()

        for idx in range(len(X_test)):
            # 构造当前样本
            current_X_raw = X_test.iloc[idx:idx+1].copy()
            for i in range(1, lag_features + 1):
                current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i] if len(history_y) >= i else 0

            current_window = pd.concat([history_X_feat.tail(seq_len-1), current_X_raw])
            X_seq = current_window.values.reshape(1, seq_len, -1)

            # 获取LSTM和GRU预测
            lstm_pred = lstm_model.predict(X_seq, verbose=0)[0, 0]
            gru_pred = gru_model.predict(X_seq, verbose=0)[0, 0]

            # 根据策略计算最终预测
            if strategy == 'lstm':
                final_pred = lstm_pred
            elif strategy == 'gru':
                final_pred = gru_pred
            elif strategy == 'average':
                final_pred = (lstm_pred + gru_pred) / 2
            elif strategy == 'lstm_residual':
                X_flat = X_seq.reshape(1, -1)
                X_xgb = np.hstack([X_flat, [[lstm_pred, gru_pred, (lstm_pred+gru_pred)/2,
                                            abs(lstm_pred-gru_pred)]]])
                xgb_residual = xgb_model.predict(X_xgb)[0]
                final_pred = lstm_pred + xgb_residual
            elif strategy in ['gru_residual', 'gru_clipped', 'gru_weighted', 'ultimate']:
                X_flat = X_seq.reshape(1, -1)
                X_xgb = np.hstack([X_flat, [[lstm_pred, gru_pred, (lstm_pred+gru_pred)/2,
                                            abs(lstm_pred-gru_pred)]]])
                xgb_residual = xgb_model.predict(X_xgb)[0]

                if strategy == 'gru_residual':
                    final_pred = gru_pred + xgb_residual
                elif strategy == 'gru_clipped':
                    # 简化的残差剪裁
                    clipped_residual = np.clip(xgb_residual, -0.1, 0.1)
                    final_pred = gru_pred + clipped_residual
                elif strategy == 'gru_weighted':
                    final_pred = gru_pred + 0.3 * xgb_residual
                else:  # ultimate
                    clipped_residual = np.clip(xgb_residual, -0.1, 0.1)
                    final_pred = gru_pred + 0.3 * clipped_residual
            elif strategy == 'dual_residual':
                X_flat = X_seq.reshape(1, -1)
                avg_pred = (lstm_pred + gru_pred) / 2
                X_xgb = np.hstack([X_flat, [[lstm_pred, gru_pred, avg_pred,
                                            abs(lstm_pred-gru_pred)]]])
                xgb_residual = xgb_model.predict(X_xgb)[0]
                final_pred = avg_pred + xgb_residual
            else:
                final_pred = (lstm_pred + gru_pred) / 2

            predictions_scaled.append(final_pred)

            # 更新历史
            history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
            history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[idx]])])

        predictions_scaled = np.array(predictions_scaled)

        # ========== 反归一化 ==========
        predictions_original = y_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        # ========== 结果验证 ==========
        if verbose:
            r2 = r2_score(y_test_df, predictions_original)
            mae = mean_absolute_error(y_test_df, predictions_original)
            print(f"  ✓ 策略: {strategy}")
            print(f"  ✓ R²: {r2:.6f}, MAE: {mae:.2f}")

        return predictions_original

    def test_sub_periods(self, strategies=None):
        """
        子期间分析 - 测试所有策略

        Parameters:
            strategies: 要测试的策略列表
        """
        if strategies is None:
            strategies = ['gru', 'average', 'gru_residual', 'ultimate']

        print("\n" + "=" * 100)
        print("【融合模型】子期间鲁棒性测试".center(100))
        print("=" * 100)

        periods = {
            '疫情前 (2018-2019)': ('2018-01-01', '2019-12-31'),
            '疫情中 (2020-2021)': ('2020-01-01', '2021-12-31'),
            '冲突期 (2022-2023)': ('2022-01-01', '2023-12-31')
        }

        all_results = []

        for period_name, (start_date, end_date) in periods.items():
            print(f"\n{'=' * 80}")
            print(f"测试时期: {period_name}".center(80))
            print(f"{'=' * 80}")

            # 筛选数据
            mask = (self.dataset.index >= start_date) & (self.dataset.index <= end_date)
            period_data = self.dataset[mask]

            if len(period_data) < 100:
                print(f"  ⚠️ 数据量过少，跳过")
                continue

            split_idx = int(len(period_data) * 0.8)
            X = period_data.drop(columns=['Corn'], axis=1)
            y = period_data['Corn']

            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]

            print(f"\n数据: 训练{len(X_train)}样本, 测试{len(X_test)}样本")

            # 测试所有策略
            for strategy in strategies:
                try:
                    predictions = self.train_ensemble_model(
                        X_train, y_train, X_test, y_test,
                        strategy=strategy, verbose=True
                    )

                    r2 = r2_score(y_test, predictions)
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = sqrt(mean_squared_error(y_test, predictions))

                    all_results.append({
                        '时期': period_name,
                        '策略': strategy,
                        'R²': r2,
                        'MAE': mae,
                        'RMSE': rmse
                    })

                except Exception as e:
                    print(f"  ❌ 策略 {strategy} 失败: {e}")

        # 保存结果
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'{self.output_dir}sub_period_ensemble.csv', index=False)

        # 可视化
        self._plot_sub_period_results(results_df, strategies)

        return results_df

    def _plot_sub_period_results(self, results_df, strategies):
        """绘制子期间结果"""
        fig, axes = plt.subplots(1, len(strategies), figsize=(6*len(strategies), 5))

        if len(strategies) == 1:
            axes = [axes]

        periods = results_df['时期'].unique()
        x = np.arange(len(periods))

        for idx, strategy in enumerate(strategies):
            ax = axes[idx]
            strategy_data = results_df[results_df['策略'] == strategy]

            r2_values = [strategy_data[strategy_data['时期']==p]['R²'].values[0]
                        if len(strategy_data[strategy_data['时期']==p]) > 0 else 0
                        for p in periods]

            bars = ax.bar(x, r2_values, alpha=0.7, edgecolor='black')

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, r2_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels([p.split('(')[0].strip() for p in periods],
                              rotation=15, ha='right')
            ax.set_ylabel('R² Score', fontweight='bold')
            ax.set_title(f'策略: {strategy}', fontweight='bold')
            ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='良好线')
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='及格线')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('融合模型子期间鲁棒性分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}sub_period_ensemble_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ 可视化已保存: {self.output_dir}sub_period_ensemble_comparison.png")

    def generate_report(self):
        """生成综合报告"""
        print("\n" + "=" * 100)
        print("融合模型鲁棒性检验报告".center(100))
        print("=" * 100)

        # 读取结果
        try:
            results_df = pd.read_csv(f'{self.output_dir}sub_period_ensemble.csv')

            report = []
            report.append("=" * 100)
            report.append("融合模型鲁棒性检验报告")
            report.append("=" * 100)
            report.append(f"\n测试策略数: {len(results_df['策略'].unique())}")
            report.append(f"测试时期数: {len(results_df['时期'].unique())}\n")

            # 按策略汇总
            for strategy in results_df['策略'].unique():
                strategy_data = results_df[results_df['策略'] == strategy]
                report.append(f"\n【策略: {strategy}】")
                report.append("-" * 100)
                report.append(f"平均 R²: {strategy_data['R²'].mean():.4f}")
                report.append(f"R² 范围: [{strategy_data['R²'].min():.4f}, {strategy_data['R²'].max():.4f}]")
                report.append(f"R² 标准差: {strategy_data['R²'].std():.4f}")

                if strategy_data['R²'].std() < 0.1:
                    report.append("✅ 该策略在各时期表现稳定")
                else:
                    report.append("⚠️ 该策略存在时期敏感性")

            report_text = '\n'.join(report)

            with open(f'{self.output_dir}ensemble_robustness_report.txt', 'w',
                     encoding='utf-8') as f:
                f.write(report_text)

            print(report_text)
            print(f"\n✅ 报告已保存: {self.output_dir}ensemble_robustness_report.txt")

        except FileNotFoundError:
            print("⚠️ 未找到测试结果，请先运行 test_sub_periods()")


# =====================================================================
# 使用示例：将此代码添加到主代码的 SECTION 17 之后
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("开始融合模型鲁棒性测试".center(100))
    print("=" * 100)

    # 创建测试器
    tester = EnsembleRobustnessTester(
        data_path='Corn-new.csv',
        output_dir='./ensemble_robustness/'
    )

    # 测试多个策略
    test_strategies = [
        'gru',              # GRU单模型（基准）
        'average',          # 简单平均
        'gru_residual',     # 策略2-GRU残差学习
        'ultimate'          # 策略6-终极组合
    ]

    # 运行子期间测试
    results = tester.test_sub_periods(strategies=test_strategies)

    # 生成报告
    tester.generate_report()

    print("\n" + "=" * 100)
    print("✅ 融合模型鲁棒性测试完成！".center(100))
    print("📁 结果保存在: ./ensemble_robustness/".center(100))
    print("=" * 100)