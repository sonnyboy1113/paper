"""
=====================================================================================
Standalone Optuna Optimizer Module
独立Optuna优化模块 - 最终完整版
=====================================================================================

这是一个完全独立的Optuna超参数优化模块，可以无缝集成到任何时间序列预测代码中。

文件名: optuna_optimizer.py
版本: 2.0 Final
作者: AI Research Team
日期: 2025-10

使用方法:
--------
1. 将此文件保存为 optuna_optimizer.py
2. 与主程序放在同一目录
3. 在主程序中导入: from optuna_optimizer import OptunaOptimizer
4. 创建优化器并调用优化方法

快速示例:
--------
from optuna_optimizer import OptunaOptimizer

optimizer = OptunaOptimizer(X_train_seq, y_train_seq, seed=42)
best_params = optimizer.optimize_lstm(n_trials=50)
print(f"最优参数: {best_params}")

依赖包:
------
pip install optuna optuna-integration tensorflow scikit-learn xgboost kaleido

=====================================================================================
"""

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, Any
from datetime import datetime

warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class OptunaOptimizer:
    """
    独立Optuna超参数优化器

    这是一个完全独立的优化模块，可以直接集成到任何现有的时间序列预测代码中。

    主要功能:
    --------
    1. optimize_lstm()     - 优化LSTM超参数
    2. optimize_gru()      - 优化GRU超参数
    3. optimize_xgboost()  - 优化XGBoost超参数
    4. get_best_params()   - 获取最优参数
    5. visualize()         - 生成可视化
    6. generate_report()   - 生成优化报告

    参数说明:
    --------
    X_train : np.ndarray
        训练数据，形状为 (samples, timesteps, features) 或 (samples, features)
    y_train : np.ndarray
        训练标签，形状为 (samples,)
    X_val : np.ndarray, optional
        验证数据，若为None则自动从训练集分割
    y_val : np.ndarray, optional
        验证标签
    output_dir : str
        输出目录路径，默认 './optuna_results/'
    seed : int
        随机种子，默认 42
    verbose : bool
        是否显示详细信息，默认 True

    使用示例:
    --------
    >>> # 基础用法
    >>> optimizer = OptunaOptimizer(X_train_seq, y_train_seq)
    >>> best_params = optimizer.optimize_lstm(n_trials=50)
    >>> print(f"最优参数: {best_params}")

    >>> # 高级用法
    >>> optimizer = OptunaOptimizer(
    ...     X_train_seq, y_train_seq,
    ...     output_dir='./my_optuna/',
    ...     seed=42,
    ...     verbose=True
    ... )
    >>> best_params = optimizer.optimize_lstm(
    ...     n_trials=100,
    ...     timeout=7200,
    ...     enable_pruning=True
    ... )
    >>> optimizer.visualize('lstm')
    >>> optimizer.generate_report()
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        output_dir: str = './optuna_results/',
        seed: int = 42,
        verbose: bool = True
    ):
        """
        初始化Optuna优化器

        Parameters:
        -----------
        X_train : np.ndarray
            训练数据
        y_train : np.ndarray
            训练标签
        X_val : np.ndarray, optional
            验证数据
        y_val : np.ndarray, optional
            验证标签
        output_dir : str
            输出目录
        seed : int
            随机种子
        verbose : bool
            详细输出
        """
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.verbose = verbose

        # 自动处理验证集
        if X_val is None or y_val is None:
            split_idx = int(len(X_train) * 0.8)
            self.X_train_fit = X_train[:split_idx]
            self.y_train_fit = y_train[:split_idx]
            self.X_val = X_train[split_idx:]
            self.y_val = y_train[split_idx:]
            if self.verbose:
                print(f"[INFO] 自动分割验证集: train={split_idx}, val={len(X_train)-split_idx}")
        else:
            self.X_train_fit = X_train
            self.y_train_fit = y_train
            self.X_val = X_val
            self.y_val = y_val

        # 自动检测数据维度
        self.is_sequential = len(X_train.shape) == 3
        if self.verbose:
            data_type = "Sequential (LSTM/GRU)" if self.is_sequential else "Flat (XGBoost)"
            print(f"[INFO] 检测到数据类型: {data_type}")
            print(f"[INFO] 训练数据形状: {X_train.shape}")

        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 存储优化历史
        self.studies = {}
        self.best_params = {}

        # 默认搜索空间
        self.default_search_spaces = {
            'lstm': {
                'units': (50, 150, 10),
                'dropout': (0.1, 0.5),
                'recurrent_dropout': (0.0, 0.3),
                'l2_reg': (1e-4, 1e-1),
                'learning_rate': (1e-4, 1e-2),
            },
            'gru': {
                'units': (50, 150, 10),
                'dropout': (0.1, 0.5),
                'recurrent_dropout': (0.0, 0.3),
                'learning_rate': (1e-4, 1e-2),
            },
            'xgboost': {
                'n_estimators': (50, 300),
                'learning_rate': (0.001, 0.1),
                'max_depth': (2, 8),
                'min_child_weight': (1, 10),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'reg_alpha': (1e-3, 1.0),
                'reg_lambda': (1e-3, 10.0),
            }
        }

    def optimize_lstm(
        self,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        custom_search_space: Optional[Dict] = None,
        max_epochs: int = 200,
        batch_size: int = 32,
        early_stopping_patience: int = 20,
        enable_pruning: bool = True
    ) -> Dict[str, Any]:
        """
        优化LSTM超参数

        Parameters:
        -----------
        n_trials : int
            优化试验次数，默认50
        timeout : float, optional
            最大优化时间（秒），默认None（无限制）
        custom_search_space : dict, optional
            自定义搜索空间，会覆盖默认值
        max_epochs : int
            最大训练轮数，默认200
        batch_size : int
            批次大小，默认32
        early_stopping_patience : int
            早停耐心值，默认20
        enable_pruning : bool
            是否启用剪枝，默认True

        Returns:
        --------
        dict : 最优超参数字典

        Example:
        --------
        >>> best_params = optimizer.optimize_lstm(n_trials=50)
        >>> print(f"最优units: {best_params['units']}")
        """
        if not self.is_sequential:
            raise ValueError("LSTM优化需要3D序列数据 (samples, timesteps, features)")

        if self.verbose:
            print("\n" + "=" * 80)
            print("LSTM超参数优化开始".center(80))
            print("=" * 80)
            print(f"配置: n_trials={n_trials}, max_epochs={max_epochs}, "
                  f"pruning={enable_pruning}")

        # 合并搜索空间
        search_space = self.default_search_spaces['lstm'].copy()
        if custom_search_space:
            search_space.update(custom_search_space)

        def objective(trial):
            # 超参数采样
            units = trial.suggest_int('units', *search_space['units'])
            dropout = trial.suggest_float('dropout', *search_space['dropout'])
            recurrent_dropout = trial.suggest_float(
                'recurrent_dropout', *search_space['recurrent_dropout']
            )
            l2_reg = trial.suggest_float(
                'l2_reg', *search_space['l2_reg'], log=True
            )
            learning_rate = trial.suggest_float(
                'learning_rate', *search_space['learning_rate'], log=True
            )

            # 设置种子
            tf.random.set_seed(self.seed + trial.number)
            np.random.seed(self.seed + trial.number)

            # 构建模型
            model = Sequential([
                layers.LSTM(
                    units=units,
                    input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                    kernel_regularizer=l2(l2_reg),
                    recurrent_regularizer=l2(l2_reg),
                    recurrent_dropout=recurrent_dropout,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=self.seed)
                ),
                layers.Dropout(dropout, seed=self.seed),
                layers.Dense(
                    1,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
                )
            ])

            model.compile(
                loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['mae']
            )

            # 准备回调
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0
                )
            ]

            if enable_pruning:
                callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))

            # 训练
            try:
                history = model.fit(
                    self.X_train_fit, self.y_train_fit,
                    validation_data=(self.X_val, self.y_val),
                    epochs=max_epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    shuffle=False,
                    verbose=0
                )
            except optuna.TrialPruned:
                raise

            # 评估
            train_pred = model.predict(self.X_train_fit, verbose=0).flatten()
            val_pred = model.predict(self.X_val, verbose=0).flatten()

            train_r2 = r2_score(self.y_train_fit, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)

            # 过拟合惩罚
            overfitting_gap = max(0, train_r2 - val_r2)
            objective_value = val_r2 - 0.1 * overfitting_gap

            # 记录额外指标
            trial.set_user_attr('train_r2', float(train_r2))
            trial.set_user_attr('val_r2', float(val_r2))
            trial.set_user_attr('overfitting_gap', float(overfitting_gap))
            trial.set_user_attr('final_epoch', len(history.history['loss']))

            return objective_value

        # 创建Study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'lstm_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=min(20, n_trials // 5),
                multivariate=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=min(10, n_trials // 10),
                n_warmup_steps=20
            ) if enable_pruning else optuna.pruners.NopPruner()
        )

        # 优化
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.verbose
        )

        # 保存结果
        self.studies['lstm'] = study
        self.best_params['lstm'] = study.best_params
        self._save_results(study, 'lstm')

        if self.verbose:
            self._print_optimization_summary(study, 'LSTM')

        return study.best_params

    def optimize_gru(
        self,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        custom_search_space: Optional[Dict] = None,
        max_epochs: int = 200,
        batch_size: int = 32,
        early_stopping_patience: int = 20,
        enable_pruning: bool = True
    ) -> Dict[str, Any]:
        """
        优化GRU超参数

        参数与optimize_lstm相同
        """
        if not self.is_sequential:
            raise ValueError("GRU优化需要3D序列数据 (samples, timesteps, features)")

        if self.verbose:
            print("\n" + "=" * 80)
            print("GRU超参数优化开始".center(80))
            print("=" * 80)

        search_space = self.default_search_spaces['gru'].copy()
        if custom_search_space:
            search_space.update(custom_search_space)

        def objective(trial):
            units = trial.suggest_int('units', *search_space['units'])
            dropout = trial.suggest_float('dropout', *search_space['dropout'])
            recurrent_dropout = trial.suggest_float(
                'recurrent_dropout', *search_space['recurrent_dropout']
            )
            learning_rate = trial.suggest_float(
                'learning_rate', *search_space['learning_rate'], log=True
            )

            tf.random.set_seed(self.seed + trial.number)
            np.random.seed(self.seed + trial.number)

            model = Sequential([
                layers.GRU(
                    units=units,
                    input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                    recurrent_dropout=recurrent_dropout,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=self.seed)
                ),
                layers.Dropout(dropout, seed=self.seed),
                layers.Dense(
                    1,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
                )
            ])

            model.compile(
                loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['mae']
            )

            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0
                )
            ]

            if enable_pruning:
                callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))

            try:
                history = model.fit(
                    self.X_train_fit, self.y_train_fit,
                    validation_data=(self.X_val, self.y_val),
                    epochs=max_epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    shuffle=False,
                    verbose=0
                )
            except optuna.TrialPruned:
                raise

            train_pred = model.predict(self.X_train_fit, verbose=0).flatten()
            val_pred = model.predict(self.X_val, verbose=0).flatten()

            train_r2 = r2_score(self.y_train_fit, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)

            overfitting_gap = max(0, train_r2 - val_r2)
            objective_value = val_r2 - 0.1 * overfitting_gap

            trial.set_user_attr('train_r2', float(train_r2))
            trial.set_user_attr('val_r2', float(val_r2))
            trial.set_user_attr('overfitting_gap', float(overfitting_gap))

            return objective_value

        study = optuna.create_study(
            direction='maximize',
            study_name=f'gru_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=min(20, n_trials // 5),
                multivariate=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=min(10, n_trials // 10),
                n_warmup_steps=20
            ) if enable_pruning else optuna.pruners.NopPruner()
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.verbose
        )

        self.studies['gru'] = study
        self.best_params['gru'] = study.best_params
        self._save_results(study, 'gru')

        if self.verbose:
            self._print_optimization_summary(study, 'GRU')

        return study.best_params

    def optimize_xgboost(
        self,
        n_trials: int = 30,
        timeout: Optional[float] = None,
        custom_search_space: Optional[Dict] = None,
        cv_splits: int = 5
    ) -> Dict[str, Any]:
        """
        优化XGBoost超参数

        Parameters:
        -----------
        n_trials : int
            优化试验次数，默认30
        timeout : float, optional
            最大优化时间（秒）
        custom_search_space : dict, optional
            自定义搜索空间
        cv_splits : int
            时间序列交叉验证折数，默认5

        Returns:
        --------
        dict : 最优超参数字典
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("XGBoost超参数优化开始".center(80))
            print("=" * 80)

        search_space = self.default_search_spaces['xgboost'].copy()
        if custom_search_space:
            search_space.update(custom_search_space)

        # 对于扁平化数据
        X_flat = self.X_train.reshape(len(self.X_train), -1) if self.is_sequential else self.X_train

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
                'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate'], log=True),
                'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                'min_child_weight': trial.suggest_int('min_child_weight', *search_space['min_child_weight']),
                'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
                'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha'], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda'], log=True),
                'random_state': self.seed,
                'n_jobs': 1,
                'verbosity': 0
            }

            # TimeSeriesSplit交叉验证
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_flat):
                model = XGBRegressor(**params)
                model.fit(X_flat[train_idx], self.y_train[train_idx])
                val_pred = model.predict(X_flat[val_idx])
                score = r2_score(self.y_train[val_idx], val_pred)
                cv_scores.append(score)

            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            # 稳定性惩罚
            objective_value = mean_score - 0.05 * std_score

            trial.set_user_attr('mean_cv_score', float(mean_score))
            trial.set_user_attr('std_cv_score', float(std_score))

            return objective_value

        study = optuna.create_study(
            direction='maximize',
            study_name=f'xgboost_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=min(10, n_trials // 3),
                multivariate=True
            )
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.verbose
        )

        self.studies['xgboost'] = study
        self.best_params['xgboost'] = study.best_params
        self._save_results(study, 'xgboost')

        if self.verbose:
            self._print_optimization_summary(study, 'XGBoost')

        return study.best_params

    def get_best_params(self, model_type: str = None) -> Dict:
        """
        获取最优超参数

        Parameters:
        -----------
        model_type : str, optional
            模型类型 ('lstm', 'gru', 'xgboost')
            如果为None，返回所有模型的最优参数

        Returns:
        --------
        dict : 最优超参数字典
        """
        if model_type:
            return self.best_params.get(model_type, {})
        return self.best_params

    def visualize(
        self,
        model_type: str,
        save: bool = True,
        show: bool = False
    ):
        """
        生成优化可视化

        Parameters:
        -----------
        model_type : str
            模型类型 ('lstm', 'gru', 'xgboost')
        save : bool
            是否保存图像，默认True
        show : bool
            是否显示图像，默认False
        """
        if model_type not in self.studies:
            print(f"[WARNING] 未找到{model_type}的优化历史")
            return

        study = self.studies[model_type]

        try:
            # 优化历史
            fig1 = plot_optimization_history(study)
            if save:
                fig1.write_image(str(self.output_dir / f'{model_type}_optimization_history.png'))
            if show:
                fig1.show()

            # 参数重要性
            fig2 = plot_param_importances(study)
            if save:
                fig2.write_image(str(self.output_dir / f'{model_type}_param_importances.png'))
            if show:
                fig2.show()

            # 平行坐标图
            fig3 = plot_parallel_coordinate(study)
            if save:
                fig3.write_image(str(self.output_dir / f'{model_type}_parallel_coordinate.png'))
            if show:
                fig3.show()

            if self.verbose:
                print(f"[INFO] {model_type}可视化已保存至: {self.output_dir}")

        except Exception as e:
            print(f"[WARNING] 可视化失败: {e}")
            print("[INFO] 请安装kaleido: pip install kaleido")

    def generate_report(self, output_file: str = 'optimization_report.txt'):
        """
        生成优化报告

        Parameters:
        -----------
        output_file : str
            报告文件名，默认 'optimization_report.txt'
        """
        report_path = self.output_dir / output_file

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Optuna超参数优化报告\n".center(80))
            f.write("=" * 80 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"随机种子: {self.seed}\n")
            f.write(f"训练数据形状: {self.X_train.shape}\n\n")

            for model_type, study in self.studies.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{model_type.upper()}优化结果\n")
                f.write(f"{'=' * 80}\n\n")

                f.write(f"完成试验数: {len(study.trials)}\n")
                f.write(f"剪枝试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
                f.write(f"最佳目标值: {study.best_value:.6f}\n\n")

                f.write("最优超参数:\n")
                for param, value in study.best_params.items():
                    f.write(f"  {param}: {value}\n")

                # 用户属性
                best_trial = study.best_trial
                f.write("\n额外指标:\n")
                for key, value in best_trial.user_attrs.items():
                    f.write(f"  {key}: {value:.6f}\n")

        if self.verbose:
            print(f"[INFO] 优化报告已保存至: {report_path}")

    def _save_results(self, study, model_type: str):
        """保存优化结果"""
        # 保存最佳参数为JSON
        with open(self.output_dir / f'{model_type}_best_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=4)

        # 保存完整试验历史
        df = study.trials_dataframe()
        df.to_csv(self.output_dir / f'{model_type}_trials.csv', index=False)

    def _print_optimization_summary(self, study, model_name: str):
        """打印优化摘要"""
        print(f"\n{'=' * 80}")
        print(f"{model_name}优化完成".center(80))
        print(f"{'=' * 80}")
        print(f"最佳目标值: {study.best_value:.6f}")
        print(f"完成试验数: {len(study.trials)}")
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"剪枝试验数: {pruned} ({pruned/len(study.trials)*100:.1f}%)")
        print(f"\n最优超参数:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")

        # 显示用户属性
        best_trial = study.best_trial
        if best_trial.user_attrs:
            print(f"\n性能指标:")
            for key, value in best_trial.user_attrs.items():
                print(f"  {key}: {value:.6f}")
        print(f"{'=' * 80}\n")


# =====================================================================================
# 模块测试代码
# =====================================================================================

if __name__ == "__main__":
    """
    独立测试代码 - 验证模块功能
    """
    print("=" * 80)
    print("Optuna优化模块独立测试".center(80))
    print("=" * 80)

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    seq_len = 5

    # 3D序列数据（用于LSTM/GRU）
    X_seq = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randn(n_samples)

    print("\n[TEST] 模拟数据生成完成")
    print(f"  X_seq shape: {X_seq.shape}")
    print(f"  y shape: {y.shape}")

    # 测试LSTM优化
    print("\n" + "=" * 80)
    print("[TEST] 测试LSTM优化（10次试验）")
    print("=" * 80)

    optimizer = OptunaOptimizer(X_seq, y, seed=42, verbose=True)
    best_lstm_params = optimizer.optimize_lstm(n_trials=10, enable_pruning=True)

    print(f"\n✅ LSTM优化完成")
    print(f"最优参数: {best_lstm_params}")

    # 生成报告
    optimizer.generate_report()

    # 尝试生成可视化
    try:
        optimizer.visualize('lstm', show=False)
        print(f"\n✅ 可视化生成成功")
    except Exception as e:
        print(f"\n⚠️ 可视化失败: {e}")
        print("提示: 安装 kaleido 以启用可视化: pip install kaleido")

    print("\n" + "=" * 80)
    print("模块测试完成！".center(80))
    print(f"结果保存在: {optimizer.output_dir}".center(80))
    print("=" * 80)