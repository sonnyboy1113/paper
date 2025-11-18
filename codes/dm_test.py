"""
Diebold-Mariano Test Module
============================

统计检验模块，用于比较时间序列预测模型的性能差异显著性

作者: AI Assistant
日期: 2025-10-16
版本: 1.0.0

主要功能:
- 单一基准模型对比
- 两两配对比较
- 可视化分析
- 详细报告生成

使用示例:
    from dm_test import DieboldMarianoTest, quick_dm_analysis

    # 快速分析
    results = quick_dm_analysis(y_true, predictions_dict, save_dir='./results/')

    # 或使用完整类
    dm_tester = DieboldMarianoTest()
    results_df = dm_tester.compare_models(y_true, predictions_dict, 'baseline')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import sqrt
from typing import Dict, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class DieboldMarianoTest:
    """
    Diebold-Mariano检验类

    用于比较两个或多个预测模型的性能是否存在统计显著性差异

    Attributes:
        loss_function (str): 损失函数类型 ('mse', 'mae', 'mape')
        results (dict): 存储检验结果

    References:
        Diebold, F. X., & Mariano, R. S. (1995).
        Comparing predictive accuracy.
        Journal of Business & Economic Statistics, 13(3), 253-263.
    """

    def __init__(self, loss_function: str = 'mse'):
        """
        初始化DM检验

        Parameters:
            loss_function: 损失函数类型，可选 'mse', 'mae', 'mape'
        """
        self.loss_function = loss_function
        self.results = {}

    def _compute_loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """计算损失"""
        if self.loss_function == 'mse':
            return (actual - predicted) ** 2
        elif self.loss_function == 'mae':
            return np.abs(actual - predicted)
        elif self.loss_function == 'mape':
            return np.abs((actual - predicted) / (actual + 1e-8))
        else:
            raise ValueError(f"不支持的损失函数: {self.loss_function}")

    def dm_test(self,
                actual: np.ndarray,
                pred1: np.ndarray,
                pred2: np.ndarray,
                h: int = 1,
                crit: str = "MSE",
                power: int = 2) -> Tuple[float, float]:
        """
        执行Diebold-Mariano检验

        Parameters:
            actual: 真实值
            pred1: 模型1的预测值
            pred2: 模型2的预测值
            h: 预测步长（用于自相关调整）
            crit: 损失函数标准
            power: 损失函数的幂次

        Returns:
            dm_stat: DM统计量
            p_value: p值（双侧检验）
        """
        # 转换为一维数组
        actual = np.asarray(actual).flatten()
        pred1 = np.asarray(pred1).flatten()
        pred2 = np.asarray(pred2).flatten()

        # 计算预测误差
        e1 = actual - pred1
        e2 = actual - pred2

        # 计算损失差异
        if crit == "MSE":
            d = (e1 ** 2) - (e2 ** 2)
        elif crit == "MAE":
            d = np.abs(e1) - np.abs(e2)
        elif crit == "MAPE":
            d = np.abs(e1 / (actual + 1e-8)) - np.abs(e2 / (actual + 1e-8))
        else:
            d = (np.abs(e1) ** power) - (np.abs(e2) ** power)

        # 计算均值
        mean_d = np.mean(d)

        # 计算自协方差函数
        def autocovariance(Xi, N, k, Xs):
            autoCov = 0
            for i in range(0, N - k):
                autoCov += ((Xi[i] - Xs) * (Xi[i + k] - Xs))
            return (1 / (N - 1)) * autoCov

        # 计算长期方差
        gamma = [autocovariance(d, len(d), lag, mean_d) for lag in range(0, h)]
        V_d = gamma[0] + 2 * sum(gamma[1:])

        # 防止方差为0或负数
        if V_d <= 0:
            V_d = np.var(d, ddof=1)

        # 计算DM统计量
        DM_stat = mean_d / sqrt(V_d / len(d))

        # 计算p值（双侧检验）
        p_value = 2 * (1 - stats.norm.cdf(abs(DM_stat)))

        return DM_stat, p_value

    def compare_models(self,
                       actual: np.ndarray,
                       predictions_dict: Dict[str, np.ndarray],
                       baseline_model: Optional[str] = None) -> pd.DataFrame:
        """
        比较多个模型相对于基准模型的性能

        Parameters:
            actual: 真实值
            predictions_dict: 模型名称到预测值的字典
            baseline_model: 基准模型名称，如果为None则使用第一个模型

        Returns:
            results_df: 包含所有比较结果的DataFrame
        """
        if baseline_model is None:
            baseline_model = list(predictions_dict.keys())[0]

        if baseline_model not in predictions_dict:
            raise ValueError(f"基准模型 '{baseline_model}' 不在预测字典中")

        baseline_pred = predictions_dict[baseline_model]

        results = []

        for model_name, pred in predictions_dict.items():
            if model_name == baseline_model:
                continue

            # 执行DM检验
            dm_stat, p_value = self.dm_test(
                actual,
                baseline_pred,
                pred,
                h=1,
                crit="MSE"
            )

            # 计算性能指标
            mse_baseline = np.mean((actual.flatten() - baseline_pred.flatten()) ** 2)
            mse_model = np.mean((actual.flatten() - pred.flatten()) ** 2)
            mse_improvement = ((mse_baseline - mse_model) / mse_baseline) * 100

            # 判断显著性
            if p_value < 0.01:
                significance = "***"
                sig_level = "高度显著"
            elif p_value < 0.05:
                significance = "**"
                sig_level = "显著"
            elif p_value < 0.1:
                significance = "*"
                sig_level = "边际显著"
            else:
                significance = ""
                sig_level = "不显著"

            # 判断方向
            if dm_stat > 0:
                direction = f"{model_name}优于{baseline_model}"
            else:
                direction = f"{baseline_model}优于{model_name}"

            results.append({
                '比较模型': model_name,
                'DM统计量': dm_stat,
                'p值': p_value,
                '显著性': significance,
                '显著性水平': sig_level,
                'MSE改善(%)': mse_improvement,
                '是否显著': p_value < 0.05,
                '结论': direction
            })

            # 保存详细结果
            self.results[f"{baseline_model}_vs_{model_name}"] = {
                'dm_stat': dm_stat,
                'p_value': p_value,
                'significance': sig_level,
                'mse_improvement': mse_improvement
            }

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p值')

        return results_df

    def pairwise_comparison(self,
                            actual: np.ndarray,
                            predictions_dict: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        两两比较所有模型

        Parameters:
            actual: 真实值
            predictions_dict: 模型名称到预测值的字典

        Returns:
            dm_matrix_df: DM统计量矩阵
            pvalue_matrix_df: p值矩阵
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)

        # 初始化矩阵
        dm_matrix = np.zeros((n_models, n_models))
        pvalue_matrix = np.zeros((n_models, n_models))

        # 两两比较
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    dm_matrix[i, j] = 0
                    pvalue_matrix[i, j] = 1
                elif i < j:
                    pred1 = predictions_dict[model1].flatten()
                    pred2 = predictions_dict[model2].flatten()

                    dm_stat, p_value = self.dm_test(actual, pred1, pred2, h=1, crit="MSE")

                    dm_matrix[i, j] = dm_stat
                    dm_matrix[j, i] = -dm_stat
                    pvalue_matrix[i, j] = p_value
                    pvalue_matrix[j, i] = p_value

        # 创建DataFrame
        dm_df = pd.DataFrame(dm_matrix, index=model_names, columns=model_names)
        pvalue_df = pd.DataFrame(pvalue_matrix, index=model_names, columns=model_names)

        return dm_df, pvalue_df

    def plot_comparison(self,
                        results_df: pd.DataFrame,
                        figsize: Tuple[int, int] = (18, 12),
                        save_path: Optional[str] = None) -> None:
        """
        绘制DM检验比较图（优化版 - 解决文字重叠问题）

        Parameters:
            results_df: 比较结果DataFrame
            figsize: 图形大小
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 子图1: DM统计量 - 优化标签位置
        colors1 = ['green' if x > 0 else 'red' for x in results_df['DM统计量']]
        axes[0, 0].barh(range(len(results_df)), results_df['DM统计量'],
                        color=colors1, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[0, 0].set_yticks(range(len(results_df)))
        axes[0, 0].set_yticklabels([name[:25] for name in results_df['比较模型']], fontsize=9)
        axes[0, 0].set_xlabel('DM统计量', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Diebold-Mariano统计量\n(正值=优于基准)',
                             fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # 优化: 智能放置数值标签，避免重叠
        x_range = results_df['DM统计量'].max() - results_df['DM统计量'].min()
        offset = x_range * 0.08  # 动态计算偏移量

        for i, (idx, row) in enumerate(results_df.iterrows()):
            dm_val = row['DM统计量']
            sig = row['显著性']
            label = f"{dm_val:.3f}{sig}"

            # 根据数值大小和符号智能调整位置
            if abs(dm_val) < x_range * 0.15:  # 数值较小时，放在外侧
                x_pos = dm_val + (offset if dm_val >= 0 else -offset)
                ha = 'left' if dm_val >= 0 else 'right'
            else:  # 数值较大时，可以放在柱内
                x_pos = dm_val * 0.95
                ha = 'right' if dm_val > 0 else 'left'

            axes[0, 0].text(x_pos, i, label, va='center', ha=ha,
                            fontweight='bold', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor='gray',
                                      alpha=0.8))

        # 子图2: p值（无需修改）
        p_colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red'
                    for p in results_df['p值']]
        axes[0, 1].barh(range(len(results_df)), results_df['p值'],
                        color=p_colors, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0.05, color='red', linestyle='--',
                           linewidth=2, label='p=0.05', alpha=0.7)
        axes[0, 1].axvline(x=0.1, color='orange', linestyle='--',
                           linewidth=2, label='p=0.1', alpha=0.7)
        axes[0, 1].set_yticks(range(len(results_df)))
        axes[0, 1].set_yticklabels([name[:25] for name in results_df['比较模型']], fontsize=9)
        axes[0, 1].set_xlabel('p值', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('统计显著性检验\n(p<0.05为显著)',
                             fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # 子图3: MSE改善（无需修改）
        mse_colors = ['green' if m > 0 else 'red' for m in results_df['MSE改善(%)']]
        axes[1, 0].barh(range(len(results_df)), results_df['MSE改善(%)'],
                        color=mse_colors, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[1, 0].set_yticks(range(len(results_df)))
        axes[1, 0].set_yticklabels([name[:25] for name in results_df['比较模型']], fontsize=9)
        axes[1, 0].set_xlabel('MSE改善(%)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('相对基准的MSE改善\n(正值=性能更好)',
                             fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # 子图4: 散点图
        scatter_colors = ['green' if sig else 'red' for sig in results_df['是否显著']]
        axes[1, 1].scatter(results_df['DM统计量'], results_df['MSE改善(%)'],
                           c=scatter_colors, s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axes[1, 1].axvline(x=0, color='gray', linestyle='--', linewidth=1)
        axes[1, 1].axvline(x=1.96, color='red', linestyle=':', linewidth=2,
                           label='DM=±1.96 (p≈0.05)', alpha=0.7)
        axes[1, 1].axvline(x=-1.96, color='red', linestyle=':', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('DM统计量', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('MSE改善(%)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('统计显著性 vs 实际改善\n(绿色=显著, 红色=不显著)',
                             fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

        # 标注前3个模型
        for _, row in results_df.head(3).iterrows():
            axes[1, 1].annotate(
                row['比较模型'][:15],
                (row['DM统计量'], row['MSE改善(%)']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=7, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
            )

        plt.suptitle('Diebold-Mariano检验综合分析',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")

        plt.show()

    def plot_heatmap(self,
                     dm_df: pd.DataFrame,
                     pvalue_df: pd.DataFrame,
                     figsize: Tuple[int, int] = (14, 12),
                     save_path: Optional[str] = None) -> None:
        """
        绘制DM检验热力图

        Parameters:
            dm_df: DM统计量矩阵
            pvalue_df: p值矩阵
            figsize: 图形大小
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # DM统计量热力图
        sns.heatmap(dm_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'DM统计量'}, ax=axes[0],
                    linewidths=0.5, square=True)
        axes[0].set_title('Diebold-Mariano统计量热力图\n(正值表示行模型优于列模型)',
                          fontsize=13, fontweight='bold', pad=15)

        # p值热力图（带显著性标记）
        annot_matrix = pvalue_df.copy()
        for i in range(len(pvalue_df)):
            for j in range(len(pvalue_df.columns)):
                p_val = pvalue_df.iloc[i, j]
                if i == j:
                    annot_matrix.iloc[i, j] = "-"
                elif p_val < 0.01:
                    annot_matrix.iloc[i, j] = f"{p_val:.4f}***"
                elif p_val < 0.05:
                    annot_matrix.iloc[i, j] = f"{p_val:.4f}**"
                elif p_val < 0.1:
                    annot_matrix.iloc[i, j] = f"{p_val:.4f}*"
                else:
                    annot_matrix.iloc[i, j] = f"{p_val:.4f}"

        sns.heatmap(pvalue_df, annot=annot_matrix, fmt='', cmap='RdYlGn_r',
                    cbar_kws={'label': 'p值'}, ax=axes[1],
                    linewidths=0.5, square=True, vmin=0, vmax=0.1)
        axes[1].set_title('p值热力图\n(* p<0.1, ** p<0.05, *** p<0.01)',
                          fontsize=13, fontweight='bold', pad=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 热力图已保存: {save_path}")

        plt.show()

    def generate_report(self,
                        results_df: pd.DataFrame,
                        baseline_model: str,
                        save_path: Optional[str] = None) -> None:
        """
        生成DM检验报告

        Parameters:
            results_df: 比较结果DataFrame
            baseline_model: 基准模型名称
            save_path: 保存路径
        """
        print("\n" + "=" * 100)
        print(f"Diebold-Mariano检验报告 (基准模型: {baseline_model})".center(100))
        print("=" * 100)

        print(f"\n{'模型':<25} {'DM统计量':>12} {'p值':>10} {'显著性':>8} "
              f"{'MSE改善(%)':>12} {'结论':<30}")
        print("-" * 100)

        for _, row in results_df.iterrows():
            print(f"{row['比较模型']:<25} {row['DM统计量']:>12.4f} {row['p值']:>10.4f} "
                  f"{row['显著性']:>8} {row['MSE改善(%)']:>12.2f} {row['结论']:<30}")

        print("\n" + "=" * 100)
        print("统计摘要".center(100))
        print("=" * 100)

        n_significant = len(results_df[results_df['p值'] < 0.05])
        n_marginal = len(results_df[(results_df['p值'] >= 0.05) & (results_df['p值'] < 0.1)])
        n_not_significant = len(results_df[results_df['p值'] >= 0.1])

        print(f"\n总比较数: {len(results_df)}")
        print(f"  - 显著优于基准 (p < 0.05): {n_significant} 个 "
              f"({n_significant / len(results_df) * 100:.1f}%)")
        print(f"  - 边际显著 (0.05 ≤ p < 0.1): {n_marginal} 个 "
              f"({n_marginal / len(results_df) * 100:.1f}%)")
        print(f"  - 无显著差异 (p ≥ 0.1): {n_not_significant} 个 "
              f"({n_not_significant / len(results_df) * 100:.1f}%)")

        if n_significant > 0:
            best_models = results_df[results_df['p值'] < 0.05].sort_values(
                'MSE改善(%)', ascending=False
            )
            print(f"\n🏆 显著优于基准的模型 (Top 3):")
            for i, (_, row) in enumerate(best_models.head(3).iterrows(), 1):
                print(f"  {i}. {row['比较模型']}: "
                      f"DM={row['DM统计量']:.4f}, p={row['p值']:.4f}, "
                      f"MSE改善={row['MSE改善(%)']:.2f}%")

        print(f"\n💡 解释说明:")
        print(f"  - DM统计量: 正值表示该模型优于{baseline_model}")
        print(f"  - p值: 越小表示差异越显著")
        print(f"  - 显著性水平: *** p<0.01, ** p<0.05, * p<0.1")
        print(f"  - MSE改善: 正值表示该模型MSE更低（性能更好）")

        if save_path:
            results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 报告已保存至: {save_path}")


# ========================================
# 便捷函数
# ========================================

def quick_dm_analysis(y_true: np.ndarray,
                      predictions: Dict[str, np.ndarray],
                      baseline: str = None,
                      save_dir: str = './results/',
                      plot: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """
    快速执行DM检验分析（一行代码搞定）

    Parameters:
        y_true: 真实值
        predictions: 预测结果字典 {模型名: 预测值}
        baseline: 基准模型名称，默认使用第一个
        save_dir: 结果保存目录
        plot: 是否绘制图表
        verbose: 是否打印详细信息

    Returns:
        results_df: DM检验结果DataFrame

    Example:
        >>> results = quick_dm_analysis(
        ...     y_true=y_test,
        ...     predictions={
        ...         '基准模型': pred_baseline,
        ...         '策略1': pred_strategy1,
        ...         '策略2': pred_strategy2
        ...     },
        ...     save_dir='./results/'
        ... )
    """
    import os

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化
    dm_tester = DieboldMarianoTest(loss_function='mse')

    # 执行比较
    if baseline is None:
        baseline = list(predictions.keys())[0]

    results_df = dm_tester.compare_models(y_true, predictions, baseline)

    # 生成报告
    if verbose:
        dm_tester.generate_report(
            results_df,
            baseline,
            save_path=os.path.join(save_dir, 'dm_test_report.csv')
        )

    # 绘图
    if plot:
        dm_tester.plot_comparison(
            results_df,
            save_path=os.path.join(save_dir, 'dm_comparison.png')
        )

    return results_df


def pairwise_dm_analysis(y_true: np.ndarray,
                         predictions: Dict[str, np.ndarray],
                         save_dir: str = './results/',
                         plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    快速执行两两比较分析

    Parameters:
        y_true: 真实值
        predictions: 预测结果字典
        save_dir: 结果保存目录
        plot: 是否绘制热力图

    Returns:
        dm_matrix: DM统计量矩阵
        pvalue_matrix: p值矩阵
    """
    import os

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dm_tester = DieboldMarianoTest()
    dm_matrix, pvalue_matrix = dm_tester.pairwise_comparison(y_true, predictions)

    # 保存结果
    dm_matrix.to_csv(os.path.join(save_dir, 'dm_matrix.csv'))
    pvalue_matrix.to_csv(os.path.join(save_dir, 'pvalue_matrix.csv'))

    # 绘图
    if plot:
        dm_tester.plot_heatmap(
            dm_matrix,
            pvalue_matrix,
            save_path=os.path.join(save_dir, 'dm_heatmap.png')
        )

    return dm_matrix, pvalue_matrix


# ========================================
# 版本信息
# ========================================

__version__ = '1.0.0'
__author__ = 'AI Assistant'
__all__ = ['DieboldMarianoTest', 'quick_dm_analysis', 'pairwise_dm_analysis']

if __name__ == '__main__':
    # 测试代码
    print("Diebold-Mariano Test Module")
    print(f"Version: {__version__}")
    print("\n使用示例:")
    print(">>> from dm_test import quick_dm_analysis")
    print(">>> results = quick_dm_analysis(y_true, predictions_dict)")