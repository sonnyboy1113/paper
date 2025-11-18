"""
Diebold-Mariano Test Module
============================

Statistical testing module for comparing the performance significance of time series forecasting models

Author: AI Assistant
Date: 2025-10-16
Version: 1.0.0

Main Features:
- Single baseline model comparison
- Pairwise comparison
- Visualization analysis
- Detailed report generation

Usage Example:
    from dm_test import DieboldMarianoTest, quick_dm_analysis

    # Quick analysis
    results = quick_dm_analysis(y_true, predictions_dict, save_dir='./results/')

    # Or use the full class
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
    Diebold-Mariano Test Class

    Used to compare whether there are statistically significant differences in the performance of two or more prediction models

    Attributes:
        loss_function (str): Loss function type ('mse', 'mae', 'mape')
        results (dict): Store test results

    References:
        Diebold, F. X., & Mariano, R. S. (1995).
        Comparing predictive accuracy.
        Journal of Business & Economic Statistics, 13(3), 253-263.
    """

    def __init__(self, loss_function: str = 'mse'):
        """
        Initialize DM test

        Parameters:
            loss_function: Loss function type, options: 'mse', 'mae', 'mape'
        """
        self.loss_function = loss_function
        self.results = {}

    def _compute_loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Compute loss"""
        if self.loss_function == 'mse':
            return (actual - predicted) ** 2
        elif self.loss_function == 'mae':
            return np.abs(actual - predicted)
        elif self.loss_function == 'mape':
            return np.abs((actual - predicted) / (actual + 1e-8))
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

    def dm_test(self,
                actual: np.ndarray,
                pred1: np.ndarray,
                pred2: np.ndarray,
                h: int = 1,
                crit: str = "MSE",
                power: int = 2) -> Tuple[float, float]:
        """
        Execute Diebold-Mariano test

        Parameters:
            actual: True values
            pred1: Predictions from Model 1
            pred2: Predictions from Model 2
            h: Forecast horizon (for autocorrelation adjustment)
            crit: Loss function criterion
            power: Power of the loss function

        Returns:
            dm_stat: DM statistic
            p_value: p-value (two-sided test)
        """
        # Convert to 1D arrays
        actual = np.asarray(actual).flatten()
        pred1 = np.asarray(pred1).flatten()
        pred2 = np.asarray(pred2).flatten()

        # Calculate prediction errors
        e1 = actual - pred1
        e2 = actual - pred2

        # Calculate loss difference
        if crit == "MSE":
            d = (e1 ** 2) - (e2 ** 2)
        elif crit == "MAE":
            d = np.abs(e1) - np.abs(e2)
        elif crit == "MAPE":
            d = np.abs(e1 / (actual + 1e-8)) - np.abs(e2 / (actual + 1e-8))
        else:
            d = (np.abs(e1) ** power) - (np.abs(e2) ** power)

        # Calculate mean
        mean_d = np.mean(d)

        # Calculate autocovariance function
        def autocovariance(Xi, N, k, Xs):
            autoCov = 0
            for i in range(0, N - k):
                autoCov += ((Xi[i] - Xs) * (Xi[i + k] - Xs))
            return (1 / (N - 1)) * autoCov

        # Calculate long-run variance
        gamma = [autocovariance(d, len(d), lag, mean_d) for lag in range(0, h)]
        V_d = gamma[0] + 2 * sum(gamma[1:])

        # Prevent zero or negative variance
        if V_d <= 0:
            V_d = np.var(d, ddof=1)

        # Calculate DM statistic
        DM_stat = mean_d / sqrt(V_d / len(d))

        # Calculate p-value (two-sided test)
        p_value = 2 * (1 - stats.norm.cdf(abs(DM_stat)))

        return DM_stat, p_value

    def compare_models(self,
                       actual: np.ndarray,
                       predictions_dict: Dict[str, np.ndarray],
                       baseline_model: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple models relative to a baseline model

        Parameters:
            actual: True values
            predictions_dict: Dictionary mapping model names to predictions
            baseline_model: Baseline model name, if None uses the first model

        Returns:
            results_df: DataFrame containing all comparison results
        """
        if baseline_model is None:
            baseline_model = list(predictions_dict.keys())[0]

        if baseline_model not in predictions_dict:
            raise ValueError(f"Baseline model '{baseline_model}' not in predictions dictionary")

        baseline_pred = predictions_dict[baseline_model]

        results = []

        for model_name, pred in predictions_dict.items():
            if model_name == baseline_model:
                continue

            # Execute DM test
            dm_stat, p_value = self.dm_test(
                actual,
                baseline_pred,
                pred,
                h=1,
                crit="MSE"
            )

            # Calculate performance metrics
            mse_baseline = np.mean((actual.flatten() - baseline_pred.flatten()) ** 2)
            mse_model = np.mean((actual.flatten() - pred.flatten()) ** 2)
            mse_improvement = ((mse_baseline - mse_model) / mse_baseline) * 100

            # Determine significance
            if p_value < 0.01:
                significance = "***"
                sig_level = "Highly Significant"
            elif p_value < 0.05:
                significance = "**"
                sig_level = "Significant"
            elif p_value < 0.1:
                significance = "*"
                sig_level = "Marginally Significant"
            else:
                significance = ""
                sig_level = "Not Significant"

            # Determine direction
            if dm_stat > 0:
                direction = f"{model_name} superior to {baseline_model}"
            else:
                direction = f"{baseline_model} superior to {model_name}"

            results.append({
                'Comparison Model': model_name,
                'DM Statistic': dm_stat,
                'p-value': p_value,
                'Significance': significance,
                'Significance Level': sig_level,
                'MSE Improvement (%)': mse_improvement,
                'Is Significant': p_value < 0.05,
                'Conclusion': direction
            })

            # Save detailed results
            self.results[f"{baseline_model}_vs_{model_name}"] = {
                'dm_stat': dm_stat,
                'p_value': p_value,
                'significance': sig_level,
                'mse_improvement': mse_improvement
            }

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p-value')

        return results_df

    def pairwise_comparison(self,
                            actual: np.ndarray,
                            predictions_dict: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pairwise comparison of all models

        Parameters:
            actual: True values
            predictions_dict: Dictionary mapping model names to predictions

        Returns:
            dm_matrix_df: DM statistic matrix
            pvalue_matrix_df: p-value matrix
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)

        # Initialize matrices
        dm_matrix = np.zeros((n_models, n_models))
        pvalue_matrix = np.zeros((n_models, n_models))

        # Pairwise comparison
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

        # Create DataFrames
        dm_df = pd.DataFrame(dm_matrix, index=model_names, columns=model_names)
        pvalue_df = pd.DataFrame(pvalue_matrix, index=model_names, columns=model_names)

        return dm_df, pvalue_df

    def plot_comparison(self,
                        results_df: pd.DataFrame,
                        figsize: Tuple[int, int] = (18, 12),
                        save_path: Optional[str] = None) -> None:
        """
        Plot DM test comparison chart (optimized - fixed text overlap issue)

        Parameters:
            results_df: Comparison results DataFrame
            figsize: Figure size
            save_path: Save path
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Subplot 1: DM Statistic - Optimized label positioning
        colors1 = ['green' if x > 0 else 'red' for x in results_df['DM Statistic']]
        axes[0, 0].barh(range(len(results_df)), results_df['DM Statistic'],
                        color=colors1, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[0, 0].set_yticks(range(len(results_df)))
        axes[0, 0].set_yticklabels([name[:25] for name in results_df['Comparison Model']], fontsize=9)
        axes[0, 0].set_xlabel('DM Statistic', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Diebold-Mariano Statistic\n(Positive = Superior to Baseline)',
                             fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # Optimization: Intelligently place value labels to avoid overlap
        x_range = results_df['DM Statistic'].max() - results_df['DM Statistic'].min()
        offset = x_range * 0.08  # Dynamically calculate offset

        for i, (idx, row) in enumerate(results_df.iterrows()):
            dm_val = row['DM Statistic']
            sig = row['Significance']
            label = f"{dm_val:.3f}{sig}"

            # Intelligently adjust position based on value size and sign
            if abs(dm_val) < x_range * 0.15:  # For small values, place outside
                x_pos = dm_val + (offset if dm_val >= 0 else -offset)
                ha = 'left' if dm_val >= 0 else 'right'
            else:  # For large values, can place inside bar
                x_pos = dm_val * 0.95
                ha = 'right' if dm_val > 0 else 'left'

            axes[0, 0].text(x_pos, i, label, va='center', ha=ha,
                            fontweight='bold', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor='gray',
                                      alpha=0.8))

        # Subplot 2: p-value (no modification needed)
        p_colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red'
                    for p in results_df['p-value']]
        axes[0, 1].barh(range(len(results_df)), results_df['p-value'],
                        color=p_colors, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0.05, color='red', linestyle='--',
                           linewidth=2, label='p=0.05', alpha=0.7)
        axes[0, 1].axvline(x=0.1, color='orange', linestyle='--',
                           linewidth=2, label='p=0.1', alpha=0.7)
        axes[0, 1].set_yticks(range(len(results_df)))
        axes[0, 1].set_yticklabels([name[:25] for name in results_df['Comparison Model']], fontsize=9)
        axes[0, 1].set_xlabel('p-value', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Statistical Significance Test\n(p<0.05 is significant)',
                             fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # Subplot 3: MSE Improvement (no modification needed)
        mse_colors = ['green' if m > 0 else 'red' for m in results_df['MSE Improvement (%)']]
        axes[1, 0].barh(range(len(results_df)), results_df['MSE Improvement (%)'],
                        color=mse_colors, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[1, 0].set_yticks(range(len(results_df)))
        axes[1, 0].set_yticklabels([name[:25] for name in results_df['Comparison Model']], fontsize=9)
        axes[1, 0].set_xlabel('MSE Improvement (%)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('MSE Improvement vs Baseline\n(Positive = Better Performance)',
                             fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Subplot 4: Scatter plot
        scatter_colors = ['green' if sig else 'red' for sig in results_df['Is Significant']]
        axes[1, 1].scatter(results_df['DM Statistic'], results_df['MSE Improvement (%)'],
                           c=scatter_colors, s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axes[1, 1].axvline(x=0, color='gray', linestyle='--', linewidth=1)
        axes[1, 1].axvline(x=1.96, color='red', linestyle=':', linewidth=2,
                           label='DM=±1.96 (p≈0.05)', alpha=0.7)
        axes[1, 1].axvline(x=-1.96, color='red', linestyle=':', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('DM Statistic', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('MSE Improvement (%)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Statistical Significance vs Actual Improvement\n(Green=Significant, Red=Not Significant)',
                             fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

        # Annotate top 3 models
        for _, row in results_df.head(3).iterrows():
            axes[1, 1].annotate(
                row['Comparison Model'][:15],
                (row['DM Statistic'], row['MSE Improvement (%)']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=7, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
            )

        plt.suptitle('Diebold-Mariano Test Comprehensive Analysis',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved: {save_path}")

        plt.show()

    def plot_heatmap(self,
                     dm_df: pd.DataFrame,
                     pvalue_df: pd.DataFrame,
                     figsize: Tuple[int, int] = (14, 12),
                     save_path: Optional[str] = None) -> None:
        """
        Plot DM test heatmap

        Parameters:
            dm_df: DM statistic matrix
            pvalue_df: p-value matrix
            figsize: Figure size
            save_path: Save path
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # DM statistic heatmap
        sns.heatmap(dm_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'DM Statistic'}, ax=axes[0],
                    linewidths=0.5, square=True)
        axes[0].set_title('Diebold-Mariano Statistic Heatmap\n(Positive indicates row model superior to column model)',
                          fontsize=13, fontweight='bold', pad=15)

        # p-value heatmap (with significance markers)
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
                    cbar_kws={'label': 'p-value'}, ax=axes[1],
                    linewidths=0.5, square=True, vmin=0, vmax=0.1)
        axes[1].set_title('p-value Heatmap\n(* p<0.1, ** p<0.05, *** p<0.01)',
                          fontsize=13, fontweight='bold', pad=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved: {save_path}")

        plt.show()

    def generate_report(self,
                        results_df: pd.DataFrame,
                        baseline_model: str,
                        save_path: Optional[str] = None) -> None:
        """
        Generate DM test report

        Parameters:
            results_df: Comparison results DataFrame
            baseline_model: Baseline model name
            save_path: Save path
        """
        print("\n" + "=" * 100)
        print(f"Diebold-Mariano Test Report (Baseline Model: {baseline_model})".center(100))
        print("=" * 100)

        print(f"\n{'Model':<25} {'DM Statistic':>12} {'p-value':>10} {'Significance':>13} "
              f"{'MSE Improvement (%)':>20} {'Conclusion':<30}")
        print("-" * 100)

        for _, row in results_df.iterrows():
            print(f"{row['Comparison Model']:<25} {row['DM Statistic']:>12.4f} {row['p-value']:>10.4f} "
                  f"{row['Significance']:>13} {row['MSE Improvement (%)']:>20.2f} {row['Conclusion']:<30}")

        print("\n" + "=" * 100)
        print("Statistical Summary".center(100))
        print("=" * 100)

        n_significant = len(results_df[results_df['p-value'] < 0.05])
        n_marginal = len(results_df[(results_df['p-value'] >= 0.05) & (results_df['p-value'] < 0.1)])
        n_not_significant = len(results_df[results_df['p-value'] >= 0.1])

        print(f"\nTotal Comparisons: {len(results_df)}")
        print(f"  - Significantly superior to baseline (p < 0.05): {n_significant} models "
              f"({n_significant / len(results_df) * 100:.1f}%)")
        print(f"  - Marginally significant (0.05 ≤ p < 0.1): {n_marginal} models "
              f"({n_marginal / len(results_df) * 100:.1f}%)")
        print(f"  - No significant difference (p ≥ 0.1): {n_not_significant} models "
              f"({n_not_significant / len(results_df) * 100:.1f}%)")

        if n_significant > 0:
            best_models = results_df[results_df['p-value'] < 0.05].sort_values(
                'MSE Improvement (%)', ascending=False
            )
            print(f"\n🏆 Models Significantly Superior to Baseline (Top 3):")
            for i, (_, row) in enumerate(best_models.head(3).iterrows(), 1):
                print(f"  {i}. {row['Comparison Model']}: "
                      f"DM={row['DM Statistic']:.4f}, p={row['p-value']:.4f}, "
                      f"MSE Improvement={row['MSE Improvement (%)']:.2f}%")

        print(f"\n💡 Explanation:")
        print(f"  - DM Statistic: Positive values indicate the model is superior to {baseline_model}")
        print(f"  - p-value: Smaller values indicate more significant differences")
        print(f"  - Significance levels: *** p<0.01, ** p<0.05, * p<0.1")
        print(f"  - MSE Improvement: Positive values indicate lower MSE (better performance)")

        if save_path:
            results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ Report saved to: {save_path}")


# ========================================
# Convenience Functions
# ========================================

def quick_dm_analysis(y_true: np.ndarray,
                      predictions: Dict[str, np.ndarray],
                      baseline: str = None,
                      save_dir: str = './results/',
                      plot: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Quick DM test analysis (one-line solution)

    Parameters:
        y_true: True values
        predictions: Prediction results dictionary {model name: predictions}
        baseline: Baseline model name, defaults to first model
        save_dir: Results save directory
        plot: Whether to generate plots
        verbose: Whether to print detailed information

    Returns:
        results_df: DM test results DataFrame

    Example:
        >>> results = quick_dm_analysis(
        ...     y_true=y_test,
        ...     predictions={
        ...         'Baseline Model': pred_baseline,
        ...         'Strategy 1': pred_strategy1,
        ...         'Strategy 2': pred_strategy2
        ...     },
        ...     save_dir='./results/'
        ... )
    """
    import os

    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize
    dm_tester = DieboldMarianoTest(loss_function='mse')

    # Execute comparison
    if baseline is None:
        baseline = list(predictions.keys())[0]

    results_df = dm_tester.compare_models(y_true, predictions, baseline)

    # Generate report
    if verbose:
        dm_tester.generate_report(
            results_df,
            baseline,
            save_path=os.path.join(save_dir, 'dm_test_report.csv')
        )

    # Plot
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
    Quick pairwise comparison analysis

    Parameters:
        y_true: True values
        predictions: Prediction results dictionary
        save_dir: Results save directory
        plot: Whether to generate heatmap

    Returns:
        dm_matrix: DM statistic matrix
        pvalue_matrix: p-value matrix
    """
    import os

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dm_tester = DieboldMarianoTest()
    dm_matrix, pvalue_matrix = dm_tester.pairwise_comparison(y_true, predictions)

    # Save results
    dm_matrix.to_csv(os.path.join(save_dir, 'dm_matrix.csv'))
    pvalue_matrix.to_csv(os.path.join(save_dir, 'pvalue_matrix.csv'))

    # Plot
    if plot:
        dm_tester.plot_heatmap(
            dm_matrix,
            pvalue_matrix,
            save_path=os.path.join(save_dir, 'dm_heatmap.png')
        )

    return dm_matrix, pvalue_matrix


# ========================================
# Version Information
# ========================================

__version__ = '1.0.0'
__author__ = 'AI Assistant'
__all__ = ['DieboldMarianoTest', 'quick_dm_analysis', 'pairwise_dm_analysis']

if __name__ == '__main__':
    # Test code
    print("Diebold-Mariano Test Module")
    print(f"Version: {__version__}")
    print("\nUsage Example:")
    print(">>> from dm_test import quick_dm_analysis")
    print(">>> results = quick_dm_analysis(y_true, predictions_dict)")