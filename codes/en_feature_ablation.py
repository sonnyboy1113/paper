"""
Feature-Level Ablation Experiment Module
=========================================

Specialized module for quantifying the contribution of each input feature (e.g., EPU, GPR)
to prediction performance

Features:
1. Ablation experiments by removing features one by one
2. Quantifying absolute/relative contribution of each feature
3. Feature importance ranking
4. Feature combination effect analysis
5. Academic report generation

Author: Feature Ablation Module
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
import os
import time
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class FeatureAblationAnalyzer:
    """
    Feature Ablation Analyzer

    Specialized for analyzing the contribution of each input feature (e.g., EPU, GPR, VIX, etc.)
    to model performance
    """

    def __init__(self, y_test_true, feature_names, output_dir='./feature_ablation/'):
        """
        Initialize Feature Ablation Analyzer

        Parameters:
        -----------
        y_test_true : array-like
            True values of test set
        feature_names : list
            List of all feature names
        output_dir : str
            Directory to save results
        """
        self.y_test_true = np.array(y_test_true).flatten()
        self.feature_names = feature_names
        self.output_dir = output_dir

        self.experiments = {}
        self.baseline_performance = None

        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 100)
        print("【Feature-Level Ablation Study】".center(100))
        print("Quantifying the Contribution of Each Feature to Prediction Performance".center(100))
        print("=" * 100)
        print(f"\nInitialization complete:")
        print(f"  - Total features: {len(feature_names)}")
        print(f"  - Feature list: {feature_names}")
        print(f"  - Test set samples: {len(self.y_test_true)}")
        print(f"  - Output directory: {output_dir}")

    def add_baseline(self, predictions, description="Using all features"):
        """
        Add baseline experiment (using all features)

        Parameters:
        -----------
        predictions : array-like
            Predictions from baseline model
        description : str
            Experiment description
        """
        predictions = np.array(predictions).flatten()

        metrics = self._calculate_metrics(predictions)

        self.experiments['All Features (Baseline)'] = {
            'type': 'baseline',
            'predictions': predictions,
            'metrics': metrics,
            'description': description,
            'removed_features': [],
            'remaining_features': self.feature_names.copy()
        }

        self.baseline_performance = metrics

        print(f"\n✅ Baseline experiment added (using all {len(self.feature_names)} features)")
        print(f"   Baseline R^2: {metrics['R^2']:.6f}")
        print(f"   Baseline MAE: {metrics['MAE']:.6f}")
        print(f"   Baseline RMSE: {metrics['RMSE']:.6f}")

    def add_single_feature_removal(self, feature_name, predictions,
                                   train_time=None, description=""):
        """
        Add single feature removal experiment

        Parameters:
        -----------
        feature_name : str
            Name of the removed feature
        predictions : array-like
            Predictions after removing this feature
        train_time : float, optional
            Training time (seconds)
        description : str
            Experiment description
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' is not in the feature list")

        predictions = np.array(predictions).flatten()
        metrics = self._calculate_metrics(predictions)

        if train_time:
            metrics['train_time'] = train_time

        exp_name = f'w/o {feature_name}'

        self.experiments[exp_name] = {
            'type': 'single_removal',
            'predictions': predictions,
            'metrics': metrics,
            'description': description or f'Remove feature: {feature_name}',
            'removed_features': [feature_name],
            'remaining_features': [f for f in self.feature_names if f != feature_name]
        }

        # Calculate contribution
        if self.baseline_performance:
            contribution = self._calculate_contribution(metrics)
            self.experiments[exp_name]['contribution'] = contribution

            print(f"\n✓ Experiment added: {exp_name}")
            print(f"   R^2: {metrics['R^2']:.6f} (vs baseline: {contribution['r2_drop']:+.6f})")
            print(f"   Relative contribution: {contribution['relative_contribution']:.2f}%")
        else:
            print(f"\n✓ Experiment added: {exp_name}")
            print(f"   ⚠️  Please add baseline experiment first to calculate contribution")

    def _calculate_metrics(self, predictions):
        """Calculate evaluation metrics"""
        return {
            'R^2': r2_score(self.y_test_true, predictions),
            'MAE': mean_absolute_error(self.y_test_true, predictions),
            'RMSE': sqrt(mean_squared_error(self.y_test_true, predictions)),
            'MAPE': np.mean(np.abs((self.y_test_true - predictions) /
                                  (self.y_test_true + 1e-8))) * 100
        }

    def _calculate_contribution(self, ablated_metrics):
        """Calculate feature contribution"""
        baseline_r2 = self.baseline_performance['R^2']
        ablated_r2 = ablated_metrics['R^2']

        # R^2 drop (absolute contribution)
        r2_drop = baseline_r2 - ablated_r2

        # Relative contribution (%)
        relative_contribution = (r2_drop / baseline_r2) * 100 if baseline_r2 != 0 else 0

        # Importance level
        if relative_contribution >= 5:
            importance = "⭐⭐⭐ Critical"
        elif relative_contribution >= 2:
            importance = "⭐⭐ Important"
        elif relative_contribution >= 1:
            importance = "⭐ Moderate"
        else:
            importance = "○ Minor"

        return {
            'r2_drop': r2_drop,
            'relative_contribution': relative_contribution,
            'importance_level': importance
        }

    def generate_feature_importance_report(self):
        """
        Generate feature importance report

        Returns:
        --------
        report_df : DataFrame
            Feature importance report table
        """
        print("\n" + "=" * 100)
        print("【Feature Importance Report】".center(100))
        print("=" * 100)

        if self.baseline_performance is None:
            print("❌ Error: Please add baseline experiment first")
            return None

        # Collect data
        report_data = []

        for exp_name, exp_data in self.experiments.items():
            if exp_data['type'] == 'baseline':
                continue

            if exp_data['type'] == 'single_removal':
                feature = exp_data['removed_features'][0]
                metrics = exp_data['metrics']
                contribution = exp_data.get('contribution', {})

                report_data.append({
                    'Feature Name': feature,
                    'R^2 w/o': f"{metrics['R^2']:.6f}",
                    'R^2 Drop': f"{contribution.get('r2_drop', 0):.6f}",
                    'Contribution (%)': f"{contribution.get('relative_contribution', 0):.2f}%",
                    'Importance': contribution.get('importance_level', 'N/A'),
                    'MAE w/o': f"{metrics['MAE']:.6f}",
                    'RMSE w/o': f"{metrics['RMSE']:.6f}"
                })

        # Create DataFrame
        report_df = pd.DataFrame(report_data)

        # Sort (by R^2 drop in descending order)
        if len(report_df) > 0:
            report_df['_sort_key'] = report_df['R^2 Drop'].str.replace('R^2 Drop', '').astype(float)
            report_df = report_df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)

        # Print report
        print(f"\nBaseline Performance (using all {len(self.feature_names)} features):")
        print(f"  R^2 = {self.baseline_performance['R^2']:.6f}")
        print(f"  MAE = {self.baseline_performance['MAE']:.6f}")
        print(f"  RMSE = {self.baseline_performance['RMSE']:.6f}")

        print(f"\n" + "-" * 100)
        print("Feature Importance Ranking (sorted by contribution in descending order):")
        print("-" * 100)
        print(report_df.to_string(index=False))

        # Save CSV
        csv_path = os.path.join(self.output_dir, 'feature_importance_ranking.csv')
        report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ Report saved: {csv_path}")

        # Key findings
        self._print_key_findings(report_df)

        return report_df

    def _print_key_findings(self, report_df):
        """Print key findings"""
        print("\n" + "=" * 100)
        print("【Key Findings】".center(100))
        print("=" * 100)

        if len(report_df) == 0:
            return

        # Top 3 most important features
        print(f"\n🏆 Top 3 Most Important Features:")
        for i in range(min(3, len(report_df))):
            row = report_df.iloc[i]
            print(f"  {i+1}. {row['Feature Name']}")
            print(f"     - R^2 Drop: {row['R^2 Drop']}")
            print(f"     - Relative Contribution: {row['Contribution (%)']}")
            print(f"     - Importance: {row['Importance']}")

        # Statistical summary
        contributions = [float(row['Contribution (%)'].rstrip('%'))
                        for _, row in report_df.iterrows()]

        critical_count = sum(1 for c in contributions if c >= 5)
        important_count = sum(1 for c in contributions if 2 <= c < 5)
        moderate_count = sum(1 for c in contributions if 1 <= c < 2)
        minor_count = sum(1 for c in contributions if c < 1)

        print(f"\n📊 Importance Statistics:")
        print(f"  - Critical features (≥5%): {critical_count}")
        print(f"  - Important features (2-5%): {important_count}")
        print(f"  - Moderate features (1-2%): {moderate_count}")
        print(f"  - Minor features (<1%): {minor_count}")

        # Cumulative contribution
        total_contribution = sum(contributions)
        print(f"\n💡 Total Relative Contribution: {total_contribution:.2f}%")
        print(f"   Note: All features collectively contribute {total_contribution:.1f}% of the baseline model performance")

    def visualize_feature_importance(self, show=True):
        """
        Visualize feature importance

        Parameters:
        -----------
        show : bool
            Whether to display the charts
        """
        print("\nGenerating feature importance visualizations...")

        self._plot_feature_ranking(show)
        self._plot_contribution_breakdown(show)
        self._plot_performance_comparison(show)

        print(f"\n✅ All visualizations saved to: {self.output_dir}")

    def _plot_feature_ranking(self, show=True):
        """Chart 1: Feature Importance Ranking"""
        # Extract data
        features = []
        r2_drops = []
        relative_contribs = []

        for exp_name, exp_data in sorted(
            self.experiments.items(),
            key=lambda x: x[1].get('contribution', {}).get('r2_drop', 0),
            reverse=True
        ):
            if exp_data['type'] == 'single_removal':
                feature = exp_data['removed_features'][0]
                contribution = exp_data.get('contribution', {})

                features.append(feature)
                r2_drops.append(contribution.get('r2_drop', 0))
                relative_contribs.append(contribution.get('relative_contribution', 0))

        if len(features) == 0:
            print("  ⚠️ No feature data, skipping this chart")
            return

        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Color scheme
        colors = []
        for contrib in relative_contribs:
            if contrib >= 5:
                colors.append('#e74c3c')  # Red - Critical
            elif contrib >= 2:
                colors.append('#f39c12')  # Orange - Important
            elif contrib >= 1:
                colors.append('#f1c40f')  # Yellow - Moderate
            else:
                colors.append('#95a5a6')  # Gray - Minor

        # Subplot 1: Absolute contribution (R^2 drop)
        bars1 = ax1.barh(range(len(features)), r2_drops, color=colors, alpha=0.8)

        for i, (bar, drop, contrib) in enumerate(zip(bars1, r2_drops, relative_contribs)):
            label = f'{drop:.6f}\n({contrib:.2f}%)'
            ax1.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                    label, ha='left', va='center', fontsize=9, fontweight='bold')

        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=11)
        ax1.set_xlabel('R^2 Drop (Absolute Contribution)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Drop After Feature Removal\n(Larger value = More important feature)',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(0, color='black', linewidth=1)

        # Subplot 2: Relative contribution (%)
        bars2 = ax2.barh(range(len(features)), relative_contribs, color=colors, alpha=0.8)

        # Importance threshold lines
        ax2.axvline(5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Critical (≥5%)')
        ax2.axvline(2, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Important (≥2%)')
        ax2.axvline(1, color='yellow', linestyle='--', alpha=0.5, linewidth=2, label='Moderate (≥1%)')

        for i, (bar, contrib) in enumerate(zip(bars2, relative_contribs)):
            ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{contrib:.2f}%', ha='left', va='center',
                    fontsize=10, fontweight='bold')

        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features, fontsize=11)
        ax2.set_xlabel('Relative Contribution (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Feature Importance (Relative Values)\n(Percentage of baseline performance)',
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'Feature Importance Ranking\nBaseline R^2={self.baseline_performance["R^2"]:.6f}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, '01_feature_importance_ranking.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        print("✓ Chart 1: Feature Importance Ranking")

    def _plot_contribution_breakdown(self, show=True):
        """Chart 2: Contribution Breakdown (Pie chart + Bar chart)"""
        # Extract data
        features = []
        relative_contribs = []

        for exp_name, exp_data in self.experiments.items():
            if exp_data['type'] == 'single_removal':
                feature = exp_data['removed_features'][0]
                contribution = exp_data.get('contribution', {})
                relative_contribs.append(contribution.get('relative_contribution', 0))
                features.append(feature)

        if len(features) == 0:
            return

        # Sort
        sorted_indices = np.argsort(relative_contribs)[::-1]
        features = [features[i] for i in sorted_indices]
        relative_contribs = [relative_contribs[i] for i in sorted_indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Subplot 1: Pie chart (Top 5)
        top_n = min(5, len(features))
        top_features = features[:top_n]
        top_contribs = relative_contribs[:top_n]

        if len(features) > top_n:
            other_contrib = sum(relative_contribs[top_n:])
            top_features.append('Other Features')
            top_contribs.append(other_contrib)

        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(top_features)))

        wedges, texts, autotexts = ax1.pie(
            top_contribs,
            labels=top_features,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_pie,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )

        ax1.set_title(f'Top {top_n} Feature Contribution Proportion',
                     fontsize=13, fontweight='bold')

        # Subplot 2: Cumulative contribution curve
        cumulative_contribs = np.cumsum(relative_contribs)

        ax2.bar(range(len(features)), relative_contribs,
               color=plt.cm.viridis(np.linspace(0.2, 0.8, len(features))),
               alpha=0.7, label='Individual Feature Contribution')

        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(len(features)), cumulative_contribs,
                     'ro-', linewidth=3, markersize=8, label='Cumulative Contribution')
        ax2_twin.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% Threshold')

        ax2.set_xlabel('Features (Sorted by Importance)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Contribution (%)', fontsize=11, fontweight='bold')
        ax2_twin.set_ylabel('Cumulative Contribution (%)', fontsize=11, fontweight='bold', color='red')

        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels(features, rotation=45, ha='right', fontsize=9)

        ax2.set_title('Feature Contribution Cumulative Analysis',
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2_twin.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, '02_contribution_breakdown.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        print("✓ Chart 2: Contribution Breakdown")

    def _plot_performance_comparison(self, show=True):
        """Chart 3: Performance Comparison Matrix"""
        # Extract data
        exp_names = []
        r2_scores = []
        mae_scores = []
        rmse_scores = []

        # Add baseline first
        if self.baseline_performance:
            exp_names.append('All Features\n(Baseline)')
            r2_scores.append(self.baseline_performance['R^2'])
            mae_scores.append(self.baseline_performance['MAE'])
            rmse_scores.append(self.baseline_performance['RMSE'])

        # Add each ablation experiment
        for exp_name, exp_data in sorted(
            self.experiments.items(),
            key=lambda x: x[1]['metrics']['R^2'],
            reverse=True
        ):
            if exp_data['type'] == 'single_removal':
                feature = exp_data['removed_features'][0]
                exp_names.append(f'w/o\n{feature}')
                r2_scores.append(exp_data['metrics']['R^2'])
                mae_scores.append(exp_data['metrics']['MAE'])
                rmse_scores.append(exp_data['metrics']['RMSE'])

        if len(exp_names) <= 1:
            return

        # Create chart
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        x_pos = np.arange(len(exp_names))
        colors = ['gold'] + ['lightblue'] * (len(exp_names) - 1)

        # Subplot 1: R^2
        bars1 = axes[0].bar(x_pos, r2_scores, color=colors, alpha=0.8, edgecolor='black')
        axes[0].axhline(self.baseline_performance['R^2'], color='red',
                       linestyle='--', linewidth=2, alpha=0.7, label='Baseline')

        for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(exp_names, fontsize=9)
        axes[0].set_ylabel('R^2 Score', fontsize=11, fontweight='bold')
        axes[0].set_title('R^2 Comparison\n(↑ Higher is better)', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Subplot 2: MAE
        bars2 = axes[1].bar(x_pos, mae_scores, color=colors, alpha=0.8, edgecolor='black')
        axes[1].axhline(self.baseline_performance['MAE'], color='red',
                       linestyle='--', linewidth=2, alpha=0.7, label='Baseline')

        for i, (bar, score) in enumerate(zip(bars2, mae_scores)):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(exp_names, fontsize=9)
        axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
        axes[1].set_title('MAE Comparison\n(↓ Lower is better)', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        # Subplot 3: RMSE
        bars3 = axes[2].bar(x_pos, rmse_scores, color=colors, alpha=0.8, edgecolor='black')
        axes[2].axhline(self.baseline_performance['RMSE'], color='red',
                       linestyle='--', linewidth=2, alpha=0.7, label='Baseline')

        for i, (bar, score) in enumerate(zip(bars3, rmse_scores)):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(exp_names, fontsize=9)
        axes[2].set_ylabel('RMSE', fontsize=11, fontweight='bold')
        axes[2].set_title('RMSE Comparison\n(↓ Lower is better)', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Performance Comparison After Removing Different Features',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, '03_performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        print("✓ Chart 3: Performance Comparison Matrix")

    def export_to_latex(self, filename='feature_importance.tex'):
        """Export LaTeX table"""
        report_df = self.generate_feature_importance_report()

        if report_df is None:
            return

        latex_content = []
        latex_content.append("% Feature Importance Table for LaTeX\n")
        latex_content.append("% Auto-generated by FeatureAblationAnalyzer\n\n")

        latex_content.append(report_df.to_latex(index=False, escape=False))

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(latex_content)

        print(f"\n✅ LaTeX table saved: {filepath}")


# =====================================================================================
# Convenience Functions
# =====================================================================================

def quick_feature_ablation(y_test, feature_names, baseline_pred,
                           ablation_predictions_dict, output_dir='./feature_ablation/'):
    """
    Quick Feature Ablation Experiment

    Parameters:
    -----------
    y_test : array-like
        True values of test set
    feature_names : list
        List of feature names
    baseline_pred : array-like
        Baseline predictions (using all features)
    ablation_predictions_dict : dict
        Ablation predictions dictionary, format: {feature_name: predictions}
    output_dir : str
        Output directory

    Returns:
    --------
    analyzer : FeatureAblationAnalyzer
        Feature ablation analyzer object

    Examples:
    ---------
    >>> feature_names = ['EPU', 'GPR', 'VIX', 'OIL']
    >>> ablation_preds = {
    ...     'EPU': pred_without_epu,
    ...     'GPR': pred_without_gpr,
    ...     'VIX': pred_without_vix,
    ...     'OIL': pred_without_oil
    ... }
    >>> analyzer = quick_feature_ablation(
    ...     y_test, feature_names, baseline_pred, ablation_preds
    ... )
    """
    analyzer = FeatureAblationAnalyzer(y_test, feature_names, output_dir)

    # Add baseline
    analyzer.add_baseline(baseline_pred)

    # Add each ablation experiment
    for feature_name, predictions in ablation_predictions_dict.items():
        analyzer.add_single_feature_removal(feature_name, predictions)

    # Generate report and visualizations
    analyzer.generate_feature_importance_report()
    analyzer.visualize_feature_importance(show=False)
    analyzer.export_to_latex()

    return analyzer


if __name__ == "__main__":
    print("Feature-Level Ablation Study Module v1.0.0")
    print("\nMain Features:")
    print("  ✓ Quantify contribution of each feature")
    print("  ✓ Feature importance ranking")
    print("  ✓ Generate academic reports and visualizations")
    print("\nUsage Example:")
    print("  from feature_ablation import FeatureAblationAnalyzer")
    print("  analyzer = FeatureAblationAnalyzer(y_test, feature_names)")
    print("  analyzer.add_baseline(baseline_predictions)")
    print("  analyzer.add_single_feature_removal('EPU', pred_without_epu)")
    print("  analyzer.generate_feature_importance_report()")