"""
特征级别消融实验模块
====================

专门用于量化每个输入特征（如EPU、GPR）对预测性能的贡献度

功能：
1. 逐个移除特征的消融实验
2. 量化每个特征的绝对/相对贡献
3. 特征重要性排序
4. 特征组合效应分析
5. 学术报告生成

作者：Feature Ablation Module
版本：1.0.0
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
    特征消融分析器

    专门用于分析每个输入特征（如EPU、GPR、VIX等）对模型性能的贡献
    """

    def __init__(self, y_test_true, feature_names, output_dir='./feature_ablation/'):
        """
        初始化特征消融分析器

        Parameters:
        -----------
        y_test_true : array-like
            测试集真实值
        feature_names : list
            所有特征名称列表
        output_dir : str
            结果保存目录
        """
        self.y_test_true = np.array(y_test_true).flatten()
        self.feature_names = feature_names
        self.output_dir = output_dir

        self.experiments = {}
        self.baseline_performance = None

        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 100)
        print("【特征级别消融实验（Feature-Level Ablation Study）】".center(100))
        print("量化每个特征对预测性能的贡献度".center(100))
        print("=" * 100)
        print(f"\n初始化完成:")
        print(f"  - 特征总数: {len(feature_names)}")
        print(f"  - 特征列表: {feature_names}")
        print(f"  - 测试集样本数: {len(self.y_test_true)}")
        print(f"  - 输出目录: {output_dir}")

    def add_baseline(self, predictions, description="使用全部特征"):
        """
        添加基线实验（使用所有特征）

        Parameters:
        -----------
        predictions : array-like
            基线模型的预测结果
        description : str
            实验描述
        """
        predictions = np.array(predictions).flatten()

        metrics = self._calculate_metrics(predictions)

        self.experiments['全部特征（基线）'] = {
            'type': 'baseline',
            'predictions': predictions,
            'metrics': metrics,
            'description': description,
            'removed_features': [],
            'remaining_features': self.feature_names.copy()
        }

        self.baseline_performance = metrics

        print(f"\n✅ 基线实验已添加（使用全部{len(self.feature_names)}个特征）")
        print(f"   基线R^2: {metrics['R^2']:.6f}")
        print(f"   基线MAE: {metrics['MAE']:.6f}")
        print(f"   基线RMSE: {metrics['RMSE']:.6f}")

    def add_single_feature_removal(self, feature_name, predictions,
                                   train_time=None, description=""):
        """
        添加单个特征移除实验

        Parameters:
        -----------
        feature_name : str
            被移除的特征名称
        predictions : array-like
            移除该特征后的预测结果
        train_time : float, optional
            训练时间（秒）
        description : str
            实验描述
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"特征 '{feature_name}' 不在特征列表中")

        predictions = np.array(predictions).flatten()
        metrics = self._calculate_metrics(predictions)

        if train_time:
            metrics['train_time'] = train_time

        exp_name = f'w/o {feature_name}'

        self.experiments[exp_name] = {
            'type': 'single_removal',
            'predictions': predictions,
            'metrics': metrics,
            'description': description or f'移除特征: {feature_name}',
            'removed_features': [feature_name],
            'remaining_features': [f for f in self.feature_names if f != feature_name]
        }

        # 计算贡献度
        if self.baseline_performance:
            contribution = self._calculate_contribution(metrics)
            self.experiments[exp_name]['contribution'] = contribution

            print(f"\n✓ 实验已添加: {exp_name}")
            print(f"   R^2: {metrics['R^2']:.6f} (vs基线: {contribution['r2_drop']:+.6f})")
            print(f"   相对贡献: {contribution['relative_contribution']:.2f}%")
        else:
            print(f"\n✓ 实验已添加: {exp_name}")
            print(f"   ⚠️  请先添加基线实验以计算贡献度")

    def _calculate_metrics(self, predictions):
        """计算评估指标"""
        return {
            'R^2': r2_score(self.y_test_true, predictions),
            'MAE': mean_absolute_error(self.y_test_true, predictions),
            'RMSE': sqrt(mean_squared_error(self.y_test_true, predictions)),
            'MAPE': np.mean(np.abs((self.y_test_true - predictions) /
                                  (self.y_test_true + 1e-8))) * 100
        }

    def _calculate_contribution(self, ablated_metrics):
        """计算特征贡献度"""
        baseline_r2 = self.baseline_performance['R^2']
        ablated_r2 = ablated_metrics['R^2']

        # R^2下降（绝对贡献）
        r2_drop = baseline_r2 - ablated_r2

        # 相对贡献（%）
        relative_contribution = (r2_drop / baseline_r2) * 100 if baseline_r2 != 0 else 0

        # 重要性等级
        if relative_contribution >= 5:
            importance = "⭐⭐⭐ 关键"
        elif relative_contribution >= 2:
            importance = "⭐⭐ 重要"
        elif relative_contribution >= 1:
            importance = "⭐ 一般"
        else:
            importance = "○ 轻微"

        return {
            'r2_drop': r2_drop,
            'relative_contribution': relative_contribution,
            'importance_level': importance
        }

    def generate_feature_importance_report(self):
        """
        生成特征重要性报告

        Returns:
        --------
        report_df : DataFrame
            特征重要性报告表
        """
        print("\n" + "=" * 100)
        print("【特征重要性报告（Feature Importance Report）】".center(100))
        print("=" * 100)

        if self.baseline_performance is None:
            print("❌ 错误：请先添加基线实验")
            return None

        # 收集数据
        report_data = []

        for exp_name, exp_data in self.experiments.items():
            if exp_data['type'] == 'baseline':
                continue

            if exp_data['type'] == 'single_removal':
                feature = exp_data['removed_features'][0]
                metrics = exp_data['metrics']
                contribution = exp_data.get('contribution', {})

                report_data.append({
                    '特征名称 (Feature)': feature,
                    '移除后R^2 (R^2 w/o)': f"{metrics['R^2']:.6f}",
                    'R^2下降 (R^2 Drop)': f"{contribution.get('r2_drop', 0):.6f}",
                    '相对贡献(%) (Contribution)': f"{contribution.get('relative_contribution', 0):.2f}%",
                    '重要性 (Importance)': contribution.get('importance_level', 'N/A'),
                    'MAE w/o': f"{metrics['MAE']:.6f}",
                    'RMSE w/o': f"{metrics['RMSE']:.6f}"
                })

        # 创建DataFrame
        report_df = pd.DataFrame(report_data)

        # 排序（按R^2下降降序）
        if len(report_df) > 0:
            report_df['_sort_key'] = report_df['R^2下降 (R^2 Drop)'].str.replace('R^2下降 \\(R^2 Drop\\)', '').astype(float)
            report_df = report_df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)

        # 打印报告
        print(f"\n基线性能（使用全部{len(self.feature_names)}个特征）：")
        print(f"  R^2 = {self.baseline_performance['R^2']:.6f}")
        print(f"  MAE = {self.baseline_performance['MAE']:.6f}")
        print(f"  RMSE = {self.baseline_performance['RMSE']:.6f}")

        print(f"\n" + "-" * 100)
        print("特征重要性排序（按贡献度降序）：")
        print("-" * 100)
        print(report_df.to_string(index=False))

        # 保存CSV
        csv_path = os.path.join(self.output_dir, 'feature_importance_ranking.csv')
        report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 报告已保存: {csv_path}")

        # 关键发现
        self._print_key_findings(report_df)

        return report_df

    def _print_key_findings(self, report_df):
        """打印关键发现"""
        print("\n" + "=" * 100)
        print("【关键发现（Key Findings）】".center(100))
        print("=" * 100)

        if len(report_df) == 0:
            return

        # Top 3 最重要特征
        print(f"\n🏆 Top 3 最重要特征:")
        for i in range(min(3, len(report_df))):
            row = report_df.iloc[i]
            print(f"  {i+1}. {row['特征名称 (Feature)']}")
            print(f"     - R^2下降: {row['R^2下降 (R^2 Drop)']}")
            print(f"     - 相对贡献: {row['相对贡献(%) (Contribution)']}")
            print(f"     - 重要性: {row['重要性 (Importance)']}")

        # 统计摘要
        contributions = [float(row['相对贡献(%) (Contribution)'].rstrip('%'))
                        for _, row in report_df.iterrows()]

        critical_count = sum(1 for c in contributions if c >= 5)
        important_count = sum(1 for c in contributions if 2 <= c < 5)
        moderate_count = sum(1 for c in contributions if 1 <= c < 2)
        minor_count = sum(1 for c in contributions if c < 1)

        print(f"\n📊 重要性统计:")
        print(f"  - 关键特征 (≥5%): {critical_count}")
        print(f"  - 重要特征 (2-5%): {important_count}")
        print(f"  - 一般特征 (1-2%): {moderate_count}")
        print(f"  - 轻微特征 (<1%): {minor_count}")

        # 累计贡献度
        total_contribution = sum(contributions)
        print(f"\n💡 累计相对贡献度: {total_contribution:.2f}%")
        print(f"   说明: 所有特征共同贡献了基线模型{total_contribution:.1f}%的性能")

    def visualize_feature_importance(self, show=True):
        """
        可视化特征重要性

        Parameters:
        -----------
        show : bool
            是否显示图表
        """
        print("\n生成特征重要性可视化...")

        self._plot_feature_ranking(show)
        self._plot_contribution_breakdown(show)
        self._plot_performance_comparison(show)

        print(f"\n✅ 所有可视化已保存至: {self.output_dir}")

    def _plot_feature_ranking(self, show=True):
        """图1: 特征重要性排序"""
        # 提取数据
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
            print("  ⚠️ 无特征数据，跳过该图表")
            return

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # 配色方案
        colors = []
        for contrib in relative_contribs:
            if contrib >= 5:
                colors.append('#e74c3c')  # 红色 - 关键
            elif contrib >= 2:
                colors.append('#f39c12')  # 橙色 - 重要
            elif contrib >= 1:
                colors.append('#f1c40f')  # 黄色 - 一般
            else:
                colors.append('#95a5a6')  # 灰色 - 轻微

        # 子图1: 绝对贡献（R^2下降）
        bars1 = ax1.barh(range(len(features)), r2_drops, color=colors, alpha=0.8)

        for i, (bar, drop, contrib) in enumerate(zip(bars1, r2_drops, relative_contribs)):
            label = f'{drop:.6f}\n({contrib:.2f}%)'
            ax1.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                    label, ha='left', va='center', fontsize=9, fontweight='bold')

        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=11)
        ax1.set_xlabel('R^2 下降（绝对贡献）', fontsize=12, fontweight='bold')
        ax1.set_title('移除特征后的性能下降\n(数值越大=特征越重要)',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(0, color='black', linewidth=1)

        # 子图2: 相对贡献（%）
        bars2 = ax2.barh(range(len(features)), relative_contribs, color=colors, alpha=0.8)

        # 重要性阈值线
        ax2.axvline(5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='关键 (≥5%)')
        ax2.axvline(2, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='重要 (≥2%)')
        ax2.axvline(1, color='yellow', linestyle='--', alpha=0.5, linewidth=2, label='一般 (≥1%)')

        for i, (bar, contrib) in enumerate(zip(bars2, relative_contribs)):
            ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{contrib:.2f}%', ha='left', va='center',
                    fontsize=10, fontweight='bold')

        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features, fontsize=11)
        ax2.set_xlabel('相对贡献度 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('特征重要性（相对值）\n(占基线性能的百分比)',
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'特征重要性排序（Feature Importance Ranking）\n基线R^2={self.baseline_performance["R^2"]:.6f}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, '01_feature_importance_ranking.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        print("✓ 图1: 特征重要性排序")

    def _plot_contribution_breakdown(self, show=True):
        """图2: 贡献度分解（饼图+柱状图）"""
        # 提取数据
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

        # 排序
        sorted_indices = np.argsort(relative_contribs)[::-1]
        features = [features[i] for i in sorted_indices]
        relative_contribs = [relative_contribs[i] for i in sorted_indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # 子图1: 饼图（Top 5）
        top_n = min(5, len(features))
        top_features = features[:top_n]
        top_contribs = relative_contribs[:top_n]

        if len(features) > top_n:
            other_contrib = sum(relative_contribs[top_n:])
            top_features.append('其他特征')
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

        ax1.set_title(f'Top {top_n} 特征贡献度占比',
                     fontsize=13, fontweight='bold')

        # 子图2: 累计贡献曲线
        cumulative_contribs = np.cumsum(relative_contribs)

        ax2.bar(range(len(features)), relative_contribs,
               color=plt.cm.viridis(np.linspace(0.2, 0.8, len(features))),
               alpha=0.7, label='单个特征贡献')

        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(len(features)), cumulative_contribs,
                     'ro-', linewidth=3, markersize=8, label='累计贡献')
        ax2_twin.axhline(80, color='red', linestyle='--', alpha=0.5, label='80%阈值')

        ax2.set_xlabel('特征（按重要性排序）', fontsize=12, fontweight='bold')
        ax2.set_ylabel('相对贡献度 (%)', fontsize=11, fontweight='bold')
        ax2_twin.set_ylabel('累计贡献度 (%)', fontsize=11, fontweight='bold', color='red')

        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels(features, rotation=45, ha='right', fontsize=9)

        ax2.set_title('特征贡献度累计分析',
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

        print("✓ 图2: 贡献度分解")

    def _plot_performance_comparison(self, show=True):
        """图3: 性能对比矩阵"""
        # 提取数据
        exp_names = []
        r2_scores = []
        mae_scores = []
        rmse_scores = []

        # 先添加基线
        if self.baseline_performance:
            exp_names.append('全部特征\n(基线)')
            r2_scores.append(self.baseline_performance['R^2'])
            mae_scores.append(self.baseline_performance['MAE'])
            rmse_scores.append(self.baseline_performance['RMSE'])

        # 添加各个消融实验
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

        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        x_pos = np.arange(len(exp_names))
        colors = ['gold'] + ['lightblue'] * (len(exp_names) - 1)

        # 子图1: R^2
        bars1 = axes[0].bar(x_pos, r2_scores, color=colors, alpha=0.8, edgecolor='black')
        axes[0].axhline(self.baseline_performance['R^2'], color='red',
                       linestyle='--', linewidth=2, alpha=0.7, label='基线')

        for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(exp_names, fontsize=9)
        axes[0].set_ylabel('R^2 Score', fontsize=11, fontweight='bold')
        axes[0].set_title('R^2 对比\n(↑ 越高越好)', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # 子图2: MAE
        bars2 = axes[1].bar(x_pos, mae_scores, color=colors, alpha=0.8, edgecolor='black')
        axes[1].axhline(self.baseline_performance['MAE'], color='red',
                       linestyle='--', linewidth=2, alpha=0.7, label='基线')

        for i, (bar, score) in enumerate(zip(bars2, mae_scores)):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(exp_names, fontsize=9)
        axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
        axes[1].set_title('MAE 对比\n(↓ 越低越好)', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        # 子图3: RMSE
        bars3 = axes[2].bar(x_pos, rmse_scores, color=colors, alpha=0.8, edgecolor='black')
        axes[2].axhline(self.baseline_performance['RMSE'], color='red',
                       linestyle='--', linewidth=2, alpha=0.7, label='基线')

        for i, (bar, score) in enumerate(zip(bars3, rmse_scores)):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(exp_names, fontsize=9)
        axes[2].set_ylabel('RMSE', fontsize=11, fontweight='bold')
        axes[2].set_title('RMSE 对比\n(↓ 越低越好)', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.suptitle('移除不同特征后的性能对比',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, '03_performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        print("✓ 图3: 性能对比矩阵")

    def export_to_latex(self, filename='feature_importance.tex'):
        """导出LaTeX表格"""
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

        print(f"\n✅ LaTeX表格已保存: {filepath}")


# =====================================================================================
# 便捷函数
# =====================================================================================

def quick_feature_ablation(y_test, feature_names, baseline_pred,
                           ablation_predictions_dict, output_dir='./feature_ablation/'):
    """
    快速执行特征消融实验

    Parameters:
    -----------
    y_test : array-like
        测试集真实值
    feature_names : list
        特征名称列表
    baseline_pred : array-like
        基线预测（使用全部特征）
    ablation_predictions_dict : dict
        消融预测字典，格式：{feature_name: predictions}
    output_dir : str
        输出目录

    Returns:
    --------
    analyzer : FeatureAblationAnalyzer
        特征消融分析器对象

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

    # 添加基线
    analyzer.add_baseline(baseline_pred)

    # 添加各个消融实验
    for feature_name, predictions in ablation_predictions_dict.items():
        analyzer.add_single_feature_removal(feature_name, predictions)

    # 生成报告和可视化
    analyzer.generate_feature_importance_report()
    analyzer.visualize_feature_importance(show=False)
    analyzer.export_to_latex()

    return analyzer


if __name__ == "__main__":
    print("Feature-Level Ablation Study Module v1.0.0")
    print("\n主要功能：")
    print("  ✓ 量化每个特征的贡献度")
    print("  ✓ 特征重要性排序")
    print("  ✓ 生成学术报告和可视化")
    print("\n使用示例：")
    print("  from feature_ablation import FeatureAblationAnalyzer")
    print("  analyzer = FeatureAblationAnalyzer(y_test, feature_names)")
    print("  analyzer.add_baseline(baseline_predictions)")
    print("  analyzer.add_single_feature_removal('EPU', pred_without_epu)")
    print("  analyzer.generate_feature_importance_report()")