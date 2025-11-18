"""
消融实验配置文件
"""

# 基准模型名称
BASELINE_NAME = '完整模型'

# 输出目录
OUTPUT_DIR = './ablation_analysis/'

# 统计检验参数
BOOTSTRAP_N_ITERATIONS = 1000
SIGNIFICANCE_LEVEL = 0.05

# 可视化参数
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
SHOW_PLOTS = True

# 重要性阈值（相对贡献度%）
IMPORTANCE_THRESHOLDS = {
    'critical': 10,      # ≥10% 为关键
    'important': 5,      # ≥5% 为重要
    'moderate': 2        # ≥2% 为一般
}

# LaTeX导出设置
LATEX_FILENAME = 'ablation_tables.tex'

# CSV导出设置
CSV_ENCODING = 'utf-8-sig'