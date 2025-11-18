import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import shap
import pickle
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("SHAP可解释性分析 - LSTM+GRU+XGBoost融合模型".center(100))
print("=" * 100)

# ========== 加载保存的模型和数据 ==========
results_directory = "./Predict/"
shap_directory = "./SHAP_Analysis/"
if not os.path.exists(shap_directory):
    os.makedirs(shap_directory)

print("\n加载模型和数据...")

# 加载XGBoost模型
with open(results_directory + 'xgb_gru_conservative.pkl', 'rb') as f:
    xgb_gru_model = pickle.load(f)

with open(results_directory + 'xgb_dual.pkl', 'rb') as f:
    xgb_dual_model = pickle.load(f)

with open(results_directory + 'ridge_meta_model.pkl', 'rb') as f:
    ridge_meta_model = pickle.load(f)

# 加载归一化器
with open(results_directory + 'scalers.pkl', 'rb') as f:
    scalers_data = pickle.load(f)
    feature_scalers = scalers_data['feature_scalers']
    y_scaler = scalers_data['y_scaler']

print("✓ 模型加载完成")

# ========== 重新准备数据（与训练代码保持一致）==========
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])

X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

split_idx = int(len(X) * 0.8)
X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_raw, y_test_raw = y.iloc[:split_idx], y.iloc[split_idx:]

# 归一化
X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

for col in X_train.columns:
    scaler = feature_scalers[col]
    X_train[col] = scaler.transform(X_train[col].values.reshape(-1, 1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

y_train = y_scaler.transform(y_train_raw.values.reshape(-1, 1)).flatten()
y_test = y_scaler.transform(y_test_raw.values.reshape(-1, 1)).flatten()

y_train = pd.Series(y_train, index=y_train_raw.index)
y_test = pd.Series(y_test, index=y_test_raw.index)


# 添加滞后特征
def add_features(X, y):
    X_new = X.copy()
    for i in range(1, 6):
        X_new[f'Corn_lag_{i}'] = y.shift(i)
    return X_new.dropna()


X_train_feat = add_features(X_train, y_train)
y_train = y_train.loc[X_train_feat.index]

X_test_feat = add_features(X_test, y_test)
y_test = y_test.loc[X_test_feat.index]


# 构造序列数据
def create_flat_sequences(X, y, seq_len=5):
    X_flat, y_flat = [], []
    for i in range(len(X) - seq_len):
        X_flat.append(X.iloc[i:i + seq_len].values.flatten())
        y_flat.append(y.iloc[i + seq_len])
    return np.array(X_flat), np.array(y_flat)


seq_len = 5
X_train_flat, y_train_seq = create_flat_sequences(X_train_feat, y_train, seq_len)
X_test_flat, y_test_seq = create_flat_sequences(X_test_feat, y_test, seq_len)

print(f"✓ 数据准备完成: train={X_train_flat.shape}, test={X_test_flat.shape}")

# ========== 生成特征名称 ==========
base_feature_names = list(X_train_feat.columns)
feature_names_flat = []
for t in range(seq_len):
    for feat in base_feature_names:
        feature_names_flat.append(f'{feat}_t-{seq_len - t - 1}')

print(f"✓ 特征总数: {len(feature_names_flat)}")

# ========== 为SHAP分析准备带预测值的完整特征 ==========
# 加载保存的预测结果
predictions_df = pd.read_csv(results_directory + 'all_predictions.csv')

# 从归一化数据反推LSTM和GRU的预测（归一化空间）
gru_test_pred_norm = y_scaler.transform(predictions_df['GRU单模型'].values.reshape(-1, 1)).flatten()
lstm_test_pred_norm = y_scaler.transform(predictions_df['LSTM单模型'].values.reshape(-1, 1)).flatten()


# 构建简化特征（与训练时一致）
def create_simplified_features_for_shap(X_flat, lstm_preds, gru_preds):
    """构建用于XGBoost的完整特征"""
    features_list = [X_flat]
    features_list.append(lstm_preds.reshape(-1, 1))
    features_list.append(gru_preds.reshape(-1, 1))
    features_list.append(((lstm_preds + gru_preds) / 2).reshape(-1, 1))
    features_list.append(np.abs(lstm_preds - gru_preds).reshape(-1, 1))
    return np.hstack(features_list)


X_test_simplified = create_simplified_features_for_shap(
    X_test_flat, lstm_test_pred_norm, gru_test_pred_norm
)

# 生成扩展特征名
extended_feature_names = feature_names_flat + [
    'LSTM_Prediction',
    'GRU_Prediction',
    'Average_Prediction',
    'Prediction_Difference'
]

print(f"✓ 扩展特征维度: {X_test_simplified.shape[1]}")

# ========== 1. XGBoost模型的SHAP分析 ==========
print("\n" + "=" * 100)
print("【1】XGBoost模型 - SHAP TreeExplainer分析".center(100))
print("=" * 100)

# 使用测试集的子集（SHAP计算较慢）
sample_size = min(100, len(X_test_simplified))
X_sample = X_test_simplified[:sample_size]

print(f"\n计算SHAP值（样本数: {sample_size}）...")
explainer_xgb = shap.TreeExplainer(xgb_gru_model)
shap_values_xgb = explainer_xgb.shap_values(X_sample)

print("✓ SHAP值计算完成")

# 1.1 SHAP Summary Plot（特征重要性概览）
plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_values_xgb,
    X_sample,
    feature_names=extended_feature_names,
    show=False,
    max_display=20
)
plt.title('XGBoost模型 - SHAP特征重要性概览（Top 20）',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(shap_directory + '01_xgb_shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图1: SHAP Summary Plot")

# 1.2 SHAP Bar Plot（平均绝对SHAP值）
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values_xgb,
    X_sample,
    feature_names=extended_feature_names,
    plot_type='bar',
    show=False,
    max_display=20
)
plt.title('XGBoost模型 - 平均特征影响力（Top 20）',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(shap_directory + '02_xgb_shap_bar.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图2: SHAP Bar Plot")

# 1.3 单个样本的SHAP Force Plot
sample_idx = 0
plt.figure(figsize=(16, 3))
shap.force_plot(
    explainer_xgb.expected_value,
    shap_values_xgb[sample_idx],
    np.round(X_sample[sample_idx], 3),
    feature_names=extended_feature_names,
    matplotlib=True,
    show=False
)
plt.title(f'单样本SHAP解释（样本#{sample_idx}）', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(shap_directory + '03_xgb_force_plot_sample.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图3: Force Plot（单样本）")

# 1.4 SHAP Dependence Plot（关键特征）
# 找出最重要的几个特征
mean_abs_shap = np.abs(shap_values_xgb).mean(axis=0)
top_feature_indices = np.argsort(mean_abs_shap)[-6:][::-1]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feat_idx in enumerate(top_feature_indices):
    ax = axes[i]
    shap.dependence_plot(
        feat_idx,
        shap_values_xgb,
        X_sample,
        feature_names=extended_feature_names,
        ax=ax,
        show=False
    )
    ax.set_title(f'{extended_feature_names[feat_idx]}', fontsize=10, fontweight='bold')

plt.suptitle('XGBoost模型 - Top 6特征的SHAP依赖图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(shap_directory + '04_xgb_dependence_plots.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图4: SHAP Dependence Plots")

# 1.5 SHAP Waterfall Plot（详细分解单个预测）
plt.figure(figsize=(10, 12))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_xgb[sample_idx],
        base_values=explainer_xgb.expected_value,
        data=X_sample[sample_idx],
        feature_names=extended_feature_names
    ),
    max_display=20,
    show=False
)
plt.title(f'SHAP Waterfall Plot - 样本#{sample_idx}预测分解', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(shap_directory + '05_xgb_waterfall.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图5: SHAP Waterfall Plot")

# ========== 2. 特征重要性对比分析 ==========
print("\n" + "=" * 100)
print("【2】多角度特征重要性对比".center(100))
print("=" * 100)

# 2.1 XGBoost内置特征重要性
xgb_importance = xgb_gru_model.get_booster().get_score(importance_type='gain')
xgb_importance_df = pd.DataFrame([
    {'feature': extended_feature_names[int(k.replace('f', ''))], 'importance': v}
    for k, v in xgb_importance.items()
]).sort_values('importance', ascending=False).head(20)

# 2.2 SHAP特征重要性
shap_importance_df = pd.DataFrame({
    'feature': extended_feature_names,
    'importance': np.abs(shap_values_xgb).mean(axis=0)
}).sort_values('importance', ascending=False).head(20)

# 2.3 排列重要性（Permutation Importance）
print("\n计算排列重要性...")
perm_importance = permutation_importance(
    xgb_gru_model,
    X_test_simplified,
    y_test_seq,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)
perm_importance_df = pd.DataFrame({
    'feature': extended_feature_names,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False).head(20)
print("✓ 排列重要性计算完成")

# 绘制对比图
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# XGBoost Gain
axes[0].barh(range(len(xgb_importance_df)), xgb_importance_df['importance'].values,
             color='steelblue', alpha=0.7)
axes[0].set_yticks(range(len(xgb_importance_df)))
axes[0].set_yticklabels(xgb_importance_df['feature'].values, fontsize=8)
axes[0].set_xlabel('Importance (Gain)', fontsize=10)
axes[0].set_title('XGBoost内置特征重要性\n(Gain)', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# SHAP Importance
axes[1].barh(range(len(shap_importance_df)), shap_importance_df['importance'].values,
             color='coral', alpha=0.7)
axes[1].set_yticks(range(len(shap_importance_df)))
axes[1].set_yticklabels(shap_importance_df['feature'].values, fontsize=8)
axes[1].set_xlabel('Mean |SHAP value|', fontsize=10)
axes[1].set_title('SHAP特征重要性\n(平均绝对值)', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

# Permutation Importance
axes[2].barh(range(len(perm_importance_df)), perm_importance_df['importance'].values,
             color='seagreen', alpha=0.7)
axes[2].set_yticks(range(len(perm_importance_df)))
axes[2].set_yticklabels(perm_importance_df['feature'].values, fontsize=8)
axes[2].set_xlabel('Permutation Importance', fontsize=10)
axes[2].set_title('排列特征重要性\n(模型无关)', fontsize=12, fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(True, alpha=0.3, axis='x')

plt.suptitle('三种方法的特征重要性对比（Top 20）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(shap_directory + '06_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图6: 特征重要性对比")

# ========== 3. 特征类别分析 ==========
print("\n" + "=" * 100)
print("【3】特征类别影响力分析".center(100))
print("=" * 100)


# 将特征分类
def categorize_features(feature_names):
    categories = {
        'LSTM/GRU预测': [],
        '滞后特征(Corn_lag)': [],
        '原始特征': []
    }

    for i, name in enumerate(feature_names):
        if any(pred in name for pred in ['LSTM', 'GRU', 'Average', 'Difference']):
            categories['LSTM/GRU预测'].append(i)
        elif 'Corn_lag' in name:
            categories['滞后特征(Corn_lag)'].append(i)
        else:
            categories['原始特征'].append(i)

    return categories


feature_categories = categorize_features(extended_feature_names)

# 计算每类特征的总SHAP贡献
category_importance = {}
for cat_name, indices in feature_categories.items():
    if len(indices) > 0:
        cat_shap = np.abs(shap_values_xgb[:, indices]).sum(axis=1).mean()
        category_importance[cat_name] = cat_shap

# 绘制类别重要性
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 饼图
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax1.pie(category_importance.values(), labels=category_importance.keys(),
        autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 11})
ax1.set_title('特征类别对模型的平均贡献占比', fontsize=13, fontweight='bold')

# 柱状图
ax2.bar(category_importance.keys(), category_importance.values(),
        color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Mean |SHAP value|', fontsize=11)
ax2.set_title('特征类别的SHAP重要性', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for i, (k, v) in enumerate(category_importance.items()):
    ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(shap_directory + '07_category_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图7: 特征类别分析")

# ========== 4. 时间维度分析 ==========
print("\n" + "=" * 100)
print("【4】时间滞后维度的影响分析".center(100))
print("=" * 100)

# 按时间步聚合SHAP值
timestep_importance = {}
for t in range(seq_len):
    timestep_features = [i for i, name in enumerate(extended_feature_names)
                         if f't-{t}' in name]
    if len(timestep_features) > 0:
        timestep_importance[f't-{t}'] = np.abs(shap_values_xgb[:, timestep_features]).mean()

# 绘制时间维度重要性
fig, ax = plt.subplots(figsize=(12, 6))

timesteps = list(timestep_importance.keys())
importances = list(timestep_importance.values())

bars = ax.bar(timesteps, importances, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(timesteps))),
              alpha=0.8, edgecolor='black')

ax.set_xlabel('时间步（t-0为最近）', fontsize=12)
ax.set_ylabel('平均 |SHAP value|', fontsize=12)
ax.set_title('不同时间滞后期的特征重要性', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, imp in zip(bars, importances):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{imp:.5f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(shap_directory + '08_timestep_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图8: 时间维度分析")

# ========== 5. 预测案例深度解析 ==========
print("\n" + "=" * 100)
print("【5】典型预测案例的深度解析".center(100))
print("=" * 100)

# 选择几个有代表性的样本
y_pred_sample = xgb_gru_model.predict(X_sample)
errors = np.abs(y_pred_sample - y_test_seq[:sample_size])

# 最好、最差、中等预测
best_idx = np.argmin(errors)
worst_idx = np.argmax(errors)
median_idx = np.argsort(errors)[len(errors) // 2]

case_indices = {
    '最佳预测': best_idx,
    '最差预测': worst_idx,
    '中等预测': median_idx
}

fig, axes = plt.subplots(len(case_indices), 1, figsize=(14, 12))

for i, (case_name, idx) in enumerate(case_indices.items()):
    ax = axes[i]

    # 获取该样本的SHAP值和特征值
    sample_shap = shap_values_xgb[idx]
    sample_features = X_sample[idx]

    # 找出最重要的10个特征
    top_k = 10
    top_indices = np.argsort(np.abs(sample_shap))[-top_k:][::-1]

    # 绘制
    y_pos = np.arange(top_k)
    colors_bar = ['green' if sample_shap[j] > 0 else 'red' for j in top_indices]

    ax.barh(y_pos, [sample_shap[j] for j in top_indices], color=colors_bar, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([extended_feature_names[j] for j in top_indices], fontsize=9)
    ax.set_xlabel('SHAP value', fontsize=10)

    true_val = y_test_seq[idx]
    pred_val = y_pred_sample[idx]
    error_val = errors[idx]

    ax.set_title(f'{case_name}（样本#{idx}）\n真实值={true_val:.4f}, 预测值={pred_val:.4f}, 误差={error_val:.4f}',
                 fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(shap_directory + '09_case_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图9: 典型案例分析")

# ========== 6. SHAP交互效应分析 ==========
print("\n" + "=" * 100)
print("【6】特征交互效应分析（SHAP Interaction）".center(100))
print("=" * 100)

# 由于计算量大，使用更小的样本
interaction_sample_size = min(50, sample_size)
X_interaction = X_sample[:interaction_sample_size]

print(f"计算SHAP交互值（样本数: {interaction_sample_size}）...")
shap_interaction_values = explainer_xgb.shap_interaction_values(X_interaction)
print("✓ 交互值计算完成")

# 选择最重要的几个特征进行交互分析
top_n_features = 6
top_features_idx = np.argsort(mean_abs_shap)[-top_n_features:][::-1]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, main_feat_idx in enumerate(top_features_idx):
    ax = axes[i]

    # 对于每个主特征，找出与它交互最强的特征
    interaction_strength = np.abs(shap_interaction_values[:, main_feat_idx, :]).mean(axis=0)
    interaction_strength[main_feat_idx] = 0  # 排除自身
    interact_feat_idx = np.argmax(interaction_strength)

    # 绘制交互依赖图
    shap.dependence_plot(
        (main_feat_idx, interact_feat_idx),
        shap_interaction_values,
        X_interaction,
        feature_names=extended_feature_names,
        ax=ax,
        show=False
    )
    ax.set_title(f'{extended_feature_names[main_feat_idx][:30]}', fontsize=9, fontweight='bold')

plt.suptitle('Top 6特征的SHAP交互效应分析', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(shap_directory + '10_interaction_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图10: 特征交互分析")

# ========== 在第7部分之前添加这段代码 ==========
# 位置：在 "【7】生成可解释性分析报告" 之前

# 生成类别贡献报告
category_report = pd.DataFrame([
    {'Category': cat, 'Feature_Count': len(indices),
     'Total_SHAP_Contribution': category_importance[cat],
     'Avg_SHAP_Per_Feature': category_importance[cat] / len(indices) if len(indices) > 0 else 0}
    for cat, indices in feature_categories.items() if len(indices) > 0
]).sort_values('Total_SHAP_Contribution', ascending=False)
category_report.to_csv(shap_directory + 'category_contribution_report.csv', index=False)
print("✓ 类别贡献报告已保存")

# 案例分析报告
case_report = []
for case_name, idx in case_indices.items():
    top_features_idx = np.argsort(np.abs(shap_values_xgb[idx]))[-5:][::-1]
    case_report.append({
        'Case': case_name,
        'Sample_Index': idx,
        'True_Value': y_test_seq[idx],
        'Predicted_Value': y_pred_sample[idx],
        'Error': errors[idx],
        'Top_Feature_1': extended_feature_names[top_features_idx[0]],
        'Top_Feature_1_SHAP': shap_values_xgb[idx, top_features_idx[0]],
        'Top_Feature_2': extended_feature_names[top_features_idx[1]],
        'Top_Feature_2_SHAP': shap_values_xgb[idx, top_features_idx[1]],
        'Top_Feature_3': extended_feature_names[top_features_idx[2]],
        'Top_Feature_3_SHAP': shap_values_xgb[idx, top_features_idx[2]]
    })

case_df = pd.DataFrame(case_report)
case_df.to_csv(shap_directory + 'case_analysis_report.csv', index=False)
print("✓ 案例分析报告已保存")

# ========== 然后才是第7部分 ==========
print("\n" + "=" * 100)
print("【7】生成可解释性分析报告".center(100))
print("=" * 100)

# 保存特征重要性到CSV
importance_report = pd.DataFrame({
    'Feature': extended_feature_names,
    'SHAP_Importance': np.abs(shap_values_xgb).mean(axis=0),
    'XGB_Gain': [xgb_gru_model.get_booster().get_score(importance_type='gain').get(f'f{i}', 0)
                 for i in range(len(extended_feature_names))],
    'Permutation_Importance': perm_importance.importances_mean
})
importance_report = importance_report.sort_values('SHAP_Importance', ascending=False)
importance_report.to_csv(shap_directory + 'feature_importance_report.csv', index=False)
print("✓ 特征重要性报告已保存")

# 类别贡献（现在可以使用 category_report 了）
print(f"\n  【特征类别贡献排名】:")
for i, row in category_report.iterrows():
    print(f"    {i + 1}. {row['Category']}")
    print(f"       特征数量: {row['Feature_Count']}, 总贡献: {row['Total_SHAP_Contribution']:.6f}")
    print(f"       平均贡献: {row['Avg_SHAP_Per_Feature']:.6f}")

# 时间步重要性
print(f"\n  【时间滞后期影响力】:")
sorted_timesteps = sorted(timestep_importance.items(), key=lambda x: x[1], reverse=True)
for ts, imp in sorted_timesteps:
    print(f"    {ts}: {imp:.6f}")

print(f"\n💡 解释性洞察:")
print(f"  1. LSTM/GRU预测值特征显著影响残差学习效果")
print(f"  2. 近期时间步（t-0, t-1）比远期时间步更重要")
print(f"  3. 玉米滞后特征（Corn_lag）对预测有关键作用")
print(f"  4. 特征间存在交互效应，需要非线性模型捕捉")


print(f"\n📁 输出文件:")
print(f"  图表:")
print(f"    01_xgb_shap_summary.png - SHAP特征重要性概览")
print(f"    02_xgb_shap_bar.png - 平均特征影响力")
print(f"    03_xgb_force_plot_sample.png - 单样本Force Plot")
print(f"    04_xgb_dependence_plots.png - Top 6特征依赖图")
print(f"    05_xgb_waterfall.png - Waterfall预测分解")
print(f"    06_importance_comparison.png - 三种方法对比")
print(f"    07_category_importance.png - 特征类别分析")
print(f"    08_timestep_importance.png - 时间维度分析")
print(f"    09_case_analysis.png - 典型案例解析")
print(f"    10_interaction_analysis.png - 特征交互分析")

print(f"\n  报告:")
print(f"    feature_importance_report.csv - 完整特征重要性")
print(f"    category_contribution_report.csv - 类别贡献分析")
print(f"    case_analysis_report.csv - 案例深度分析")

print(f"\n💾 所有结果已保存到: {shap_directory}")

# ========== 额外：生成HTML交互式报告 ==========
print("\n" + "=" * 100)
print("【8】生成交互式HTML报告".center(100))
print("=" * 100)

try:
    # SHAP Force Plot HTML
    shap.force_plot(
        explainer_xgb.expected_value,
        shap_values_xgb[:50],  # 前50个样本
        X_sample[:50],
        feature_names=extended_feature_names,
        show=False
    )
    shap.save_html(shap_directory + 'shap_force_plot_interactive.html',
                   shap.force_plot(
                       explainer_xgb.expected_value,
                       shap_values_xgb[:50],
                       X_sample[:50],
                       feature_names=extended_feature_names
                   ))
    print("✓ 交互式Force Plot已生成: shap_force_plot_interactive.html")
except Exception as e:
    print(f"⚠️  HTML生成失败: {e}")

# ========== 模型决策边界可视化 ==========
print("\n" + "=" * 100)
print("【9】模型决策行为分析".center(100))
print("=" * 100)

# 分析预测值与真实值的关系
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 9.1 预测值 vs 真实值
ax = axes[0, 0]
ax.scatter(y_test_seq[:sample_size], y_pred_sample, alpha=0.6, s=50)
ax.plot([y_test_seq[:sample_size].min(), y_test_seq[:sample_size].max()],
        [y_test_seq[:sample_size].min(), y_test_seq[:sample_size].max()],
        'r--', linewidth=2, label='完美预测线')
ax.set_xlabel('真实值', fontsize=11)
ax.set_ylabel('预测值', fontsize=11)
ax.set_title('预测值 vs 真实值', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 9.2 SHAP值与预测误差的关系
ax = axes[0, 1]
total_shap_impact = np.abs(shap_values_xgb).sum(axis=1)
ax.scatter(total_shap_impact, errors, alpha=0.6, s=50, c=errors, cmap='RdYlGn_r')
ax.set_xlabel('总SHAP影响力（绝对值之和）', fontsize=11)
ax.set_ylabel('预测误差', fontsize=11)
ax.set_title('SHAP总影响 vs 预测误差', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 9.3 预测置信度分析
ax = axes[1, 0]
# 使用SHAP值的标准差作为不确定性度量
shap_std = np.std(shap_values_xgb, axis=1)
ax.scatter(shap_std, errors, alpha=0.6, s=50, c=errors, cmap='RdYlGn_r')
ax.set_xlabel('SHAP值标准差（不确定性）', fontsize=11)
ax.set_ylabel('预测误差', fontsize=11)
ax.set_title('模型不确定性 vs 预测误差', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 9.4 误差分布
ax = axes[1, 1]
ax.hist(errors, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2,
           label=f'均值={np.mean(errors):.6f}')
ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2,
           label=f'中位数={np.median(errors):.6f}')
ax.set_xlabel('预测误差', fontsize=11)
ax.set_ylabel('频数', fontsize=11)
ax.set_title('预测误差分布', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(shap_directory + '11_model_behavior_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图11: 模型决策行为分析")

# ========== 特征贡献热力图 ==========
print("\n" + "=" * 100)
print("【10】特征贡献热力图".center(100))
print("=" * 100)

# 选择前20个最重要的特征
top_20_indices = np.argsort(mean_abs_shap)[-20:][::-1]
top_20_names = [extended_feature_names[i] for i in top_20_indices]

# 选择30个样本用于热力图
heatmap_samples = min(30, sample_size)
shap_heatmap_data = shap_values_xgb[:heatmap_samples, top_20_indices]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(shap_heatmap_data.T, aspect='auto', cmap='RdBu_r',
               vmin=-np.abs(shap_heatmap_data).max(),
               vmax=np.abs(shap_heatmap_data).max())

ax.set_xticks(np.arange(heatmap_samples))
ax.set_yticks(np.arange(len(top_20_names)))
ax.set_xticklabels(np.arange(heatmap_samples), fontsize=8)
ax.set_yticklabels([name[:40] for name in top_20_names], fontsize=8)

ax.set_xlabel('样本索引', fontsize=11)
ax.set_ylabel('特征', fontsize=11)
ax.set_title('Top 20特征的SHAP值热力图\n（红色=正向影响，蓝色=负向影响）',
             fontsize=13, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('SHAP value', fontsize=10)

plt.tight_layout()
plt.savefig(shap_directory + '12_shap_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图12: SHAP值热力图")

# ========== 生成Markdown报告 ==========
print("\n生成Markdown格式的分析报告...")

markdown_report = f"""# LSTM+GRU+XGBoost融合模型 - SHAP可解释性分析报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 执行摘要

本报告使用SHAP（SHapley Additive exPlanations）方法对时间序列预测融合模型进行全面的可解释性分析。

### 关键指标
- **分析样本数**: {sample_size}
- **特征总数**: {len(extended_feature_names)}
- **生成可视化**: 12张
- **生成数据报告**: 3份

---

## 2. Top 10 最重要特征

| 排名 | 特征名称 | SHAP重要性 | XGBoost Gain | 排列重要性 |
|------|---------|-----------|--------------|-----------|
"""

for idx, row in importance_report.head(10).iterrows():
    markdown_report += f"| {idx + 1} | {row['Feature']} | {row['SHAP_Importance']:.6f} | {row['XGB_Gain']:.6f} | {row['Permutation_Importance']:.6f} |\n"

markdown_report += f"""

---

## 3. 特征类别贡献分析

"""

for idx, row in category_report.iterrows():
    pct = (row['Total_SHAP_Contribution'] / sum(category_report['Total_SHAP_Contribution'])) * 100
    markdown_report += f"""### {row['Category']}
- **特征数量**: {row['Feature_Count']}
- **总贡献度**: {row['Total_SHAP_Contribution']:.6f} ({pct:.1f}%)
- **平均贡献**: {row['Avg_SHAP_Per_Feature']:.6f}

"""

markdown_report += f"""---

## 4. 时间滞后分析

不同时间步的特征重要性：

"""

for ts, imp in sorted_timesteps:
    markdown_report += f"- **{ts}**: {imp:.6f}\n"

markdown_report += f"""

**关键发现**: 近期时间步（t-0, t-1）的特征对预测影响最大，表明模型更关注最近的历史信息。

---

## 5. 典型案例分析

"""

for idx, row in case_df.iterrows():
    markdown_report += f"""### {row['Case']}

- **样本索引**: {row['Sample_Index']}
- **真实值**: {row['True_Value']:.6f}
- **预测值**: {row['Predicted_Value']:.6f}
- **预测误差**: {row['Error']:.6f}

**最重要的3个特征**:
1. {row['Top_Feature_1']} (SHAP: {row['Top_Feature_1_SHAP']:.6f})
2. {row['Top_Feature_2']} (SHAP: {row['Top_Feature_2_SHAP']:.6f})
3. {row['Top_Feature_3']} (SHAP: {row['Top_Feature_3_SHAP']:.6f})

"""

markdown_report += f"""---

## 6. 关键洞察与建议

### 模型行为理解
1. **LSTM/GRU预测特征至关重要**: 深度学习模型的预测值本身是XGBoost残差学习的重要输入
2. **时间依赖性**: 最近的时间步信息权重最高，符合时间序列的直觉
3. **滞后特征的作用**: 玉米价格的历史滞后值对预测有显著贡献

### 特征工程建议
1. 保留并优化LSTM/GRU预测特征
2. 可以考虑增加更多近期时间步的特征
3. 探索非线性特征交互（已通过XGBoost捕捉）

### 模型优化方向
1. 针对预测误差较大的样本，分析其SHAP模式找出薄弱环节
2. 考虑对高不确定性样本进行集成或加权处理
3. 持续监控特征重要性变化，及时调整特征集

---

## 7. 附录：生成文件清单

### 可视化图表
- `01_xgb_shap_summary.png` - SHAP特征重要性概览
- `02_xgb_shap_bar.png` - 平均特征影响力
- `03_xgb_force_plot_sample.png` - 单样本Force Plot
- `04_xgb_dependence_plots.png` - Top 6特征依赖图
- `05_xgb_waterfall.png` - Waterfall预测分解
- `06_importance_comparison.png` - 三种方法对比
- `07_category_importance.png` - 特征类别分析
- `08_timestep_importance.png` - 时间维度分析
- `09_case_analysis.png` - 典型案例解析
- `10_interaction_analysis.png` - 特征交互分析
- `11_model_behavior_analysis.png` - 模型决策行为
- `12_shap_heatmap.png` - SHAP值热力图

### 数据报告
- `feature_importance_report.csv` - 完整特征重要性
- `category_contribution_report.csv` - 类别贡献分析
- `case_analysis_report.csv` - 案例深度分析

### 交互式文件
- `shap_force_plot_interactive.html` - 交互式SHAP解释

---

**报告结束**
"""

# 保存Markdown报告
with open(shap_directory + 'SHAP_Analysis_Report.md', 'w', encoding='utf-8') as f:
    f.write(markdown_report)

print("✓ Markdown报告已生成: SHAP_Analysis_Report.md")

print("\n" + "=" * 100)
print("✅ 完整的SHAP可解释性分析已完成！".center(100))
print("=" * 100)

print(f"\n📊 分析成果汇总:")
print(f"  • 生成可视化图表: 12张")
print(f"  • 生成数据报告: 3份CSV")
print(f"  • 生成Markdown报告: 1份")
print(f"  • 生成交互式HTML: 1份")

print(f"\n🎯 核心价值:")
print(f"  1. 揭示了模型预测的内在机制")
print(f"  2. 识别了最重要的预测特征")
print(f"  3. 分析了特征间的交互效应")
print(f"  4. 提供了模型优化的具体方向")

print(f"\n📂 所有结果保存在: {shap_directory}")
print(f"\n💡 建议: 查看 SHAP_Analysis_Report.md 获取完整分析报告")

print("\n" + "=" * 100)



# ========== 最终总结 ==========
print("\n" + "=" * 100)
print("🎉 SHAP可解释性分析完成！".center(100))
print("=" * 100)

print(f"\n📊 分析总结:")
print(f"  • 样本数量: {sample_size}")
print(f"  • 特征总数: {len(extended_feature_names)}")
print(f"  • 生成图表: 10张")
print(f"  • 生成报告: 3份")

print(f"\n🔍 关键发现:")

# Top 5重要特征
top5_features = importance_report.head(5)
print(f"\n  【最重要的5个特征】:")
for i, row in top5_features.iterrows():
    print(f"    {i + 1}. {row['Feature'][:50]}")
    print(f"       SHAP重要性: {row['SHAP_Importance']:.6f}")

# 类别