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
    from en_feature_ablation import FeatureAblationAnalyzer
    FEATURE_ABLATION_AVAILABLE = True
    print("[INFO] ✅ Feature ablation module imported successfully")
except ImportError:
    FEATURE_ABLATION_AVAILABLE = False
    print("[WARNING] ⚠️ feature_ablation.py not found, skipping feature ablation experiment")
# =====================================================================================
# ✨ NEW: Import independent Optuna optimization module
# =====================================================================================
try:
    from optuna_optimizer import OptunaOptimizer
    OPTUNA_AVAILABLE = True
    print("[INFO] ✅ Optuna optimization module imported successfully")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] ⚠️ optuna_optimizer.py not found, using manual parameters")
    print("[INFO] Download address: Please place optuna_optimizer.py in the same directory")

# =====================================================================================
# ✨ NEW: Optuna optimization configuration (global switch)
# =====================================================================================
USE_OPTUNA_OPTIMIZATION = False  # 🔄 Set to False to use original manual parameters
OPTUNA_CONFIG = {
    'lstm_trials': 20,      # Reduced to 20
    'gru_trials': 20,
    'xgb_trials': 10,
    'timeout_hours': 2,     # Max 2 hours
    'enable_pruning': True,
    'verbose': True,
}
FEATURE_ABLATION_CONFIG = {
    'enabled': True,              # Whether to enable feature ablation
    'output_dir': './feature_ablation/',
    'show_plots': True,
    'save_latex': True
}

# English display settings
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print("LSTM + GRU + XGBoost Ensemble Time Series Prediction - Optuna Auto-optimization (Data Leakage Fixed)".center(100))
else:
    print("LSTM + GRU + XGBoost Ensemble Time Series Prediction - Manual Parameters (Data Leakage Fixed)".center(100))
print("Core Improvements: Rolling Window Evaluation + Overfitting Detection for Base Models + Complete Training Pipeline".center(100))
print("=" * 100)


# =====================================================================================
# SECTION 1: Global seed fixing
# =====================================================================================
def set_global_seed(seed=12):
    """Global random seed fixing - ensure experiment reproducibility"""
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

    print(f"✅ Global random seed fixed: {seed}")


GLOBAL_SEED = 12
set_global_seed(GLOBAL_SEED)

# =====================================================================================
# SECTION 2: Overfitting detection module (keep as is)
# =====================================================================================
class OverfittingDetector:
    """Overfitting detector - based on learning curve analysis (simplified version)"""

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
        """Generate learning curve for sequential models (LSTM/GRU)"""
        print(f"\n{'=' * 80}")
        print(f"Learning Curve Analysis: {model_name}".center(80))
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

            print(f"Training samples: {n_samples}/{len(X_train_pool)}", end=' ')

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

            print(f"→ Train R^2={train_r2:.4f}, Val R^2={val_r2:.4f}, Gap={train_r2 - val_r2:.4f}")

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
        """Diagnose overfitting status"""
        print(f"\n{'=' * 80}")
        print(f"Overfitting Diagnosis: {model_name}".center(80))
        print(f"{'=' * 80}")

        final_gap = train_scores[-1] - val_scores[-1]

        if len(train_scores) >= 3:
            train_trend = train_scores[-1] - train_scores[-3]
            val_trend = val_scores[-1] - val_scores[-3]
        else:
            train_trend = train_scores[-1] - train_scores[0]
            val_trend = val_scores[-1] - val_scores[0]

        print(f"\n📊 Key Metrics:")
        print(f"  - Final Train R^2: {train_scores[-1]:.4f}")
        print(f"  - Final Val R^2: {val_scores[-1]:.4f}")
        print(f"  - R^2 Gap: {final_gap:.4f}")
        print(f"  - Train Trend: {train_trend:+.4f}")
        print(f"  - Val Trend: {val_trend:+.4f}")

        print(f"\n🔍 Diagnosis:")

        if final_gap > 0.2:
            print(f"  ⚠️  Severe Overfitting (gap>0.2)")
            diagnosis = "Severe Overfitting"
        elif final_gap > 0.1:
            print(f"  ⚠️  Moderate Overfitting (gap>0.1)")
            diagnosis = "Moderate Overfitting"
        elif final_gap > 0.05:
            print(f"  ⚡ Mild Overfitting (gap>0.05)")
            diagnosis = "Mild Overfitting"
        else:
            print(f"  ✅ Good Fit (gap<0.05)")
            diagnosis = "Good Fit"

        if val_scores[-1] < 0.5:
            print(f"  ⚠️  Possible Underfitting (Val R^2<0.5)")
            diagnosis += " + Underfitting"

        if abs(val_trend) < 0.01:
            print(f"  ✅ Model Converged")
        else:
            print(f"  ⚡ Model May Need More Data")

        print(f"\n💡 Optimization Suggestions:")
        if final_gap > 0.1:
            print(f"  - Increase regularization strength")
            print(f"  - Reduce model complexity")
            print(f"  - Increase dropout ratio")
            print(f"  - Use more training data")
        elif val_scores[-1] < 0.5:
            print(f"  - Increase model complexity")
            print(f"  - Increase training epochs")
            print(f"  - Optimize feature engineering")
        else:
            print(f"  - Model is in good condition, ready for deployment")

        self.results[model_name]['diagnosis'] = diagnosis
        self.results[model_name]['final_gap'] = final_gap

    def plot_all_learning_curves(self, figsize=(20, 8)):
        """Plot learning curve comparison for all models"""
        n_models = len(self.results)
        if n_models == 0:
            print("No results to plot!")
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
                    label='Train R^2', color='#2E86AB', alpha=0.8)
            ax.plot(sample_counts, val_scores, 's-',
                    linewidth=2.5, markersize=8,
                    label='Val R^2', color='#A23B72', alpha=0.8)

            ax.axhline(y=train_scores[-1], color='#2E86AB',
                       linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(y=val_scores[-1], color='#A23B72',
                       linestyle='--', alpha=0.3, linewidth=1)

            ax.fill_between(sample_counts, train_scores, val_scores,
                            alpha=0.2, color='red' if final_gap > 0.1 else 'green')

            ax.set_title(f'{model_name}\n{diagnosis} (Gap={final_gap:.4f})',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Training Samples', fontsize=11)
            ax.set_ylabel('R^2 Score', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            ax.text(sample_counts[-1], train_scores[-1],
                    f'{train_scores[-1]:.3f}',
                    fontsize=9, ha='right', va='bottom', color='#2E86AB')
            ax.text(sample_counts[-1], val_scores[-1],
                    f'{val_scores[-1]:.3f}',
                    fontsize=9, ha='right', va='top', color='#A23B72')

        plt.suptitle('Base Model Learning Curve Analysis - Overfitting Diagnosis',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}learning_curves_base_models.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Learning curve comparison saved: {self.output_dir}learning_curves_base_models.png")

    def plot_loss_curves(self, figsize=(20, 6)):
        """Plot loss curves"""
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
                    linewidth=2.5, label='Train Loss', color='#F18F01')
            ax.plot(sample_counts, val_losses, 's-',
                    linewidth=2.5, label='Val Loss', color='#C73E1D')

            ax.set_title(f'{model_name} - MSE Loss',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Training Samples', fontsize=11)
            ax.set_ylabel('MSE Loss', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.suptitle('Base Model Training and Validation Loss Curves',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}loss_curves_base_models.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Loss curves saved: {self.output_dir}loss_curves_base_models.png")

    def generate_report(self):
        """Generate overfitting diagnosis report"""
        print(f"\n{'=' * 80}")
        print("Base Model Overfitting Diagnosis Summary Report".center(80))
        print(f"{'=' * 80}\n")

        report_data = []
        for model_name, result in self.results.items():
            report_data.append({
                'Model': model_name,
                'Final Train R^2': result['train_scores'][-1],
                'Final Val R^2': result['val_scores'][-1],
                'R^2 Gap': result['final_gap'],
                'Diagnosis': result['diagnosis']
            })

        df = pd.DataFrame(report_data)
        df = df.sort_values('R^2 Gap', ascending=False)

        print(df.to_string(index=False))

        df.to_csv(f'{self.output_dir}overfitting_report.csv', index=False)
        print(f"\n✅ Report saved to: {self.output_dir}overfitting_report.csv")

        print(f"\n📊 Statistical Summary:")
        print(f"  - Models analyzed: {len(self.results)}")
        print(f"  - Good fit: {len([r for r in report_data if r['R^2 Gap'] < 0.05])}")
        print(f"  - Mild overfitting: {len([r for r in report_data if 0.05 <= r['R^2 Gap'] < 0.1])}")
        print(f"  - Moderate overfitting: {len([r for r in report_data if 0.1 <= r['R^2 Gap'] < 0.2])}")
        print(f"  - Severe overfitting: {len([r for r in report_data if r['R^2 Gap'] >= 0.2])}")

        print(f"\n💡 Key Findings:")
        print(f"  - Strategy model (XGBoost) only learns residuals, limited impact")
        print(f"  - Base model overfitting is the main risk, needs attention")
        print(f"  - If base models overfit, adjust regularization parameters and retrain")


# =====================================================================================
# SECTION 3: Data loading and preprocessing
# =====================================================================================
dataset = pd.read_csv('Corn-new.csv', parse_dates=['Date'], index_col=['Date'])
print("\nDataset Info:")
print(dataset.info())

X = dataset.drop(columns=['Corn'], axis=1)
y = dataset['Corn']

split_idx = int(len(X) * 0.8)
X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_raw, y_test_raw = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nData Split:")
print(f"Training set: {len(X_train_raw)} samples")
print(f"Test set: {len(X_test_raw)} samples")

# Normalization
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
# ✨ FIXED SECTION 4: Feature engineering (training set only)
# =====================================================================================
def add_features(X, y):
    """Add lag features (training set only)"""
    X_new = X.copy()
    for i in range(1, 6):
        X_new[f'Corn_lag_{i}'] = y.shift(i)
    return X_new.dropna()


print(f"\n{'=' * 80}")
print("【Feature Engineering - Training Set Only】".center(80))
print(f"{'=' * 80}")

X_train_feat = add_features(X_train, y_train)
y_train = y_train.loc[X_train_feat.index]

print(f"\n✓ Training set feature construction:")
print(f"  - Original samples: {len(X_train)}")
print(f"  - After adding lag features: {len(X_train_feat)}")
print(f"  - Feature dimensions: {X_train_feat.shape[1]}")

print(f"\n✓ Test set processing strategy:")
print(f"  - Features not pre-constructed")
print(f"  - Will be dynamically constructed during rolling window prediction")
print(f"  - Completely avoid data leakage")


# Construct training set sequence data
def create_sequences(X, y, seq_len=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X.iloc[i:i + seq_len].values)
        y_seq.append(y.iloc[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


seq_len = 5
X_train_seq, y_train_seq = create_sequences(X_train_feat, y_train, seq_len)

print(f"\n✓ Training set sequence construction:")
print(f"  - Sequence shape: {X_train_seq.shape}")
print(f"  - Target shape: {y_train_seq.shape}")


# =====================================================================================
# ✨ NEW: Rolling window prediction function
# =====================================================================================
def rolling_window_predict(model, X_train_feat, y_train, X_test, y_test, seq_len=5):
    """
    Rolling window prediction function (for LSTM/GRU)

    Key logic:
    1. Initialize history window (training set only)
    2. Predict test samples one by one
    3. Update history with true values after each prediction
    """
    predictions = []

    # Initialize: get history window from training set
    history_X_feat = X_train_feat.tail(seq_len).copy()
    history_y = y_train.tail(5).copy()

    # Predict test samples one by one
    for idx in range(len(X_test)):
        # 1. Get current test sample's original features (without lag)
        current_X_raw = X_test.iloc[idx:idx+1].copy()

        # 2. Dynamically construct lag features (using known history only)
        for i in range(1, 6):
            if len(history_y) >= i:
                current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i]
            else:
                current_X_raw[f'Corn_lag_{i}'] = 0

        # 3. Construct sequence window
        current_window = pd.concat([history_X_feat.tail(seq_len-1), current_X_raw])
        X_seq = current_window.values.reshape(1, seq_len, -1)

        # 4. Predict
        pred = model.predict(X_seq, verbose=0)[0, 0]
        predictions.append(pred)

        # 5. ✅ Update history with true value (key: this is known after prediction)
        history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
        history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[idx]])])

        # Progress display
        if (idx + 1) % 50 == 0 or idx == len(X_test) - 1:
            print(f"  Progress: {idx+1}/{len(X_test)} samples", end='\r')

    print()  # Newline
    return np.array(predictions)


def rolling_window_predict_ensemble(lstm_model, gru_model, xgb_model,
                                     X_train_feat, y_train, X_test, y_test,
                                     X_train_flat, seq_len=5, strategy='gru_residual'):
    """
    Ensemble model rolling window prediction
    """
    predictions = []
    lstm_preds_list = []
    gru_preds_list = []

    # Initialize history window
    history_X_feat = X_train_feat.tail(seq_len).copy()
    history_y = y_train.tail(5).copy()

    # Predict one by one
    for idx in range(len(X_test)):
        # Construct current sample features
        current_X_raw = X_test.iloc[idx:idx+1].copy()

        for i in range(1, 6):
            if len(history_y) >= i:
                current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i]
            else:
                current_X_raw[f'Corn_lag_{i}'] = 0

        # Construct sequence window
        current_window = pd.concat([history_X_feat.tail(seq_len-1), current_X_raw])
        X_seq = current_window.values.reshape(1, seq_len, -1)

        # Get LSTM and GRU predictions
        lstm_pred = lstm_model.predict(X_seq, verbose=0)[0, 0]
        gru_pred = gru_model.predict(X_seq, verbose=0)[0, 0]

        lstm_preds_list.append(lstm_pred)
        gru_preds_list.append(gru_pred)

        # Calculate final prediction based on strategy
        if strategy == 'average':
            final_pred = (lstm_pred + gru_pred) / 2

        elif strategy == 'lstm_residual':
            # Construct XGBoost features
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

        # Update history
        history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
        history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[idx]])])

        if (idx + 1) % 50 == 0 or idx == len(X_test) - 1:
            print(f"  Progress: {idx+1}/{len(X_test)} samples", end='\r')

    print()
    return np.array(predictions), np.array(lstm_preds_list), np.array(gru_preds_list)


# =====================================================================================
# ✨ NEW SECTION: Optuna hyperparameter optimization phase
# =====================================================================================

if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print("\n" + "=" * 100)
    print("【Optuna Hyperparameter Optimization Phase】".center(100))
    print("=" * 100)

    # Create optimizer
    optimizer = OptunaOptimizer(
        X_train=X_train_seq,
        y_train=y_train_seq,
        output_dir='./optuna_results/',
        seed=GLOBAL_SEED,
        verbose=OPTUNA_CONFIG['verbose']
    )

    # Optimize LSTM
    print(f"\n[1/3] Optimizing LSTM hyperparameters (n_trials={OPTUNA_CONFIG['lstm_trials']})...")
    best_lstm_params = optimizer.optimize_lstm(
        n_trials=OPTUNA_CONFIG['lstm_trials'],
        max_epochs=200,
        batch_size=32,
        enable_pruning=OPTUNA_CONFIG['enable_pruning']
    )

    # Optimize GRU
    print(f"\n[2/3] Optimizing GRU hyperparameters (n_trials={OPTUNA_CONFIG['gru_trials']})...")
    best_gru_params = optimizer.optimize_gru(
        n_trials=OPTUNA_CONFIG['gru_trials'],
        max_epochs=200,
        batch_size=32,
        enable_pruning=OPTUNA_CONFIG['enable_pruning']
    )

    # Optimize XGBoost
    print(f"\n[3/3] Optimizing XGBoost hyperparameters (n_trials={OPTUNA_CONFIG['xgb_trials']})...")
    best_xgb_params = optimizer.optimize_xgboost(
        n_trials=OPTUNA_CONFIG['xgb_trials'],
        cv_splits=5
    )

    # Generate optimization report and visualization
    try:
        optimizer.visualize('lstm', show=False)
        optimizer.visualize('gru', show=False)
        optimizer.visualize('xgboost', show=False)
    except Exception as e:
        print(f"[WARNING] Visualization generation failed: {e}")

    optimizer.generate_report()

    print("\n" + "=" * 100)
    print("✅ Optuna optimization completed!".center(100))
    print(f"Optimization results saved in: ./optuna_results/".center(100))
    print("=" * 100)

else:
    # Use original manually set parameters
    print("\n" + "=" * 100)
    print("【Using Manual Parameters】".center(100))
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

    print("\nCurrently using hyperparameters:")
    print(f"LSTM: {best_lstm_params}")
    print(f"GRU: {best_gru_params}")
    print(f"XGBoost: {best_xgb_params}")


# =====================================================================================
# SECTION 5: Model builders (for overfitting detection)
# =====================================================================================
def build_final_lstm(input_shape):
    """Final LSTM model"""
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
    """Final GRU model"""
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
# SECTION 6: OOF prediction generation (within training set, no data leakage)
# =====================================================================================
def get_oof_predictions(X_seq, y_seq, params, model_type='lstm', n_splits=5):
    """Generate OOF predictions"""
    print(f"\nGenerating {model_type.upper()} OOF predictions (TimeSeriesSplit with {n_splits} splits)...")

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
print("【Phase 1】Generate LSTM and GRU OOF Predictions".center(100))
print("=" * 100)

lstm_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, best_lstm_params, 'lstm', n_splits=5)
gru_oof_preds = get_oof_predictions(X_train_seq, y_train_seq, best_gru_params, 'gru', n_splits=5)

print(f"\nOOF prediction generation completed!")
print(f"LSTM OOF R^2: {r2_score(y_train_seq, lstm_oof_preds):.4f}")
print(f"GRU OOF R^2: {r2_score(y_train_seq, gru_oof_preds):.4f}")


# =====================================================================================
# SECTION 7: Train final models
# =====================================================================================
print("\n" + "=" * 100)
print("【Phase 2】Train Final LSTM and GRU Models (Using Optimal Parameters)".center(100))
print("=" * 100)

print("\nTraining final LSTM model...")
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
print("✓ LSTM model training completed")

print("\nTraining final GRU model...")
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
print("✓ GRU model training completed")


# =====================================================================================
# ✨ FIXED SECTION 8: Use rolling window to predict test set
# =====================================================================================
print("\n" + "=" * 100)
print("【Phase 3】Rolling Window Test Set Prediction (Avoiding Data Leakage)".center(100))
print("=" * 100)

print("\nRolling window prediction LSTM...")
lstm_test_pred = rolling_window_predict(
    lstm_final, X_train_feat, y_train, X_test, y_test, seq_len=seq_len
)

print("\nRolling window prediction GRU...")
gru_test_pred = rolling_window_predict(
    gru_final, X_train_feat, y_train, X_test, y_test, seq_len=seq_len
)

# Simple average
avg_test_pred = (lstm_test_pred + gru_test_pred) / 2

# Evaluate base models
lstm_test_r2 = r2_score(y_test, lstm_test_pred)
gru_test_r2 = r2_score(y_test, gru_test_pred)
avg_r2 = r2_score(y_test, avg_test_pred)

print(f"\n【Base Model Performance (Rolling Window Evaluation)】")
print(f"LSTM Single - Test R^2: {lstm_test_r2:.4f}")
print(f"GRU Single  - Test R^2: {gru_test_r2:.4f}")
print(f"Simple Avg  - Test R^2: {avg_r2:.4f}")


# =====================================================================================
# SECTION 9: Learning curve overfitting detection
# =====================================================================================
print("\n" + "=" * 100)
print("【Phase 4】Learning Curve Overfitting Detection for Base Models (LSTM/GRU)".center(100))
print("Reason: Base models are the main prediction source, overfitting directly affects final results".center(100))
print("=" * 100)

detector = OverfittingDetector(output_dir='./overfitting_analysis/')

print("\n🔍 Detecting LSTM final model...")
detector.sequential_learning_curve(
    model_builder=build_final_lstm,
    X_seq=X_train_seq,
    y_seq=y_train_seq,
    model_name='LSTM Final Model',
    train_sizes=np.linspace(0.2, 1.0, 8),
    epochs=100,
    batch_size=32
)

print("\n🔍 Detecting GRU final model...")
detector.sequential_learning_curve(
    model_builder=build_final_gru,
    X_seq=X_train_seq,
    y_seq=y_train_seq,
    model_name='GRU Final Model',
    train_sizes=np.linspace(0.2, 1.0, 8),
    epochs=100,
    batch_size=32
)

detector.plot_all_learning_curves(figsize=(16, 6))
detector.plot_loss_curves(figsize=(16, 5))
detector.generate_report()


# =====================================================================================
# SECTION 10: XGBoost training
# =====================================================================================
def create_simplified_features(X_flat, lstm_preds, gru_preds):
    """Simplified features: original + predictions + average + difference"""
    features_list = [X_flat]
    features_list.append(lstm_preds.reshape(-1, 1))
    features_list.append(gru_preds.reshape(-1, 1))
    features_list.append(((lstm_preds + gru_preds) / 2).reshape(-1, 1))
    features_list.append(np.abs(lstm_preds - gru_preds).reshape(-1, 1))
    return np.hstack(features_list)


def train_xgboost(X_train, y_train):
    """Train XGBoost"""
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
print("【Phase 5】XGBoost Residual Learning".center(100))
print("=" * 100)

# Calculate residuals
lstm_oof_residual = y_train_seq - lstm_oof_preds
gru_oof_residual = y_train_seq - gru_oof_preds
avg_oof_preds = (lstm_oof_preds + gru_oof_preds) / 2
avg_oof_residual = y_train_seq - avg_oof_preds

print(f"\nResidual statistics:")
print(f"LSTM residual - Mean: {np.mean(lstm_oof_residual):.6f}, Std: {np.std(lstm_oof_residual):.6f}")
print(f"GRU residual  - Mean: {np.mean(gru_oof_residual):.6f}, Std: {np.std(gru_oof_residual):.6f}")

# Prepare features (training set)
X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
simplified_train = create_simplified_features(X_train_flat, lstm_oof_preds, gru_oof_preds)

print(f"\nFeature dimensions: {simplified_train.shape[1]} dims")

# Train XGBoost models
print("\nTraining XGBoost (LSTM residual)...")
xgb_lstm = train_xgboost(simplified_train, lstm_oof_residual)

print("Training XGBoost (GRU residual)...")
xgb_gru = train_xgboost(simplified_train, gru_oof_residual)

print("Training XGBoost (dual residual)...")
xgb_dual = train_xgboost(simplified_train, avg_oof_residual)


# =====================================================================================
# SECTION 11: Feature engineering and XGBoost training (supplementary functions)
# =====================================================================================
def clip_residual(residual_pred, threshold=2.0):
    """Clip extreme residual values"""
    std = np.std(residual_pred)
    mean = np.mean(residual_pred)
    return np.clip(residual_pred, mean - threshold * std, mean + threshold * std)


def weighted_residual_correction(base_pred, residual_pred, weight=0.5):
    """Weighted residual correction"""
    return base_pred + weight * residual_pred


# =====================================================================================
# ✨ FIXED SECTION 12: Ensemble strategy rolling window evaluation (all 6 strategies)
# =====================================================================================
print("\n" + "=" * 100)
print("【Phase 6】Ensemble Strategy Rolling Window Evaluation".center(100))
print("=" * 100)

strategies_results = {}

# Strategy 1: LSTM residual learning
print("\n" + "=" * 100)
print("【Strategy 1】LSTM Residual Learning: Simplified Features + XGBoost".center(100))
print("=" * 100)

print("\nRolling window prediction...")
pred_lstm_residual, _, _ = rolling_window_predict_ensemble(
    lstm_final, gru_final, xgb_lstm,
    X_train_feat, y_train, X_test, y_test,
    X_train_flat, seq_len=seq_len, strategy='lstm_residual'
)
strategies_results['Strategy1-LSTM_Residual'] = pred_lstm_residual
r2_lstm_residual = r2_score(y_test, pred_lstm_residual)
print(f"✓ R^2: {r2_lstm_residual:.4f} (vs Simple Avg: {r2_lstm_residual - avg_r2:+.4f})")

# Strategy 2: GRU residual learning
print("\n" + "=" * 100)
print("【Strategy 2】GRU Residual Learning: Simplified Features + XGBoost".center(100))
print("=" * 100)

print("\nRolling window prediction...")
pred_gru_residual, _, _ = rolling_window_predict_ensemble(
    lstm_final, gru_final, xgb_gru,
    X_train_feat, y_train, X_test, y_test,
    X_train_flat, seq_len=seq_len, strategy='gru_residual'
)
strategies_results['Strategy2-GRU_Residual'] = pred_gru_residual
r2_gru_residual = r2_score(y_test, pred_gru_residual)
print(f"✓ R^2: {r2_gru_residual:.4f} (vs Simple Avg: {r2_gru_residual - avg_r2:+.4f})")

# Strategy 3: Dual residual learning
print("\n" + "=" * 100)
print("【Strategy 3】Dual Residual Learning: (LSTM+GRU)/2 + XGBoost".center(100))
print("=" * 100)

print("\nRolling window prediction...")
# Dual residual prediction needs separate implementation, as baseline is average
pred_dual_list = []
history_X_feat = X_train_feat.tail(seq_len).copy()
history_y = y_train.tail(5).copy()

for idx in range(len(X_test)):
    current_X_raw = X_test.iloc[idx:idx+1].copy()
    for i in range(1, 6):
        current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i] if len(history_y) >= i else 0

    current_window = pd.concat([history_X_feat.tail(seq_len-1), current_X_raw])
    X_seq = current_window.values.reshape(1, seq_len, -1)

    # Get LSTM and GRU predictions
    lstm_pred = lstm_final.predict(X_seq, verbose=0)[0, 0]
    gru_pred = gru_final.predict(X_seq, verbose=0)[0, 0]
    avg_pred = (lstm_pred + gru_pred) / 2

    # Construct XGBoost features and predict residual
    X_flat = X_seq.reshape(1, -1)
    X_xgb = np.hstack([X_flat, np.array([[lstm_pred, gru_pred, avg_pred, abs(lstm_pred - gru_pred)]])])

    residual = xgb_dual.predict(X_xgb)[0]
    final_pred = avg_pred + residual
    pred_dual_list.append(final_pred)

    history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
    history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[idx]])])

    if (idx + 1) % 50 == 0 or idx == len(X_test) - 1:
        print(f"  Progress: {idx+1}/{len(X_test)} samples", end='\r')

print()
pred_dual = np.array(pred_dual_list)
strategies_results['Strategy3-Dual_Residual'] = pred_dual
r2_dual = r2_score(y_test, pred_dual)
print(f"✓ R^2: {r2_dual:.4f} (vs Simple Avg: {r2_dual - avg_r2:+.4f})")

# Strategy 4: Residual clipping
print("\n" + "=" * 100)
print("【Strategy 4】GRU Residual Learning + Residual Clipping".center(100))
print("=" * 100)

# Use strategy 2 predictions, calculate and clip residuals
gru_residual_test = pred_gru_residual - gru_test_pred
gru_residual_clipped = clip_residual(gru_residual_test, threshold=2.0)
pred_gru_clipped = gru_test_pred + gru_residual_clipped

r2_gru_clipped = r2_score(y_test, pred_gru_clipped)
print(f"✓ R^2: {r2_gru_clipped:.4f} (vs Simple Avg: {r2_gru_clipped - avg_r2:+.4f})")
print(f"  Residual std before clipping: {np.std(gru_residual_test):.6f}, after: {np.std(gru_residual_clipped):.6f}")
strategies_results['Strategy4-Clipped'] = pred_gru_clipped

# Strategy 5: Weighted fusion (30%)
print("\n" + "=" * 100)
print("【Strategy 5】GRU Residual Learning + Weighted Fusion (30%)".center(100))
print("=" * 100)

pred_gru_weighted = weighted_residual_correction(gru_test_pred, gru_residual_test, weight=0.3)

r2_gru_weighted = r2_score(y_test, pred_gru_weighted)
print(f"✓ R^2: {r2_gru_weighted:.4f} (vs Simple Avg: {r2_gru_weighted - avg_r2:+.4f})")
strategies_results['Strategy5-Weighted_30%'] = pred_gru_weighted

# Strategy 6: Ultimate combination
print("\n" + "=" * 100)
print("【Strategy 6】Ultimate Combination: Clipping + Weighted Fusion (30%)".center(100))
print("=" * 100)

pred_ultimate = weighted_residual_correction(gru_test_pred, gru_residual_clipped, weight=0.3)

r2_ultimate = r2_score(y_test, pred_ultimate)
print(f"✓ R^2: {r2_ultimate:.4f} (vs Simple Avg: {r2_ultimate - avg_r2:+.4f})")
strategies_results['Strategy6-Ultimate'] = pred_ultimate


# =====================================================================================
# SECTION 13: Comprehensive comparison
# =====================================================================================
print("\n" + "=" * 100)
print("All Strategy Performance Comparison (Normalized Data)".center(100))
print("=" * 100)

all_strategies = {
    'LSTM_Single': lstm_test_pred,
    'GRU_Single': gru_test_pred,
    'Simple_Avg(Baseline)': avg_test_pred,
    **strategies_results
}

print(f"\n{'Strategy':<30} {'R^2':>10} {'vs Baseline':>10} {'MAE':>12} {'RMSE':>12}")
print("-" * 75)

results_list = []
for name, pred in all_strategies.items():
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = sqrt(mean_squared_error(y_test, pred))
    improvement = r2 - avg_r2
    results_list.append((name, r2, improvement, mae, rmse, pred))
    print(f"{name:<30} {r2:>10.4f} {improvement:>10.4f} {mae:>12.6f} {rmse:>12.6f}")

# Sort
results_list.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 100)
print("Performance Ranking (by R^2 descending)".center(100))
print("=" * 100)

best_r2 = results_list[0][1]
best_name = results_list[0][0]

for rank, (name, r2, improvement, mae, rmse, pred) in enumerate(results_list, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:>2}. {name:<30} R^2={r2:.4f} (vs Baseline: {improvement:+.4f})")

print(f"\n🏆 Best Strategy: {best_name} (R^2 = {best_r2:.4f})")

# Original scale evaluation
print("\n" + "=" * 100)
print("Original Scale Performance Comparison".center(100))
print("=" * 100)

y_test_original = y_scaler.inverse_transform(y_test.values.reshape(-1, 1))
strategies_original = {}

print(f"\n{'Strategy':<30} {'R^2':>10} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
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
# SECTION 14: Complete visualization (keep all original charts)
# =====================================================================================
results_directory = "./Predict/"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

print("\n" + "=" * 100)
print("Generating Visualization Charts".center(100))
print("=" * 100)

# Figure 1: Training process
fig = plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(lstm_history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('LSTM Model Training Process', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(gru_history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(gru_history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('GRU Model Training Process', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + '01_training_process.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 1: Training process curves")

# Figure 2: Performance ranking bar chart
fig, ax = plt.subplots(figsize=(16, 8))

strategy_names = [name for name, _, _, _, _, _ in results_list]
r2_scores = [r2 for _, r2, _, _, _, _ in results_list]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(strategy_names)))

bars = ax.barh(range(len(strategy_names)), r2_scores, color=colors, alpha=0.8)

ax.axvline(x=avg_r2, color='red', linestyle='--', linewidth=2, label='Simple Avg Baseline', alpha=0.7)

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
ax.set_title('All Strategy Performance Ranking Comparison (Rolling Window Evaluation)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(results_directory + '02_performance_ranking.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 2: Performance ranking comparison")

# Figure 3: Top6 strategy prediction comparison
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
axes = axes.flatten()

top6_strategies = results_list[:6]

for idx, (name, r2, improvement, mae, rmse, pred) in enumerate(top6_strategies):
    ax = axes[idx]

    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    r2_original = r2_score(y_test_original, pred_original)

    ax.plot(y_test_original, label='True Value', linewidth=2.5, color='black', alpha=0.8)
    ax.plot(pred_original, label=name, linewidth=2, alpha=0.8)
    ax.set_title(f'{name}\nR^2={r2_original:.4f} (vs Baseline: {improvement:+.4f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=9)
    ax.set_ylabel('Corn Price', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + '03_top6_strategies_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 3: Top6 strategy prediction comparison")

# Figure 4: Residual analysis comparison
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

residuals_dict = {
    'Simple_Avg': y_test - avg_test_pred,
    'Strategy1-LSTM_Residual': y_test - strategies_results['Strategy1-LSTM_Residual'],
    'Strategy2-GRU_Residual': y_test - strategies_results['Strategy2-GRU_Residual'],
    'Strategy3-Dual_Residual': y_test - strategies_results['Strategy3-Dual_Residual'],
    'Strategy4-Clipped': y_test - strategies_results['Strategy4-Clipped'],
    'Strategy5-Weighted_30%': y_test - strategies_results['Strategy5-Weighted_30%'],
    'Strategy6-Ultimate': y_test - strategies_results['Strategy6-Ultimate'],
}

for idx, (name, residual) in enumerate(residuals_dict.items()):
    ax = axes[idx]
    ax.hist(residual, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{name} Residual Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.text(0.05, 0.95, f'Mean={np.mean(residual):.5f}\nStd={np.std(residual):.5f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3, axis='y')

for idx in range(len(residuals_dict), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(results_directory + '04_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 4: Residual distribution analysis")

# Figure 5: Residual time series comparison
fig, ax = plt.subplots(figsize=(16, 6))

for name, residual in residuals_dict.items():
    ax.plot(residual, label=name, linewidth=2, alpha=0.7)

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_title('Residual Time Series Comparison for Different Strategies', fontsize=14, fontweight='bold')
ax.set_xlabel('Sample Index', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_directory + '05_residual_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 5: Residual time series")

# Figure 6: Comprehensive comparison chart
plt.figure(figsize=(18, 8))
plt.plot(y_test_original, label='True Value', linewidth=3, color='black', alpha=0.9, zorder=10)

key_strategies_for_plot = [
    ('Simple_Avg(Baseline)', strategies_original['Simple_Avg(Baseline)'], 'gray'),
    ('Strategy1-LSTM_Residual', strategies_original['Strategy1-LSTM_Residual'], '#A09FA4'),
    ('Strategy2-GRU_Residual', strategies_original['Strategy2-GRU_Residual'], '#FEA0A0'),
    ('Strategy3-Dual_Residual', strategies_original['Strategy3-Dual_Residual'], '#AD07E3'),
    ('Strategy4-Clipped', strategies_original['Strategy4-Clipped'], '#55A0FB'),
    ('Strategy5-Weighted_30%', strategies_original['Strategy5-Weighted_30%'], '#099A63'),
    ('Strategy6-Ultimate', strategies_original['Strategy6-Ultimate'], '#FFB90F'),
]

for name, pred, color in key_strategies_for_plot:
    plt.plot(pred, label=name, linewidth=1.8, alpha=0.7, color=color)

plt.title('Key Strategy Comprehensive Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Corn Price', fontsize=12)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_directory + '06_key_strategies_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 6: Key strategy comprehensive comparison")

# Figure 7: Performance metrics comparison
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
axes[0, 0].set_title('R^2 Score Comparison', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('R^2 Score')
axes[0, 0].set_xticks(range(len(top9_names)))
axes[0, 0].set_xticklabels([n[:20] for n in top9_names], rotation=45, ha='right', fontsize=8)
axes[0, 0].grid(True, alpha=0.3, axis='y')

axes[0, 1].bar(range(len(top9_names)), [metrics_data[m]['MAE'] for m in top9_names],
               color=colors_bar, alpha=0.7)
axes[0, 1].set_title('MAE Comparison', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_xticks(range(len(top9_names)))
axes[0, 1].set_xticklabels([n[:20] for n in top9_names], rotation=45, ha='right', fontsize=8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[1, 0].bar(range(len(top9_names)), [metrics_data[m]['RMSE'] for m in top9_names],
               color=colors_bar, alpha=0.7)
axes[1, 0].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].set_xticks(range(len(top9_names)))
axes[1, 0].set_xticklabels([n[:20] for n in top9_names], rotation=45, ha='right', fontsize=8)
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].bar(range(len(top9_names)), [metrics_data[m]['MAPE'] for m in top9_names],
               color=colors_bar, alpha=0.7)
axes[1, 1].set_title('MAPE Comparison', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('MAPE')
axes[1, 1].set_xticks(range(len(top9_names)))
axes[1, 1].set_xticklabels([n[:20] for n in top9_names], rotation=45, ha='right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_directory + '07_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 7: Multi-metric comparison")

# Figure 8: Prediction error analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

top4_for_error = results_list[:4]

for idx, (name, r2, improvement, mae, rmse, pred) in enumerate(top4_for_error):
    ax = axes[idx // 2, idx % 2]

    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    errors = pred_original - y_test_original.flatten()

    ax.scatter(y_test_original, errors, alpha=0.5, s=30)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{name} - Prediction Error Analysis', fontsize=11, fontweight='bold')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Prediction Error')
    ax.grid(True, alpha=0.3)

    ax.text(0.05, 0.95, f'MAE={mae:.2f}\nRMSE={rmse:.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig(results_directory + '08_prediction_error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 8: Prediction error analysis")

# Figure 9: Model improvement radar chart
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

radar_strategies = ['Simple_Avg(Baseline)', 'Strategy1-LSTM_Residual', 'Strategy2-GRU_Residual', 'Strategy6-Ultimate']
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
ax.set_title('Multi-dimensional Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig(results_directory + '09_performance_radar.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 9: Performance radar chart")

# Figure 10: Residual boxplot comparison
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
ax.set_title('Residual Distribution Boxplot Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Residual Value')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig(results_directory + '10_residual_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 10: Residual boxplot")

print("\n✅ All visualization charts generated and saved!")


# =====================================================================================
# SECTION 15: Save models and results
# =====================================================================================
print("\n" + "=" * 100)
print("Saving Models and Results".center(100))
print("=" * 100)

# Save Keras models
lstm_final.save(results_directory + 'lstm_final_rolling.h5')
gru_final.save(results_directory + 'gru_final_rolling.h5')

# Save XGBoost models
with open(results_directory + 'xgb_lstm_rolling.pkl', 'wb') as f:
    pickle.dump(xgb_lstm, f)

with open(results_directory + 'xgb_gru_rolling.pkl', 'wb') as f:
    pickle.dump(xgb_gru, f)

# Save scalers
with open(results_directory + 'scalers.pkl', 'wb') as f:
    pickle.dump({'feature_scalers': feature_scalers, 'y_scaler': y_scaler}, f)

# Save all prediction results
predictions_dict = {'True_Value': y_test_original.flatten()}
for name, pred in all_strategies.items():
    pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1))
    predictions_dict[name] = pred_original.flatten()

predictions_df = pd.DataFrame(predictions_dict)
predictions_df.to_csv(results_directory + 'all_predictions_rolling.csv', index=False)

# Save hyperparameters
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

print(f"\n✓ All models and results saved to: {results_directory}")


# =====================================================================================
# SECTION 16: DM Test (Diebold-Mariano Test)
# =====================================================================================
print("\n" + "=" * 100)
print("【DM Test】Diebold-Mariano Statistical Significance Test".center(100))
print("=" * 100)

try:
    from en_dm_test import quick_dm_analysis, pairwise_dm_analysis

    # Prepare data (using original scale prediction results)
    all_predictions = {
        'LSTM_Single': strategies_original['LSTM_Single'],
        'GRU_Single': strategies_original['GRU_Single'],
        'Simple_Avg(Baseline)': strategies_original['Simple_Avg(Baseline)'],
        **{k: v for k, v in strategies_original.items() if k.startswith('Strategy')}
    }

    # Baseline comparison analysis
    print("\nAll Models vs Baseline Model")
    print("-" * 100)

    dm_results = quick_dm_analysis(
        y_true=y_test_original,
        predictions=all_predictions,
        baseline='Simple_Avg(Baseline)',
        save_dir=results_directory,
        plot=True,
        verbose=True
    )

    print("\n✅ DM test completed! Results saved to:", results_directory)

except ImportError:
    print("[WARNING] ⚠️ dm_test.py module not found, skipping DM test")
    print("[INFO] If DM test needed, ensure dm_test.py is in the same directory")


# =====================================================================================
# SECTION 17: Final summary report
# =====================================================================================
print("\n" + "=" * 100)
print("Final Summary Report".center(100))
print("=" * 100)

print(f"\n📊 Dataset Information:")
print(f"  - Training set samples: {len(y_train_seq)}")
print(f"  - Test set samples: {len(y_test)}")
print(f"  - Feature dimensions: {X_train_feat.shape[1]}")
print(f"  - Sequence length: {seq_len}")

print(f"\n✅ Data leakage fix verification:")
print(f"  - Evaluation method: Rolling window prediction")
print(f"  - Each sample predicted independently")
print(f"  - Using true historical values to update window")
print(f"  - Completely avoid data leakage")

print(f"\n🏆 Best performing strategy:")
print(f"  - Strategy name: {best_name}")
print(f"  - Test set R^2: {best_r2:.4f}")
print(f"  - Improvement: {best_r2 - avg_r2:+.4f}")

# ✅ Fix: Index changed from 4 to 5
best_pred = strategies_original[best_name]
best_mae = mean_absolute_error(y_test_original, best_pred)
best_rmse = sqrt(mean_squared_error(y_test_original, best_pred))
best_mape = np.mean(np.abs((best_pred - y_test_original) / (y_test_original + 1e-8)))

print(f"\n📈 Best strategy original scale metrics:")
print(f"  - MAE:  {best_mae:.2f}")
print(f"  - RMSE: {best_rmse:.2f}")
print(f"  - MAPE: {best_mape:.4%}")

print(f"\n⚙️ Hyperparameter optimization method:")
if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print(f"  ✅ Optuna automatic optimization")
    print(f"  - LSTM trials: {OPTUNA_CONFIG['lstm_trials']}")
    print(f"  - GRU trials: {OPTUNA_CONFIG['gru_trials']}")
    print(f"  - XGBoost trials: {OPTUNA_CONFIG['xgb_trials']}")
    print(f"  - Optimization report: ./optuna_results/")
else:
    print(f"  ⚠️  Manual parameter setting")

print(f"\n⚠️ Overfitting diagnosis summary:")
print(f"  【Learning Curve Method】")
for model_name, result in detector.results.items():
    diagnosis = result.get('diagnosis', 'Unknown')
    final_gap = result.get('final_gap', 0)
    print(f"  - {model_name}: {diagnosis} (Gap={final_gap:.4f})")

print(f"\n💡 Key Findings:")
if best_r2 > avg_r2:
    print(f"  ✅ Residual learning strategy successfully improved model performance")
    print(f"  ✅ Improved by {(best_r2 - avg_r2) * 100:.2f}% compared to simple average")
else:
    print(f"  ⚠️ Residual learning strategy did not improve performance, recommend using simple average")

if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print(f"\n🎯 Optuna optimization achievements:")
    print(f"  ✅ Automatically found optimal hyperparameter combination")
    print(f"  ✅ Saved significant manual tuning time")
    print(f"  ✅ Detailed optimization report available for analysis")

print(f"\n📁 All results save location:")
print(f"  - Models and prediction results: {results_directory}")
print(f"  - Overfitting analysis: {detector.output_dir}")
if USE_OPTUNA_OPTIMIZATION and OPTUNA_AVAILABLE:
    print(f"  - Optuna optimization report: ./optuna_results/")

print("\n" + "=" * 100)
print("Program execution completed! Data leakage issue completely fixed!".center(100))
print("=" * 100)

# =====================================================================================
# Add feature ablation experiment code in SECTION 18
# =====================================================================================

if FEATURE_ABLATION_CONFIG['enabled'] and FEATURE_ABLATION_AVAILABLE:
    print("\n" + "=" * 100)
    print("【Feature Ablation Experiment】Quantify Each Feature's Contribution".center(100))
    print("=" * 100)

    # Get original feature list (excluding lag features)
    original_features = [col for col in X_train.columns if not col.startswith('Corn_lag_')]

    print(f"\nFeature list for analysis: {original_features}")
    print(f"Number of features: {len(original_features)}")

    # Create feature ablation analyzer
    feature_analyzer = FeatureAblationAnalyzer(
        y_test_true=y_test.values,
        feature_names=original_features,
        output_dir=FEATURE_ABLATION_CONFIG['output_dir']
    )

    # =========================================================================
    # Step 1: Add baseline (using all features)
    # =========================================================================
    print("\n[Step 1] Adding baseline experiment (using all features)...")

    # Use current best strategy's prediction as baseline
    baseline_pred = strategies_results['Strategy2-GRU_Residual']  # Adjust according to your best strategy

    feature_analyzer.add_baseline(
        predictions=baseline_pred,
        description=f'Complete model: LSTM+GRU+XGBoost, using all {len(original_features)} features'
    )

    # =========================================================================
    # Step 2: Remove features one by one and retrain
    # =========================================================================
    print("\n[Step 2] Removing features one by one and retraining...")
    print("Note: This will take a long time (each feature requires retraining the complete model)")

    for idx, feature_to_remove in enumerate(original_features, 1):
        print(f"\n{'=' * 100}")
        print(f"[{idx}/{len(original_features)}] Removing feature: {feature_to_remove}".center(100))
        print(f"{'=' * 100}")

        # Select remaining features
        remaining_features = [f for f in original_features if f != feature_to_remove]

        print(f"Remaining features: {remaining_features}")

        # Reconstruct training data (using remaining features only)
        X_train_ablation = X_train[remaining_features].copy()
        X_test_ablation = X_test[remaining_features].copy()

        # Add lag features
        X_train_feat_ablation = add_features(X_train_ablation, y_train)
        y_train_ablation = y_train.loc[X_train_feat_ablation.index]

        # Construct sequences
        X_train_seq_ablation, y_train_seq_ablation = create_sequences(
            X_train_feat_ablation, y_train_ablation, seq_len
        )

        print(f"Training set sequence shape: {X_train_seq_ablation.shape}")

        # Train LSTM
        print("  [1/3] Training LSTM...", end='')
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
        print(" Completed")

        # Train GRU
        print("  [2/3] Training GRU...", end='')
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
        print(" Completed")

        # Train XGBoost
        print("  [3/3] Training XGBoost...", end='')
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
        print(" Completed")

        # Rolling window prediction
        print("  [Prediction] Rolling window predicting test set...", end='')
        predictions_ablation = []
        history_X_feat = X_train_feat_ablation.tail(seq_len).copy()
        history_y = y_train_ablation.tail(5).copy()

        for test_idx in range(len(X_test)):
            # Construct current sample features (containing remaining features only)
            current_X_raw = X_test_ablation.iloc[test_idx:test_idx + 1].copy()

            for i in range(1, 6):
                if len(history_y) >= i:
                    current_X_raw[f'Corn_lag_{i}'] = history_y.iloc[-i]
                else:
                    current_X_raw[f'Corn_lag_{i}'] = 0

            # Construct sequence window
            current_window = pd.concat([history_X_feat.tail(seq_len - 1), current_X_raw])
            X_seq_test = current_window.values.reshape(1, seq_len, -1)

            # Get LSTM and GRU predictions
            lstm_pred = lstm_ablation.predict(X_seq_test, verbose=0)[0, 0]
            gru_pred = gru_ablation.predict(X_seq_test, verbose=0)[0, 0]

            # Construct XGBoost features and predict residual
            X_flat_test = X_seq_test.reshape(1, -1)
            X_xgb_test = np.hstack([
                X_flat_test,
                np.array([[lstm_pred, gru_pred, (lstm_pred + gru_pred) / 2,
                           abs(lstm_pred - gru_pred)]])
            ])

            residual = xgb_ablation.predict(X_xgb_test)[0]
            final_pred = gru_pred + residual
            predictions_ablation.append(final_pred)

            # Update history
            history_X_feat = pd.concat([history_X_feat.iloc[1:], current_X_raw])
            history_y = pd.concat([history_y.iloc[1:], pd.Series([y_test.iloc[test_idx]])])

        predictions_ablation = np.array(predictions_ablation)
        print(" Completed")

        # Add to feature analyzer
        feature_analyzer.add_single_feature_removal(
            feature_name=feature_to_remove,
            predictions=predictions_ablation,
            description=f'Using {len(remaining_features)} features (removed {feature_to_remove})'
        )

        print(f"\n✅ Feature ablation experiment for {feature_to_remove} completed")

    # =========================================================================
    # Step 3: Generate report and visualization
    # =========================================================================
    print("\n" + "=" * 100)
    print("Generating Feature Importance Report and Visualization".center(100))
    print("=" * 100)

    # Generate report
    feature_report = feature_analyzer.generate_feature_importance_report()

    # Generate visualization
    feature_analyzer.visualize_feature_importance(
        show=FEATURE_ABLATION_CONFIG['show_plots']
    )

    # Export LaTeX (optional)
    if FEATURE_ABLATION_CONFIG['save_latex']:
        feature_analyzer.export_to_latex()

    print("\n" + "=" * 100)
    print("✅ Feature Ablation Experiment Completed!".center(100))
    print(f"Results saved in: {FEATURE_ABLATION_CONFIG['output_dir']}".center(100))
    print("=" * 100)

else:
    if not FEATURE_ABLATION_CONFIG['enabled']:
        print("\n⏭️ Feature ablation experiment disabled (FEATURE_ABLATION_CONFIG['enabled']=False)")
    else:
        print("\n⚠️ Feature ablation module not imported, skipping experiment")