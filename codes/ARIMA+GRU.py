# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib

# 设置Matplotlib后端为Agg
matplotlib.use('Agg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class HybridModel:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.scaler = MinMaxScaler()
        self.arima_model = None
        self.gru_model = None

    def load_data(self, path):
        """加载并预处理数据"""
        data = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        data = data.interpolate()
        print("数据加载完成，前5行示例:")
        print(data.head())
        return data

    def prepare_data(self, test_size=0.2):
        """准备训练测试数据"""
        train_size = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:train_size]
        self.test_data = self.data.iloc[train_size:]

        # 训练ARIMA
        self.train_arima()

        # 准备GRU数据
        self.prepare_gru_data()

    def train_arima(self):
        """训练ARIMA模型"""
        try:
            self.arima_model = ARIMA(self.train_data['Corn'], order=(0, 2, 1)).fit()
            self.residuals = self.arima_model.resid.dropna()
            print("ARIMA模型训练成功")
        except Exception as e:
            print(f"ARIMA训练错误: {str(e)}")
            raise

    def prepare_gru_data(self, seq_len=5):
        """准备GRU训练数据"""
        # 添加ARIMA残差
        train_features = self.train_data.drop('Corn', axis=1)
        train_features['Residual'] = self.residuals.reindex(self.train_data.index).fillna(0)

        # 归一化
        self.train_features_scaled = pd.DataFrame(
            self.scaler.fit_transform(train_features),
            columns=train_features.columns,
            index=train_features.index
        )

        # 创建序列数据
        self.X_train, self.y_train = self.create_sequences(
            self.train_features_scaled,
            self.train_data['Corn'],
            seq_len
        )
        print(f"GRU训练数据准备完成，输入形状: {self.X_train.shape}")

    @staticmethod
    def create_sequences(features, target, seq_len):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(features) - seq_len):
            X.append(features.iloc[i:i + seq_len].values)
            y.append(target.iloc[i + seq_len])
        return np.array(X), np.array(y)

    def build_gru_model(self):
        """构建并训练GRU模型"""
        try:
            model = Sequential([
                GRU(64, input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                    return_sequences=True),
                Dropout(0.2),
                GRU(32),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            print("\nGRU模型结构摘要:")
            model.summary()

            self.gru_model = model
            return model
        except Exception as e:
            print(f"GRU模型构建错误: {str(e)}")
            raise

    def train_gru(self, epochs=100, batch_size=32):
        """训练GRU模型"""
        history = self.gru_model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        self.plot_training_history(history)

    def predict(self):
        """进行混合预测"""
        # 准备测试数据
        test_features = self.test_data.drop('Corn', axis=1)
        test_features['Residual'] = 0
        test_features_scaled = pd.DataFrame(
            self.scaler.transform(test_features),
            columns=test_features.columns,
            index=test_features.index
        )

        # ARIMA预测
        arima_pred = self.arima_model.get_forecast(steps=len(self.test_data)).predicted_mean

        # GRU预测
        X_test, _ = self.create_sequences(test_features_scaled, self.test_data['Corn'], 5)
        gru_residual_pred = self.gru_model.predict(X_test).flatten()

        # 组合预测
        min_len = min(len(arima_pred), len(gru_residual_pred))
        self.hybrid_pred = arima_pred[:min_len] + gru_residual_pred[:min_len]
        self.true_values = self.test_data['Corn'].iloc[5:5 + min_len]

        self.evaluate()
        self.plot_results()

    def evaluate(self):
        """评估模型性能"""
        print("\n" + "=" * 50 + " 模型评估 " + "=" * 50)
        print(f"混合模型 MAE: {mean_absolute_error(self.true_values, self.hybrid_pred):.4f}")
        print(f"混合模型 R²: {r2_score(self.true_values, self.hybrid_pred):.4f}")

    def plot_training_history(self, history):
        """绘制训练历史"""
        plt.figure()
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('GRU模型训练过程')
        plt.ylabel('MSE损失值')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
        print("训练过程图表已保存为training_history.png")

    def plot_results(self):
        """绘制预测结果"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.true_values.index, self.true_values, label='真实值')
        plt.plot(self.true_values.index, self.hybrid_pred, label='混合预测', linestyle='--')
        plt.title('ARIMA-GRU混合模型预测效果')
        plt.legend(loc='best', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.show(block=True)


if __name__ == '__main__':
    try:
        # 初始化模型
        model = HybridModel('Corn-new.csv')

        # 准备数据
        model.prepare_data()

        # 构建和训练GRU
        model.build_gru_model()
        model.train_gru(epochs=100)  # 减少epochs以便快速测试

        # 进行预测
        model.predict()

        print("\n模型运行完成，结果图表已保存")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")