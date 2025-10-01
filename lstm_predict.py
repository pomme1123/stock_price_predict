import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math
import random

#ランダム値を固定することによって、再現性を持たせる
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# データ読み込み
data = pd.read_csv("stock_price.csv")

# 日付けをdatetime型に変換し、インデックスに設定
data['日付け'] = pd.to_datetime(data['日付け'])
data.set_index('日付け', inplace=True)
data.sort_index(inplace=True)

# 訓練データとテストデータに分割
train_data = data.loc['1987-02-12':'2021-12-30']
test_data = data.loc['2022-01-04':]

# グラフ描画
sns.set(style="darkgrid", font="Hiragino Maru Gothic Pro")
plt.figure(figsize=(15, 8))
plt.plot(train_data.index, train_data["終値"], label="NTT平均株価終値", color="blue")
plt.title("1987年〜2021年のNTT平均株価", fontsize=16)
plt.xlabel("日付", fontsize=12)
plt.ylabel("終値", fontsize=12)
plt.legend()
plt.savefig(f'ntt_stock_price/{pd.Timestamp.now()}.png')

# 終値データを0〜1の範囲にスケーリング
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[["終値"]].values.reshape(-1, 1))
scaled_test_data = scaler.transform(test_data[["終値"]].values.reshape(-1, 1))


def create_dataset(data, time_step=90):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# 訓練データの作成
X_train, y_train = create_dataset(scaled_train_data)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# テストデータの作成
X_test, y_test = create_dataset(scaled_test_data)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# LSTMモデルの定義
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# モデルのコンパイルとトレーニング
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

# 訓練データの予測
train_predicted = model.predict(X_train)
train_predicted_prices = scaler.inverse_transform(train_predicted)
train_real_prices = scaler.inverse_transform(y_train.reshape(-1, 1))

# テストデータの予測
test_predicted = model.predict(X_test)
test_predicted_prices = scaler.inverse_transform(test_predicted)
test_real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# グラフ描画（訓練・テスト両方）
plt.figure(figsize=(15, 8))
plt.plot(train_data.index[-len(train_real_prices):], train_real_prices, color='blue', label='Train Real')
plt.plot(train_data.index[-len(train_predicted_prices):], train_predicted_prices, color='cyan', label='Train Predicted')
plt.plot(test_data.index[-len(test_real_prices):], test_real_prices, color='green', label='Test Real')
plt.plot(test_data.index[-len(test_predicted_prices):], test_predicted_prices, color='red', label='Test Predicted')
plt.title('NTT株価予測（訓練・テスト比較）')
plt.xlabel('日付')
plt.ylabel('終値')
plt.legend()
plt.savefig(f'stock_price_prediction/compare_{pd.Timestamp.now()}.png')

# 訓練データの評価指標
train_rmse = math.sqrt(mean_squared_error(train_real_prices, train_predicted_prices))
train_mae = mean_absolute_error(train_real_prices, train_predicted_prices)
print(f'Train RMSE: {train_rmse}')


# テストデータの評価指標
test_rmse = math.sqrt(mean_squared_error(test_real_prices, test_predicted_prices))
test_mae = mean_absolute_error(test_real_prices, test_predicted_prices)
print(f'Test RMSE: {test_rmse}')


# 過学習の程度を評価する比率
rmse_ratio = train_rmse / test_rmse

# カスタム評価指標の計算
# w1: テストRMSEの重み (例: 0.8)
# w2: 比率の重み (例: 0.2)
w1 = 0.8
w2 = 0.2

# 評価ポイントを計算
# 比率が1に近いほど良いモデルなので、(rmse_ratio - 1)の絶対値
custom_score = (w1 * test_rmse) + (w2 * abs(rmse_ratio - 1))


print(f'訓練・テストRMSEの比率: {rmse_ratio:.2f}')
print(f'カスタム評価ポイント（重み付け）: {custom_score:.2f}')
