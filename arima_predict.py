import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

warnings.filterwarnings("ignore") # statsmodelsの警告を非表示にする

# データの読み込み
data = pd.read_csv("stock_price.csv")

# 日付をdatetime型に変換し、インデックスに設定
data['日付け'] = pd.to_datetime(data['日付け'])
data.set_index('日付け', inplace=True)
data.sort_index(inplace=True)

# 訓練データとテストデータに分割
# ARIMAモデルは単変量モデルなので、予測したい列（'終値'）だけを抽出します。
train = data.loc['1987-02-12':'2022-12-30', '終値']
test = data.loc['2023-01-04':, '終値']

# 訓練データセットの可視化
sns.set(style="darkgrid", font="Hiragino Maru Gothic Pro")
plt.figure(figsize=(15, 8))
plt.plot(train.index, train, label="NTT平均株価終値", color="blue")
plt.title("1987年〜2022年のNTT平均株価", fontsize=16)
plt.xlabel("日付", fontsize=12)
plt.ylabel("終値", fontsize=12)
plt.legend()
plt.savefig(f'ntt_stock_price_arima/training_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png')


# ARIMAモデルの構築と訓練
# 予測対象を「終値」のSeriesとして渡します。
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# 予測
predictions = model_fit.forecast(steps=len(test))

# 予測結果の可視化
plt.figure(figsize=(15, 8))
plt.plot(train, label='訓練データ', color='blue')
plt.plot(test.index, test, label='実際の株価', color='green')
plt.plot(test.index, predictions, label='ARIMA予測', color='red', linestyle='--')
plt.title('ARIMAモデルによる株価予測', fontsize=16)
plt.xlabel('日付', fontsize=12)
plt.ylabel('終値', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig(f'stock_price_prediction_arima/comparison_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png')

# 予測結果の評価（RMSE）
rmse = sqrt(mean_squared_error(test, predictions))
print(f'テストデータのRMSE: {rmse:.2f}')