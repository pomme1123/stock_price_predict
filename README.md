Stock_Price_Predict_LSTM
"Stock_Price_Predict_LSTM" is an LSTM-based tool for predicting stock prices using historical time series data.

#DEMO
The tool focuses on utilizing LSTM's strength in capturing sequential patterns and evaluating model risk using RMSE.

This demo shows the predicted stock prices (e.g., closing price) against the actual historical data, demonstrating the model's performance on unseen data.

#Features
Stock_Price_Predict_LSTM uses the Keras/TensorFlow backend for building the recurrent neural network.

#Key Features:
LSTM Time Series Modeling: Utilizes 2-layer LSTM architecture for robust sequence learning.

Data Preparation: Includes necessary steps for scaling data (0∼1 normalization) and transforming it into the required 3D format (samples, timesteps, features).

RMSE Evaluation: Employs Root Mean Squared Error (RMSE) as the primary evaluation metric, prioritizing the penalization of large prediction errors inherent in financial risk assessment.

Hyperparameter Focus: Highlights the role of timesteps (sequence length) and epochs in optimizing prediction performance.

#Requirement
Python 3.x

NumPy

Pandas

TensorFlow / Keras

scikit-learn (for MinMaxScaler)

Environments under [Anaconda for Windows/Mac] are recommended.

#Usage
Please ensure your historical data is stored in a .csv file (e.g., stock_price.csv) with the required features (Date, Close, etc.).

Create a python code file named "lstm_predict.py" and insert your modeling code.

Run the prediction script:

Bash

python lstm_predict.py
The script will output the model's training and testing RMSE and generate a final price prediction.