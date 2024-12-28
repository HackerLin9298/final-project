import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
import matplotlib.pyplot as plt

# Function to calculate additional technical indicators
def add_technical_indicators(data):
    data['MA7'] = data['Price'].rolling(window=7).mean()  # 7-day moving average
    data['MA21'] = data['Price'].rolling(window=21).mean()  # 21-day moving average
    data['EMA'] = data['Price'].ewm(span=14, adjust=False).mean()  # Exponential moving average
    data['RSI'] = calculate_rsi(data['Price'], window=14)  # Relative Strength Index
    data['High-Low'] = data['High'] - data['Low']  # High-Low price difference
    data['Price-Open'] = data['Price'] - data['Open']  # Price-Open difference
    data.fillna(0, inplace=True)  # Replace NaN values with 0
    return data

# Function to calculate RSI
def calculate_rsi(series, window):
    delta = series.diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 1. Load and clean the new data
file_path = '/Users/linyoucheng/ml/Final project/比特幣歷史數據.csv'  # Update to your file path
new_bitcoin_data = pd.read_csv(file_path)

# Rename columns to match the previous format
new_bitcoin_data.rename(columns={
    '日期': 'Date',
    '收市': 'Price',
    '開市': 'Open',
    '高': 'High',
    '低': 'Low',
    '成交量': 'Vol.',
    '升跌（%）': 'Change %'
}, inplace=True)

# Clean data
new_bitcoin_data['Date'] = pd.to_datetime(new_bitcoin_data['Date'])
new_bitcoin_data['Price'] = new_bitcoin_data['Price'].str.replace(',', '').astype(float)
new_bitcoin_data['Open'] = new_bitcoin_data['Open'].str.replace(',', '').astype(float)
new_bitcoin_data['High'] = new_bitcoin_data['High'].str.replace(',', '').astype(float)
new_bitcoin_data['Low'] = new_bitcoin_data['Low'].str.replace(',', '').astype(float)
new_bitcoin_data['Vol.'] = new_bitcoin_data['Vol.'].str.replace('K', 'e3').str.replace('M', 'e6').str.replace('B', 'e9')
new_bitcoin_data['Vol.'] = new_bitcoin_data['Vol.'].str.replace(',', '').astype(float)
new_bitcoin_data['Change %'] = new_bitcoin_data['Change %'].str.replace('%', '').astype(float)

# Sort by date
new_bitcoin_data.sort_values('Date', inplace=True)

# Add technical indicators
new_bitcoin_data = add_technical_indicators(new_bitcoin_data)

# 2. Feature engineering
features = new_bitcoin_data[['Open', 'High', 'Low', 'Vol.', 'MA7', 'MA21', 'EMA', 'RSI', 'High-Low', 'Price-Open']].values
target = new_bitcoin_data['Price'].values

# Scale features and target
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target.reshape(-1, 1))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42, shuffle=False)

# Reshape for LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3. Build the first LSTM model
model_1 = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1))),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1)  # Output layer
])

model_1.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the first model
history_1 = model_1.fit(X_train_reshaped, y_train, epochs=30, batch_size=16, validation_data=(X_test_reshaped, y_test), verbose=1)

# 4. Build the second LSTM model with different parameters
model_2 = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer
])

model_2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the second model
history_2 = model_2.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)

# 5. Gaussian Process Regression model
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
gpr_model.fit(X_train, y_train.ravel())

# Make predictions for Gaussian Process Regression
predictions_gpr_scaled, _ = gpr_model.predict(X_test, return_std=True)
predictions_gpr = target_scaler.inverse_transform(predictions_gpr_scaled.reshape(-1, 1))

# 6. Make predictions for both LSTM models
predictions_1_scaled = model_1.predict(X_test_reshaped)
predictions_2_scaled = model_2.predict(X_test_reshaped)

# Inverse scale the predictions
y_test_inversed = target_scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_1 = target_scaler.inverse_transform(predictions_1_scaled)
predictions_2 = target_scaler.inverse_transform(predictions_2_scaled)

# 7. Visualize results
plt.figure(figsize=(12, 6))
plt.plot(y_test_inversed, label="Actual Price", color='blue')
plt.plot(predictions_1, label="Model 1 Predictions", linestyle='--', color='orange')
plt.plot(predictions_2, label="Model 2 Predictions", linestyle='--', color='green')
plt.plot(predictions_gpr, label="Gaussian Process Regression Predictions", linestyle='--', color='red')
plt.title("Comparison of LSTM Models and Gaussian Process Regression")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# Visualize training and validation loss for both LSTM models
plt.figure(figsize=(12, 6))
plt.plot(history_1.history['loss'], label='Model 1 Training Loss', color='orange')
plt.plot(history_1.history['val_loss'], label='Model 1 Validation Loss', color='orange', linestyle='--')
plt.plot(history_2.history['loss'], label='Model 2 Training Loss', color='green')
plt.plot(history_2.history['val_loss'], label='Model 2 Validation Loss', color='green', linestyle='--')
plt.title('Training and Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize training and validation accuracy (MAE) for both LSTM models
plt.figure(figsize=(12, 6))
plt.plot(history_1.history['mean_absolute_error'], label='Model 1 Training MAE', color='orange')
plt.plot(history_1.history['val_mean_absolute_error'], label='Model 1 Validation MAE', color='orange', linestyle='--')
plt.plot(history_2.history['mean_absolute_error'], label='Model 2 Training MAE', color='green')
plt.plot(history_2.history['val_mean_absolute_error'], label='Model 2 Validation MAE', color='green', linestyle='--')
plt.title('Training and Validation MAE Comparison')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# 8. Future prediction
# Prepare the last available data point for prediction
last_data_point = features_scaled[-1].reshape(1, -1, 1)

# Predict the next day's price for both LSTM models
future_prediction_1_scaled = model_1.predict(last_data_point)
future_prediction_2_scaled = model_2.predict(last_data_point)

future_prediction_1 = target_scaler.inverse_transform(future_prediction_1_scaled)
future_prediction_2 = target_scaler.inverse_transform(future_prediction_2_scaled)

# Gaussian Process Regression Future Prediction
future_prediction_gpr_scaled, _ = gpr_model.predict(features_scaled[-1].reshape(1, -1), return_std=True)
future_prediction_gpr = target_scaler.inverse_transform(future_prediction_gpr_scaled.reshape(-1, 1))

print(f"Predicted Future Price by Model 1: {future_prediction_1[0][0]}")
print(f"Predicted Future Price by Model 2: {future_prediction_2[0][0]}")
print(f"Predicted Future Price by Gaussian Process Regression: {future_prediction_gpr[0][0]}")
