import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
import pickle

# Add technical indicators
def add_technical_indicators(data):
    data['MA7'] = data['Price'].rolling(window=7).mean()  # 7-day moving average
    data['MA21'] = data['Price'].rolling(window=21).mean()  # 21-day moving average
    data['EMA'] = data['Price'].ewm(span=14, adjust=False).mean()  # Exponential moving average
    data['High-Low'] = data['High'] - data['Low']  # High-Low price difference
    data['Price-Open'] = data['Price'] - data['Open']  # Price-Open difference
    data.fillna(0, inplace=True)  # Replace NaN values with 0
    return data

# Load and clean data
file_path = "/Users/linyoucheng/ml/Final project/比特幣歷史數據.csv"
data = pd.read_csv(file_path)
data.rename(columns={
    '日期': 'Date',
    '收市': 'Price',
    '開市': 'Open',
    '高': 'High',
    '低': 'Low',
    '成交量': 'Vol.',
    '升跌（%）': 'Change %'
}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Price'] = data['Price'].str.replace(',', '').astype(float)
data['Open'] = data['Open'].str.replace(',', '').astype(float)
data['High'] = data['High'].str.replace(',', '').astype(float)
data['Low'] = data['Low'].str.replace(',', '').astype(float)
data['Vol.'] = data['Vol.'].str.replace('K', 'e3').str.replace('M', 'e6').str.replace('B', 'e9').str.replace(',', '').astype(float)
data['Change %'] = data['Change %'].str.replace('%', '').astype(float)
data.sort_values('Date', inplace=True)

# Add technical indicators
data = add_technical_indicators(data)

# Save cleaned data
data.to_csv("bitcoin_cleaned.txt", index=False, sep='\t')

# Prepare features and target
features = data[['Open', 'High', 'Low', 'Vol.', 'MA7', 'MA21', 'EMA', 'High-Low', 'Price-Open']].values
target = data['Price'].values

# Scale data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.reshape(-1, 1))

# Save scaler for later use
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42, shuffle=False)

# Reshape for LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1))),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train model
model.fit(X_train_reshaped, y_train, epochs=30, batch_size=16, validation_data=(X_test_reshaped, y_test), verbose=1)

# Save trained model
model.save("lstm_model.h5")
print("Model and scaler saved successfully.")
print(f"Training features shape: {features.shape}") 