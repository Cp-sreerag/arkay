import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

start = "2010-01-01"
end = "2028-01-01"
st.title('Stock trend prediction')
user_input = st.text_input("Enter stock ticker", 'AAPL')

# Download stock data using yfinance
krt = yf.Ticker(user_input)
df = krt.history(start=start, end=end)
df = df.reset_index()

st.write(df)

# Describing data
st.subheader("Data from 2010")
st.write(df.describe())

# Visualization of Closing Price vs Time
st.subheader('Closing price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Visualization of Closing price vs Time with 100-day moving average
st.subheader('Closing price vs Time chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

# Visualization of Closing price vs Time with 100 and 200-day moving averages
st.subheader('Closing price vs Time chart with 100 and 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Prepare data for LSTM model
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Only using 'Close' price for this example
data = df[['Close']]

# Normalize the data using MinMaxScaler (scales data between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create the dataset with input (X) and output (y)
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # Use past 'time_step' days
        y.append(data[i + time_step, 0])  # Next day's stock price
    return np.array(X), np.array(y)

# Prepare the dataset (60 days for training)
X, y = create_dataset(scaled_data, time_step=60)

# Reshape the input to be 3D for LSTM: [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing datasets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Load the pre-trained model
model = load_model('stock_models.keras')

# Make predictions
predictions = list(model.predict(X_test))
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize Predicted vs Original stock prices
st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, 'b', label='Original Price')
plt.plot(predictions, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Example: Predict the next day's stock price (61st day)
# For this, use the most recent 60 days from the test set
last_60_days = scaled_data[-60:]  # Get the last 60 days from your dataset
last_60_days = last_60_days.reshape(1, 60, 1)  # Reshape for LSTM input

# Predict the next day's price (61st day)
predicted_61st_day = model.predict(last_60_days)

# Inverse transform to get the price in original scale
predicted_61st_day = scaler.inverse_transform(predicted_61st_day)
st.subheader(f"Predicted next day stock close price of {user_input}: {predicted_61st_day[0][0]}")
st.subheader(f"Last day stock close price of {user_input}: {df['Close'][-1]}")
if df['Close'][-1] > predicted_61st_day[0][0]:
    st.subheader(f"Possibility of an uptrend")
else :
     st.subheader(f"Possibility of a downtrend")




# #splitting data into training and testing
# data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
# print(data_training.shape)
# print(data_testing.shape)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))
# data_training_array = scaler.fit_transform(data_training)

# loading model



# #Testing part
# past_100_data = data_training.tail(100)
# final_df = pd.concat([past_100_data, data_testing], ignore_index=True)
# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test = []

# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i - 100:i])
#     y_test.append(input_data[i, 0])

# y_predicted = model.predict(x_test)
# scaler = scaler.scale_
# scale_factor= 1/scaler

# y_predicted = y_predicted*scale_factor
# y_test = np.array(y_test)
# y_test = y_test*scale_factor
# # print(y_predicted)
# #final graph

