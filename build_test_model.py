import time
start_time = time.time()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

filename = 'csv_files/A.csv'

dataset = pd.read_csv(filename)
stock_data = dataset[['Open', 'Close']].values

# Scale data to values between 0-1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_stock_data = scaler.fit_transform(stock_data)

# Split the data into training 70%, testing the rest.  The data size varies, otherwise I would have used slicing.
def split_data (data):
    # we need data to be 4586 days or less.
    data_size = len(data) 
    expected_length = 4586
    training_percent = 0.67
    start_index = 0

    if data_size > expected_length:
        start_index = data_size - expected_length

    # 70% of data
    stop_index = round(expected_length * training_percent)+ start_index
    return (data[start_index:stop_index], data[stop_index:])

# Populate the training and testing datasets.
training_data, testing_data = split_data(scaled_stock_data)

# Timesteps for LSTM
def create_timestep(data):
    X_values = []
    y_values = []

    for i in range(60, len(data)):
        X_values.append(data[i-60:i,0])
        y_values.append(data[i,0])

    X_values = np.array(X_values)
    y_values = np.array(y_values)
    X_values = np.reshape(X_values, (X_values.shape[0], X_values.shape[1], 1))
    return (X_values, y_values)

# Obtain timestep data
X_train_timestep, y_train_timestep = create_timestep(training_data)
X_test_timestep, y_test_timestep = create_timestep(testing_data)

# Define the LSTM Model
#def create_lstm_model():
lstm_model = Sequential()
lstm_model.add(LSTM(units=32, 
                return_sequences=True, 
                input_shape=(X_train_timestep.shape[1], 1),
                name='layer_1'
                ))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=32, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=32, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=32))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Tensorboard Logger
logger = TensorBoard(
    log_dir='logs',
    histogram_freq= 0,
    write_graph=True
)

# Train the model 
lstm_model.fit(
    X_train_timestep,
    y_train_timestep, 
    epochs=10, 
    verbose=2,
    shuffle=True,
    callbacks=[logger]
    )

# Persist LSTM model
lstm_model.save('trained_models/trained_model.h5')

predicted_price = lstm_model.predict(X_test_timestep)

#predicted_price = scaler.inverse_transform(predicted_price)

# Plotting the results
plt.plot(y_test_timestep, color = 'cyan', label = filename)
plt.plot(predicted_price, color = 'red', label = 'Predicted Price')
plt.title(''.join([filename, ' Stock Prediction']))
plt.xlabel('Days')
plt.ylabel(''.join([filename, ' Stock Price']))
plt.legend()
print("--- %s seconds ---" % (time.time() - start_time))   # To keep track of how long does it take the script to finish.
plt.show()