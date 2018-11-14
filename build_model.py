import time
start_time = time.time()  # To find out how long the program takes to execute

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
np.random.seed(1337)  # Dreams are allowed

def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

def scale_data(data):
    # Scale data to values between 0-1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_stock_data = scaler.fit_transform(data)
    return scaled_stock_data, scaler

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

def create_testing_training_data(scaled_data):
    # Populate the training and testing datasets.
    training_data, testing_data = split_data(scaled_data)
    return training_data, testing_data 

# Timesteps for LSTM
def timestep(data):
    X_values = []
    y_values = []

    for i in range(60, len(data)):
        X_values.append(data[i-60:i,0])
        y_values.append(data[i,0])

    X_values = np.array(X_values)
    y_values = np.array(y_values)
    X_values = np.reshape(X_values, (X_values.shape[0], X_values.shape[1], 1))
    return X_values, y_values

# Obtain timestep data
def create_timestep(data):
    X_timestep, y_timestep = timestep(data)
    return X_timestep, y_timestep

# Define the LSTM Model
def create_model(input_shape):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=32, 
                    return_sequences=True, 
                    input_shape=input_shape,
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
    return lstm_model

# Tensorboard Logger
def tensorboard_logger():
    logger = TensorBoard(
        log_dir='logs',
        histogram_freq= 0,
        write_graph=True
    )
    return logger

# Train the model 
def train_the_model(lstm_model, X_train_timestep, y_train_timestep):
    logger = tensorboard_logger()
    lstm_model.fit(
        X_train_timestep,
        y_train_timestep, 
        epochs=1, 
        verbose=2,
        shuffle=True,
        callbacks=[logger]
        )
    return lstm_model

def plot_results(actual_value, predicted_value, filename):
    # Plotting the results
    plt.plot(actual_value, color = 'cyan', label = filename)
    plt.plot(predicted_value, color = 'red', label = 'Predicted Price')
    plt.title(''.join([filename, ' Stock Prediction']))
    plt.xlabel('Days')
    plt.ylabel(''.join([filename, ' Stock Price']))
    plt.legend()
    plt.show()
    return None

if __name__ == "__main__":
    #This is mostly to test the script and it works, except the part to invert the scaler values
    filename = 'csv_files/A.csv'
    dataset = get_data(filename)
    data_values = dataset[['Open', 'Close']].values
    scaled_data, scaler = scale_data(data_values)
    training_data, testing_data = create_testing_training_data(scaled_data)
    X_training_timestep, y_training_timestep = timestep(training_data)
    X_testing_timestep, y_testing_timestep = timestep(testing_data)
    input_shape = (X_training_timestep.shape[1],1)
    model = create_model(input_shape)
    trained_model = train_the_model(model, X_training_timestep, y_training_timestep)
    trained_model.save('trained_models/trained_model.h5')
    predicted_value = model.predict(X_testing_timestep)
    actual_value = y_testing_timestep
    # Scale the values back to their original values
    #predicted_price = scaler.inverse_transform(predicted_value)
    #actual_value = scaler.inverse_transform(actual_value.reshape(-1,1))
    print("--- %s seconds ---" % (time.time() - start_time))   # How long it took to finish executing the program
    print(input_shape)
    plot_results(actual_value, predicted_value, filename)