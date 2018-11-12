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

def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

def scale_data(data):
    # Scale data to values between 0-1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_stock_data = scaler.fit_transform(data)
    return scaled_stock_data

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
    #X_train_timestep, y_train_timestep = create_timestep(training_data)
    #X_test_timestep, y_test_timestep = create_timestep(testing_data)
    X_timestep, y_timestep = timestep(data)
    return X_timestep, y_timestep

# Define the LSTM Model
def define_model(input_shape):
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

# Create the model
def create_model(input_shape):
    model = define_model(input_shape)
    return model

# Train the model 
def train_the_model(lstm_model, X_train_timestep, y_train_timestep):
    logger = tensorboard_logger()
    lstm_model.fit(
        X_train_timestep,
        y_train_timestep, 
        epochs=10, 
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
    print("--- %s seconds ---" % (time.time() - start_time))   # To keep track of how long does it take the script to finish.
    plt.show()
    return None

#main():
if __name__ == "__main__":
    lock = thread.allocate_lock()

    # Persist LSTM model
    lstm_model.save('trained_models/trained_model.h5')
    predicted_value = lstm_model.predict(X_test_timestep)
    #predicted_price = scaler.inverse_transform(predicted_price)
    
    filename = 'csv_files/A.csv'
    stock_data = dataset[['Open', 'Close']].values

    
#   input_shape = (X_train_timestep.shape[1],1)
'''The main goal is to have functions and no global variables and to be able to call this script from the final program to run the code, basically be able
to call this program from another program in order to not repeat the code'''