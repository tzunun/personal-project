import time
start_time = time.time()
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import build_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# Load the matrix that was created when loading the data


if __name__ == "__main__":


    sp500_matrix = np.load('./data_matrix.npy')
#    sp500_data = sp500_matrix[:,:,[0,3]]
    sp500_data = sp500_matrix.reshape(2315930,4)  # In order to scale, as it is expecting a 2d array
    sp500_data_scaler = MinMaxScaler(feature_range=(0,1))
    sp500_scaled_data = sp500_data_scaler.fit_transform(sp500_data)
    sp500_scaled_data = sp500_scaled_data.reshape(4586, 505,4)

    # Flatten to reduce the dimensions later
    sp500_2D = []
    for index in range (len(sp500_scaled_data)):
        sp500_2D.append(sp500_scaled_data[index].flatten())

    sp500_pca = PCA(n_components=1)
    sp500_pca.fit(sp500_2D)
    sp500_reduced_dimensions = sp500_pca.transform(sp500_2D)
    sp500_array = np.asarray(sp500_reduced_dimensions)
    sp500_1D = sp500_array.flatten()
    print(sp500_1D.shape)

    # SPRD data
    spdr_data = pd.read_csv('csv_files/testSPY.csv')
    y_scaler = MinMaxScaler(feature_range=(0,1))
    y_data = spdr_data['Close'].values
    y_data = y_data.reshape(-1,1)
    y_scaled_data = y_scaler.fit_transform(y_data)
    y_scaled_data = y_scaled_data.flatten()
    
    # Combine historical data from SPDR and the components of the sp500, built 
    sp500_y_combined = np.stack((sp500_1D, y_scaled_data), axis=-1)

    # Build the LSTM
    training_data, testing_data = build_model.create_testing_training_data(sp500_y_combined)
    X_training_timestep, y_training_timestep = build_model.timestep(training_data)
    X_testing_timestep, y_testing_timestep = build_model.timestep(testing_data)
    input_shape = (X_training_timestep.shape[1],1)
    model = build_model.create_model(input_shape)
    trained_model = build_model.train_the_model(model, X_training_timestep, y_training_timestep)
    trained_model.save('trained_models/trained_model.h5')
    predicted_value = model.predict(X_testing_timestep)
    actual_value = y_testing_timestep
    predicted_value = y_scaler.inverse_transform(predicted_value.reshape(-1,1))
    actual_value = y_scaler.inverse_transform(actual_value.reshape(-1,1))
    print("--- %s seconds ---" % (time.time() - start_time))   # How long it took to finish executing the program
    build_model.plot_results(actual_value, predicted_value, 'Actual SPRD Price')