import pandas as pd
from numba import jit
import numpy as np
import os.path   # To check whether a file exist

# Load the s&p500 csv file to read the csv names
sp500_components = pd.read_csv('sp500.csv')

# Create a matrix of 4586 by ~504 by 4, I only need 4 columns per stock, per day, for 4586 days of data or ~18 years.
days = 4586   
components = len(sp500_components)
columns = 4    # I'm only using the High, Low, Open and Close columns 
prepared_data_matrix = np.zeros((days,components,columns))


@jit
def add_to_data_matrix(csv_file, index):

    csv_data = pd.read_csv(csv_file)
    date = (days - 1)
    data_size = (len(csv_data) -1)
    
    if date <= data_size:
        counter = date   # 4586 spots counting from the 0th index
    else:
        counter = data_size # Some number less than 4586
    
    print('counter =', counter)
    
    while counter >= 0:
        prepared_data_matrix[date][index][0] = csv_data['Open'][counter]
        prepared_data_matrix[date][index][1] = csv_data['High'][counter]
        prepared_data_matrix[date][index][2] = csv_data['Low'][counter]
        prepared_data_matrix[date][index][3] = csv_data['Close'][counter]
        date -= 1
        counter -= 1

# Create each stock's csv filename 
for index in range (len(sp500_components) - 1):
    
    stock_symbol = str(sp500_components['Symbol'][index])    # Get the stock symbol
    csv_file = ''.join(['csv_files/', stock_symbol, '.csv'])    # This is the actual file to read from the local directory 
    print(csv_file)
    
    # Read files found in the local directory
    if os.path.isfile(csv_file):
        add_to_data_matrix(csv_file, index)    #  Add the contents of the file to the proper location within the prepared_data_matrix(4586x504x4)
    else:
        print("File not found!")
        
print('Done creating prepared_data_matrix!')    # This process takes a while.

# Save combined components data which now it's prepared_data_matrix as a numpy array to be used in the future and avoid reassembling the matrix again.
np.save('data_matrix.npy', prepared_data_matrix, allow_pickle=False)
