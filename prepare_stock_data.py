
# coding: utf-8

# In[1]:
import pandas as pd

# Load the s&p500 csv file to read the csv names
sp500_components = pd.read_csv('sp500.csv')


# In[2]:
from numba import jit
import numpy as np

# Create a matrix of 4586 by ~504 by 4, I only need 4 columns per stock, per day, for 4586 days of data or ~18 years.
days = 4586   
components = len(sp500_components)
columns = 4    # I'm only using High, Low, Open and Close columns 

prepared_data_matrix = np.zeros((days,components,columns))

# In[4]:
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
        
    #print('Finished adding', csv_file, 'data to the prepared_data_matrix!')


# In[5]:
import os.path   # To check whether a file exist

for index in range (len(sp500_components) - 1):
    
    stock_symbol = str(sp500_components['Symbol'][index])    # Get the stock symbol
    csv_file = ''.join([stock_symbol, '.csv'])    # Create the filename to read from the local directory 
    print(csv_file)
    
    # Read files found in the local directory
    if os.path.isfile(csv_file):
        add_to_data_matrix(csv_file, index)
    else:
        print("File not found!")
        
print('Done creating prepared_data_matrix!')

#%%
# Use PCA to reduce dimensionality, ~500 points per day to one?
from sklearn.decomposition import PCA
@jit
def reduce_dimensions(daily_data):
    pca = PCA(n_components=1)
    pca.fit(daily_data)
    return pca.singular_values_

reduced_dimensionality_matrix = np.zeros([days,1])

# In[9]:
# Creating the reduced dimension matrix
for day in range(days - 1):
    reduced_dimensionality_matrix[day] = reduce_dimensions(prepared_data_matrix[day])

#%%
print(y_data)

#%%
# X and Y data
y_data = pd.read_csv('SPY.csv')
y_data = y_data['Close'].values
y_data = y_data.reshape(-1,1)
x_data = reduced_dimensionality_matrix #.reshape(-1,1)
#y_data = y_data.reshape(-1,1)
#x_data = reduced_dimensionality_matrix.reshape(-1,1)


# In[7]:
# Split the data intor training set and test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 0)

#%%
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(x_data)
y = sc_y.fit_transform(y_data)
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#y_train = scaler.fit_transform(y_train)
#
#X_test = scaler.transform(X_test)
#y_test = scaler.transform(y_test)


#%%
# Predicting the unpredictable, yeah right!
from sklearn.svm import SVR
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score

svr_classifier = SVR(kernel="rbf",degree=1)
svr_classifier.fit(X, y)


#%%
# Plot the SVR results

import matplotlib.pyplot as plt
# Actual Data
plt.scatter(X,y, color = 'cyan')
# Prediction 
plt.plot(X, svr_classifier.predict(X), color= 'red')
plt.xlabel('sp&500 stocks')
plt.ylabel('SPY')
plt.show()


#%%

y_prediction = sc_y.inverse_transform(sc_X.transform())
