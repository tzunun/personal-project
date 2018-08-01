
# coding: utf-8

# In[38]:


import pandas as pd

# Load the s&p500 csv file to read the csv names
sp500_components = pd.read_csv('sp500.csv')


# In[39]:


import numpy as np

# Create a matrix of 4482 by ~540 by 4, I only need 4 columns per stock, per day, for 4482 days of data or 18 years.
days = 4482   
components = len(sp500_components)
columns = 4    # I'm only using High, Low, Open and Close columns 

prepared_data_matrix = np.zeros((days,components,columns))


# In[40]:


def add_to_data_matrix(csv_file, index):
    
    csv_data = pd.read_csv(csv_file)
    
    date = (len(prepared_data_matrix) - 1)
    data_size = (len(csv_data) -1)
    
    if date <= data_size:
        counter = date   # 4528 spots counting from the 0th index
    else:
        counter = data_size # Some number less than 4528
    
    print('counter =', counter)
    
    while counter >= 0:
        
        prepared_data_matrix[date][index][0] = csv_data['Open'][counter]
        prepared_data_matrix[date][index][1] = csv_data['High'][counter]
        prepared_data_matrix[date][index][2] = csv_data['Low'][counter]
        prepared_data_matrix[date][index][3] = csv_data['Close'][counter]
        date -= 1
        counter -= 1
        
    print('Finished adding', csv_file, 'data to the prepared_data_matrix!')


# In[41]:


import os.path   # To check whether a file exist

for index in range (len(sp500_components) - 1):
    
    stock_symbol = str(sp500_components['Symbol'][index])    # Get the stock symbol
    csv_file = ''.join([stock_symbol, '.csv'])    # Create the filename to read from the local directory 
    
    # Read files found in the local directory
    if os.path.isfile(csv_file):
        print(index, csv_file)
        add_to_data_matrix(csv_file, index)
    else:
        print("File not found!")
print('Done creating prepared_data_matrix!')


# In[46]:


# Load the SPY also known as SPYDR
spydr_data = pd.read_csv('SPY.csv')
y = spydr_data[['Open', 'High', 'Low', 'Close']].values


# In[52]:


X = prepared_data_matrix

