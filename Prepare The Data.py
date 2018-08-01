
# coding: utf-8

# In[16]:


import pandas as pd

# Load the s&p500 csv file to read the csv names
sp500_components = pd.read_csv('sp500.csv')


# In[144]:


from datetime import datetime

def add_to_database(csv_file, prepared_data_matrix, index):
    
    csv_data = pd.read_csv(csv_file)
    csv_file_starting_date = csv_data['Date'][0]
    
    # 2000-01-03 January 3, 2000 will be the starting date for the data.  Some stocks don't go back to that day.
    target_starting_date = '2000-01-03'
    
    # 2018-01-03 January 3, 2018
    target_ending_date = '2018-01-03'
    
    
    for date in range(len(prepared_data_matrix)):
        prepared_data_matrix[date][index][0] = csv_data['Open'][date]
        prepared_data_matrix[date][index][1] = csv_data['High'][date]
        prepared_data_matrix[date][index][2] = csv_data['Low'][date]
        prepared_data_matrix[date][index][3] = csv_data['Close'][date]
        
        print(date)
""" Note to self:
This function and loop is working, What needs to be done is to start inserting the data in the appropriate place
either the data from the target_ending_date, or the csv_starting_date which would be the most recent.  I need to check the date, get the index of those either the target_starting_date or the csv_starting_date and insert it accordingly otherwise I will keep getting the error from the loop that calls the function with an invalid key."""


# In[145]:


import numpy as np

# Create a matrix of 4529x500x5, I only need 4 columns per stock, per day, for 4529 days of data or 18 years.
prepared_data_matrix = np.zeros((4529,500,4))


# In[147]:


import os.path   # To check whether a file exist

for index in range (len(sp500_components) - 1):
    
    stock_symbol = str(sp500_components['Symbol'][index])    # Get the stock symbol
    csv_file = ''.join([stock_symbol, '.csv'])    # Create the filename to read from the local directory 
    
    # Read files found in the local directory
    if os.path.isfile(csv_file):
        print(index, csv_file)
        add_to_database(csv_file, prepared_data_matrix, index)
    else:
        print("File not found!")


# In[71]:


# 2000-01-03 January 3, 2000 will be the starting date for the data.  Some stocks don't go back to that day.
string_date = '2000-01-03'


# In[72]:


csv_data = pd.read_csv('A.csv')


# In[73]:


i = 0
while str(csv_data['Date'][i]) != string_date:
    print(i, csv_data['Date'][i])
    i += 1


# In[78]:


i


# In[74]:


datetime(1990, 10, 10) < datetime(2000, 1, 1)


# In[75]:


string_date = string_date.split('-')


# In[76]:


csv_data_date =[int(float(string)) for string in string_date]

