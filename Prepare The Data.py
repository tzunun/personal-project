
# coding: utf-8

# In[13]:


import pandas as pd

# Load the s&p500 csv file to read the csv names
sp500_components = pd.read_csv('sp500.csv')


# In[15]:


import os.path   # To check whether a file exist

# Read files found in the local directory
if os.path.isfile('A.csv'):
    a_csv = pd.read_csv('A.csv')
    a_csv.head()
else:
    print("File not found!")


for i in range (len(sp500_components) - 1):
    
    stock_symbol = str(sp500_components['Symbol'][i])    # Get the stock symbol
    filename = ''.join([stock_symbol, '.csv'])    # Create the filename to read from the local directory 
    
    # Read files found in the local directory
    if os.path.isfile(filename):
        #csv_data = pd.read_csv(filename)
        print(filename)
    else:
        print("File not found!")

