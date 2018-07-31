
# coding: utf-8

# In[ ]:


import pandas as pd
# Load the file that would be used to obtain the components of the S&P500 index
sp500 = pd.read_csv('sp500.csv')


# In[ ]:


# Open the file containing the api key
api = open('quandl_api_key', 'r')
if api.mode == 'r':
    api_key = api.read()
    
# Close the api file
api.close()


# In[ ]:


import quandl
quandl.ApiConfig.api_key = api_key.strip()


# In[ ]:


# Test the connection to the Quandl API
test = quandl.get('WIKI/IBM')
test.head()


# In[ ]:


trouble_stock = []    # List to keep track of the unobtained data.
for x in range (len(sp500) - 1):    # Minus 1 because the last line of the sp500 file refers to the source
        
    try:
        stock_symbol = str(sp500['Symbol'][x])    # Get the stock symbol
        stock = ''.join(['WIKI/', stock_symbol])       # String required for quandl api ex.  WIKI/AAPL
        file_name = ''.join([stock_symbol, '.csv'])    # Create file name for the csv file
        print(x, file_name)
        
        stock_data = quandl.get(stock)    # Get the data from Quandl
        stock_data.to_csv(file_name)    # Save the data to disk
        
    except Exception:
        trouble_stock.append(x)    # Keep track of stock_symbol causing issues with a request to quandl
        print(x)
        pass
        
        


# In[ ]:


''' List the stocks that were not acquired from Quandl,
BF.B, BRK.B, EVRG, JEF. Some data is not free and some sites use a slight variation of the stock symbol
I was reading the symbols from a csv I obtained from barchart. I manually downloaded the missing data from Yahoo Finance.''' 


# In[ ]:


'''Show stock symbol and name of the data not obtained from Quandl.  These represent 1 percent of all the data
At this point I decided to not include them in the data, although I wonder how this will affect the model. '''
for x in trouble_stock:
    print(sp500['Symbol'][x], sp500['Name'][x])

