
# coding: utf-8

# In[1]:


import pandas as pd

# Load the s&p500 csv file to read the csv names
sp500_components = pd.read_csv('sp500.csv')


# In[2]:


import numpy as np

# Create a matrix of 4482 by ~504 by 4, I only need 4 columns per stock, per day, for 4482 days of data or 18 years.
days = 4482   
components = len(sp500_components)
columns = 4    # I'm only using High, Low, Open and Close columns 

prepared_data_matrix = np.zeros((days,components,columns))


# In[3]:


# Data to create a 3d graph
length = days * components
x = []
y = []
z = []


# In[4]:


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
        x.append(date)
        y.append(index)
        z.append(csv_data['Close'][counter])
        date -= 1
        counter -= 1
        
    print('Finished adding', csv_file, 'data to the prepared_data_matrix!')


# In[5]:


import os.path   # To check whether a file exist

for index in range (len(sp500_components) - 1):
    
    stock_symbol = str(sp500_components['Symbol'][index])    # Get the stock symbol
    csv_file = ''.join([stock_symbol, '.csv'])    # Create the filename to read from the local directory 
    
    # Read files found in the local directory
    if os.path.isfile(csv_file):
        #print(index, csv_file)
        add_to_data_matrix(csv_file, index)
    else:
        print("File not found!")
        
print('Done creating prepared_data_matrix!')


# In[6]:


# Load the SPY also known as SPYDR
spydr_data = pd.read_csv('SPY.csv')
y_hat = spydr_data['Close'].values


# In[7]:


# Use PCA to reduce dimensionality, ~500 points per day to one
from sklearn.decomposition import PCA

def reduce_dimensions(daily_data):
    X = daily_data
    pca = PCA(n_components = 1)
    pca.fit(X)
    #print('Variance ration', pca.explained_variance_ratio_)
    #print('PCA singular values', pca.singular_values_)
    return pca.singular_values_


# In[9]:


# Creating the reduced dimension matrix
reduced_dimensionality_matrix = np.zeros(days)

for day in range(days -1):
    reduced_dimensionality_matrix[day] = (reduce_dimensions(prepared_data_matrix[day]))

print(reduced_dimensionality_matrix)


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


# This takes over 10 minutes in my laptop
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure(figsize=(30, 15), dpi= 100, facecolor='lightblue', edgecolor='m')
ax = fig.gca(projection='3d')
ax.plot_trisurf(x,y,z, cmap='RdBu',linewidth=0,antialiased=True)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
    


# In[12]:


fig=plt.figure(figsize=(25, 15), dpi= 100, facecolor='lightblue', edgecolor='m')
x = np.arange(days -1)
plt.plot(reduced_dimensionality_matrix[:], 'b--', y_hat, 'm^')
plt.show()


# In[13]:


print(reduced_dimensionality_matrix[:])

