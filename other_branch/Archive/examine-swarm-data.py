#modules
import scipy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import geopandas

#This file is used for examining the SWARM following extraction from #CDF
#08-Sep-21 - Plot Te, Ti, TiM and Ne against long and lat. 
    #This is purely for visual inspection and has NO ML elements

def loadCSV():
    csv_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data/'
    file_name = r'IPD-Dayside-Cleaned.csv'
    load_csv = csv_path + file_name
    
    return pd.read_csv(load_csv)

#Splits data into a training (80%) and test set (20%)
def trainTestSplit(data):
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size = 0.2, random_state=42)

    return train_set, test_set

#train

#print(training_only.head())

#ti_data = ti_data.dropna()

def trainModel(ti_data, ti_labels):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    lin_reg = LinearRegression()
    lin_reg.fit(ti_data, ti_labels)
    ti_predictions = lin_reg.predict(ti_data)

    #return ti_data, ti_predictions
    
    lin_mse = mean_squared_error(ti_labels, ti_predictions)
    lin_rmse = np.sqrt(lin_mse)
    return lin_rmse

#ti_real, ti_fit = trainModel(ti_data[:20], ti_labels[:20])


#ti_real, ti_fit = trainModel(ti_data[:20], ti_labels[:20])

#print (ti_data[:20])
#print (ti_labels[:20])

#linear_rmse = trainModel(ti_data[:20], ti_labels[:20])
#print(linear_rmse)

'''
sns.lineplot(data = ti_real, palette='plasma')
sns.lineplot(data = ti_fit, palette ='mako', dashes = True)

plt.show()'''


#print("Predictions:", lin_reg)

#plasma = loadCSV()
#print(plasma)

def PlotLatLong():

    #load csv and split functions
    plasma = loadCSV()
    train_set, test_set = trainTestSplit(plasma)

    #print('Train len:',len(train_set))
    #print('Test len:', len(test_set))

    training_only = train_set
    #print(training_only.head())

    #Compute Correlation Matrix
    corr_matrix = training_only.corr()
    #corr_matrix = corr_matrix["Ne"].sort_values(ascending=False)
    #print(corr_matrix)

    #ax = sns.heatmap(corr_matrix, fmt = '.2f', annot= True)
    
    #sns.kdeplot(data=training_only['Ne'], x=training_only["long"], y=training_only["lat"],fill=True, thresh=0, levels=10, cmap="mako", shade=True, 
    #cbar=True, shade_lowest=False)

    
    figs, axs = plt.subplots(figsize=(9.5,6.5), sharex=False, sharey=True) #3.5 for single, #5.5 for double
    axs = axs.flatten()

    training_only.plot(kind="scatter", x="long", y="lat", alpha=0.4, ax = axs[0],
    c="Ti", cmap=plt.get_cmap("jet"), colorbar=True)

    training_only.plot(kind="scatter", x="long", y="lat", alpha=0.4, ax = axs[1],
    c="b_field_int", cmap=plt.get_cmap("jet"), colorbar=True)
    
    den = r'm$^{-3}$'
    training_only.plot(kind="scatter", x="long", y="lat", alpha=0.4, ax = axs[2],
    c="Te", cmap=plt.get_cmap("jet"), colorbar=True)

    rod = r'm$^{-3}$/s'
    training_only.plot(kind="scatter", x="long", y="lat", alpha=0.4, ax = axs[3],
    c="Ne", cmap=plt.get_cmap("jet"), colorbar=True)
    
    plt.legend()
    plt.show()

PlotLatLong()
