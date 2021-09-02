#modules
import scipy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def loadCSV():
    csv_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data/'
    file_name = r'IPD-Dayside-Cleaned.csv'
    load_csv = csv_path + file_name
    return pd.read_csv(load_csv)

def createTestTrain(data, test_ratio):
    shuffle_index = np.random.permutation(len(data)) #permute = rearrange sequence / order of set
    #shuffle_index = np.random.seed(42)

    test_set_size = int(len(data) * test_ratio) #testing
    test_index = shuffle_index[:test_set_size]

    train_index = shuffle_index[test_set_size:] #training

    return data.iloc[train_index], data.iloc[test_index]

plasma = loadCSV()
train_set, test_set = createTestTrain(plasma, 0.2)
print('Train len (1):',len(train_set))
print('Test len:', len(test_set))

def trainTestSplit(data, size, random ):
    from sklearn.model_selection import train_test_split

    train_set, test_set = train_test_split(data, test_size = size, random_state=random )

    return train_set, test_set

train_set, test_set = trainTestSplit(plasma, 0.18, 42)
print('Train len (2):',len(train_set))
print('Test len:', len(test_set))