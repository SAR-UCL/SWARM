import scipy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

#Loading and exporting
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data/'
file_name = r'IPD-Dayside-Cleaned.csv'
load_csv = path + file_name
load_csv = pd.read_csv(load_csv)
#print(load_csv)

def selectNScale(df):
    from sklearn.preprocessing import StandardScaler

    global features

    df = df.replace({'reg': {0: "equator", 1: "mid-lat", 2: "polar", 3:"auroral"}})
    df = df[(df.reg != 'polar') & (df.reg != 'auroral')] #Remove auroral and polar classes
    #df = df[(df.reg != 2) & (df.reg != 3)] 

    #df = df[(df.hemi != 'night')] #remove day or night

    #Select and scale the x data
    features = ['Ne','b_field_int','rod','Te']
    x_data = df[features]
    scaler = StandardScaler()
    scaler.fit(x_data) #compute mean for removal and std
    x_data = scaler.transform(x_data)

    #check for na values
    #print(df.isna().sum())

    #select and flatten y data
    labels = 'reg'
    y_data = df[[labels]]
    y_data = y_data[[labels]].to_numpy()
    y_data = np.concatenate(y_data).ravel().tolist()

    #Visualise pre-ML
    '''
    plt.figure(figsize=(5,3.5), dpi=90)
    plt.rcParams['font.size'] = '9.5' 
    #plt.title('Potential vs. Density\n')
    sns.scatterplot(data = df, x = 'rod', y='b_field_int', hue = 'reg', palette='Set2')
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()'''

    return x_data, y_data

x_data, y_data = selectNScale(load_csv)

def trainTestSplit(x_data, y_data):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state=0)

    #print(len(X_train))
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = trainTestSplit(x_data, y_data)

#print(x_data)

def kMeans():
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances_argmin

    '''
    #Check Elbow point for best number of clusters
    ks = range(1, 10)
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(X_train)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8,5))
    plt.style.use('bmh')
    plt.plot(ks, inertias, '-o')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.xticks(ks)
    plt.show()

    #Principle Component Analysis. Check variance and select most important features
    from sklearn.decomposition import PCA
    pca = PCA(random_state=0)
    pca.fit(X_train)
    features = range(pca.n_components_)
    print(features)
    print (pca.explained_variance_)

    plt.figure(figsize=(8,4))
    plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
    plt.xlabel('PCA feature')
    plt.ylabel('Variance')
    plt.xticks(features[:15])
    plt.show()'''
    

    #K means modelling
    k = 4
  
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train)
    centroids = kmeans.cluster_centers_
    palette = kmeans.predict(X_train)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=palette, s=10, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, alpha = 0.7)

    plt.show()

kMeans()


'''
def loadCSV():
    csv_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data/'
    file_name = r'IPD-Dayside-Cleaned.csv'
    load_csv = csv_path + file_name
    
    return pd.read_csv(load_csv)

#Splits data into a training (80%) and test set (20%)
def trainTestSplit(data):
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size = 0.2, random_state=42)

    return train_set[::], test_set[::]


def scaleData():

    plasma = loadCSV()
    train_set, test_set = trainTestSplit(plasma)

    from sklearn.preprocessing import StandardScaler

    df = train_set[['Te','Ne']]

    df_ref = train_set[['reg']].reset_index().drop(columns=['index'])

    scaler = StandardScaler()

    scaler.fit(df) #compute mean and std for scaling
    transform = pd.DataFrame(scaler.transform(df)) #perform scaling and centring 

    appended = transform.add(df_ref, fill_value=0)

    #return appended

    return appended

    #return transform

scaled_data = scaleData()

scaled_data = scaled_data.replace({'reg':{0:"equator",1:"mid-lat",2:"auroral",3:"polar"}})
scaled_data = scaled_data.rename(columns={0:"Te",1:"Ne"})
print(scaled_data)

#scaled_data = pd.DataFrame(scaleData())
#
#scaled_data.



plasma = loadCSV()
train_set, test_set = trainTestSplit(plasma)

df = train_set[['mlt','Te','Ti','Ne','lat','long','reg']].reset_index().drop(columns=['index'])
df = df.replace({'reg':{0:"equator",1:"mid-lat",2:"auroral",3:"polar"}})

#plt.figure(figsize=(5.5,3.5), dpi=90)
plt.rcParams['font.size'] = '10.5'
#sns.scatterplot(data = df, x = "Te", y = "Ne", hue = "reg", alpha = 0.6)

figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.5,3.5)) #3.5 for single, #5.5 for double
axs = axs.flatten()
plt.rcParams['font.size'] = '10.5'

sns.scatterplot(data = df, x = "Te", y = "Ne", hue = "reg", alpha = 0.6, ax = axs[0])
sns.scatterplot(data = scaled_data, x = "Te", y = "Ne", hue = "reg", alpha = 0.6, ax = axs[1], legend = False)

axs[0].set_title('Unscaled Data')
axs[1].set_title('Scaled Data')

#plt.yscale('log')

plt.tight_layout()
plt.show()


print('Train len:',len(train_set))
print('Test len:', len(test_set))

print(train_set)



#https://datatofish.com/k-means-clustering-python/

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import axes3d

df = train_set[['mlt','Te','Ti','Ne','lat','long','reg']].reset_index().drop(columns=['index'])
#df = train_set[['mlt','Ne']].reset_index().drop(columns=['index'])
#print(df)

k = 4
  
kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
predict = KMeans
centroids = kmeans.cluster_centers_
palette = kmeans.labels_.astype(float)
labels = np.array(kmeans.labels_)
#print(centroids)

df['labels'] = labels.tolist()
print(df)

#plt.scatter(df['Te'], df['Ne'], c= palette, s=25, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=25)
#plt.yscale('log')
#plt.show()


#Compares the clustering labels with the regional labels from SWARM. Not very insightful yet

figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.5,3.5), sharex=True, sharey=True) #3.5 for single, #5.5 for double
axs = axs.flatten()
plt.rcParams['font.size'] = '10.5'

df.plot(kind="scatter", x="Te", y="Ne", c= 'reg', alpha=0.4, ax = axs[0], cmap=plt.get_cmap("jet"), colorbar=False)
df.plot(kind="scatter", x="Te", y="Ne", c= 'labels', alpha=0.4, ax = axs[1], cmap=plt.get_cmap("jet"), colorbar=True)

axs[0].set_title('Region Based')
axs[1].set_title('Clustering')

plt.tight_layout()
#plt.yscale('log')
plt.show()'''
