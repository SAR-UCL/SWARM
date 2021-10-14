#. Created by Sachin A. Reddy @ MSSL, UCL
# October 2021

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

    #Region filtering
    df = df.replace({'reg': {0: "equator", 1: "mid-lat", 2: "polar", 3:"auroral"}})
    df = df[(df.reg != 'polar') & (df.reg != 'auroral')] #Remove auroral and polar classes
    #df = df[(df.reg != 2) & (df.reg != 3)]
    
    #Fluctuations Index 
    #Calculated by working out rate of change
    #df = df[(df.IPIR != 1) & (df.IPIR != 5) & (df.IPIR != 6)] 

    #Noon or Midnight
    df = df.loc[df['mid-noon'] != 'other'] 
    print(df)

    #Select and scale the x data
    features = ['Ne','Ti','Te','pot']
    x_data = df[features]
    scaler = StandardScaler()
    scaler.fit(x_data) #compute mean for removal and std
    x_data = scaler.transform(x_data)

    #select and flatten y data
    labels = 'hemi'
    y_data = df[[labels]]
    y_data = y_data[[labels]].to_numpy()
    y_data = np.concatenate(y_data).ravel().tolist()

    '''
    #Visualise pre-ML
    plt.figure(figsize=(5,3.5), dpi=90)
    plt.rcParams['font.size'] = '9.5' 
    #plt.title('Potential vs. Density\n')
    sns.scatterplot(data = df, x = 'rod', y='F', hue = 'IPIR', palette='Set2')
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()'''

    return x_data, y_data

x_data, y_data = selectNScale(load_csv)

'''Split the data into a training and test set''' 
def trainTestSplit(x_data, y_data):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state=0) #test 0.2 = 20%

    #print(len(X_train))
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = trainTestSplit(x_data, y_data)

def randomForest():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=75, min_samples_leaf=2, bootstrap=True, random_state=0,
        class_weight='balanced')
    model = model.fit(X_train, y_train)

    model.n_features_

    return model

def svm():
    from sklearn.svm import SVC
    #create a model
    model = SVC(C=20, kernel='rbf', probability=True) 
    model.fit(X_train, y_train)

    return model

def gaussian():
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    kernel = 1 * RBF(1)
    gpc = GaussianProcessClassifier(kernel=kernel, max_iter_predict=50).fit(X_train, y_train)

    return gpc

def sgd():
    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier(loss="log", penalty="l2", max_iter=100)
    model.fit(X_train, y_train)

    return model

#Model
model_name = 'random-forest.pkl'
model_pathfile = path + model_name

def saveModel():

    from sklearn.model_selection import cross_val_score
    import pickle
    from sklearn import metrics

    #model = gaussian()
    model = randomForest()

    #Split set and cross-validate. Reduces risk of over-fitting.
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print("μ accuracy: %0.2f, σ: %0.2f" % (scores.mean(), scores.std()))

    #See (hyper)parameters. Useful for optimisation
    #print(model.get_params())

    with open(model_pathfile, 'wb') as file:
        pickle.dump(model, file)
    
    print('Model exported \n')
    
    #y_pred = model.predict(X_test) 
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

saveModel()

def pltSKL():

    from sklearn import metrics
    import scikitplot as skplt
    import pickle

    #Load the model
    model_pathfile
    with open(model_pathfile, 'rb') as file:
        model = pickle.load(file)
    
    y_pred = model.predict(X_test) #based on the model, predict classes (equator, mid-lat, etc) of test set 
    y_probas = model.predict_proba(X_test) #based on the model, predict probability of classes: equator 0.8%, mid-lat 0.65%

    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy = float("{:.2f}".format(accuracy)) * 100

    #print("Recall:",metrics.recall_score(y_test, y_pred))
    #print("Precision:",metrics.precision_score(y_test, y_pred))

    #print out the specific values and the prediction and predition %
    #Good for testing
    #for i in range(10):
    #    print("X:%s, Pred:%s Prob:%s" % (X_test[i], y_pred[i], y_probas[i]))
    
    #Plot the different metrics
    figs, axs = plt.subplots(ncols=2, nrows=2, figsize=(8.5,5.5), dpi=90) #3.5 for single, #5.5 for double
    axs = axs.flatten()

    #Decor
    #plt.rcParams['font.size'] = '9.5' 
    figs.suptitle(f'Determining Diurnality \n Classifier accuracy: {accuracy}%, Cadence: 30s')
    #plt.rcParams['font.size'] = '9.5'  
    #figs.subplots_adjust(top=0.88)

    #skplt.metrics.plot_roc(y_test, y_probas, ax=axs[1])
    skplt.estimators.plot_feature_importances(model, feature_names=features, ax=axs[1])
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax = axs[3])
    skplt.metrics.plot_precision_recall(y_test, y_probas, ax=axs[2])
    skplt.estimators.plot_learning_curve(model, X_train, y_train, ax=axs[0])
    

    #axs[0].legend(prop={'size': 9.5})
    #axs[1].legend(prop={'size': 9.5})
    #axs[1].get_legend().remove()

    
    plt.tight_layout()
    plt.show()

pltSKL()
