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
load_csv = load_csv[::] #skips every 60 is a cadency of 1 minute, so 1440 per day





def selectNScale(df):
    from sklearn.preprocessing import StandardScaler

    df = df.replace({'reg': {0: "equator", 1: "mid-lat", 2: "polar", 3:"auroral"}})
    df = df[(df.reg != 'polar') & (df.reg != 'auroral')] #Remove auroral and polar classes
    #df = df[(df.reg != 2) & (df.reg != 3)] 

    #df = df[(df.hemi != 'night')] #remove day or night

    #Select and scale the x data
    x_data = df[['Te','Ti','Ne','pot']]
    scaler = StandardScaler()
    scaler.fit(x_data) #compute mean for removal and std
    x_data = scaler.transform(x_data)

    #select and flatten y data
    y_data = df[['hemi']]
    y_data = y_data[['hemi']].to_numpy()
    y_data = np.concatenate(y_data).ravel().tolist()

    #Visualise pre-ML
    plt.figure(figsize=(7,3.5), dpi=90)
    sns.scatterplot(data = df, x = 'pot', y='Ne', hue = 'reg', style = 'hemi', palette='Set2')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    return x_data, y_data

x_data, y_data = selectNScale(load_csv)

def trainTestSplit(x_data, y_data):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state=0)

    return X_train, X_test, y_train, y_test

#X_train, X_test, y_train, y_test = trainTestSplit(x_data, y_data)

def randomForest():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=75, min_samples_leaf=2, max_features="log2", bootstrap=True, random_state=0,
        class_weight='balanced', criterion='gini')
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

    scores = cross_val_score(model, X_train, y_train, cv=5)
    print("μ accuracy: %0.2f, σ: %0.2f" % (scores.mean(), scores.std()))

    print(model.get_params())

    with open(model_pathfile, 'wb') as file:
        pickle.dump(model, file)
    
    print('Model exported \n')
    
    #y_pred = model.predict(X_test) 
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#saveModel()

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

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print("Recall:",metrics.recall_score(y_test, y_pred))
    #print("Precision:",metrics.precision_score(y_test, y_pred))

    #print out the specific values and the prediction and predition %
    #Good for testing
    #for i in range(10):
    #    print("X:%s, Pred:%s Prob:%s" % (X_test[i], y_pred[i], y_probas[i]))
    
    #Plot the different metrics
    figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(8.5,3.5)) #3.5 for single, #5.5 for double
    axs = axs.flatten()

    #skplt.metrics.plot_precision_recall(y_test, y_probas, ax=axs[0])
    #skplt.metrics.plot_roc(y_test, y_probas, ax=axs[1])
    skplt.estimators.plot_feature_importances(model, feature_names=['Te','Ti','Ne','pot'], ax=axs[0])
    #skplt.estimators.plot_learning_curve(model, X_train, y_train, ax=axs[2])
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax = axs[1])

    axs[0].legend(prop={'size': 9.5})
    #axs[1].legend(prop={'size': 9.5})
    #axs[3].get_legend().remove()


    plt.tight_layout()
    plt.show()

#pltSKL()
