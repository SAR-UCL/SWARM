#. Created by Sachin A. Reddy @ MSSL, UCL
# October 2021

import scipy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from datetime import date

#Loading and exporting
#path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/April-16/'
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/decadal/'

#csv
#file_name = r'may18-cleaned.csv'
#load_csv = path + file_name
#load_csv = pd.read_csv(load_csv)

#HDF
#file_name = 'joined-data-2022-01-06.h5'
today =  str(date.today())
file_name = 'wrangled-EPB-'+ today +'.h5'
load_hdf = path + file_name
load_hdf = pd.read_hdf(load_hdf)
#print(load_hdf)

def featureEng(df):

    #Remove non-MLT and dayside data
    df = df[df['b_ind'] != -1]

    #Remove non-low lat. Useful is day/night required
    def lowLat(df):
        df = df[df['lat'].between(-30,30)]
        return df
    df = lowLat(df)

    #Create daynight class. Useful if all latitudes required
    def dayNight(df):
        df = df[~df['mlt'].between(6,18)]
        return df
    df = dayNight(df)

    #Remove SSA 
    #Heirtzler, J. R. (2002). The future of the South Atlantic anomaly and implications for radiation damage in space. Journal of Atmospheric and Solar-Terrestrial Physics, 64(16), 1701-1708.
    def removeSSA(df):
        df = df[df['long'].between(10,180)]
        return df
    df = removeSSA(df)

    '''
        def potentialCalc(x):
            if 6 <= x <= 18:
                return -1.5
            else:
                return -2.5
    '''

    #Calcualte Major: Minor ratio
    dfc = df.groupby(['epb']).count()
    #print('counts',dfc)
    major = dfc['date'].iloc[0]
    minor = dfc['date'].iloc[1]
    print('Major : Minor = ', major//minor,': 1')

    #df['pot_s'] = df['mlt'].apply(potentialCalc)

    #Fluctuations Index 
    #Calculated by working out rate of change
    #df = df[(df.IPIR != 1) & (df.IPIR != 5) & (df.IPIR != 6)] 

    #Noon or Midnight
    #df = df.loc[df['mid-noon'] != 'other'] 
    
    #For IPD Indexing
    #df = df.loc[df['hemi'] == 'day']
    #df = df[df['lat'].between(-45,45)]
    #df = df.loc[df['bubble'] == -1] #1 Confirmed Bubble, 0 unconfirmed bubble, -1 unanalyzable bubble

    return df

engFeats = featureEng(load_hdf)
#print(engFeats)

def selectNScale(df):
    from sklearn.preprocessing import StandardScaler

    global features

    #Select and scale the x data
    features = ['long','Ne','Ti','pot']
    #features = ['Ne','Ti','Te','F','vfm_x','vfm_y','vfm_z']
    x_data = df[features]
    scaler = StandardScaler()
    scaler.fit(x_data) #compute mean for removal and std
    x_data = scaler.transform(x_data)

    #select and flatten y data
    labels = 'epb'
    y_data = df[[labels]]
    y_data = y_data[[labels]].to_numpy()
    y_data = np.concatenate(y_data).ravel().tolist()

    #print(y_data)

    '''
        #Visualise pre-ML
        plt.figure(figsize=(5,3.5), dpi=90)
        plt.rcParams['font.size'] = '9.5' 
        #plt.title('Potential vs. Density\n')
        sns.scatterplot(data = df, x = 'rod', y='F', hue = 'IPIR', palette='Set2')
        #plt.yscale('log')
        plt.tight_layout()
        plt.show()
    '''

    return x_data, y_data

x_data, y_data = selectNScale(engFeats)

'''Split the data into a training and test set''' 
def trainTestSplit(x_data, y_data):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.1, random_state=42) #test 0.2 = 20%

    #print(len(X_train))
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = trainTestSplit(x_data, y_data)

def resample(X, y):
    from collections import Counter
    from imblearn.datasets import make_imbalance
    from imblearn.over_sampling import SMOTE

    #https://imbalanced-learn.org/dev/references/generated/
    #imblearn.over_sampling.SMOTE.html
    sm = SMOTE(random_state = 42)
    X_rs, y_rs = sm.fit_resample(X,y)

    #print('Orignal data shape%s' %Counter(y))
    #print('Resampled data shape%s' %Counter(y_rs))
    
    return X_rs, y_rs

#X_smote_train, y_smote_train = resample(X_train, y_train)
X_train, y_train = resample(X_train, y_train)

def randomForest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    model = RandomForestClassifier(n_estimators=175, random_state=42,
    min_samples_leaf=2)
    #model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
    #        max_depth=3, random_state=42)

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
model_name = today + '_rf_conservative.pkl'
model_pathfile = path + model_name

def saveModel():

    from sklearn.model_selection import cross_val_score
    import pickle
    from sklearn import metrics

    try:
        print('Creating ML model...')
        #model = gaussian()
        model = randomForest()
        #model = sgd()

        #Split set and cross-validate. Reduces risk of over-fitting.
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print("μ accuracy: %0.2f, σ: %0.2f" % (scores.mean(), scores.std()))
        #print("xv scores: %f" % scores)

        #See (hyper)parameters. Useful for optimisation
        #print(model.get_params())

        with open(model_pathfile, 'wb') as file:
            pickle.dump(model, file)
    
    except RuntimeError:
        raise Exception('Problems with model')

    
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
    
    y_pred = model.predict(X_test) #based on the model, predict EPB or not EPB 
    y_probas = model.predict_proba(X_test) #based on the model, predict probability of classes: EPB 0.8%, not EPB 0.65% etc

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("Accuracy:", accuracy )
    print("Precision:", precision)
    print("Recall:", recall)
    print ('F1:', f1)

    #print out the specific values and the prediction and predition %
    #Good for testing
    #for i in range(10):
    #    print("X:%s, Pred:%s Prob:%s" % (X_test[i], y_pred[i], y_probas[i]))

    #skplt.estimators.plot_feature_importances(model, feature_names=features)
    #skplt.metrics.plot_confusion_matrix(y_test, y_pred)
    #skplt.metrics.plot_precision_recall(y_test, y_probas) #appropirate for imbalanced data
    #skplt.metrics.plot_roc(y_test, y_probas) #approporate for balanced data
    #skplt.estimators.plot_learning_curve(model, X_train, y_train)
    
    
    #Plot the different metrics
    figs, axs = plt.subplots(ncols=2, nrows=2, figsize=(8.5,5.5), dpi=90) #3.5 for single, #5.5 for double
    axs = axs.flatten()

    #Decor
    #plt.rcParams['font.size'] = '9.5' 
    #figs.suptitle(f'Determining Diurnality \n Classifier accuracy: {accuracy}%, Cadence: 30s')
    #plt.rcParams['font.size'] = '9.5'  
    #figs.subplots_adjust(top=0.88)

    #skplt.estimators.plot_learning_curve(model, X_train, y_train, ax=axs[0]) #very slow
    skplt.estimators.plot_feature_importances(model, feature_names=features, ax=axs[1])
    skplt.metrics.plot_roc(y_test, y_probas, ax=axs[2]) #For balanced data
    #skplt.metrics.plot_precision_recall(y_test, y_probas, ax=axs[2]) #for imbalanced data
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax = axs[3])


    #axs[0].legend(prop={'size': 9.5})
    #axs[1].legend(prop={'size': 9.5})
    #axs[1].get_legend().remove()
    
    
    plt.tight_layout()
    plt.show()

#pltSKL()

