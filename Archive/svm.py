import scipy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

def loadCSV():
    csv_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data/'
    file_name = r'IPD-Dayside-Cleaned.csv'
    load_csv = csv_path + file_name
    
    return pd.read_csv(load_csv)

#Splits data into a training (80%) and test set (20%)
def trainTestSplit(data):
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size = 0.2, random_state=42)

    #Remove auroral and polar classes. They smaller by comparison
    def removeClasses(df):
        df = df[(df.reg != 2) & (df.reg != 3)]
        #df = df
        return df 

    train_set = removeClasses(train_set)
    test_set = removeClasses(test_set)

    return train_set[::1000], test_set[::1000]



#load data and split into train & test
plasma_data = loadCSV()
train_set, test_set = trainTestSplit(plasma_data)

print(plasma_data)


def scaleTrainData():

    from sklearn.preprocessing import StandardScaler

    df = train_set[['Ti','Ne','Te']] #Isolate the data to be scaled
    df_ref = train_set[['reg']].reset_index().drop(columns=['index']) #regions don't need scaling
    
    scaler = StandardScaler()
    scaler.fit(df) #compute mean for removal and std
    transform = pd.DataFrame(scaler.transform(df)) #perform scaling and centring 

    appended_df = transform.add(df_ref, fill_value=0)
    appended_df = appended_df.rename(columns={0:"Ti",1:"Ne",2:"Te"})
    #appended_df = appended_df.replace({'reg':{0:"equator",1:"mid-lat"}})

    return appended_df

scaled_train = scaleTrainData()

def scaleTestData():

    from sklearn.preprocessing import StandardScaler

    df = test_set[['Ti','Ne','Te']] #Isolate the data to be scaled
    df_ref = test_set[['reg']].reset_index().drop(columns=['index']) #regions don't need scaling
    
    scaler = StandardScaler()
    scaler.fit(df) #compute mean for removal and std
    transform = pd.DataFrame(scaler.transform(df)) #perform scaling and centring 

    appended_df = transform.add(df_ref, fill_value=0)
    appended_df = appended_df.rename(columns={0:"Ti",1:"Ne",2:"Te"})
    #appended_df = appended_df.replace({'reg':{0:"equator",1:"mid-lat"}})

    return appended_df

scaled_test = scaleTestData()


#sns.scatterplot(data = scaled, x = "Ti", y = "Ne", hue = "reg", alpha = 0.6)

def gaussianKernel():
    from sklearn import datasets
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.metrics import plot_confusion_matrix

    #print(scaled)

    #Take data from df to numpy array
    X = scaled_train[["Ti", "Ne"]].to_numpy()
    y = scaled_train[["reg"]].to_numpy()
    y = np.concatenate(y).ravel().tolist()

    kernel = 1.0 * RBF([1.0])
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)

    score = gpc.score(X, y)
    print(score)

    #figure.plt(figsize=(5, 3.5))
    plot_confusion_matrix(gpc, X, y)
    plt.show()


gaussian = gaussianKernel()

def supportVectorMachine():
    from sklearn.svm import SVC
    from sklearn.metrics import plot_confusion_matrix
    from sklearn import metrics
    import scikitplot as skplt
    

    X_train = scaled_train[["Ti","Ne","Te"]].to_numpy()

    y_train = scaled_train[["reg"]].to_numpy()
    y_train = np.concatenate(y_train).ravel().tolist()

    clf = SVC(C=8, kernel='rbf', probability=True)
    clf.fit(X_train, y_train)

    
    X_test = scaled_test[["Ti","Ne","Te"]].to_numpy()
    y_test = scaled_test[["reg"]].to_numpy()
    y_test = np.concatenate(y_test).ravel().tolist()
    
    #predict
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred, average = 'micro'))
    print("Recall:",metrics.recall_score(y_test, y_pred, average = 'micro')) # Model Recall: what percentage of positive tuples are labelled as such?


    #Precision-Recall Curve & ROC plot
    y_probas = clf.predict_proba(X_test)

    figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.5,3.5), sharex=True, sharey=True) #3.5 for single, #5.5 for double
    axs = axs.flatten()

    skplt.metrics.plot_precision_recall(y_test, y_probas, ax=axs[0])
    skplt.metrics.plot_roc(y_test, y_probas, ax=axs[1])


    axs[0].legend(prop={'size': 9})
    axs[1].legend(prop={'size': 9})

    plt.tight_layout()
    plt.show()

    #print(predict)
    
    #plot_confusion_matrix(clf, X_train, y_train, normalize='true')
    #plt.title(f'Confusion Matrix' )
    

    #skplt.metrics.plot_roc(X_train, y_train, title="Roc Curve for XBG Classifier", figsize=(10, 8))

    #plt.show()


#supportVectorMachine()

#plt.show()
