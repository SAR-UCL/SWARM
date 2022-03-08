'''
    This file uses a k-means classifier to assses the similarity / disimilarity 
    between the EPB data

    Created by Sachin A. Reddy

    Mar 2022.

'''

import scipy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#import seaborn as sns
#from matplotlib.colors import LogNorm
from datetime import date
import pickle


path = (r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions'
        '/SWARM/Non-Flight Data/Analysis/Mar-22/data/solar_max/ml_model/')

filename = path + 'ml-2015_with-std.csv'
load_hdf = pd.read_csv(filename)

print('loading data...')
#print(load_hdf)



def selectNScale(df, features, y_label):
    from sklearn.preprocessing import StandardScaler

    print('re-scaling data...')
    #Select and scale the x data
    #features = ['long','Ne','Ti','pot']
    x_data = df[features]
    scaler = StandardScaler()
    scaler.fit(x_data) #compute mean for removal and std
    x_data = scaler.transform(x_data)

    #select and flatten y data
    #labels = y_label
    y_data = df[[y_label]]
    y_data = y_data[[y_label]].to_numpy()
    y_data = np.concatenate(y_data).ravel().tolist()

    return x_data, y_data

y_label = 'sg_smooth'
features = ['pot','Ne']
x_data, y_data = selectNScale(load_hdf, features, y_label)

def train_test_split(x_data, y_data):
    from sklearn.model_selection import train_test_split

    print('splitting data...')

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, 
            test_size = 0.1, random_state=42)

    #print(len(X_train))
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data)

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

k = 3

kmeans = KMeans(n_clusters=k, random_state=0).fit(X_test)
centroids = kmeans.cluster_centers_
palette = kmeans.predict(X_test)

plt.scatter(X_test[:, 0], X_test[:, 1], c=palette, s=10, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, alpha = 0.7)

plt.show()

class RF_classifer():

    def featureEng(self, df):

        print('feature engineering...')

        #df = df[df['b_ind'] != -1] #remove non MLT data
        #df = df[df['lat'].between(-30,30)] #only include low lat
        #df = df[~df['mlt'].between(6, 18)] #include only nightside 
        #df = df[df['long'].between(10,180)] #remove SSA

        #dfc = df.groupby(['sg_smooth']).count()
        #major = dfc['date'].iloc[0]
        #minor = dfc['date'].iloc[1]
        #print('Major : Minor = ', major//minor,': 1')
        
        return df

    def selectNScale(self, df, features, y_label):
        from sklearn.preprocessing import StandardScaler

        print('re-scaling data...')
        #Select and scale the x data
        #features = ['long','Ne','Ti','pot']
        x_data = df[features]
        scaler = StandardScaler()
        scaler.fit(x_data) #compute mean for removal and std
        x_data = scaler.transform(x_data)

        #select and flatten y data
        #labels = y_label
        y_data = df[[y_label]]
        y_data = y_data[[y_label]].to_numpy()
        y_data = np.concatenate(y_data).ravel().tolist()

        return x_data, y_data

    def train_test_split(self, x_data, y_data):
        from sklearn.model_selection import train_test_split

        print('splitting data...')

        x_data, X_test, y_train, y_test = train_test_split(x_data, y_data, 
                test_size = 0.1, random_state=42)

        #print(len(X_train))
        return X_train, X_test, y_train, y_test

    def resample_class(self, X, y):
        from collections import Counter
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import NearMiss 

        print('re-sampling data...')

        #OVERSAMPLING
        #https://imbalanced-learn.org/dev/references/generated/
        #imblearn.over_sampling.SMOTE.html
        #sm = SMOTE(random_state = 42)
        #X_rs, y_rs = sm.fit_resample(X,y)
        
        #UNDERSAMPLING
        #https://imbalanced-learn.org/stable/references/generated/
        #imblearn.under_sampling.NearMiss.html#imblearn.under_sampling.NearMiss
        nm = NearMiss()
        X_rs, y_rs = nm.fit_resample(X, y)
    
        print('Orignal data shape%s' %Counter(y))
        print('Resampled data shape%s' %Counter(y_rs))
        
        return X_rs, y_rs

    def build_rf_model(self, X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier

        import pickle
        #from sklearn import metrics

        print('creating RF...')
        model = RandomForestClassifier(n_estimators=175, random_state=42,
        min_samples_leaf=3)
        #model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
        #        max_depth=3, random_state=42)

        model = model.fit(X_train, y_train)
        model.n_features_
        
        #return model

        with open(model_pathfile, 'wb') as file:
            pickle.dump(model, file)
        print('model exported\n')
        
        return model_pathfile

    def load_model(self):
        #print('loading model...')
        with open(model_pathfile, 'rb') as file:
            model = pickle.load(file)
        return model

    def model_info(self, feature_labs):

        from sklearn import metrics

        #model = self.build_rf_model(X_train, y_train)
        model = self.load_model()

        y_pred = model.predict(X_test) #based on the model, predict EPB or not EPB 
        y_probas = model.predict_proba(X_test) #based on the model, predict probability of classes: EPB 0.8%, not EPB 0.65% etc
        
        def format_int(value):
            return round(value, 2)

        accuracy = format_int(metrics.accuracy_score(y_test, y_pred))
        recall = format_int(metrics.recall_score(y_test, y_pred))
        precision = format_int(metrics.precision_score(y_test, y_pred))
        f1 = format_int(2 * (precision * recall) / (precision + recall))

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print ('F1:', f1)

        feature_imp = pd.Series(model.feature_importances_, 
                index = feature_labs).round(2).sort_values(ascending=False)
        print('Feature Importance:\n', feature_imp)

        #Feature importance using SHAP values
        #https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d

        return accuracy, precision, recall, f1

    def plot_rf(self, feature_labs, accuracy, precision, recall, f1):
        
        #from sklearn import metrics
        import scikitplot as skplt

        #accuracy, precision, recall, f1 = self.model_info()

        model = self.load_model()

        figs, axs = plt.subplots(ncols=2, nrows=2, figsize=(8.5,5.5), dpi=90) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        y_pred = model.predict(X_test)
        y_probas = model.predict_proba(X_test) 

        skplt.estimators.plot_feature_importances(model, feature_names=feature_labs, ax=axs[1])
        skplt.metrics.plot_roc(y_test, y_probas, ax=axs[2]) #For balanced data
        skplt.metrics.plot_precision_recall(y_test, y_probas, ax=axs[0]) #for imbalanced data
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax = axs[3])

        figs.suptitle(f'Random Forest Classifier (solar_max) \n'
                f'Accuracy: {accuracy} Precision: {precision} Recall: {recall} '
                f'F1: {f1}')

        plt.tight_layout()
        plt.show()

# rf = RF_classifer()
# feat_eng = rf.featureEng(load_hdf)

# y_label = 'sg_smooth'
# feature_labs = ['long','Ne','Ti','pot']
# x_data, y_data, = rf.selectNScale(feat_eng, feature_labs, y_label)
# X_train, X_test, y_train, y_test = rf.train_test_split(x_data, y_data)
# X_train, y_train = rf.resample_class(X_train, y_train)

# #save model
# model_name = 'rf_nm_2015_MSSL_with-std.pkl'
# model_pathfile = path + model_name
# model = rf.build_rf_model(X_train, y_train)

# #load model
# accuracy, precision, recall, f1 = rf.model_info(feature_labs) #model info
# rf_model = rf.plot_rf(feature_labs, accuracy, precision, recall, f1) #plot model

# #print(feat_eng)