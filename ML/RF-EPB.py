'''
    This file uses a Random Forest classifer to train a model to predict
    EPBs    
    Created by Sachin A. Reddy

    Dec 2021.

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
        '/SWARM/Non-Flight Data/Analysis/Jan-22/data/decadal/')

today =  str(date.today())
file_name = 'wrangled-EPB-'+ today +'.h5'
#load_hdf = path + file_name
#load_hdf = pd.read_hdf(load_hdf)

path = (r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions'
        '/SWARM/Non-Flight Data/Analysis/Feb-22/data/solar_max/classified/')

filename = path + 'EPB-sg-classified-filter_2015.csv'
load_hdf = pd.read_csv(filename)

#print(load_hdf)

class RF_classifer():

    def featureEng(self, df):

        print('feature engineering...')

        #df = df[df['b_ind'] != -1] #remove non MLT data
        #df = df[df['lat'].between(-30,30)] #only include low lat
        #df = df[~df['mlt'].between(6, 18)] #include only nightside 
        #df = df[df['long'].between(10,180)] #remove SSA

        dfc = df.groupby(['sg_smooth']).count()
        major = dfc['date'].iloc[0]
        minor = dfc['date'].iloc[1]
        
        #print('Major : Minor = ', major//minor,': 1')
        return df

    def selectNScale(self, df, features):
        from sklearn.preprocessing import StandardScaler

        print('re-scaling data...')
        #Select and scale the x data
        #features = ['long','Ne','Ti','pot']
        x_data = df[features]
        scaler = StandardScaler()
        scaler.fit(x_data) #compute mean for removal and std
        x_data = scaler.transform(x_data)

        #select and flatten y data
        labels = 'sg_smooth'
        y_data = df[[labels]]
        y_data = y_data[[labels]].to_numpy()
        y_data = np.concatenate(y_data).ravel().tolist()

        return x_data, y_data

    def train_test_split(self, x_data, y_data):
        from sklearn.model_selection import train_test_split

        print('splitting data...')

        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, 
                test_size = 0.1, random_state=42) #test 0.2 = 20%

        #print(len(X_train))
        return X_train, X_test, y_train, y_test

    def resample_class(self, X, y):
        from collections import Counter
        from imblearn.datasets import make_imbalance
        from imblearn.over_sampling import SMOTE

        print('re-sampling data...')

        #https://imbalanced-learn.org/dev/references/generated/
        #imblearn.over_sampling.SMOTE.html
        sm = SMOTE(random_state = 42)
        X_rs, y_rs = sm.fit_resample(X,y)

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

    def plot_rf(self, feature_labs):
        
        #from sklearn import metrics
        import scikitplot as skplt

        model = self.load_model()

        figs, axs = plt.subplots(ncols=2, nrows=2, figsize=(8.5,5.5), dpi=90) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        y_pred = model.predict(X_test)
        y_probas = model.predict_proba(X_test) 

        skplt.estimators.plot_feature_importances(model, feature_names=feature_labs, ax=axs[1])
        skplt.metrics.plot_roc(y_test, y_probas, ax=axs[2]) #For balanced data
        skplt.metrics.plot_precision_recall(y_test, y_probas, ax=axs[0]) #for imbalanced data
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax = axs[3])

        plt.tight_layout()
        plt.show()

rf = RF_classifer()
feat_eng = rf.featureEng(load_hdf)

feature_labs = ['long','Ne','Ti','pot']
x_data, y_data, = rf.selectNScale(feat_eng, feature_labs)
X_train, X_test, y_train, y_test = rf.train_test_split(x_data, y_data)
X_train, y_train = rf.resample_class(X_train, y_train)

#save model
model_name = today + '_rf_solar_max.pkl'
model_pathfile = path + 'ML-models/' + model_name
model = rf.build_rf_model(X_train, y_train)

#load model
rf_model = rf.model_info(feature_labs) #model info
rf_model = rf.plot_rf(feature_labs) #plot model

#print(feat_eng)