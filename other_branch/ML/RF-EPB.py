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
import datetime

path = (r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions'
        '/SWARM/Non-Flight Data/Analysis/Mar-22/data/solar_max/ml_model/')

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

    def train_test_split(self, x_data, y_data, t_size):
        from sklearn.model_selection import train_test_split

        print('splitting data...')

        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, 
                test_size = t_size, random_state=42)

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
        sm = SMOTE(random_state = 42)
        X_rs, y_rs = sm.fit_resample(X,y)
        
        #UNDERSAMPLING
        #https://imbalanced-learn.org/stable/references/generated/
        #imblearn.under_sampling.NearMiss.html#imblearn.under_sampling.NearMiss
        #nm = NearMiss()
        #X_rs, y_rs = nm.fit_resample(X, y)
    
        #print('Orignal data shape%s' %Counter(y))
        #print('Resampled data shape%s' %Counter(y_rs))
        
        return X_rs, y_rs

    def rf_grid_search(self,X_train,y_train):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        model = RandomForestClassifier(random_state=42)

        #create a dictionary of all values we want to test

        #Test 1

        '''
        param_grid = { 
                'n_estimators': [100, 200, 500],
                'min_samples_leaf':[1,2,3],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [4,5,6,7,8],
                'criterion' :['gini', 'entropy']
        } '''

        #Test 2
        param_grid = { 
                #'n_estimator':[200],
                'max_features': ['sqrt', 'log2'],
                'min_samples_leaf':[1,2],
                #'max_depth':[8],
                #'criterion' :['gini']
        }  
        
        pre_time = datetime.datetime.now()
        print('Starting GridSearchCV at', pre_time)
        CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
        CV_rfc.fit(X_train, y_train)
        post_time = datetime.datetime.now()
        print('Finished GridSearchCV at', post_time)
        diff_time = post_time - pre_time
        print('Processing time:', diff_time)
        
        print(CV_rfc.best_params_)

    def build_rf_model(self, X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier

        import pickle
        #from sklearn import metrics

        pre_time = datetime.datetime.now()
        print('Starting RFC at', pre_time)

        model = RandomForestClassifier(n_estimators=200, random_state=42,
        min_samples_leaf=1, n_jobs=-1, max_features = "sqrt")
        #model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
        #        max_depth=3, random_state=42)

        model = model.fit(X_train, y_train)
        post_time = datetime.datetime.now()
        print('Ending RFC at', post_time)
        diff_time = post_time - pre_time
        print('Processing time:', diff_time)
        
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
        model = self.load_model()
        #print(model)

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


        #df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
        #https://machinelearningmastery.com/make-predictions-scikit-learn/
        
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

        figs.suptitle(f'Random Forest Classifier \n'
                f'Accuracy: {accuracy} Precision: {precision} Recall: {recall} '
                f'F1: {f1}')

        plt.tight_layout()
        plt.show()

    def plot_dt(self,feature_labs):

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        
        # Fit the classifier with default hyper-parameters
        clf = DecisionTreeClassifier(random_state=1234)
        model = clf.fit(X, y)   

        text_representation = tree.export_text(clf)
        print(text_representation)

        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)

        plt.tight_layout()
        plt.show()

filename = path + 'SG-filtered_14-15.csv'
load_hdf = pd.read_csv(filename)
load_hdf = load_hdf[load_hdf['date'] == '2015-03-01']

print('loading data...')
print(load_hdf)

rf = RF_classifer()
feat_eng = rf.featureEng(load_hdf)

y_label = 'sg_smooth'
feature_labs = ['lat','long','mlt']
#feature_labs = ['long','pot','Ne','Ti']
x_data, y_data, = rf.selectNScale(feat_eng, feature_labs, y_label)
X_train, X_test, y_train, y_test = rf.train_test_split(x_data, y_data, t_size = 0.001)
X_train, y_train = rf.resample_class(X_train, y_train)

#save model
model_name = 'rf_2014-2015_llm.pkl'
model_pathfile = path + 'outputs/' + model_name

run_model = "no"
if run_model == "yes":
    model = rf.build_rf_model(X_train, y_train)
    #rf_grid = rf.rf_grid_search(X_train, y_train)
else:
    pass

#load model
accuracy, precision, recall, f1 = rf.model_info(feature_labs) #model info
rf_plot = rf.plot_rf(feature_labs, accuracy, precision, recall, f1) #plot model
#dt_plot = rf.plot_dt(feature_labs)


