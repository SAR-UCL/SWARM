'''
    Created by Sachin A. Reddy

    February 2022.
'''

import pandas as pd
from pathlib import Path
from datetime import date
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None) #or 10 or None

path = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/'
        'Missions/SWARM/Non-Flight Data/Analysis/Feb-22/data/solar_max/')

def open_all(filename):
    df = pd.read_csv(str(path) + '/' +filename)
    return df

filename = '2015-data-2022-02-15.csv'
df = open_all(filename)
print('Filtering Data...')
df = df[df['b_ind']!=-1]
#df = df[df['lat'].between(-30,30)]
df = df[df['date'] =='2015-02-14']


def ibi_mssl_epb_compare(df):
    
    df_new = df.groupby('p_num')
    df_arr = []
    for name, num_group in df_new:

        try:

            print('Classifying pass', name)

            def norm_data(df):
            
                #Normalise the data
                from sklearn.preprocessing import StandardScaler
                x_data = df[['Ne','pot']]
                scaler = StandardScaler()
                scaler.fit(x_data) #compute mean for removal and std
                x_data = scaler.transform(x_data)
                ne_scale = [a[0] for a in x_data]
                df['Ne_scale'] = ne_scale

                return df

            #num_group = norm_data(num_group)

            def savitzky_golay(df):
                from scipy.signal import savgol_filter
                df['Ne_savgol'] = savgol_filter(df['Ne'], window_length=23,
                    polyorder = 2) 
                df['Ne_resid'] = df['Ne'] - df['Ne_savgol']
                df.dropna()

                def sovgol_epb(x):
                    if x > 10000 or x < -10000: #non-norm
                    #if x > 0.05 or x < -0.05: #norm
                        return 1
                    else:
                        return 0

                df['sg_epb'] = df.apply(lambda x: sovgol_epb(x['Ne_resid']), axis=1)

                def smooth_savgol(x, y, z):
                    if x == 1:
                        return 1
                    elif y + z == -2:
                        return 1
                    else:
                        return 0

                df['u_c'] = df['sg_epb'] - df['sg_epb'].shift(1)
                df['l_c'] = df['sg_epb'] - df['sg_epb'].shift(-1)

                df['sg_smooth'] = df.apply(lambda x: smooth_savgol(x['sg_epb'], 
                        x['u_c'], x['l_c']), axis=1)

                df = df.drop(columns=['u_c','l_c','pot_std','Te_std','Ti_std'])

                return df

            num_group = savitzky_golay(num_group)
            
            def check_function(x,y):

                if x == 1 and y == 1:
                    return 'match'
                if x == 0 and y == 0:
                    return 'match'
                else:
                    return 'no match'
                
                '''
                if x == 1 and y == 1:
                    return 'true pos'
                elif x == 0 and y == 0:
                    return 'true neg'
                elif x == 1 and y == 0:
                    return 'false neg'
                elif x == 0 and y == 1:
                    return 'false pos'
                else:
                    return 0'''

            '''
            num_group['func_score'] = num_group.apply(lambda x: check_function(x['b_ind'], 
                    x['sg_smooth']), axis=1)

            scores = num_group.groupby('func_score', as_index=False).size()
            scores = scores.reset_index()

            m = scores.loc[scores['func_score']=='match', 'size'].values
            nm = scores.loc[scores['func_score']=='no match', 'size'].values
            

            fn = scores.loc[scores['func_score']=='false neg', 'size'].values
            fp = scores.loc[scores['func_score']=='false pos', 'size'].values
            tn = scores.loc[scores['func_score']=='true neg', 'size'].values
            tp = scores.loc[scores['func_score']=='true pos', 'size'].values

            #date = num_group['date'].iloc[0]
            p_num = num_group['p_num'].iloc[0]

            
            precision = scores.iloc[3] / (scores.iloc[3] + scores.iloc[1])
            recall = scores.iloc[3] / (scores.iloc[3] + scores.iloc[0])
            f1 = 2*((precision*recall)/(precision+recall))
            precision = "{:.2f}".format(precision) 
            recall = "{:.2f}".format(recall) 
            f1 = "{:.2f}".format(f1)

            #print('Scores',scores)
            #print ('Precision:', precision)
            #print ('Recall:', recall)
            #print('F1:',f1)

            return precision, recall, f1
            
            df = pd.DataFrame({'p_num':p_num,'match':m, 'no-match':nm})'''

            num_group = num_group.groupby(['date','p_num'], as_index=False)['sg_smooth'].sum()
        
            def count_epb(x):
                if x > 1:
                    return 1
                else:
                    return 0

            num_group['epb_num'] = num_group['sg_smooth'].apply(count_epb)
            num_group = num_group[num_group['sg_smooth'] > 50].reset_index().drop(columns=['index'])
            #df = df.sort_values(by=['date','p_num'], ascending=[True,True])


            df_arr.append(num_group)
            df = pd.concat(df_arr)
        
        except:
            print('ERROR', name)
            pass
    
    return df

compare = ibi_mssl_epb_compare(df)
compare = compare.reset_index().drop(columns=['index'])
print (compare)