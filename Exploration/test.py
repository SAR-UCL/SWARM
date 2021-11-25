import pandas as pd
import numpy as np


df = pd.DataFrame({'utc': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13],
        'A': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,0,0]})

df['B'] = df['A'] - df['A'].shift(1)
df['C'] = np.where((df.A == 1) & (df.B == 0), 1.5, 0)
#df['C'] = np.where((df.A == 1) & (df.B == 1), 1, 0)  

df = df.replace({'B':{-1:2}})

#col = 'C'
conditions = [df['B'] == 1, df['C'] == 1.5, df['B'] ==2 ]
output = [1,1.5,2]


df['cat'] = np.select(conditions, output, default = '-1')
#df = df[df['cat'].between('1','2')]
df = df.drop(columns=['B','C'], axis = 1)
print(df)

def pre_post_one(df):

    pre_epb = df[df['cat'] == '1'].index
    pre_filter = (pre_epb-1).union(pre_epb-2)

    post_epb = df[df['cat'] == '2'].index
    post_filter = (post_epb+1).union(post_epb+2)

    df_pre = df.iloc[pre_filter]
    df_post= df.iloc[post_filter]

    df = df[df['cat'].between('1','2')]


    df = pd.concat([df_pre, df, df_post], axis =0)
    print(df)

#pre_post_one(df)

#pre_epb = df.loc[df['cat']:'1'].head(2)
#pre_ind = df[df['cat'] == '1'].index
#pre_epb = df.loc[:'1',['cat']].head(3)

#cols = list(df.columns.values)
cols = df.columns
print(cols)

#pre_epb = df.loc[:'1',['cat']].head(2)
#print('\n',pre_epb)

pre_epb = df.loc[:'1',cols].head(3)
post_epb = df.loc[:'2',cols].tail(3)

#print(post_epb)

df = df[df['cat'].between('1','2')]

df = pd.concat([pre_epb, df, post_epb], axis =0)
print(df)

# print(pre_ind)
#print(pre_epb)




#df['color'] = np.where(((df.A < borderE) & ((df.B - df.C) < ex)), 'r', 'b')