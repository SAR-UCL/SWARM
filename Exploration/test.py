import pandas as pd
import numpy as np
df = pd.DataFrame({'utc': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'A': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]})

df['B'] = df['A'] - df['A'].shift(1)
df['C'] = np.where((df.A == 1) & (df.B == 0), 1.5, 0)
#df['C'] = np.where((df.A == 1) & (df.B == 1), 1, 0)  

df = df.replace({'B':{-1:2}})

#col = 'C'
conditions = [df['B'] == 1, df['C'] == 1.5, df['B'] ==2 ]
output = [1,1.5,2]


print(df)

df['cat'] = np.select(conditions, output, default = '0')

df = df[df['cat'].between('1','2')]

print(df)



#df['color'] = np.where(((df.A < borderE) & ((df.B - df.C) < ex)), 'r', 'b')