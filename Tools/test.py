import pandas as pd

df = pd.DataFrame({'B': [0, -1, -0.5, 0.51, 0]})
#df['tri'] = df.rolling(2, win_type = 'triang').sum()
df['gauss'] = df.rolling(2, win_type = 'gaussian').sum(std=1)


def gauss_check(x):
    #if -0.5<= x <=0.5:

    if x <= -0.5 or x>0.5:
        return 'epb'
    else:
        return 'no epb'

df['g_epb'] = df['B'].apply(gauss_check)

print(df)