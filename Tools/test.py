#https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
#https://chart-studio.plotly.com/~empet/14813.embed

import pandas as pd
import numpy as np
df = pd.DataFrame(
    np.random.randn(10, 3),
    index=pd.date_range("2022-01-01", periods=10),
    columns=["Ne", "Ti", "Pot"],
)

#print(df)

covs = (df[["Ne", "Pot"]].rolling(window=2)
        .cov(df[["Ne", "Pot"]], pairwise=True)
)

corrs = df['Ne'].rolling(window=3).corr(df['Pot'])

#print(df)
#print(covs)
print(corrs)

#terms of a timedelta convertible unit to specify the amount of time 
# it takes for an observation to decay to half its value when also 
# specifying a sequence
df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
times = ["2020-01-01", "2020-01-03", "2020-01-10", "2020-01-15", "2020-01-17"]
df = df.ewm(halflife="4 days", times=pd.DatetimeIndex(times)).mean()

#print(df)