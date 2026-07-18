# %% Import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from copy import deepcopy as dc

# %% device setup
device = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Available device = {device}")
df = pd.read_csv('data/AMZN_max.csv')
df.head()

# %% data pre-processsing
data = df[['Date', 'Close']]
data.head()

# %% data plotting
data['Date'] = pd.to_datetime(data["Date"], utc=True).dt.date
plt.plot(data['Date'], data['Close'])

# %% dataset creation
def data_for_lstm(df, steps):
    df = dc(df)
    df.set_index('Date', inplace=True)
    for i in range(1, steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    
    return df

lookback = 7
shifted_df = data_for_lstm(data, lookback)
shifted_df.head()
# %%
shifted_df.to_csv('data/lstm_amzn.csv')
# %%
