# %% Import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import Config

# %% Read csv and sample csv
df = pd.read_csv('data/AMZN_max.csv')
df.head()

# %% Feature selection
data = df[['Date', 'Close']]
data.head()

# %% data plotting
data['Date'] = pd.to_datetime(data["Date"], utc=True).dt.date
print(data.head())
plt.plot(data['Date'], data['Close']);

# %% dataset creation
def data_for_lstm(df, steps):
    df = dc(df)
    df.set_index('Date', inplace=True)
    for i in range(1, steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    
    return df

lookback = Config.look_back
shifted_df = data_for_lstm(data, lookback)
shifted_df.head()

# %%
np_array = shifted_df.to_numpy()
flipped = dc(np.flip(np_array, axis=1))
print(flipped[:5])

scale = MinMaxScaler(feature_range=(-1, 1))
scaled = scale.fit_transform(flipped)
print(scaled[:5])

joblib.dump(scale, 'models/scale.pkl')

# %%
np.save('data/AMZN_max_norm.npy', scaled)

# %%
