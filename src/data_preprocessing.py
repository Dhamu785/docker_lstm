# %% Import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import Config

# %% Read csv and sample csv
stock_name = 'HDFC_max'
df = pd.read_csv(f'data/{stock_name}.csv')
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

joblib.dump(scale, 'models/scale_hdfc.pkl')

# %%
np.save(f'data/{stock_name}_norm.npy', scaled)

# %%
