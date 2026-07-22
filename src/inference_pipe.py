# %%
import os
import numpy as np
import torch as t
from sklearn.preprocessing import MinMaxScaler
from joblib import load

def scale(data):
    scale = load('models/scale.pkl')
    norm = scale.transform(data)
# %%
a = np.random.randn(10,8)

s = scale(a)
# %%
a
# %%
