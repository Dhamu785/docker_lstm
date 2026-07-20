# %%
import numpy as np
from torch.utils.data import Dataset
from config import Config
import torch as t
# %%
def get_dataset(look_back: int, split_ratio: float=0.90) -> t.Tensor:
    data = np.load('data/AMZN_max_norm.npy')
    print(f"{"Shape of the file":<15} : {data.shape}")
    X = data[:,:-1]
    Y = data[:,-1]
    print(f"{"Shape of X":<15} : {X.shape}, Y : {Y.shape}")
    split_ratio = len(X) * split_ratio
    split_ratio = round(split_ratio)
    print(f"{"Split ratio":<15} : {split_ratio}")

    # Splitting
    X_train, X_test, Y_train, Y_test = X[:split_ratio].reshape((-1, look_back, 1)), X[split_ratio:].reshape((-1, look_back, 1)), Y[:split_ratio].reshape((-1, 1)), Y[split_ratio:].reshape((-1, 1))
    print(f"{"Reshaped values":<15} : {X_train.shape = }, {X_test.shape = }, {Y_train.shape = }, {Y_test.shape = }")
    
    return X_train, X_test, Y_train, Y_test

# %%
x_train, x_test, y_train, y_test = get_dataset(Config.look_back)

# %%
