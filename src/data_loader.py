# %%
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import Config
import torch as t

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
    
    return t.from_numpy(X_train).float(), t.from_numpy(X_test).float(), t.from_numpy(Y_train).float(), t.from_numpy(Y_test).float()


class TimeSeries_data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def get_trainable_data():
    x_train, x_test, y_train, y_test = get_dataset(Config.look_back)
    train_dataset = TimeSeries_data(x_train, y_train)
    test_dataset = TimeSeries_data(x_test, y_test)

    batch_size = Config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# %%
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_dataset(Config.look_back)
    train_dataset = TimeSeries_data(x_train, y_train)
    test_dataset = TimeSeries_data(x_test, y_test)
    print(type(x_train))

    batch_size = Config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"{len(train_loader) = }, {len(test_loader) = }")

    for x,y in train_loader:
        print(x.shape, y.shape)
        break
# %%
