import torch as t

class Config:
    look_back = 7
    epochs = 50
    batch_size = 16
    lr = 1e-5
    loss = t.nn.MSELoss()
    optim = t.optim.Adam
    device = 'cuda' if t.cuda.is_available() else 'cpu'