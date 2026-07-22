# %%
import os
import numpy as np
import numpy.typing as npt
import torch as t
from joblib import load
from config import Config
from model import lstm_model

scale = load('models/scale.pkl')

model = lstm_model(1,4,1, Config.device).to(Config.device)
ckpt = t.load('runs/latest.pth', map_location=Config.device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()


def scale_data(data:npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norm = scale.transform(data)
    return norm

def predict(data: npt.NDArray[np.float32], scale=False) -> npt.NDArray[np.float32]:
    if scale:
        scaled = scale_data(data)
    else:
        scaled = data
    with t.inference_mode():
        predictions, hidden = model(t.from_numpy(scaled[:,:-1].reshape(-1,Config.look_back,1)).float().to(Config.device))

    pred = predictions.cpu().squeeze().numpy()
    data[:,-1] = pred

    data = scale.inverse_transform(data)

    return data

# %%
if __name__ == "__main__":
    a = np.random.randn(10,8)
    b = np.random.randn(1,8)

    a_scale = scale_data(a)
    b_scale = scale_data(b)

    print(a_scale.shape, b_scale.shape)

    res = predict(b, True)
    print(res)
