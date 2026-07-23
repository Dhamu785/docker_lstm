# %%
from data_loader import get_dataset
from config import Config
from inference_pipe import predict, scale_inverse
import matplotlib.pyplot as plt
import numpy as np

# %% sample dataset

def get_samples():
    x_train, x_test, y_train, y_test = get_dataset(Config.look_back)
    x_test = x_test.squeeze().numpy()
    y_test = y_test.squeeze().numpy()
    print(f"{x_test.shape, y_test.shape}")
    return x_test, y_test

def predict_samples(x_test):
    dummy_x = np.zeros((733,8))
    dummy_x[:,:-1] = x_test
    prediction = predict(dummy_x)
    return prediction

def get_real_num(x_test, y_test):
    combine = np.zeros((733,8))
    combine[:,:-1] = x_test
    combine[:,-1] = y_test
    unscaled = scale_inverse(combine)
    return unscaled

# %%
x_test, y_test = get_samples()
predictions = predict_samples(x_test)
real_val = get_real_num(x_test, y_test)

print("==="*20)
print(predictions.shape)
print(real_val.shape)
# %%
plt.plot(predictions[:,-1], label="predictions")
plt.plot(real_val[:,-1], label="actual")
plt.xlabel('Days')
plt.ylabel('Close')
plt.show()
# %%

def predict_next_day(data):
    dummy = np.zeros((1,8))
    dummy[:,:-1] = data
    # print(dummy)
    # print(dummy.shape)
    res = predict(dummy, need_scale=True)
    return res

a = np.array([247.48, 255.07, 249.74, 247.23, 249.98, 247.48, 244.85])

res = predict_next_day(a)
# %%
res
# %%
