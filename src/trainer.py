from train import train_model
from data_loader import get_trainable_data
from model import lstm_model
from config import Config

train_data, test_data = get_trainable_data()
model = lstm_model(1,4,1, Config.device)
train_loss, test_loss = train_model(lstm_model=model, train_data=train_data, test_data=test_data)