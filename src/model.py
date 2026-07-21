# %%
import torch as t

class lstm_model(t.nn.Module):
    def __init__(self, input_shape: int, hidden_shape: int, num_stacks: int, device: str) -> None:
        super().__init__()
        self.hidden_shape = hidden_shape
        self.num_stacks = num_stacks
        self.device = device
        self.lstm = t.nn.LSTM(input_size=input_shape, hidden_size=hidden_shape, num_layers=num_stacks, batch_first=True)
        self.fc = t.nn.Linear(in_features=hidden_shape, out_features=1)

    def forward(self, x: t.Tensor) -> t.Tensor:
        batch_size = x.size(0)
        # h0 = t.zeros(self.num_stacks, batch_size, self.hidden_shape).to(self.device)
        # c0 = t.zeros(self.num_stacks, batch_size, self.hidden_shape).to(self.device)

        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out, (hn, cn)
# %%
if __name__ == "__main__":
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    x = t.randn((16,7,1), device=device)

    model = lstm_model(1, 4, 2, device=device).to(device)
    pred, lst_layer = model(x)

    print(f"Shape of prediction = {pred.shape}")
