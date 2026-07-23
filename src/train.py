import os
import random
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn

import torch as t
import torch.nn as nn

from config import Config

progress = Progress(
    SpinnerColumn(spinner_name='dots', style='green'),
    TextColumn('[bold blue]{task.description}'),
    BarColumn(complete_style='bright_green', finished_style='green', style='grey35'),
    TextColumn('[yellow]{task.percentage:3.0f}%'),
    TextColumn('[red] Loss: {task.fields[loss]}'),
    TextColumn('[yellow] Elasped:'),
    TimeElapsedColumn(),
    TextColumn('[cyan] Remaining:'),
    TimeRemainingColumn()
)

cwd = os.getcwd()
sav_loc = os.path.join(cwd, 'runs')
if not os.path.exists(sav_loc):
    os.mkdir(sav_loc)

def train_model(lstm_model, train_data, test_data, epochs=Config.epochs, batch_size=Config.batch_size, 
            lr=Config.lr, loss=Config.loss, optimizer=Config.optim, resume=False):
    train_loss = []
    test_loss = []
    start_epoch = 1
    lstm_model.to(Config.device)
    print(f"Model on {next(lstm_model.parameters()).device}")
    optim = optimizer(lstm_model.parameters(), lr=lr)

    if resume:
        ckpt = t.load(os.path.join(sav_loc, 'latest.pt'), map_location=Config.device, weights_only=False)
        lstm_model.load_state_dict(ckpt['model_state_dict'])
        optim.load(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epochs']+1
        train_loss = ckpt['train_loss']
        test_loss = ckpt['test_loss']
        print("Resume training")

    with progress:
        epoch_task = progress.add_task("Learning : ", total=(epochs+1)-start_epoch, loss=0.0)
        for epoch in range(start_epoch, epochs+1):
            batch_train_task = progress.add_task(f"Epoch [{epoch}/{epochs}]", total=len(train_data), loss=0.0)

            lstm_model.train()
            epoch_train_loss = 0
            for x_train, y_train in train_data:
                optim.zero_grad()
                pred, hidden = lstm_model(x_train.to(Config.device))
                ls = loss(pred, y_train.to(Config.device))
                ls.backward()
                t.nn.utils.clip_grad_norm_(lstm_model.parameters(), 3)
                optim.step()

                epoch_train_loss += ls.item()
                progress.update(batch_train_task, advance=1, description=f"Epoch [{epoch}/{epochs}]", loss=f"{ls.item():.4f}")

            train_loss.append(epoch_train_loss/len(train_data))

            batch_test_task = progress.add_task(f"Epoch [{epoch}/{epochs}]", total=len(test_data), loss=0.0)
            lstm_model.eval()
            batch_test_loss = 0
            for x_test, y_test in test_data:
                with t.inference_mode():
                    pred, hidden = lstm_model(x_test.to(Config.device))
                    ls = loss(pred, y_test.to(Config.device))
                batch_test_loss += ls.item()
                progress.update(batch_test_task, advance=1, description=f"Epoch [{epoch}/{epochs}]", loss=f"{ls.item():.4f}")
            test_loss.append(batch_test_loss/len(test_data))

            # Save check-point
            checkpoint = {
                'epoch': epoch, 'train_loss': train_loss, 'test_loss' : test_loss,
                'lr' : lr, 'model_state_dict' : lstm_model.state_dict(), 
                'optimizer_state_dict' : optim.state_dict()
            }
            t.save(checkpoint, os.path.join(sav_loc, 'latest_hdfc.pth'))
            progress.remove_task(batch_train_task)
            progress.remove_task(batch_test_task)
            progress.update(epoch_task, advance=1, loss=f"{test_loss[-1]:.4f}")

    return train_loss, test_loss