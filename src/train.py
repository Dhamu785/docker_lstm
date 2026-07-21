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

def train(lstm_model, train_data, epochs=Config.epochs, batch_size=Config.batch_size, 
            lr=Config.lr, loss=Config.loss, optimizer=Config.optim, resume=False):
    all_losses = []
    start_epoch = 1
    lstm_model.to(Config.device)
    print(f"Model on {next(lstm_model.parameters()).device}")
    optim = optimizer(lstm_model.parameters(), lr=lr)