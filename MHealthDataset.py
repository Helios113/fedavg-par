import torch
from pathlib import Path
import pandas as pd
import numpy as np
import torch


def MHealthDataset(dir: str, training: float, test: float):
    data_dir = Path(dir)
    data = torch.Tensor(pd.read_csv(data_dir, delimiter=",", header=None).values)
    data_len = data.shape[0]
    labels = data[:, -1]
    data = data[:, :-1]

    training = int(training * data_len)
    test = int(test * data_len)

    data = data.reshape(data_len, -1, 23)

    sampling = range(data_len)
    train_idx = np.random.choice(sampling, size=training, replace=False).tolist()
    sampling = list(set(sampling) - set(train_idx))
    test_idx = np.random.choice(sampling, size=test, replace=False).tolist()

    # Temp for now
    return (data[train_idx], labels[train_idx]), (data[test_idx], labels[test_idx])
