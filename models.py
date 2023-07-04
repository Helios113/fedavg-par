import re
from tkinter.messagebox import NO
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import copy
import math
torch.manual_seed(0)
np.random.seed(0)


class Composite(nn.Module):
    def __init__(self, local: nn.Module, glob: nn.Module):
        super(Composite, self).__init__()
        self.local = local
        self.glob = glob

    def forward(self, x):
        x, v = self.local(x)
        return self.glob(x), x, v

class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Dropout())    
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(0)
        x = self.linear(x)
        return x


class MLP_drop(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_drop, self).__init__()
        self.seq = nn.Sequential(nn.Dropout(0.4), nn.Linear(input_dim, output_dim))

    def forward(self, x):
        x = x.squeeze()
        x = self.seq(x)
        return x


class MLP_CL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_CL, self).__init__()
        self.input_dim = input_dim
        self.stack = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Dropout(p=0.4),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.stack(x)


class SLC(nn.Module):
    def __init__(self, modalities: dict, hidden_dims, output_dim, penalty):
        super(SLC, self).__init__()
        self.penalty = penalty
        self.modalities = modalities
        self.layer_list = nn.ModuleDict()
        self.output_dim = output_dim
        for i in modalities:
            self.layer_list[i] = nn.LSTM(
                        input_size = len(modalities[i]),
                        hidden_size = output_dim,
                        num_layers = hidden_dims,
                        batch_first=True,
                        dropout = 0.5
                    )

    def forward(self, x):
        h = None
        h_list = []
        penalty = None
        size = len(self.modalities)
        for i in self.modalities:
            X = x[..., self.modalities[i]]
            X, (h_n, c_n) = self.layer_list[i](X)
            h_n = h_n[-1]
            h_list.append(h_n)
            if h is not None:
                h += h_n
            else:
                h = h_n
        h = h/len(self.modalities)
        cntt=1
        if self.penalty:
            for comb in combinations(h_list, 2):
                cntt+=1
                if penalty is None:
                    penalty = torch.abs(comb[0] - comb[1])
                else:
                    penalty = torch.add(penalty, torch.abs(comb[0] - comb[1]))
            if penalty is None:
                penalty = torch.zeros_like(h)
            return torch.subtract(h, penalty),  h_list
        return h, h_list

class SLC_co(nn.Module):
    def __init__(self, modalities: dict, hidden_dims, output_dim, penalty, sig, diff):
        super(SLC_co, self).__init__()
        self.penalty = penalty
        self.modalities = modalities
        self.layer_list = nn.ModuleDict()
        self.output_dim = output_dim
        if sig:
            self.lam = torch.nn.ParameterList([torch.ones(1)])
        else:
            self.lam = torch.nn.ParameterList([torch.ones(output_dim)])
            
        self.diff = diff
        
        for i in modalities:
            self.layer_list[i] = nn.LSTM(
                        input_size = len(modalities[i]),
                        hidden_size = output_dim,
                        num_layers = hidden_dims,
                        batch_first=True,
                        dropout = 0.5
                    )

    def forward(self, x):
        h = None
        h_list = []
        penalty = None
        size = len(self.modalities)
        for i in self.modalities:
            X = x[..., self.modalities[i]]
            X, (h_n, c_n) = self.layer_list[i](X)
            h_n = h_n[-1]
            h_list.append(h_n)
            if h is not None:
                h += h_n
            else:
                h = h_n
        h = h/len(self.modalities)
        if self.penalty:
            for comb in combinations(h_list, 2):
                if penalty is None:
                    penalty = torch.abs(comb[0] - comb[1])
                else:
                    penalty = torch.add(penalty, torch.abs(comb[0] - comb[1]))
            if penalty is None:
                penalty = torch.zeros_like(h)
            if self.diff:    
                return self.lam[0] * torch.subtract(h, penalty),  h_list
            return torch.subtract(h, self.lam[0] * penalty),  h_list
        if self.diff: 
            return self.lam[0] * h, h_list
        return h, h_list
