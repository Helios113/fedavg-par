import torch
from torch import nn
from torch.utils.data import Dataset
from MHealthDataset import MHealthDataset
import numpy as np
import pandas as pd
import random


class FedDataset(Dataset):
    def __init__(self, data, device="mps"):
        self.data = (
            torch.Tensor(data[:, :-1])
            .to(device)
            .unsqueeze(1)
            .reshape((data.shape[0], -1, 23))
        )
        self.target = torch.LongTensor(data[:, -1]).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return [self.data[item], self.target[item]]


def get_data(path, clients, iid, data_info):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    dataset_train, dataset_test = load_data(
        path, data_info.train_data_portion, data_info.test_data_portion
    )
    if iid:
        dict_users_train = iid_data(dataset_train, clients)
        dict_users_test = iid_data(dataset_test, clients)
    else:
        if data_info.classes_per_client_testing == data_info.classes_per_client_training:
            initalK = random.randint(0, 12 - 1)
        else:
            initalK = -1
        dict_users_train = non_iid_data(
            dataset_train,
            clients,
            data_info.classes_per_client_training, initalK
        )
        dict_users_test = non_iid_data(
            dataset_test, clients, data_info.classes_per_client_testing, initalK
        )
    return dict_users_train, dict_users_test


def iid_data(data, num_users):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    num_items = [data.shape[0] // num_users] * num_users

    lefover = data.shape[0] - num_items[0] * num_users
    i = 0
    while lefover > 0:
        num_items[i] += 1
        lefover -= 1
        i += 1
        i = i % num_users

    dict_users, inds = {}, [i for i in range(data.shape[0])]
    for i in range(num_users):
        dict_users[i] = None
        B = random.sample(range(len(inds)), num_items[i])  # Random saple of
        B = np.flip(np.sort(B))
        for j in B:
            if dict_users[i] is None:
                dict_users[i] = data[inds.pop(j)]
            else:
                dict_users[i] = np.vstack((dict_users[i], data[inds.pop(j)]))
    return dict_users


def non_iid_data(daTa, num_users, classesPerClient, initialK = None):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if type(daTa) != list:
        daTa = [daTa]
    if len(daTa) == num_users:
        num_users = 1
    dict_users = {}
    for cnt, data in enumerate(daTa):
        unique_classes = np.unique(data[:, -1])
        number_classes = len(unique_classes)
        inds = {}
        for c in unique_classes:
            inds[c] = np.where(data[:, -1] == c)[0].tolist()
        data_size = len(data)
        budgetPerClass = np.ceil(data_size / (num_users * classesPerClient))
        for i in range(num_users):
            dict_users[cnt+i] = None
            budgetPerDevice = data_size // (num_users - i)
            data_size -= budgetPerDevice
            if initialK < 0:
                k = random.randint(0, number_classes - 1)
            else:
                k = initialK
            while budgetPerDevice > 0:
                t = int(min(budgetPerDevice, budgetPerClass, len(inds[k])))
                budgetPerDevice -= t
                B = np.flip(np.sort(random.sample(range(len(inds[k])), t)))
                for j in B:
                    if dict_users[cnt+i] is None:
                        dict_users[cnt+i] = data[inds[k].pop(j)]
                    else:
                        dict_users[cnt+i] = np.vstack((dict_users[cnt+i], data[inds[k].pop(j)]))
                k = (k + 1) % number_classes

    return dict_users


def load_data(path, train_data_portion, test_data_portion):
    if type(path) == list:
        train = []
        test = []
        for i in path:
            tmp = pd.read_csv(i, header=None, index_col=False)
            tr = tmp.sample(frac=train_data_portion, random_state=10)
            te = tmp.drop(tr.index)
            te = te.sample(frac=test_data_portion/(1-train_data_portion) , random_state=10)
            train.append(tr.values)
            test.append(te.values)
        return train, test
            
    data = pd.read_csv(path, header=None, index_col=False)
    train = data.sample(frac=train_data_portion, random_state=10)
    test = data.drop(train.index)
    test = test.sample(frac=test_data_portion/(1-train_data_portion) , random_state=10)
    # train = train.sample(frac=1, random_state=10).reset_index(drop=True).values
    # test = test.sample(frac=1, random_state=10).reset_index(drop=True).values
    return train.values, test.values


class DataInfo:
    def __init__(
        self,
        train_data_portion,
        test_data_portion,
        classes_per_client_training,
        classes_per_client_testing,
        num_sets_train,
        num_sets_test,
    ) -> None:
        self.train_data_portion = train_data_portion
        self.test_data_portion = test_data_portion
        self.classes_per_client_training = classes_per_client_training
        self.classes_per_client_testing = classes_per_client_testing
        self.num_sets_train = num_sets_train
        self.num_sets_test = num_sets_test
