import os
import sys
import torch
import numpy as np
import torchvision
import torch.utils.data as Data
from torch.utils.data import Subset


def MNIST_transform():
    return torchvision.transforms.ToTensor()


def CIFA_transform(training=False):
    if training:
        trans = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
                    std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
                ),
            ]
        )
    else:
        trans = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
                    std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
                ),
            ]
        )

    return trans


class DataLorder:

    size = 0
    dataset_cls = {
        "MNIST": torchvision.datasets.MNIST,
        "CIFA10": torchvision.datasets.CIFAR10,
        "CIFA100": torchvision.datasets.CIFAR100,
    }
    dataset_trans = {
        "MNIST": MNIST_transform(),
        "CIFA10": CIFA_transform(),
        "CIFA100": CIFA_transform(),
    }

    @classmethod
    def get(cls, dataset, indices):
        return Data.Subset(dataset, indices)

    @classmethod
    def create(cls, dataset, batch_size=None, shuffle=False):
        if batch_size == None:
            batch_size = len(dataset)

        rtn = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
        )
        return rtn

    @classmethod
    def load(cls, args, training, loader=False, batch_size=64, shuffle=False):
        data_dir = "./data"
        dataset = cls.dataset_cls[args.dataset](
            root=data_dir,
            train=training,
            transform=cls.dataset_trans[args.dataset],
            download=True,
        )

        if loader:
            rtn = Data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=2,
            )
        else:
            rtn = dataset

        cls.size = len(dataset)

        return rtn


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    """
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [
        np.argwhere(train_labels[train_idcs] == y).flatten() for y in range(n_classes)
    ]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(
            np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
        ):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


class CustomSubset(Data.Subset):
    """A custom subset class with customizable data transformation"""

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y
