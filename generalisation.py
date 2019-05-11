#!/usr/bin/env python3

import cmath
import collections

# from copy import deepcopy
# import cProfile
# import importlib
from itertools import islice
from functools import reduce
import logging
import math

# import os
# import sys
import time
from typing import Dict, List, Tuple, Optional
import tempfile

# import click
# import mpmath  # Just to be safe: for accurate computation of L2 norms
from numba import jit, jitclass, uint8, int64, uint64, float32

# from numba.types import Bytes
# import numba.extending
import numpy as np
import scipy.special

# from scipy.sparse.linalg import lgmres, LinearOperator
import torch
import torch.utils.data

import lightgbm as lgb
# import torch.nn as nn
# import torch.nn.functional as F


def _make_checkpoints_for(n: int, steps: int = 10):
    if n <= steps:
        return list(range(0, n))
    important_iterations = list(range(0, n, n // steps))
    if important_iterations[-1] != n - 1:
        important_iterations.append(n - 1)
    return important_iterations


def split_dataset(dataset, fraction=0.01):
    n = dataset[0].size(0)
    indices = torch.randperm(n)
    middle = int(fraction * n)
    train = tuple(x[indices[:middle]] for x in dataset)
    test = tuple(x[indices[middle:]] for x in dataset)
    # train_x, test_x = x[indices[:middle]], x[indices[middle:]]
    # train_y, test_y = y[indices[:middle]], y[indices[middle:]]
    return train, test # (train_x, train_y), (test_x, test_y)


class EarlyStopping(object):
    def __init__(self, patience: int = 7, verbose: bool = False):
        assert patience > 0, "`patience` must be positive"
        self._patience = patience
        self._verbose = verbose
        self._best_loss = math.inf
        self._should_stop = False
        self._checkpoint = tempfile.SpooledTemporaryFile(max_size=100 * 1024 * 1024)
        self._counter = 0

    def __call__(self, loss: float, model):
        loss = float(loss)
        if loss < self._best_loss:
            self._save_checkpoint(loss, model)
            self._counter = 0
        else:
            self._counter += 1
            if self._verbose:
                print(
                    "[EarlyStopping] Test loss increased: {:.3e} -> {:.3e}".format(
                        self._best_loss, loss
                    )
                )
            if self._counter >= self._patience:
                print("[EarlyStopping] Stop now!")
                self._should_stop = True

    @property
    def should_stop(self):
        return self._should_stop

    @property
    def best_loss(self):
        return self._best_loss

    def load_best(self, model):
        self._checkpoint.seek(0)
        model.load_state_dict(torch.load(self._checkpoint))
        return model

    def _save_checkpoint(self, loss, model):
        if self._verbose:
            print(
                "[EarlyStopping] Test loss decreased: {:.3e} -> {:.3e}. Saving the model...".format(
                    self._best_loss, loss
                )
            )
        self._best_loss = loss
        self._should_stop = False
        self._checkpoint.seek(0)
        self._checkpoint.truncate()
        torch.save(model.state_dict(), self._checkpoint)

@torch.jit.script
def negative_log_overlap_real(predicted, expected):
    predicted = predicted.view([-1])
    expected = expected.view([-1])
    sqr_l2_expected = torch.dot(expected, expected)
    sqr_l2_predicted = torch.dot(predicted, predicted)
    expected_dot_predicted = torch.dot(expected, predicted)
    return -0.5 * torch.log(
        expected_dot_predicted
        * expected_dot_predicted
        / (sqr_l2_expected * sqr_l2_predicted)
    )

def train(ψ, train_set, test_set, config):
    print("Training...")
    start = time.time()

    epochs = config["epochs"]
    optimiser = config["optimiser"](ψ)
    loss_fn = config["loss"]
    check_frequency = config["frequency"]

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_set),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=1,
    )
    test_x, test_y = test_set
    checkpoints = set(_make_checkpoints_for(epochs, steps=100))
    early_stopping = EarlyStopping(config["patience"])
    update_count = 0
    for i in range(epochs):
        losses = []
        for batch_x, batch_y in dataloader:
            optimiser.zero_grad()
            loss = loss_fn(ψ(batch_x), batch_y)
            losses.append(loss.item())
            loss.backward()
            optimiser.step()
            update_count += 1
            if update_count == check_frequency:
                update_count = 0
                with torch.no_grad():
                    loss = loss_fn(ψ(test_x), test_y).item()
                early_stopping(loss, ψ)
                if early_stopping.should_stop:
                    break
        if early_stopping.should_stop:
            print(
                "Stopping at epoch {}: test loss = {:.3e}".format(
                    i, early_stopping.best_loss
                )
            )
            early_stopping.load_best(ψ)
            break
        if i in checkpoints:
            losses = torch.tensor(losses)
            print(
                "{}%: train loss = {:.3e} ± {:.2e}; train loss ∈ [{:.3e}, {:.3e}]".format(
                    100 * (i + 1) // epochs,
                    torch.mean(losses).item(),
                    torch.std(losses).item(),
                    torch.min(losses).item(),
                    torch.max(losses).item(),
                )
            )
    if not early_stopping.should_stop:
        print("test loss = {:.3e}".format(early_stopping.best_loss))
        early_stopping.load_best(ψ)

    finish = time.time()
    print("Finished training in {:.2f} seconds!".format(finish - start))
    return ψ


def all_spins(n: int, m: Optional[int]) -> torch.Tensor:
    if m is not None:
        n_ups = (n + m) // 2
        n_downs = (n - m) // 2
        size = int(scipy.special.comb(n, n_ups))
        spins = torch.empty((size, n), dtype=torch.float32)
        for i, s in enumerate(
            map(
                lambda x: torch.tensor(x, dtype=torch.float32).view(1, -1),
                perm_unique([1] * n_ups + [-1] * n_downs),
            )
        ):
            spins[i, :] = s
        return spins
    else:
        raise NotImplementedError()


def accuracy(ψ, test_set):
    x, expected = test_set
    with torch.no_grad():
        predicted = torch.max(ψ(x), dim=1)[1]
        print("total sign: {}".format(torch.sum(predicted)))
        return torch.sum(predicted == expected).item() / x.size(0)


_DATASETS = {
    "1x18": "1x18.dataset.pickle",
    "1x19": "1x19.dataset.pickle",
    "1x20": "1x20.dataset.pickle",
    "Kagome-18": "Kagome-18.dataset.pickle",
    "test": "Kagome-18.dataset.full.pickle",
}


def add_features(samples, edges):
    xs = torch.empty(samples.size(0), samples.size(1) + len(edges), dtype=torch.float32)
    xs[:, :samples.size(1)] = samples
    for n, (i, j) in enumerate(edges):
        xs[:, samples.size(1) + n] = samples[:, i] * samples[:, j]
    return xs


def main():
    dataset = torch.load(_DATASETS["test"])
    dataset = dataset[0], dataset[1]
    # dataset = add_features(dataset[0], [(i, (i + 1) % 19) for i in range(19)]), dataset[1]
    train_set, rest_set = split_dataset(dataset, 0.05)
    test_set, _ = split_dataset(rest_set, 100 / 95 * 0.05)
    print(
        [x.size() for x in train_set],
        [x.size() for x in rest_set],
        [x.size() for x in test_set],
    )
    print(train_set[0])

    number_spins = train_set[0].size(1)
    ψ = torch.jit.trace(
        torch.nn.Sequential(
            torch.nn.Linear(number_spins, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Softplus(),
        ),
        torch.rand(32, number_spins),
        check_trace=False,
    )

    print("total number of samples: {}".format(dataset[0].size(0)))
    print(
        "train set length: {}; test set length: {}".format(
            train_set[0].size(0), test_set[0].size(0)
        )
    )
    # print(
    #     "accuracy: train = {}, test = {}, rest = {}".format(
    #         accuracy(ψ, train_set), accuracy(ψ, test_set), accuracy(ψ, rest_set)
    #     )
    # )
    train(
        ψ,
        train_set,
        test_set,
        {
            "optimiser": lambda m: torch.optim.Adam(m.parameters(), lr=0.001),
            "epochs": 200,
            "batch_size": 32,
            "loss": negative_log_overlap_real, # torch.nn.CrossEntropyLoss(),
            "frequency": 10,
            "patience": 1000,
        },
    )
    # print(
    #     "accuracy: train = {}, test = {}, rest = {}".format(
    #         accuracy(ψ, train_set), accuracy(ψ, test_set), accuracy(ψ, rest_set)
    #     )
    # )
    torch.save(ψ.state_dict(), "pre-training.1.pickle")


if __name__ == "__main__":
    main()
