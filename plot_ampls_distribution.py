#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

from copy import deepcopy
import datetime

# import cProfile
from shutil import copyfile
import json
import math
import os
import os.path
import pickle
import re
import sys
import tempfile
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.utils.data

import scipy.io as sio
from scipy.special import comb
from itertools import combinations



# "Borrowed" from pytorch/torch/serialization.py.
# All credit goes to PyTorch developers.
def _with_file_like(f, mode, body):
    """
    Executes a body function with a file object for f, opening
    it in 'mode' if it is a string filename.
    """
    new_fd = False
    if (
        isinstance(f, str)
        or (sys.version_info[0] == 2 and isinstance(f, unicode))
        or (sys.version_info[0] == 3 and isinstance(f, pathlib.Path))
    ):
        new_fd = True
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()


def _make_checkpoints_for(n: int, steps: int = 10):
    if n <= steps:
        return list(range(0, n))
    important_iterations = list(range(0, n, n // steps))
    if important_iterations[-1] != n - 1:
        important_iterations.append(n - 1)
    return important_iterations



def get_info(system_folder, j2=None, rt=None):
    if not os.path.exists(os.path.join(system_folder, "info.json")):
        raise ValueError(
            "Could not find {!r} in the system directory {!r}".format(
                "info.json", system_folder
            )
        )
    with open(os.path.join(system_folder, "info.json"), "r") as input:
        info = json.load(input)
    if j2 is not None:
        info = [x for x in info if x["j2"] in j2]
    return info



def load_dataset_TOM(dataset):
    # Load the dataset using pickle
    dataset = tuple(
        torch.from_numpy(x) for x in _with_file_like(dataset, "rb", pickle.load)
    )
    
    # Pre-processing
    dataset = (
        dataset[0],
        torch.where(dataset[1] >= 0, torch.tensor([0]), torch.tensor([1])).squeeze(),
        torch.abs(dataset[1]) ** 2,
    )
    return dataset

def load_dataset_K(dataset):
    print("load")
    # Load the dataset and basis
    mat = sio.loadmat(dataset)
    dataset = dataset.split("_");
    dataset[1] = "basis"; 
    basis = "_".join(dataset[:2]+dataset[-2:])[:-4]
    basis = _with_file_like(basis, "rb", pickle.load)

    print("construction")
    # Construction of full basis
    psi = np.ones((comb(basis.N,basis.N//2,exact=True),basis.N), dtype=np.float32)
    for ix, item in enumerate(combinations(range(basis.N), basis.N//2)):
        psi[ix,item] = -1

    print("expansion")
    # Expansion of eigenstate
    phi = basis.get_vec(mat['psi'][:,0], sparse = False, pcon=True).astype(np.float32)    
    
    dataset = (psi, phi)
    
    dataset = tuple(
        torch.from_numpy(x) for x in dataset
    )
    # Pre-processing
    dataset = (
        dataset[0],
        torch.where(dataset[1] >= 0, torch.tensor([0]), torch.tensor([1])).squeeze(),
        (torch.abs(dataset[1]) ** 2).unsqueeze(1),
    )
    return dataset

def main():
    config = _with_file_like(sys.argv[1], "r", json.load)
    system_folder = config["system"]
    lrs = config.get("lr")
    info = get_info(system_folder, config.get("j2"))

    for _obj,lr in zip(info, lrs):
        j2 = _obj["j2"]

        dataset = _obj["dataset"]
        if dataset.endswith("pickle"):
            dataset = load_dataset_TOM(dataset)
        elif dataset.endswith("mat"):
            dataset = load_dataset_K(dataset)
        else:
            raise Exception('dataset should be either *.mat or *.pickle file, received: ' + dataset)

        amplitudes_squared = dataset[2][:, 0].numpy()
        amplitudes_squared = np.sort(amplitudes_squared)
        #print(amplitudes_squared)
        cumsum = np.cumsum(amplitudes_squared)
        #plt.hist(amplitudes_squared, label = str(j2), alpha=0.15, bins=np.linspace(0.0, 0.0001, 100))
        plt.plot(np.arange(len(cumsum)) / 1.0 / len(cumsum), cumsum, label = str(j2))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlim([0.6, 1])
    plt.xscale('log')
    plt.xlabel('states fraction')
    plt.ylabel('total weight')
    plt.savefig(sys.argv[2])
    return


if __name__ == "__main__":
    main()
