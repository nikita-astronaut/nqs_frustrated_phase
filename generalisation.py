#!/usr/bin/env python3

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

import scipy
import numpy as np
import torch
import torch.utils.data

import scipy.io as sio
from scipy.special import comb
from itertools import combinations
from nqs_frustrated_phase.hphi import load_eigenvector
from nqs_frustrated_phase._core import sector
# import quspin
# from quspin import basis
number_spins = None

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

def index_to_spin(index):
    global number_spins
    if number_spins == 24:
        return index
    """
    Generates spins out of indexes given the total spin number
    P.S. Can be slow, but intuitive (I believe that the bottleneck is not here)
    """

    return torch.from_numpy((((index.numpy()[:, None] & (1 << np.arange(number_spins)))) > 0).astype(int) * 2. - 1.).type(torch.FloatTensor)

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


def split_dataset(dataset, fractions, sampling='uniform'):
    """
    Randomly splits a dataset into ``len(fractions) + 1`` parts
    """
    n = dataset[0].size(0)

    def parts(xs):
        first = 0
        for x in xs:
            last = first + x
            yield first, last
            first = last
        yield first, n

    weights = None
    if sampling == "quadratic":
        print("sampling with |A|^2 weights")
        weights = (dataset[2] / torch.sum(dataset[2])).numpy()
        # weights = weights / np.sum(weights)
    else:
        print("sampling with uniform weights")

    t = time.time()
    print('started sampling of ', n, ' elements', flush = True)
    if weights is None:
        indices = torch.randperm(n)
    else:
        #indices = torch.multinomial(weights[:int(len(weights) * 0.01)], int(n * sum(fractions))).numpy()
        #indices = np.random.choice(np.arange(int(n / 100.)), size = int(n * sum(fractions)), replace=False, p=weights[:int(len(weights) / 100.)] / np.sum(weights[:int(len(weights) / 100.)]))
        # print('indices selection took = ', time.time() - t, flush = True)
        #t = time.time()
        indices = np.random.choice(np.arange(n), size = int(n * sum(fractions)), replace=True, p=weights)
        indices = torch.from_numpy(np.concatenate([indices, np.setdiff1d(np.arange(n), indices)], axis = 0))

    print('indexes concatenation took = ', time.time() - t, flush = True)
    t = time.time()

    sets = []
    for first, last in parts(map(lambda x: int(round(x * n)), fractions)):
        sets.append(tuple(x[indices[first:last]] for x in dataset))
    print('sets creation took = ', time.time() - t, flush = True)
    t = time.time()
    print(sets[0][0].size(), sets[1][0].size(), sets[2][0].size())
    return sets


class EarlyStopping(object):
    def __init__(self, patience: int = 7, verbose: bool = False):
        """
        Initialises the early stopping observer.

        :param int patience:
            for how many steps the validation loss is allowed to increase
            before we stop the learning process.
        :param bool verbose:
            whether to print a message every time the loss changes.
        """
        assert patience > 0, "`patience` must be positive"
        self._patience = patience
        self._verbose = verbose
        self._best_loss = math.inf
        self._should_stop = False
        # Reserve a 100MB in RAM for storing weights before we turn to using
        # the filesystem (which is slower).
        self._checkpoint = tempfile.SpooledTemporaryFile(max_size=100 * 1024 * 1024)
        # Number of iterations since the validation loss has last decreased.
        self._counter = 0

    def __call__(self, loss: float, model):
        """
        Updates internal state.

        This function should be called every time validation loss is computed.

        :param float loss:            validation loss.
        :param torch.nn.Module model: neural network model.
        """
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
        """
        Returns whether the learning process should be stopped.
        """
        return self._should_stop

    @property
    def best_loss(self):
        """
        Returns the best validation loss achieved during training.
        """
        return self._best_loss

    def load_best(self, model):
        """
        Loads the weights for which the best validation loss was achieved.
        """
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


def train(ψ, train_set, test_set, gpu, lr, **config):
    if gpu:
        ψ = ψ.cuda()
    
    epochs = config["epochs"]
    optimiser = config['optimiser'](ψ)

    loss_fn = config["loss"]
    check_frequency = config["frequency"]
    load_best = True  # config["load_best"]
    verbose = config["verbose"]
    print_info = print if verbose else lambda *_1, **_2: None
    accuracy_fn = config.get("accuracy")
    if accuracy_fn is None:
        accuracy_fn = lambda _1, _2, _3: 0.0

    print_info("Training on {} spin configurations...".format(train_set[0].size(0)))
    start = time.time()
    test_x, test_y, test_weight = test_set
    

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_set),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=1,
    )
    checkpoints = set(_make_checkpoints_for(epochs, steps=100)) if verbose else set()
    early_stopping = EarlyStopping(config["patience"])
    train_loss_history = []
    test_loss_history = []


    def training_loop():
        update_count = -1
        for epoch_index in range(epochs):
            important = epoch_index in checkpoints
            if important:
                losses = []
                accuracies = []
            for batch_index, (batch_x, batch_y, batch_weight) in enumerate(dataloader):
                batch_x = index_to_spin(batch_x)
                if gpu:
                    batch_x, batch_y, batch_weight = batch_x.cuda(), batch_y.cuda(), batch_weight.cuda()
                optimiser.zero_grad()
                predicted = ψ(batch_x)
                loss = loss_fn(predicted, batch_y, batch_weight)
                loss.backward()
                optimiser.step()
                update_count += 1
                with torch.no_grad():
                    accuracy = accuracy_fn(predicted, batch_y, batch_weight)
                train_loss_history.append(
                    (update_count, epoch_index, loss.item(), accuracy)
                )
                if important:
                    losses.append(loss.item())
                    accuracies.append(accuracy)
                if update_count % check_frequency == 0:
                    with torch.no_grad():
                        predicted = predict_large_data(ψ, test_x, gpu, config["type"], no_model_movement = True)
                        loss = loss_fn(predicted, test_y, test_weight).item()
                        accuracy = accuracy_fn(predicted, test_y, test_weight)
                        print('test loss = {:.3e}, test accuracy = {:.3e}'.format(loss, accuracy))
                    early_stopping(loss, ψ)
                    test_loss_history.append(
                        (update_count, epoch_index, loss, accuracy)
                    )
                    if early_stopping.should_stop or 1 - accuracy <= 1e-5:
                        print_info(
                            "Stopping at epoch {}, batch {}: test loss = {:.3e}".format(
                                epoch_index, batch_index, early_stopping.best_loss
                            )
                        )
                        return True

            if important:
                losses = torch.tensor(losses)
                accuracies = torch.tensor(accuracies)
                print_info(
                    "{:3d}%: train loss     = {:.3e} ± {:.2e}; train loss     ∈ [{:.3e}, {:.3e}]".format(
                        100 * (epoch_index + 1) // epochs,
                        torch.mean(losses).item(),
                        torch.std(losses).item(),
                        torch.min(losses).item(),
                        torch.max(losses).item(),
                    )
                )
                print_info(
                    "      train accuracy = {:.3e} ± {:.2e}; train accuracy ∈ [{:.3e}, {:.3e}]".format(
                        torch.mean(accuracies).item(),
                        torch.std(accuracies).item(),
                        torch.min(accuracies).item(),
                        torch.max(accuracies).item(),
                    )
                )
        return False

    stopped_early = training_loop()
    if load_best:
        print_info("Loading best weights...")
        early_stopping.load_best(ψ)
    finish = time.time()
    print_info("Finished training in {:.2f} seconds!".format(finish - start))
    if gpu:
        ψ = ψ.cpu()
    return ψ, train_loss_history, test_loss_history

def predict_large_data(model, dataset_x, gpu, train_type, no_model_movement = False):
    size = dataset_x.size()[0]
    if train_type == "phase":
        result = torch.zeros((size, 2)).type(torch.FloatTensor)
    else:
        result = torch.zeros((size)).type(torch.FloatTensor)
    if gpu:
        model = model.cuda()
    
    size = dataset_x.size()[0]
    for idxs in np.split(np.arange(size), np.arange(0, size, 10000))[1:]:
        if gpu:
            predicted = torch.squeeze(model(index_to_spin(dataset_x[idxs]).cuda()).cpu()).type(torch.FloatTensor)
        else:
            predicted = torch.squeeze(model(index_to_spin(dataset_x[idxs]))).type(torch.FloatTensor)
        result[idxs, ...] = predicted
    if gpu and not no_model_movement:
        model = model.cpu()
    return result


def import_network(filename: str):
    import importlib

    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the network from {!r}: not a Python source file.".format(
                filename
            )
        )
    if not os.path.exists(filename):
        raise ValueError(
            "Could not import the network from {!r}: no such file or directory".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module.Net


def get_number_spins(config):
    number_spins = config.get("number_spins")
    if number_spins is not None:
        return number_spins
    match = re.match(r".+/data/(.+)/([0-9]+)/.+", config["system"])
    if len(match.groups()) != 2:
        raise ValueError(
            "Configuration file does not specify the number of spins "
            "and it could not be extracted from the system folder {!r} either"
        )
    return int(match.group(2))

def load_dataset_Tom(dataset):
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
    print(dataset[0].size(), dataset[1].size(), dataset[2].size())
    return dataset

def load_dataset_K(dataset_name):
    global number_spins
    t = time.time()
    magnetisation = number_spins % 2
    number_ups = (number_spins + magnetisation) // 2
    shift = number_ups * (number_ups - 1) // 2 if number_ups > 0 else 0

    print("load", flush = True)
    # Load the dataset and basis
    print(dataset_name, flush = True)
    mat = sio.loadmat(dataset_name)
    dataset_name = dataset_name.split("--");
    dataset_name[1] = "basis";
    basis = "--".join(dataset_name[:2]+dataset_name[-2:])[:-4]
    print(basis, flush = True)
    basis = _with_file_like(basis, "rb", pickle.load)
    print('loading took = ', time.time() - t, flush = True)
    t = time.time()
    all_spins = quspin.basis.spin_basis_general(number_spins, pauli=0, Nup = number_spins // 2)
    # all_spins = np.fromiter(
    #     sector(number_spins, magnetisation),
    #     dtype=np.uint64,
    #     count=int(scipy.special.comb(number_spins, number_ups)),
    # ).astype(np.int64)    
    print('spins set generation took = ', time.time() - t, flush= True)
    t = time.time()
    print("expansion", flush = True)
    # Expansion of eigenstate
    # phi = basis.get_vec(mat['psi'][:, 0], sparse = False, pcon=True).astype(np.float32)

    # print('expansion took = ', time.time() - t, flush = True)
    # t = time.time()
    dataset_name[1] = "dump_vector"
    vector = "--".join(dataset_name[:2]+dataset_name[-2:])[:-4]
    # np.save(vector, phi)
    phi = np.load(vector + '.npy')

    print('saving/loading took = ', time.time() - t, flush = True)
    t = time.time()

    dataset = torch.from_numpy(phi)
    # Pre-processing
    print('from numpy done', flush = True)
    norm = torch.sum(torch.abs(dataset) ** 2).item()
    dataset = (
        torch.from_numpy(all_spins.states.astype(np.int64)),
        torch.where(dataset >= 0, torch.tensor([0]), torch.tensor([1])).squeeze(),
        (torch.abs(dataset) ** 2 / norm).unsqueeze(1)[:, 0].type(torch.FloatTensor),
    )

    print('to torch format took = ', time.time() - t, flush = True)
    t = time.time()

    # dataset_name[1] = "dump_vector"
    # vector = "--".join(dataset_name[:2]+dataset_name[-2:])[:-4]
    # with open(vector, 'wb') as f:
    #    pickle.dump(dataset, f)

    print('dump took = ', time.time() - t, flush = True)
    t = time.time()

    print('total norm = ', torch.sum(dataset[2]).item(), norm, flush = True)
    return dataset


def accuracy(predicted, expected, weight, apply_weights_loss = False):
    predicted = torch.max(predicted, dim=1)[1]
    if not apply_weights_loss:
        return torch.sum(predicted == expected).item() / float(expected.size(0))
    agreement = predicted == expected
    print(agreement.size(), weight.size(), 'accuracy')
    return torch.sum(agreement.type(torch.FloatTensor) * torch.tensor(weight, dtype = torch.float32)).item()  # / float(expected.size(0))
 
def overlap(train_type, ψ, samples, target, weights, gpu):
    if train_type == 'phase':
        return overlap_phase(ψ, samples, target, weights, gpu)
    else:
        return overlap_amplitude(ψ, samples, target, weights, gpu)

def overlap_amplitude(ψ, samples, target, weights, gpu):
    predicted_amplitudes = predict_large_data(ψ, samples, gpu, "amplitude")
    overlap = torch.sum(torch.sqrt(predicted_amplitudes.type(torch.FloatTensor)) * torch.sqrt(weights)).item()
    norm_bra = torch.sum(predicted_amplitudes.type(torch.FloatTensor)).item()
    norm_ket = torch.sum(weights).item()

    return overlap / np.sqrt(norm_bra) / np.sqrt(norm_ket)

def overlap_phase(ψ, samples, target, weights, gpu):
    predicted_signs = 2.0 * torch.max(predict_large_data(ψ, samples, gpu, "phase"), dim=1)[1] - 1.0
    print(predicted_signs.size(), flush = True)
    target_signs = 2.0 * target - 1.0
    print(predicted_signs.size(), target_signs.size(), weights.size(), 'overlap phase')
    overlap = torch.sum(predicted_signs.type(torch.FloatTensor) * target_signs.type(torch.FloatTensor) * weights.type(torch.FloatTensor)).item()
    return overlap / torch.sum(weights).item()

def load_dataset_large(dataset_name):
    global number_spins
    magnetisation = number_spins % 2
    number_ups = (number_spins + magnetisation) // 2
    shift = number_ups * (number_ups - 1) // 2 if number_ups > 0 else 0
    all_spins = np.fromiter(
        sector(number_spins, magnetisation),
        dtype=np.uint64,
        count=int(scipy.special.comb(number_spins, number_ups)),
    ).astype(np.int64)

    all_amplitudes = torch.from_numpy(load_eigenvector(dataset_name))
    print('norm = ', torch.sum(all_amplitudes ** 2).item())
    ipr = int(torch.sum(all_amplitudes ** 2).item() ** 2 / torch.sum(all_amplitudes ** 4).item())
    print('IPR = ', str(ipr))
    sort_ampls = np.sort(all_amplitudes)[-ipr:]
    print('1/IPR states contains weigth: ', np.sum(sort_ampls ** 2))
    # return -1
    dataset = (
        torch.from_numpy(all_spins),
        torch.where(all_amplitudes >= 0, torch.tensor([0]), torch.tensor([1])).squeeze(),
        (torch.abs(all_amplitudes) ** 2).type(torch.FloatTensor),
    )


    return dataset

def load_dataset_TOM(dataset):
    # Load the dataset using pickle
    # it is expected that dataset[0] is the array of integers (n < 2 ** 30)! (not of spin configurations)
    dataset = tuple(
        torch.from_numpy(x) for x in _with_file_like(dataset, "rb", pickle.load)
    )
    
    # Pre-processing
    dataset = (
        dataset[0],
        torch.where(dataset[1] >= 0, torch.tensor([0]), torch.tensor([1])).squeeze(),
        (torch.abs(dataset[1]) ** 2).squeeze(),
    )
    return dataset

def try_one_dataset(dataset, output, Net, number_runs, train_options, rt = 0.02, lr = 0.0003, gpu = False, sampling = "uniform"):
    global number_spins
    weights = None

    class Loss(object):
        def __init__(self, train_type = 'phase'):
            if train_type == 'phase':
                self._fn = torch.nn.CrossEntropyLoss(reduction = 'none')
            else:
                self._fn = torch.nn.MSELoss()
            self.type = train_type

        def __call__(self, predicted, expected, weight, apply_weights_loss = False):
            if self.type == 'phase':
                if not apply_weights_loss:
                    return torch.mean(self._fn(predicted, expected))
                return torch.sum(self._fn(predicted, expected) * weight)
            else:
                return self._fn(predicted, torch.log(weight))

    loss_fn = Loss(train_type = train_options["type"])
    train_options = deepcopy(train_options)
    train_options["loss"] = loss_fn
    if train_options["type"] == "phase":
        train_options["accuracy"] = accuracy
    print(train_options["optimiser"][:-1] + str(', lr = ') + str(lr) + ')')
    train_options["optimiser"] = eval(train_options["optimiser"][:-1] + str(', lr = ') + str(lr) + ')')

    stats = []
    rest_overlaps = []
    for i in range(number_runs):
        module = Net(number_spins)
        train_set, test_set, rest_set = split_dataset(
            dataset, [rt, rt * 0.1], sampling = sampling
        )
        print("splitted DS", flush = True)
        module, train_history, test_history = train(
            module, train_set, test_set, gpu, lr, **train_options
        )
        print("finished training", flush = True)
        resampled_set, _, _ = split_dataset(
            dataset, [rt, rt * 0.2], sampling = sampling
        )

        if gpu:
            module = module.cuda()

        rest_set_amplitudes = rest_set[2] / torch.sum(rest_set[2])
        if sampling == "quadratic":
            rest_set = (rest_set[0], rest_set[1], rest_set[2] / torch.sum(rest_set[2]))
        else:
            rest_set = (rest_set[0], rest_set[1], rest_set[2] * 0.0 + 1.0 / rest_set[2].size()[0])
        
        with torch.no_grad():
            predicted_rest = predict_large_data(module, rest_set[0], gpu, train_options["type"])
            predicted_resampled = predict_large_data(module, resampled_set[0], gpu, train_options["type"])

            rest_loss = loss_fn(predicted_rest, rest_set[1], rest_set[2], apply_weights_loss = True).item()
            rest_accuracy = accuracy(predicted_rest, rest_set[1], rest_set_amplitudes, apply_weights_loss = True)
            resampled_loss = loss_fn(predicted_resampled, resampled_set[1], resampled_set[2]).item()
            resampled_acc = accuracy(predicted_resampled, resampled_set[1], resampled_set[2])
        
            best_overlap = overlap(train_options["type"], module, *dataset, gpu)
            print('total dataset overlap = ' + str(best_overlap) + ', rest dataset accuracy = ' + str(rest_accuracy))
            print('resampled loss =  = ' + str(resampled_loss) + ', resampled dataset accuracy = ' + str(resampled_acc))

            rest_overlap = overlap(train_options["type"], module, rest_set[0], rest_set[1], rest_set_amplitudes, gpu)
            rest_overlaps.append(rest_overlap)
            print('rest dataset overlap = ' + str(rest_overlap))

        if gpu:
            module = module.cpu()
            if sampling != 'uniform':
                dataset = (dataset[0].cpu(), dataset[1], dataset[2])
        best = min(test_history, key=lambda t: t[2])
        best_train = min(train_history, key=lambda t: t[2])
        train_unique = torch.sum(torch.unique(train_set[2])).item()
        train_nonunique = torch.sum(train_set[2]).item()
        stats.append((*best[2:], *best_train[2:], rest_loss, rest_accuracy, resampled_loss, resampled_acc, best_overlap, rest_overlap, train_unique, train_nonunique))
        print("train_unique = {:.10e}, train_nonunique = {:.10e}".format(train_unique, train_nonunique))
        folder = os.path.join(output, str(i + 1))
        os.makedirs(folder, exist_ok=True)
        print("test_acc = {:.10e}, train_acc = {:.10e}, rest_acc = {:.10e}, resampled_acc = {:.10e}".format(best[3], best_train[3], rest_accuracy, resampled_acc))
        print("test_loss = {:.10e}, train_loss = {:.10e}, rest_loss = {:.10e}, resampled_loss = {:.10e}".format(best[2], best_train[2], rest_loss, resampled_loss))
        torch.save(module.state_dict(), os.path.join(folder, "state_dict.pickle"))
        # np.savetxt(os.path.join(folder, "train_history.dat"), np.array(train_history))
        # np.savetxt(os.path.join(folder, "test_history.dat"), np.array(test_history))

    #  Andrey asked to check the total expressibility of the model and also plot it
    '''
    module = Net(dataset[0].size(1))
    train_options['patience'] *= 5
    module, train_history, test_history = train(
            module, dataset, dataset, gpu, lr, **train_options
    )
    '''
    best_expression = min(train_history, key=lambda t: t[2])
    rest_overlaps = np.array(rest_overlaps) 
    # stats = np.array(stats)
    # if len(rest_overlaps) == np.sum(rest_overlaps < 0.03):
    #     best_runs_ids = np.arange(len(rest_overlaps))
    # else:
    #     best_runs_ids = np.where(rest_overlaps > 0.03)[0]
    # best_runs_ids = np.argsort(-np.array(rest_overlaps))[:number_best]
    #stats = stats[best_runs_ids, ...]
    np.savetxt(os.path.join(output, "loss.dat"), stats)
    return np.concatenate([np.vstack((np.mean(stats, axis=0), np.std(stats, axis=0))).T.reshape(-1), np.array([*best_expression[2:]])], axis = 0)


def main():
    global number_spins
    if not len(sys.argv) == 2:
        print(
            "Usage: python3 {} <path-to-json-config>".format(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    config = _with_file_like(sys.argv[1], "r", json.load)
    output = config["output"]
    number_spins = get_number_spins(config)
    number_runs = config["number_runs"]
    gpu = config["gpu"]
    sampling = config["sampling"]
    lrs = config.get("lr")
    if number_spins != 24:
        j2_list = config.get("j2")
        j2_iter = j2_list
    else:
        system_folder = config["system"]
        info = get_info(system_folder, config.get("j2"))
        j2_iter = info
    Net = import_network(config["model"])
    
    

    if config["use_jit"]:
        _dummy_copy_of_Net = Net
        Net = lambda n: torch.jit.trace(
            _dummy_copy_of_Net(n), torch.rand(config["training"]["batch_size"], n)
        )

    os.makedirs(output, exist_ok=True)
    results_filename = os.path.join(output, "results.dat")
    copyfile(sys.argv[1], os.path.join(output, "config.dat"))  # copy config file to the same location where the results.txt file is
    if os.path.isfile(results_filename):
        results_file = open(results_filename, "a")
    else:
        results_file = open(results_filename, "w")

    results_file.write("# process with pid = " + str(os.getpid()) + ', launched at ' + str(datetime.datetime.now()) + '\n')
    results_file.write(
            "# <j2> <train_ratio> <test_loss> <test_loss_err> <test_accuracy> "
            "<test_accuracy_err> "
            "<train_loss> <train_loss_err> <train_accuracy> <train_accuracy_err> "
            "<rest_loss> <rest_loss_err> "
            "<rest_accuracy> <rest_accuracy_err> "
            "<resampled_loss> <resampled_loss_err> "
            "<resampled_accuracy> <resampled_accuracy_err>"
            " <total_overlap> <total_overlap_err> <rest_overlap> <rest_overlap_err> <tr_unique> <tr_unique_err> <tr_nonunique> <tr_nonunique_err> <total_expr> <total_acc> \n")
    results_file.flush()

    
    for j2_i, lr in zip(j2_iter, lrs):
        for rt in config.get("train_fractions"):
            if number_spins == 24:
                j2 = j2_i["j2"]
                dataset = load_dataset_TOM(j2_i["dataset"])
            else:
                j2 = j2_i
                dataset = load_dataset_large(os.path.join(config['system'] + '/' + str(j2) + '/output/zvo_eigenvec_0_rank_0.dat'))

            local_output = os.path.join(output, "j2={}rt={}".format(j2, rt))
            os.makedirs(local_output, exist_ok=True)
            local_result = try_one_dataset(
                dataset, local_output, Net, number_runs, config["training"], rt = rt, lr = lr, gpu = gpu, sampling = sampling
            )
            # continue   ### DEBUG ###
            with open(results_filename, "a") as results_file:
                results_file.write(
                        ("{:.3f} {:.7f}" + " {:.10e}" * 26 + "\n").format(j2, rt, *tuple(local_result))
                )
                results_file.flush()
    return


if __name__ == "__main__":
    main()
