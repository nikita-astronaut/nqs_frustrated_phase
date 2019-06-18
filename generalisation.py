#!/usr/bin/env python3

from copy import deepcopy

# import cProfile
import json
import math
import os
import pickle
import re
import sys
import tempfile
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.utils.data


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
        weights = dataset[2] / torch.sum(dataset[2])
    else:
        print("sampling with uniform weights")

    if weights is None:
        indices = torch.randperm(n)
    else:
        indices = np.random.choice(np.arange(n), size = int(n * sum(fractions)), replace=False, p=weights.numpy()[:,0])
        indices = torch.from_numpy(np.concatenate([indices, np.setdiff1d(np.arange(n), indices)], axis = 0))

    sets = []
    for first, last in parts(map(lambda x: int(round(x * n)), fractions)):
        sets.append(tuple(x[indices[first:last]] for x in dataset))
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
    if gpu:
        test_x, test_y, test_weight = test_x.cuda(), test_y.cuda(), test_weight.cuda()
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
        update_count = 0
        for epoch_index in range(epochs):
            important = epoch_index in checkpoints
            if important:
                losses = []
                accuracies = []
            for batch_index, (batch_x, batch_y, batch_weight) in enumerate(dataloader):
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
                        predicted = ψ(test_x)
                        loss = loss_fn(predicted, test_y, test_weight).item()
                        accuracy = accuracy_fn(predicted, test_y, test_weight)
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


def accuracy(predicted, expected, weight, apply_weights_loss = False):
    predicted = torch.max(predicted, dim=1)[1]
    if not apply_weights_loss:
        return torch.sum(predicted == expected).item() / float(expected.size(0))
    agreement = predicted == expected
    return torch.sum(agreement.type(torch.FloatTensor) * torch.tensor(weight, dtype = torch.float32)[:, 0] ).item() / float(expected.size(0))
 
def overlap(ψ, samples, target, weights, gpu):
    if gpu:
        ψ = ψ.cuda()
        samples = samples.cuda()
    BATCH_SIZE = 1024
    overlap = 0.0
    target = 2.0 * target.type(torch.FloatTensor) - 1.0
    for i in range(int(samples.size()[0] / BATCH_SIZE) + 1):
        idx_min = i * BATCH_SIZE
        idx_max = (i + 1) * BATCH_SIZE
        if idx_max > len(samples):
            idx_max = len(samples)
        predicted_signs = torch.max(ψ(samples[idx_min:idx_max]), dim=1)[1].cpu().type(torch.FloatTensor)
        predicted_signs = 2.0 * predicted_signs.type(torch.FloatTensor) - 1.0
        overlap += torch.sum(predicted_signs.type(torch.FloatTensor) * target[idx_min:idx_max].type(torch.FloatTensor) * weights[idx_min:idx_max].type(torch.FloatTensor)).item() / (idx_max - idx_min)
    if gpu:
        ψ = ψ.cpu()
        samples = samples.cpu()

    return overlap / torch.sum(weights).item()

def try_one_dataset(dataset, output, Net, number_runs, train_options, rt = 0.02, lr = 0.0003, gpu = False, sampling = "uniform"):
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

    weights = None

    class Loss(object):
        def __init__(self):
            self._fn = torch.nn.CrossEntropyLoss(reduction = 'none')

        def __call__(self, predicted, expected, weight, apply_weights = False):
            if not apply_weights:
                return torch.sum(self._fn(predicted, expected))
            return torch.sum(self._fn(predicted, expected) * weight)

    loss_fn = Loss()
    train_options = deepcopy(train_options)
    train_options["loss"] = loss_fn
    train_options["accuracy"] = accuracy
    train_options["optimiser"] = eval(train_options["optimiser"][:-1] + str(', lr = ') + str(lr) + ')')

    stats = []
    for i in range(number_runs):
        module = Net(dataset[0].size(1))
        train_set, test_set, rest_set = split_dataset(
            dataset, [rt, train_options["test_fraction"]], sampling = sampling
        )
        module, train_history, test_history = train(
            module, train_set, test_set, gpu, lr, **train_options
        )
        if gpu:
            module = module.cuda()
            rest_set = (rest_set[0].cuda(), rest_set[1], rest_set[2])
        predicted = torch.zeros([0, 2], dtype=torch.float32)
        with torch.no_grad():
            for idxs in np.split(np.arange(rest_set[0].size()[0]), np.arange(0, rest_set[0].size()[0], 10000))[1:]:
                predicted_local = module(rest_set[0][idxs]).cpu()
                predicted = torch.cat((predicted, predicted_local), dim = 0)
            rest_loss = loss_fn(predicted, *rest_set[1:], ).item()
            rest_accuracy = accuracy(predicted, *rest_set[1:], apply_weights_loss = True)  # rest accuracy and loss are computed with 
        best_overlap = overlap(module, *dataset, gpu)
        if gpu:
            module = module.cpu()

        best = min(test_history, key=lambda t: t[2])
        best_train = min(train_history, key=lambda t: t[2])
        stats.append((*best[2:], *best_train[2:], rest_loss, rest_accuracy, best_overlap))

        folder = os.path.join(output, str(i + 1))
        os.makedirs(folder, exist_ok=True)
        torch.save(module.state_dict(), os.path.join(folder, "state_dict.pickle"))
        np.savetxt(os.path.join(folder, "train_history.dat"), np.array(train_history))
        np.savetxt(os.path.join(folder, "test_history.dat"), np.array(test_history))

    #  Andrey asked to check the total expressibility of the model and also plot it
    module = Net(dataset[0].size(1))
    module, train_history, test_history = train(
            module, dataset, dataset, gpu, lr, **train_options
    )
    best_expression = min(train_history, key=lambda t: t[2])

    stats = np.array(stats)
    np.savetxt(os.path.join(output, "loss.dat"), stats)
    return np.concatenate([np.vstack((np.mean(stats, axis=0), np.std(stats, axis=0))).T.reshape(-1), np.array([*best_train[2:]])], axis = 0)


def main():
    if not len(sys.argv) == 2:
        print(
            "Usage: python3 {} <path-to-json-config>".format(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    config = _with_file_like(sys.argv[1], "r", json.load)
    system_folder = config["system"]
    output = config["output"]
    number_spins = get_number_spins(config)
    number_runs = config["number_runs"]
    gpu = config["gpu"]
    sampling = config["sampling"]
    lrs = config.get("lr")
    info = get_info(system_folder, config.get("j2"))
    Net = import_network(config["model"])
    if config["use_jit"]:
        _dummy_copy_of_Net = Net
        Net = lambda n: torch.jit.trace(
            _dummy_copy_of_Net(n), torch.rand(config["training"]["batch_size"], n)
        )

    os.makedirs(output, exist_ok=True)
    results_filename = os.path.join(output, "results.dat")
    with open(results_filename, "w") as results_file:
        results_file.write(
		"# <j2> <train_ratio> <test_loss> <test_loss_err> <test_accuracy> "
            "<test_accuracy_err> "
            "<train_loss> <train_loss_err> <train_accuracy> <train_accuracy_err> "
            "<rest_loss> <rest_loss_err> "
            "<rest_accuracy> <rest_accuracy_err> <overlap> <overlap_err>\n")

    for _obj,lr in zip(info, lrs):
        for rt in config.get("train_fractions"):
            j2 = _obj["j2"]
 
            dataset = _obj["dataset"]
            local_output = os.path.join(output, "j2={}rt={}".format(j2, rt))
            os.makedirs(local_output, exist_ok=True)
            local_result = try_one_dataset(
                dataset, local_output, Net, number_runs, config["training"], rt = rt, lr = lr, gpu = gpu, sampling = sampling
            )
            with open(results_filename, "a") as results_file:
                results_file.write(
                        ("{:.3f}\t{:.3f}" + "\t{:.10e}" * 14 + "\n").format(j2, rt, *tuple(local_result))
                )
    return


if __name__ == "__main__":
    main()
