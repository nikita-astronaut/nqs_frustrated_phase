import os
import time
import sys
import numpy as np
import torch
import importlib
import tqdm

def import_network(filename: str):
    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the network from '{}': not a python source file.".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module.Net



def load_ground_state(number_spins, name):
	f = open(name, 'r')

	all_spins = []
	signs = []
	ampls = []

	for line in tqdm.tqdm(f):
		spin, re, im = line.split('\t')
		spin = np.array(list(spin))[np.newaxis, :].astype(int) * 2 - 1
		all_spins.append(spin)
		signs.append(np.sign(float(re)) * 0.5 + 0.5)
		ampls.append(np.abs(float(re)))

	all_spins = np.array(all_spins)[:, 0, :]
	return torch.from_numpy(all_spins).type(torch.FloatTensor), (torch.from_numpy(np.array(signs))).type(torch.LongTensor), np.array(ampls)


def train_phase(ψ: torch.nn.Module, samples_train, target_train, config, samples_val, target_val, GPU = False, validate = False):
    start = time.time()

    if GPU:
        ψ = ψ.cuda()

    batch_size = config["batch_size"]
    optimiser = config["optimiser"](ψ)
    loss_fn = config["loss"]

    if GPU:
        samples_train = samples_train.cuda()
        target_train = target_train.cuda()
        samples_val = samples_val.cuda()
        target_val = target_val.cuda()

    def accuracy(train):
        with torch.no_grad():
            if train:
                _, predicted = torch.max(ψ(samples_train), dim=1)
                return float(torch.sum(target_train == predicted)) / target_train.size(0)
            _, predicted = torch.max(ψ(samples_val), dim=1)
            return float(torch.sum(target_val == predicted)) / target_val.size(0)
    val_losses = []
    print("Initial accuracy train : {:.2f}%".format(100 * accuracy(True)))
    print("Initial accuracy val : {:.2f}%".format(100 * accuracy(False)))


    idxs = np.arange(len(samples_train))
    np.random.shuffle(idxs)

    losses = []
    for j in range(len(samples_train) // batch_size):
        samples = samples_train[j * batch_size:j * batch_size + batch_size]
        target = target_train[j * batch_size:j * batch_size + batch_size]

        optimiser.zero_grad()
        loss = loss_fn(ψ(samples), target)
        losses.append(loss.item())
        loss.backward()
        optimiser.step()
        del samples
        del target
        del loss

    del idxs
        
    th = 100 * accuracy(True)
    vh = 100 * accuracy(False)

    print("Final accuracy train : {:.2f}%".format(th))
    print("Final accuracy val : {:.2f}%".format(vh))

    finish = time.time()
    print("Done in {:.2f} seconds!".format(finish - start))
    if GPU:
        ψ = ψ.cpu()
    return ψ, [th], [vh]




#################### parameters setting here ###########################
n_trials = 1 # how many times we run learning process from the beginning
parameters_set = ['0.55']#, '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8']#, '0.85', '0.9', '0.95', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0'] # set possible values of J2 or any other tunable frustration parameter
# parameters_set = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
# parameters_set = ['2.0', '3.0', '4.0', '5.0', '6.0', '7.0']
# parameters_set = ['0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
# parameters_set = ['0.5', '0.8']
exact_ground_states_signs = []

importance_ratio = 0.2

train_ratios = [0.02 * 1.0]

log_file = open('./logs/J1J2_log_square_importance_0.20.dat', 'w') # specify logfile 

# other options
number_spins = 24
magnetization = number_spins % 2

for parameter in parameters_set:
	exact_gs_vector_name = "./4x6.exact/" + parameter + '_vector_0_0.txt' # specify how the vector name is obtained
	exact_ground_states_signs.append(load_ground_state(number_spins, exact_gs_vector_name))


PhaseNet = import_network("./models/J1J2_model.py") # explicit path to model.py file
optimiser = lambda p: torch.optim.Adam(p.parameters(), lr=0.003)
epochs = 20000
epoch_split = 1
batch_size = 2 ** 10
loss = torch.nn.CrossEntropyLoss()
# train_ratio = 0.02 # train on 1 % of the total number of spin configurations
val_ratio = 0.1 * 1.0

################### parameters setting ended #########################


options_dict = {"optimiser": optimiser, "batch_size": batch_size, "loss": loss}

for parameter, gs in zip(parameters_set, exact_ground_states_signs):
    for train_ratio in train_ratios:
        log_file.write(str(parameter) + ' : ' + str(train_ratio) + ' : ')
        for trial_number in range(n_trials):
            samples = gs[0]
            phase_signs = gs[1]
            ampls = gs[2]
            
            idxs = np.random.choice(np.arange(len(samples)), size = len(samples), replace=False, p=ampls ** 2 / np.sum(ampls ** 2))
            samples_trial = samples[idxs]
            phase_signs_trial = phase_signs[idxs]

            ψ_phase = PhaseNet(number_spins)

            idxs = np.arange(len(samples_trial))
            np.random.shuffle(idxs)
            train_idxs = idxs[0:int(train_ratio * len(samples_trial))]
            val_idxs = idxs[int(train_ratio * len(samples_trial)):int(train_ratio * len(samples_trial) + val_ratio * len(samples_trial))]
            train_history = []
            val_history = []

            for i in range(epochs):
                print("epoch number: " + str(i))
                np.random.shuffle(train_idxs)
                ψ_phase, th, vh = train_phase(ψ_phase, samples_trial[train_idxs[:len(train_idxs) // epoch_split]], phase_signs_trial[train_idxs[:len(train_idxs) // epoch_split]], options_dict, samples_trial[val_idxs], phase_signs_trial[val_idxs], GPU = True)
                train_history = train_history + th
                val_history = val_history + vh
                # print("For parameter = " + str(parameter) + ", iteration " + str(i) + " train accuracy: " + str(th[0]) + ", val accuracy " + str(vh[0]))
            log_file.write(str(train_history[-1]) + '/')
            log_file.write(str(val_history[-1]) + ' ')
            log_file.flush()
            print("For parameter = " + str(parameter) + ", train ratio = " + str(train_ratio) + ", iteration " + str(i) + " train accuracy: " + str(th[0]) + ", val accuracy " + str(vh[0]))
        log_file.write('\n')
