import os
import sys
import numpy as np
import torch


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
    f = open(name, 'rb')

    all_spins = np.zeros((0, number_spins))
    signs = []

    for line in f:
    	spin, re, im = line.split('\t')
    	spin = np.array(list(spin)) * 2 - 1
    	all_spins = np.concatenate([all_spins, spin], axis = 0)
    	signs.append(np.sign(float(re)))

    return torch.from_numpy(all_spins), (torch.from_numpy(signs)).type(torch.LongTensor)


def train_phase(ψ: torch.nn.Module, samples_train, target_train, config, samples_val, target_val, GPU = False):
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
    logging.info("Initial accuracy train : {:.2f}%".format(100 * accuracy(True)))
    logging.info("Initial accuracy val : {:.2f}%".format(100 * accuracy(False)))


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

    logging.info("Final accuracy train : {:.2f}%".format(th))
    logging.info("Final accuracy val : {:.2f}%".format(vh))

    finish = time.time()
    logging.info("Done in {:.2f} seconds!".format(finish - start))
    if GPU:
        ψ = ψ.cpu()
    return ψ, [th], [vh]




#################### parameters setting here ###########################
n_trials = 10 # how many times we run learning process from the beginning
parameters_set = [] # set possible values of J2 or any other tunable frustration parameter
hamiltonians = []
exact_ground_states_signs = []

log_file = open('J1J2_log.dat', 'w') # specify logfile 

# other options
number_spins = 25
magnetization = number_spins % 2
samples = all_spins(number_spins, magnetization)


for parameter in parameters_set:
	hamiltonian_path_name = "./hamiltonians/" + str(parameter) + '.hamiltonian' # specify how the name is obtained
	hamiltonians.append(read_hamiltonian(hamiltonian_path_name).to_cxx())

	exact_gs_vector_name = "./vectors/" + str(parameter) + '.vector' # specify how the vector name is obtained
	exact_ground_states_signs.append(load_ground_state(samples, exact_gs_vector_name))


PhaseNet = import_network("../../nqs-playground/nqs_playground/extra_files/phase.py") # explicit path to model.py file
optimiser = lambda p: torch.optim.Adam(p.parameters(), lr=0.003)
epochs = 100
batch_size = 512
loss = torch.nn.CrossEntropyLoss()
train_ratio = 0.01 # train on 1 % of the total number of spin configurations

################### parameters setting ended #########################





options_dict = {"optimiser": optimiser, "batch_size": batch_size, "loss": loss}

for parameter in parameters_set:
	log_file.write(str(parameter) + ' : ')
    for trial_number in range(n_trials):
        ψ_phase = PhaseNet(number_spins)

        idxs = np.arange(len(samples))
        np.random.shuffle(idxs)
        train_idxs = idxs[0:int(train_ratio * len(samples))]
        val_idxs = idxs[int(train_ratio * len(samples)):]
        
        for i in range(epochs):
            ψ_phase, th, vh = train_phase(ψ_phase, samples[train_idxs], phase_signs[train_idxs], 
                                          options_dict, samples[val_idxs], phase_signs[val_idxs], GPU = True)
            train_history = train_history + th
            val_history = val_history + vh
        log_file.write(str(train_history[-1]) + '/')
        log_file.write(str(val_history[-1]) + ' ')
        log_file.flush()
        print("For parameter = " + str(parameters) + ", iteration " + str(i) + " train accuracy: " + str(th[0]) + ", val accuracy " + str(vh[0]))

