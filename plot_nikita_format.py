import matplotlib.pyplot as plt
import numpy as np
import sys


J2 = []
ntrials = 7
val = np.zeros((0, ntrials))
train = np.zeros((0, ntrials))
for line in open('plotlog2.txt', 'r'):
    v = np.zeros((1, ntrials))
    t = np.zeros((1, ntrials))
    val = np.concatenate([val, v], axis = 0)
    train = np.concatenate([train, t], axis = 0)

    j2, rest = line.split(':')
    J2.append(float(j2))
    pairs = rest.split(' ')
    for idx, pair in enumerate(pairs[1:-3]):
        t, v = pair.split('/')
        val[-1, idx] = float(v)
        train[-1, idx] = float(t)
    print(train[-1])

J2 = np.array(J2)

plt.rc('text', usetex = True)
plt.rc('font', family = 'TeX Gyre Adventor', size = 14)
plt.rc("pdf", fonttype=42)

plt.errorbar(J2 + 0.005, train.mean(axis = 1), train.std(axis = 1), fmt = 'o', markersize = 5,
						 mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
						 label = 'train, 100%', color = 'red')

plt.errorbar(J2, val.mean(axis = 1), val.std(axis = 1), fmt = 'o', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'validation, 100%', color = 'blue')

J2 = []
ntrials = 3
val = np.zeros((0, ntrials))
train = np.zeros((0, ntrials))
for line in open('plotlog.txt', 'r'):
    v = np.zeros((1, ntrials))
    t = np.zeros((1, ntrials))
    val = np.concatenate([val, v], axis = 0)
    train = np.concatenate([train, t], axis = 0)

    j2, rest = line.split(':')
    J2.append(float(j2))
    pairs = rest.split(' ')
    for idx, pair in enumerate(pairs[1:]):
        t, v = pair.split('/')
        val[-1, idx] = float(v)
        train[-1, idx] = float(t)
    print(train[-1])
J2 = np.array(J2)

plt.errorbar(J2 + 0.005, train.mean(axis = 1), train.std(axis = 1), fmt = '*', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'train, 50%', color = 'red')

plt.errorbar(J2, val.mean(axis = 1), val.std(axis = 1), fmt = '*', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'validation, 50%', color = 'blue')

plt.xlabel('$J_2$')
plt.ylabel('accuracy, \\%')
plt.legend(loc = 'lower left', ncol = 1, numpoints = 1, fontsize = 16)
plt.grid(True)
plt.show()
