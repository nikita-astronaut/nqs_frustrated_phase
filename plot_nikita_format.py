import matplotlib.pyplot as plt
import numpy as np
import sys


J2 = []
val = np.zeros((0, int(sys.argv[2])))
train = np.zeros((0, int(sys.argv[2])))
for line in open(sys.argv[1], 'r'):
    v = np.zeros((1, int(sys.argv[2])))
    t = np.zeros((1, int(sys.argv[2])))
    val = np.concatenate([val, v], axis = 0)
    train = np.concatenate([train, t], axis = 0)

    j2, rest = line.split(':')
    J2.append(float(j2))
    pairs = line.split(' ')
    for idx, pair in enumerate(pairs[1:-1]):
        t, v = pair.split('/')
        val[-1, idx] = float(v)
        train[-1, idx] = float(t)

J2 = np.array(J2)

plt.rc('text', usetex = True)
plt.rc('font', family = 'TeX Gyre Adventor', size = 14)
plt.rc("pdf", fonttype=42)

plt.errorbar(J2 + 0.005, train.mean(axis = 1), train.std(axis = 1), fmt = 'o', markersize = 5,
						 mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
						 label = 'train', color = 'red')

plt.errorbar(J2, val.mean(axis = 1), val.std(axis = 1), fmt = 'o', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'validation', color = 'blue')

plt.xlabel('$J_2$')
plt.ylabel('accuracy, \\%')
plt.legend(loc = 'lower left', ncol = 1, numpoints = 1, fontsize = 16)
plt.grid(True)
plt.show()
