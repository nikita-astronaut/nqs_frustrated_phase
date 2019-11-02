import pickle
import quspin
import numpy as np
import scipy.io as sio

dataset_name = '/zfs/hybrilit.jinr.ru/user/a/astrakh/nqs_frustrated_phase/data/square/36/1.0/AFM--states--1.0--1.0--6--6.mat'
print(dataset_name, flush = True)
psi = sio.loadmat(dataset_name)['psi'][:, 0]
dataset_name = dataset_name.split("--");
dataset_name[1] = "basis";
basis = "--".join(dataset_name[:2]+dataset_name[-2:])[:-4]

with open(basis, "rb") as f:
    basis = pickle.load(f)
'''
dataset_name[1] = "dump_vector"
vector = "--".join(dataset_name[:2]+dataset_name[-2:])[:-4]
phi = np.load(vector + '.npy')
'''
def sample(basis, repr, repr_ix, psi, n):
    repr_sampled = basis.states[np.random.choice(len(psi), p=psi**2, size=n)]
    res = np.zeros(n, dtype = np.int64); i = 0
    for i, r in enumerate(repr_sampled):
        res[i] = repr_ix[np.random.randint(np.searchsorted(repr, r, 'left'), np.searchsorted(repr, r, 'right'))]
    return res

from quspin.basis import spin_basis_general
fullbasis = spin_basis_general(basis.N, pauli=0, Nup = basis.N//2)
dataset_name[1] = "fullbasisstates";
fullbasisstates_name = "--".join(dataset_name[:2]+dataset_name[-2:])[:-4]
np.save(fullbasisstates_name, fullbasis.states)
# exit(-1)
repr = basis.representative(fullbasis.states)
repr_ix = np.argsort(repr)
repr = repr[repr_ix]
dataset_name[1] = "repr_ix";
repr_ix_name = "--".join(dataset_name[:2]+dataset_name[-2:])[:-4]
np.save(repr_ix_name, repr_ix)
dataset_name[1] = "repr";
repr_name = "--".join(dataset_name[:2]+dataset_name[-2:])[:-4]
np.save(repr_name, repr)

res = sample(basis, repr, repr_ix, psi, 10**5)
print(np.mean(phi[res]**2))
print(np.sum(phi**4))
