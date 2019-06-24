from __future__ import print_function, division
import sys,os
import scipy.io as sio
from quspin.operators import hamiltonian, exp_op
from quspin.basis import spin_basis_general
import numpy as np
from optparse import OptionParser
import sys
import pickle

dir = "ed/"

parser = OptionParser()
parser.add_option("--J1", dest = "J1", default = "1", type = "float")
parser.add_option("--J2", dest = "J2", default = "0.6", type = "float")
parser.add_option("--Lx", dest = "Lx", default = "4", type = "int")
parser.add_option("--Ly", dest = "Ly", default = "4", type = "int")
parser.add_option("--Ex", dest = "Ex", default = "0", type = "int")
parser.add_option("--Ey", dest = "Ey", default = "0", type = "int")
parser.add_option("-k", dest = "k", default = "1", type = "int")
parser.add_option("-d", dest="d", default = False, action="store_true")
parser.add_option("-b", dest="b", default = False, action="store_true")

(options, args) = parser.parse_args(sys.argv[1:])
Lx = options.Lx; Ly = options.Ly
J1 = options.J1; J2 = options.J2
Ex = options.Ex; Ey = options.Ey
k = options.k; d = options.d; b = options.b

fname_states = "_".join(["AFM","states",str(J1),str(J2),str(Lx),str(Ly)])
fname_basis = "_".join(["AFM","basis",str(Lx),str(Ly)])

N = Lx*Ly
s = np.arange(N)
x = s%Lx; y = s//Lx
T_x = (x+1)%Lx + Lx*y
T_mx = (x-1)%Lx + Lx*y
T_y = x +Lx*((y+1)%Ly)
T_my = x +Lx*((y-1)%Ly)
P_x = x + Lx*(Ly-y-1)
P_y = (Lx-x-1) + Lx*y
Z   = -(s+1)

if b:
    print(fname_basis)
    print("Generating the basis")
    basis = spin_basis_general(N,kxblock=(T_x,0), kyblock=(T_y,0), zblock=(Z,0), pxblock=(P_x,Ex), pyblock=(P_y,Ey), pauli=0, Nup = N//2)
    print("Size of H-space: {Ns:d}".format(Ns=basis.Ns))
    with open(dir+fname_basis, 'wb') as f:
        pickle.dump(basis, f, pickle.HIGHEST_PROTOCOL)
else:
    with open(dir+fname_basis, 'rb') as f:
        basis = pickle.load(f)
    
if d:
    print(fname_states)
    print("Diagonalization")
    J1_2d = [[J1,i,T_x[i]] for i in range(N)]+[[J1,i,T_y[i]] for i in range(N)]
    J2_2d = [[J2,i,T_y[T_x[i]]] for i in range(N)]+[[J2,i,T_my[T_x[i]]] for i in range(N)]
    conn = [["xx",J1_2d],["yy",J1_2d],["zz",J1_2d],["xx",J2_2d],["yy",J2_2d],["zz",J2_2d]]
    H = hamiltonian(conn,[],basis=basis,dtype=np.float64)
    E, psi = H.eigsh(k=k,which="SA")        
    sio.savemat(dir+fname_states,{"E": E,"psi": psi})