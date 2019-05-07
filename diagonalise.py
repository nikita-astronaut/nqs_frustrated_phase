#!/usr/bin/env python3

# Copyright Tom Westerhout (c) 2019
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy.sparse


class _Hamiltonian(object):
    def __init__(self, matrix, g_to_l, l_to_g):
        self.matrix = matrix
        self.g_to_l = g_to_l
        self.l_to_g = l_to_g


def get_entries(n, edges):
    """
    :param n:     Number of spins.
    :param edges: Edges of the graph.
    """
    empty_value = 2 ** 63 - 1
    m = n % 2
    number_ups = (n + m) // 2

    def get_bit(x: int, i: int) -> int:
        """
        Returns the ``i``'th most significant bit.

        :param x: Our spin configuration interpreted as an ``int``.
        :param i: Index of the spin.
        """
        return (x >> (n - 1 - i)) & 1

    def get_flipped(x: int, i: int, j: int) -> int:
        """
        Returns the spin configuration with ``i``'th and ``j``'th spins
        **flipped** (not swapped).

        :param x: Original spin configuration.
        :param i: Index of the spin to flip.
        :param j: Index of another spin to flip.
        """
        x ^= 1 << (n - 1 - i)
        x ^= 1 << (n - 1 - j)
        return x

    def get_row(i: int):
        """
        Returns the sparse representation of the ``i``'th row of the
        Hamiltonian. The row is a list of pairs (cⱼ, j), where cⱼ's are the matrix
        elements ⟨i|H|j⟩'s.
        """
        c = 0.0
        for J, edge in edges:

            aligned = get_bit(i, edge[0]) == get_bit(i, edge[1])
            c += J * (-1.0 + 2.0 * int(aligned))
            if not aligned:
                yield (2.0, get_flipped(i, *edge))
        if c != 0.0:
            yield (c, i)

    g_to_l = np.empty(2 ** n, dtype=np.int64)
    g_to_l[:] = empty_value
    l_to_g = []
    entries = []
    k = 0
    for i in range(2 ** n):
        if bin(i).count("1") == number_ups:
            g_to_l[i] = k
            l_to_g.append(i)
            k += 1
            for (c, j) in get_row(i):
                entries.append((c, i, j))

    def matrix_from_entries():
        data = np.fromiter(
            (c for (c, _, _) in entries), dtype=np.float64, count=len(entries)
        )
        row_ind = np.fromiter(
            (g_to_l[i] for (_, i, _) in entries), dtype=np.int64, count=len(entries)
        )
        col_ind = np.fromiter(
            (g_to_l[j] for (_, _, j) in entries), dtype=np.int64, count=len(entries)
        )
        assert not np.any(row_ind == empty_value)
        assert not np.any(col_ind == empty_value)
        return scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(len(l_to_g), len(l_to_g))
        )

    return _Hamiltonian(matrix_from_entries(), g_to_l, np.array(l_to_g))


def J1J2_Heisenberg(J2 = 0.0, nx = 5, ny = 5):
    edges = []
    for x in range(nx):
        for y in range(ny):
            edges.append((1.0, (x * ny + y, ((x + 1) % nx) * ny + (y) % ny)))
            edges.append((1.0, (x * ny + y, ((x) % nx) * ny + (y + 1) % ny)))
            edges.append((J2, (x * ny + y, ((x + 1) % nx) * ny + (y + 1) % ny)))
            edges.append((J2, (x * ny + y, ((x - 1) % nx) * ny + (y + 1) % ny)))
    return edges

edges = J1J2_Heisenberg(J2 = 0.0, nx = 5, ny = 5)

nvectors = 1
j1j2_25_H_sparse = get_entries(25, edges)

from scipy.sparse.linalg import eigsh
values, vectors = eigsh(j1j2_25_H_sparse.matrix, k = nvectors, which = 'SA')

values = (values * 1e+5).astype(np.int64) / 1e+5
values_unique = np.sort(np.unique(values))

val = values[0]
vec = vectors[:, 0]
f = open('./vectors/j1j2_0.0.txt', 'w')

for ampl, global_index in zip(vec, j1j2_25_H_sparse.l_to_g):
    f.write("{0:{fill}25b}".format(global_index, fill='0') + '\t' + "{:.10E}".format(ampl) + '\t0.0\n')

f.flush()
f.close()
