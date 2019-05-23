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

from ._core import System


def make_j1j2_graph(L_x, L_y):
    """
    Returns adjacency lists of a J1-J2 model on a square lattice with given
    size. Periodic boundary conditions are used.
    """

    def coord2index(x, y):
        return x + y * L_x

    def left(x, y):
        return (x - 1 if x > 0 else L_x - 1, y)

    def right(x, y):
        return (x + 1 if x < L_x - 1 else 0, y)

    def down(x, y):
        return (x, y - 1 if y > 0 else L_y - 1)

    def up(x, y):
        return (x, y + 1 if y < L_y - 1 else 0)

    def nearest_neighbours(p):
        site = coord2index(*p)
        return filter(
            lambda t: t[0] < t[1],
            map(
                lambda t: (site, coord2index(*t)),
                [left(*p), right(*p), down(*p), up(*p)],
            ),
        )

    def next_nearest_neighbours(p):
        site = coord2index(*p)
        return filter(
            lambda t: t[0] < t[1],
            map(
                lambda t: (site, coord2index(*t)),
                [up(*left(*p)), down(*left(*p)), up(*right(*p)), down(*right(*p))],
            ),
        )

    edges_j1 = []
    edges_j2 = []
    for i in range(L_x):
        for j in range(L_y):
            for edge in nearest_neighbours((i, j)):
                edges_j1.append(edge)
            for edge in next_nearest_neighbours((i, j)):
                edges_j2.append(edge)
    return edges_j1, edges_j2


class Square24(System):
    POSITIONS = [(i, j) for i in range(6) for j in range(4)]
    J1_EDGES, J2_EDGES = make_j1j2_graph(6, 4)
