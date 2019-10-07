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

from ._core import System


if False:  # Funcitons for plotting which are now obsolete

    class Kagome(System):
        @staticmethod
        def _to_xy(a, b):
            return (1.0 * a + 0.5 * b, 0.0 * a + math.sqrt(3.0) / 2.0 * b)

        @staticmethod
        def _are_nn(pos_1, pos_2):
            decision = (
                np.linalg.norm(
                    np.array(Kagome._to_xy(*pos_1)) - np.array(Kagome._to_xy(*pos_2))
                )
                - 1.0
                < 1.0e-1
            )
            return decision

        @classmethod
        def positions_for_gnuplot(cls):
            for pos in cls.POSITIONS:
                sys.stdout.write("{}\t{}\n".format(*Kagome._to_xy(*pos)))

        @classmethod
        def solid_edges_for_gnuplot(cls, edge_type):
            edge_type = edge_type.lower()
            assert edge_type in {"j1", "j2"}, "Invalid edge_type"
            edges = {"j1": cls.J1_EDGES, "j2": cls.J2_EDGES}[edge_type]
            for (i, j) in filter(
                lambda t: Kagome._are_nn(cls.POSITIONS[t[0]], cls.POSITIONS[t[1]]),
                edges,
            ):
                sys.stdout.write(
                    "{}\t{}\n{}\t{}\n\n".format(
                        *Kagome._to_xy(*cls.POSITIONS[i]),
                        *Kagome._to_xy(*cls.POSITIONS[j])
                    )
                )

        @classmethod
        def dashed_edges_for_gnuplot(cls, edge_type):
            L_x = max(map(lambda t: t[0], cls.POSITIONS)) + 1
            L_y = max(map(lambda t: t[1], cls.POSITIONS)) + 1

            def shorten(*positions):
                def mirror(p):
                    a = p[0] if p[0] < L_x - 1 else p[0] - L_x
                    b = p[1] if p[1] < L_y - 1 else p[1] - L_y
                    return a, b

                return tuple(mirror(p) for p in positions)

            edge_type = edge_type.lower()
            assert edge_type in {"j1", "j2"}, "Invalid edge_type"
            edges = {"j1": cls.J1_EDGES, "j2": cls.J2_EDGES}[edge_type]

            for pos_1, pos_2 in map(
                lambda t: shorten(*t),
                filter(
                    lambda t: not Kagome._are_nn(*t),
                    map(lambda t: (cls.POSITIONS[t[0]], cls.POSITIONS[t[1]]), edges),
                ),
            ):
                sys.stdout.write(
                    "{}\t{}\n{}\t{}\n\n".format(
                        *Kagome._to_xy(*pos_1), *Kagome._to_xy(*pos_2)
                    )
                )

        # @classmethod
        # def diagonalise(cls, j2s, out_dir=None):
        #     import json
        #     import pickle
        #     import scipy.sparse.linalg
        #     from . import diagonalise

        #     number_of_spins = len(cls.POSITIONS)
        #     if out_dir is None:
        #         this_folder = os.path.dirname(os.path.realpath(__file__))
        #         out_dir = os.path.join(this_folder, "..", "data")
        #     model_folder = os.path.join(out_dir, "kagome", number_of_spins, "exact")
        #     H_j1 = diagonalise.make_hamiltonian(cls.J1_EDGES, number_of_spins)
        #     H_j2 = diagonalise.make_hamiltonian(cls.J2_EDGES, number_of_spins)

        #     xs = np.empty((len(H_j1.l_to_g), H_j1.n), dtype=np.float32)
        #     for i, σ in enumerate(H_j1.l_to_g):
        #         spin = "{sigma:0{n}b}".format(sigma=σ, n=number_of_spins)
        #         for k in range(number_of_spins):
        #             xs[i, k] = spin[k] == "1"
        #     xs *= 2
        #     xs -= 1

        #     os.makedirs(model_folder, exist_ok=True)

        #     info = []
        #     for j2 in j2s:
        #         j2 = round(1000 * j2) / 1000
        #         H = H_j1.matrix + j2 * H_j2.matrix
        #         E, ys = scipy.sparse.linalg.eigsh(H, k=1, which="SA")
        #         H = None
        #         E = E[0]
        #         ys = ys.astype(np.float32)
        #         dataset_file = os.path.join(model_folder, "dataset_{:04d}.pickle".format(int(round(1000 * j2))))
        #         with open(dataset_file, "wb") as out:
        #             pickle.dump((xs, ys), out)
        #         info.append({"j2": j2, "energy": E, "dataset": dataset_file})

        #     if os.path.exists(os.path.join(model_folder, "info.json")):
        #         with open(os.path.join(model_folder, "info.json"), "r") as input:
        #             for _obj in json.load(input):
        #                 j2 = _obj["j2"]
        #                 if j2 not in (x["j2"] for x in info):
        #                     info.append(_obj)

        #     info = sorted(info, key=lambda x: x["j2"])
        #     with open(os.path.join(model_folder, "info.json"), "w") as out:
        #         json.dump(info, out)


class Kagome18(System):

    POSITIONS = [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (3, 0),
        (2, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (2, 2),
        (3, 2),
        (2, 3),
        (0, 4),
        (1, 4),
        (0, 5),
        (2, 4),
        (3, 4),
        (2, 5),
    ]

    J1_EDGES = [
        (0, 1),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 17),
        (2, 10),
        (3, 4),
        (4, 5),
        (4, 14),
        (5, 7),
        (6, 7),
        (6, 10),
        (7, 8),
        (7, 9),
        (8, 16),
        (9, 10),
        (10, 11),
        (11, 13),
        (12, 13),
        (12, 16),
        (13, 14),
        (13, 15),
        (15, 16),
        (16, 17),
    ]

    J2_EDGES = [
        (0, 2),
        (0, 14),
        (2, 6),
        (3, 17),
        (3, 5),
        (5, 9),
        (6, 8),
        (8, 12),
        (9, 11),
        (11, 15),
        (12, 14),
        (15, 17),
    ]


class Kagome24(System):

    POSITIONS = [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (3, 0),
        (2, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (2, 2),
        (3, 2),
        (2, 3),
        (0, 4),
        (1, 4),
        (0, 5),
        (2, 4),
        (3, 4),
        (2, 5),
        (0, 6),
        (1, 6),
        (0, 7),
        (2, 6),
        (3, 6),
        (2, 7),
    ]

    J1_EDGES = [
        (0, 1),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 23),
        (2, 10),
        (3, 4),
        (4, 5),
        (4, 20),
        (5, 7),
        (6, 7),
        (6, 10),
        (7, 8),
        (7, 9),
        (8, 16),
        (9, 10),
        (10, 11),
        (11, 13),
        (12, 13),
        (12, 16),
        (13, 14),
        (13, 15),
        (14, 22),
        (15, 16),
        (16, 17),
        (17, 19),
        (18, 19),
        (18, 22),
        (19, 20),
        (19, 21),
        (21, 22),
        (22, 23),
    ]

    J2_EDGES = [
        (0, 2),
        (0, 20),
        (2, 6),
        (3, 23),
        (3, 5),
        (5, 9),
        (6, 8),
        (8, 12),
        (9, 11),
        (11, 15),
        (12, 14),
        (14, 18),
        (15, 17),
        (17, 21),
        (18, 20),
        (21, 23),
    ]

class Kagome30(System):
    POSITIONS = [(0, 0) for _ in range(30)]

    J1_EDGES = [
        (1, 2),
        (4, 6),
        (5, 7),
        (6, 9),
        (1, 11),
        (2, 5),
        (3, 5),
        (4, 27),
        (5, 8),
        (6, 10),
        (7, 12),
        (8, 13),
        (9, 13),
        (10, 14),
        (1, 22),
        (3, 27),
        (6, 29),
        (1, 30),
        (11, 14),
        (12, 15),
        (13, 16),
        (14, 18),
        (12, 19),
        (13, 17),
        (14, 19),
        (15, 20),
        (16, 20),
        (17, 21),
        (12, 26),
        (20, 22),
        (21, 24),
        (18, 21),
        (20, 23),
        (21, 25),
        (23, 27),
        (24, 27),
        (26, 28),
        (28, 29),
        (25, 28),
        (28, 30)
    ]

    J2_EDGES = [
        (2, 3),
        (3, 4),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (2, 11),
        (4, 29),
        (7, 19),
        (15, 16),
        (16, 17),
        (17, 18),
        (15, 26),
        (18, 19),
        (22, 23),
        (23, 24),
        (24, 25),
        (22, 30),
        (25, 26),
        (29, 30)
    ]

