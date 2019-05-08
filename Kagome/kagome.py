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

import math
import sys
import numpy as np


class Kagome(object):
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
            lambda t: Kagome._are_nn(cls.POSITIONS[t[0]], cls.POSITIONS[t[1]]), edges
        ):
            sys.stdout.write(
                "{}\t{}\n{}\t{}\n\n".format(
                    *Kagome._to_xy(*cls.POSITIONS[i]), *Kagome._to_xy(*cls.POSITIONS[j])
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


class Kagome18(Kagome):

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
        (3, 17),
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


class Kagome24(Kagome):

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
