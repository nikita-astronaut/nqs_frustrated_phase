#!/usr/bin/env python3

# Copyright Andrey Bagrov (c) 2019
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


class Triangle24(System):

    POSITIONS = [(i, j) for i in range(6) for j in range(4)]

    J1_EDGES = [
        (0, 1),
        (0, 6),
        (1, 2),
        (1, 7),
        (2, 3),
        (2, 8),
        (3, 4),
        (3, 9),
        (4, 5),
        (4, 10),
        (5, 11),
        (6, 7),
        (6, 12),
        (7, 8),
        (7, 13),
        (8, 9),
        (8, 14),
        (9, 10),
        (9, 15),
        (10, 11),
        (10, 16),
        (11, 17),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (12, 18),
        (13, 19),
        (14, 20),
        (15, 21),
        (16, 22),
        (17, 23),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
    ]

    J2_EDGES = [
        (1, 6),
        (2, 7),
        (7, 12),
        (3, 8),
        (8, 13),
        (4, 9),
        (9, 14),
        (5, 10),
        (10, 15),
        (11, 16),
        (13, 18),
        (14, 19),
        (15, 20),
        (16, 21),
        (17, 22),
    ]


class Triangle16(System):

    POSITIONS = [(i, j) for i in range(4) for j in range(4)]

    J1_EDGES = [
        (0, 1),
        (0, 4),
        (1, 2),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 8),
        (5, 6),
        (5, 9),
        (6, 7),
        (6, 10),
        (7, 11),
        (8, 9),
        (8, 12),
        (9, 10),
        (9, 13),
        (10, 11),
        (10, 14),
        (11, 15),
        (12, 13),
        (13, 14),
        (14, 15),
    ]

    J2_EDGES = [
        (1, 4),
        (2, 5),
        (3, 6),
        (5, 8),
        (6, 9),
        (7, 10),
        (9, 12),
        (10, 13),
        (11, 14),
    ]

class TrianglePeriodic16(System):

    POSITIONS = [(i, j) for i in range(4) for j in range(4)]

    J1_EDGES = [
        (0, 1),
        (0, 3),
        (0, 4),
        (0, 12),
        (1, 2),
        (1, 5),
        (1, 13),
        (2, 3),
        (2, 6),
        (2, 14),
        (3, 7),
        (3, 15),
        (4, 5),
        (4, 7),
        (4, 8),
        (5, 6),
        (5, 9),
        (6, 7),
        (6, 10),
        (7, 11),
        (8, 9),
        (8, 11),
        (8, 12),
        (9, 10),
        (9, 13),
        (10, 11),
        (10, 14),
        (11, 15),
        (12, 13),
        (12, 15),
        (13, 14),
        (14, 15),
    ]

    J2_EDGES = [
        (0, 7),
        (0, 13),
        (1, 4),
        (1, 14),
        (2, 5),
        (2, 15),
        (3, 6),
        (3, 12),
        (4, 11),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 15),
        (9, 12),
        (10, 13),
        (11, 14),
    ]


class TrianglePeriodic24(System):

    POSITIONS = [(i, j) for i in range(6) for j in range(4)]

    J1_EDGES = [
        (0, 1),
        (0, 6),
        (1, 2),
        (1, 7),
        (2, 3),
        (2, 8),
        (3, 4),
        (3, 9),
        (4, 5),
        (4, 10),
        (5, 11),
        (6, 7),
        (6, 12),
        (7, 8),
        (7, 13),
        (8, 9),
        (8, 14),
        (9, 10),
        (9, 15),
        (10, 11),
        (10, 16),
        (11, 17),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (12, 18),
        (13, 19),
        (14, 20),
        (15, 21),
        (16, 22),
        (17, 23),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),  # next come periodic links (connecting boundaries)
        (5, 0),
        (11, 6),
        (17, 12),
        (23, 18),
        (18, 0),
        (19, 1),
        (20, 2),
        (21, 3),
        (22, 4),
        (23, 5),
    ]

    J2_EDGES = [
        (1, 6),
        (2, 7),
        (7, 12),
        (3, 8),
        (8, 13),
        (4, 9),
        (9, 14),
        (5, 10),
        (10, 15),
        (11, 16),
        (13, 18),
        (14, 19),
        (15, 20),
        (16, 21),
        (17, 22),  # next come periodic links
        (0, 19),
        (1, 20),
        (2, 21),
        (3, 22),
        (4, 23),
        (5, 18),
        (12, 23),
        (6, 17),
        (0, 11),
    ]

class TrianglePeriodic30(System):

    POSITIONS = [(i, j) for i in range(6) for j in range(5)]

    J1_EDGES = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (0, 5),
        (6, 7),
        (7, 8),
        (8, 9),
        (10, 11),
        (6, 11),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (12, 17),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (18, 23),
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
        (28, 29),
        (24, 29),
        (0, 24),
        (1, 25),
        (2, 26),
        (3, 27),
        (4, 28),
        (5, 29),
        (0, 6),
        (1, 7),
        (2, 8),
        (3, 9),
        (4, 10),
        (5, 11),
        (6, 12),
        (7, 13),
        (8, 14),
        (9, 15),
        (10, 16),
        (11, 17),
        (12, 18),
        (13, 19),
        (14, 20),
        (15, 21),
        (16, 22),
        (17, 23),
        (18, 24),
        (19, 25),
        (20, 26),
        (21, 27),
        (22, 28),
        (23, 29)
    ]

    J2_EDGES = [
        (0, 25),
        (1, 26),
        (2, 27),
        (3, 28),
        (4, 29),
        (5, 24),
        (0, 11),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 10),
        (6, 17),
        (7, 12),
        (8, 13),
        (9, 14),
        (10, 15),
        (11, 16),
        (12, 23),
        (13, 18),
        (14, 19),
        (15, 20),
        (16, 21),
        (17, 22),
        (18, 29),
        (19, 24),
        (20, 25),
        (21, 26),
        (22, 27),
        (23, 28)
    ]
