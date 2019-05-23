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
