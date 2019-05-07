# Copyright Tom Westerhout (c) 2018
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

import torch

class Net(torch.jit.ScriptModule):
    def __init__(self, n: int):
        super().__init__()
        self._number_spins = n
        # self._conv1 = torch.nn.Conv1d(1, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self._conv2 = torch.nn.Conv1d(16, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self._dense1 = torch.nn.Linear(n, 64)
        self._dense2 = torch.nn.Linear(64, 128)
        self._dense6 = torch.nn.Linear(128, 2, bias=True)
        self.dropout = torch.nn.Dropout(0.3) 
    @torch.jit.script_method
    def forward(self, x):
        x = torch.tanh(self._dense1(x))
        x = self.dropout(x)
        x = torch.tanh(self._dense2(x))
        x = self.dropout(x)
        x = torch.tanh(self._dense6(x))
        return x

    @property
    def number_spins(self) -> int:
        return self._number_spins
