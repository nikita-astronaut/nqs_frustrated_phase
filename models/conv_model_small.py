import torch

def pad_circular(x, pad): # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
    x = torch.cat([x, x[:, :, 0:pad, :]], dim=2)
    x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
    x = torch.cat([x[:, :, -2 * pad:-pad, :], x], dim=2)
    x = torch.cat([x[:, :, :, -2 * pad:-pad], x], dim=3)

    return x

class Net(torch.jit.ScriptModule):
    def __init__(self, n: int):
        super().__init__()
        self._number_spins = n
        self._conv1 = torch.nn.Conv2d(1, 10, 5, stride=1, padding = 0, dilation=1, groups=1, bias=True)
        self._conv2 = torch.nn.Conv2d(10, 20, 5, stride=1, padding = 0, dilation=1, groups=1, bias=True)
        #self._conv3 = torch.nn.Conv2d(64, 128, 5, stride=1, padding = 0, dilation=1, groups=1, bias=True)
        self._dense6 = torch.nn.Linear(20, 2, bias=True)
        self.dropout = torch.nn.Dropout(0.3)
#    @torch.jit.script_method
    def forward(self, x):
        x = x.view((x.shape[0], 1, 4, 6))
        x = pad_circular(x, 2)
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)
        x = pad_circular(x, 2)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)
        #x = pad_circular(x, 1)
        #x = self._conv3(x)
        #x = torch.nn.functional.relu(x)
        x = x.view(x.shape[0], 20, -1)
        x = x.mean(dim = 2)
        x = self._dense6(x)
        return x

#    @property
    def number_spins(self) -> int:
        return self._number_spins
