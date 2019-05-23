import torch

Net = lambda n: torch.nn.Sequential(
    torch.nn.Linear(n, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2, bias=False),
)
