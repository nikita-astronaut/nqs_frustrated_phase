import torch

Net = lambda n: torch.nn.Sequential(
    torch.nn.Linear(n, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2, bias=False),
)
