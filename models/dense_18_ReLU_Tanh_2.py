import torch

Net = lambda n: torch.nn.Sequential(
    torch.nn.Linear(n, 18),
    torch.nn.ReLU(),
    torch.nn.Linear(18, 18),
    torch.nn.Tanh(),
    torch.nn.Linear(18, 2, bias=False),
)

