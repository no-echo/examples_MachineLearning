import torch
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 3], dtype=torch.float)
targets = torch.tensor([1, 2, 5], dtype=torch.float)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction="sum")
result = loss(inputs, targets)

print(result)