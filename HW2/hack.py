import torch

torch.squeeze()
ts = torch.tensor([[1], [2]]).squeeze(2)
print(f"ts:{ts}\n, ts.size:{ts.size()}")
