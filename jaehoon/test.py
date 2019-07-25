from torch import nn
import torch


input = torch.tensor([[1.1, 0.5, 0.3], [1.01, 0.334, 0.11], [1.001, 0.98, 0.33]], requires_grad=True)
aaa = input.detach().numpy()

target = torch.tensor([0, 0, 0], dtype=torch.int32)
target1 = torch.tensor(target, dtype = torch.int64)
bbb = target.detach().numpy()

output = nn.functional.cross_entropy(input, target)
output.backward()

loss = output.data

print()