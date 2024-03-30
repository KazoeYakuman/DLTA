
import torch


x = torch.randn((2,3), dtype=torch.long)
print(x)
print(torch.argmax(x, dim=0))
print(torch.argmax(x, dim=1))