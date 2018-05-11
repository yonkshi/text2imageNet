import torch
from torch.utils.serialization import load_lua

model = load_lua('t2.t7', unknown_classes=True)
print('hell world')