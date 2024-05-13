import pandas
import numpy
import tqdm
import common
import math
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import argparse
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.transforms as transforms
import ttach

import torch
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence

batch_sequences = [
    torch.tensor([[1, 2, 3]]),
    torch.tensor([[4, 5]]),
    torch.tensor([[6, 7, 8, 9]]),
]

# 对齐序列
padded_sequences = pad_sequence(batch_sequences, batch_first=True)

print(padded_sequences)


# 假设你有一批输入序列
# batch_sequences = [
#     torch.tensor([[1, 2, 3]]),  # [1,3]
#     torch.tensor([[4, 5]]),  # [1,2]
#     torch.tensor([[6, 7, 8, 9]]),  # [1,4]
# ]

# # 对序列进行填充
# padded_sequences = pad_sequence(batch_sequences, batch_first=False, padding_value=0)

# # 输出填充后的序列
# print("Padded Sequences:\n", padded_sequences)

# a = []
# a.append([1, 2])
# a.append([1, 2])
# print(type(a))
