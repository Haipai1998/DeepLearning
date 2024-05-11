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
import train

model = train.ImgClassifierModel().to("cuda")
ckpt = torch.load("d", map_location="cpu")
model.load_state_dict(ckpt)

# aa = []
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])
# aa.append(b)
# aa.append(a)
# print(aa)
# print(sum(aa) / len(aa))

# ts1 = torch.tensor([1, 2, 3])
# ts2 = torch.tensor([1, 2, 4])
# print((ts1 == ts2).sum().item())

# a = 1
# b = 1.0
# print(type(a))
# print(type(b))

# ts = torch.randn(4, 4)
# print(f"ts:{ts},ts.size():{ts.size()}")
# new_ts = ts.view(ts.size()[0], -1)
# print(f"new_ts:{new_ts},new_ts.size:{new_ts.size()}")

# """Plot learning curve of your DNN (train & validation loss)"""
# x_1 = range(0, 5)
# x = [1, 2, 3, 4, 100]
# print(f"x:{x}")
# figure(figsize=(6, 4))
# plt.plot(x_1, x, c="tab:red", label="train")
# plt.plot(x_1, loss_record["validation"], c="tab:cyan", label="validation")
# # plt.ylim(0.0, 4)
# plt.xlabel("Training steps")
# plt.ylabel("MSE loss")
# plt.title("Learning curve of {}".format(title))
# plt.legend()
# plt.show()
