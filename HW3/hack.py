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
