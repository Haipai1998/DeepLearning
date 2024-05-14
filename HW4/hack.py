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
import pickle

import csv

# 假设你有一个包含字符的列表
my_list = ["apple", "banana", "orange", "grape"]

# 将列表中的字符写入CSV文件
with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(my_list)


# # 假设best_accuracy是你的最佳准确度值
# best_accuracy = 0.8  # 举例

# # 保存最佳准确度到文件
# # with open("best_accuracy.pkl", "wb") as f:
# #     pickle.dump(best_accuracy, f)

# # 加载最佳准确度
# with open("HW4/best_accuracy.pkl", "rb") as f:
#     loaded_best_accuracy = pickle.load(f)

# print("Loaded best accuracy:", loaded_best_accuracy)


# layer = torch.nn.Linear(40, 128)
# ts = torch.rand(3, 60, 40)
# res = layer(ts)
# print(res.size())

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
