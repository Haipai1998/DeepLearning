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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

config = {
    "validation_ratio": 0.2,
    "seed": 1998,
    "batch_size": 256,
    "learning_rate": 1e-6,
    "n_epochs": 30000,
    "train_id_lable_path": "HW2/libriphone/train_labels.txt",
    "model_save_path": "HW2/model.pth",
    "stop_train_count": 400,
}


def GetTrainAndValDataSet():
    with open(config["train_id_lable_path"], "r") as file:
        content = file.readlines()
    line = content[0].strip().split(" ")
    print(line)
    # for line in content:
    #     line = line.strip().split(" ")


GetTrainAndValDataSet()
# train_data_set, validation_data_set = GetTrainAndValDataSet()
