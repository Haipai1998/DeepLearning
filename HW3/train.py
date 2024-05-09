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

# torch.nn.Conv2d()

config = {
    "train_data_root_path": "HW3/train/",
    "validation_ratio": 0.2,
    "seed": 1998,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "n_epochs": 50,
    "stop_train_count": 400,
    "frame_dimension": 39,
}

class ImgClassifierDataSet(torch.utils.data.Dataset):
    def __init__(self, open_imgs, lables):
        
    # def __init__(self, feature, ground_truth):

    # def __getitem__(self, index):

    # def __len__(self):



def get_train_and_val_ld():
    dir = os.listdir(config["train_data_root_path"])
    # print(f"dir_len:{len(dir)}")
    # Image.open()

    # return get_data_ld()


def train():
    get_train_and_val_ld()
    # train_loader, validation_loader = get_train_and_val_ld()
    # loss_record = train_model(
    #     train_loader,
    #     validation_loader,
    #     config["concat_num_each_side"],
    #     config["frame_dimension"],
    # )
    # plot_learning_curve(loss_record, title="deep model")


if __name__ == "__main__":
    train()
