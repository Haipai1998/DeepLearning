import pandas
import numpy
import torch.utils.data.dataset
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
import pickle
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import figure
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.transforms as transforms
import ttach
import json


def get_train_and_val_ld():
    print("get_train_and_val_ld")


def train():
    get_train_and_val_ld()
    # train_loader, validation_loader = get_train_and_val_ld()
    # loss_record = train_model(
    #     train_loader,
    #     validation_loader,
    # )
    # plot_learning_curve(loss_record, title="deep model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hack")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        required=True,
        help="Mode: 'train' or 'inference'",
    )
    parser.add_argument(
        "--load_model_to_train",
        type=str,
        choices=["true", "false"],
    )
    args = parser.parse_args()
    if args.mode == "train":
        train()
    # elif args.mode == "inference":
    #     inference()
