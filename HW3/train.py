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
    "val_data_root_path": "HW3/valid/",
    "validation_ratio": 0.2,
    "seed": 1998,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "n_epochs": 50,
    "stop_train_count": 400,
    "frame_dimension": 39,
}


class ImgClassifierDataSet(torch.utils.data.Dataset):
    def __init__(self, fpath, img_transformer, mode):
        self.fpath = fpath
        self.fnames = os.listdir(fpath)
        self.img_transformer = img_transformer
        self.mode = mode

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.fpath, self.fnames[index]))
        img_tensor = self.img_transformer(img)

        if self.mode == "test":
            return img_tensor

        lable = int(self.fnames[index].split("_")[0])

        return img_tensor, lable


class ImgClassifierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # input_dim: [3,128,128]
        self.layers = torch.nn.Sequential(
            # [3,128,128]
            # torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            # torch.nn.BatchNorm2d(64),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2, 2),
            # # [64,64,64]
            # torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2, 2),
            # #
            # torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            # torch.nn.BatchNorm2d(256),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2, 2),  # [128,32,32]
        )

    # def forward(self, x):
    #     # return self.layers(x)


def get_train_and_val_ld():
    train_img_transformer = transforms.Compose(
        [
            # Resize the image into a fixed shape (height = width = 128)
            transforms.Resize((128, 128)),
            transforms.RandomGrayscale(0.5),  # 随机灰度化
            transforms.RandomSolarize(threshold=192.0),
            transforms.ColorJitter(brightness=0.5, hue=0.5),  # 改变图像的亮度和饱和度
            transforms.RandomRotation(degrees=(0, 180)),  # 图像随机旋转
            transforms.ToTensor(),
        ]
    )
    train_ds = ImgClassifierDataSet(
        config["train_data_root_path"], train_img_transformer, "train"
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    print(f"train_ds:{len(train_ds)}")

    val_ds = ImgClassifierDataSet(
        config["val_data_root_path"], train_img_transformer, "train"
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    print(f"val_ds:{len(val_ds)}")
    return train_loader, val_loader


def train_model(train_loader, validation_loader):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


def train():
    train_loader, validation_loader = get_train_and_val_ld()
    loss_record = train_model(
        train_loader,
        validation_loader,
    )
    plot_learning_curve(loss_record, title="deep model")


def plot_learning_curve(loss_record, title=""):
    """Plot learning curve of your DNN (train & validation loss)"""
    total_steps = len(loss_record["train"])
    print(f"total_steps:{total_steps}")
    print("train_len:{}".format(len(loss_record["train"])))
    print("val_len:{}".format(len(loss_record["validation"])))
    # print(f"total_steps:{total_steps}, train_len:{len(loss_record["train"])}, val_len:{len(loss_record["validation"])}")
    x_1 = range(total_steps)
    print(f"x1:{x_1}")
    # x_2 = x_1[:: len(loss_record["train"]) // len(loss_record["validation"])]
    # print(f"x2:{x_2}")
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record["train"], c="tab:red", label="train")
    plt.plot(x_1, loss_record["validation"], c="tab:cyan", label="validation")
    # plt.ylim(0.0, 4)
    plt.xlabel("Training steps")
    plt.ylabel("MSE loss")
    plt.title("Learning curve of {}".format(title))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()
