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
    def __init__(self, fpath, img_transformer, mode):
        self.fpath = fpath
        self.fnames = os.listdir(fpath)
        self.img_transformer = img_transformer
        self.mode = mode
        print(self.fnames)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.fpath, self.fnames[index]))
        # print(f"ImgClassifierDataSet, img.type:{type(img)}")
        img_tensor = self.img_transformer(img)
        # print(f"ImgClassifierDataSet, img.type after tsf:{type(img)}")

        if self.mode == "test":
            return img_tensor

        lable = int(self.fnames[index].split("_")[0])

        return img, lable


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
        config["train_data_root_path"], train_img_transformer
    )
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
