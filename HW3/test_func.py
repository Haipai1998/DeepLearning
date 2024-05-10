import unittest
import HW3.train
import torch
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


class TestReadFunc(unittest.TestCase):
    def test_ds(self):
        train_img_transformer = transforms.Compose(
            [
                # Resize the image into a fixed shape (height = width = 128)
                transforms.Resize((128, 128)),
                transforms.RandomGrayscale(0.5),  # 随机灰度化
                transforms.RandomSolarize(threshold=192.0),
                transforms.ColorJitter(
                    brightness=0.5, hue=0.5
                ),  # 改变图像的亮度和饱和度
                transforms.RandomRotation(degrees=(0, 180)),  # 图像随机旋转
                transforms.ToTensor(),
            ]
        )

        train_ds = HW3.train.ImgClassifierDataSet(
            "HW3/ut_data/", train_img_transformer, "train"
        )
        return_transform = transforms.ToPILImage()
        # layers = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        # )
        model = HW3.train.ImgClassifierModel()
        # only one img in train_ds for testing
        for img, lable in train_ds:
            print(f"lable:{lable},img.type:{type(img)},img.size:{img.size()}")
            img = img.unsqueeze(0)
            inf_res = model(img)
            print(f"inf_res.type:{type(inf_res)},inf_res.size:{inf_res.size()}")
            assert inf_res.size() == (1, 11)
            assert lable == 2

            # plt.imshow(img)
            # img_pil = return_transform(img)
            # img_pil.save("HW3/ut_data/" + str(lable) + "_saved_image.jpg")

        # # 保存图像
        # plt.show()

    # python -m unittest HW3.test_func.TestReadFunc.test_train_logic
    def test_train_logic(self):
        train_img_transformer = transforms.Compose(
            [
                # Resize the image into a fixed shape (height = width = 128)
                transforms.Resize((128, 128)),
                transforms.RandomGrayscale(0.5),  # 随机灰度化
                transforms.RandomSolarize(threshold=192.0),
                transforms.ColorJitter(
                    brightness=0.5, hue=0.5
                ),  # 改变图像的亮度和饱和度
                transforms.RandomRotation(degrees=(0, 180)),  # 图像随机旋转
                transforms.ToTensor(),
            ]
        )

        train_ds = HW3.train.ImgClassifierDataSet(
            "HW3/ut_data/", train_img_transformer, "train"
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            pin_memory=True,
        )
        loss_record = HW3.train.train_model(train_loader, val_loader)
        HW3.train.plot_learning_curve(loss_record, title="deep model")


# if __name__ == "__main__":
#     unittest.main()
