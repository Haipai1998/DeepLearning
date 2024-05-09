import unittest
import train
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

import train


class TestReadFunc(unittest.TestCase):
    def test_ds(self):
        train_img_transformer = transforms.Compose(
            [
                # Resize the image into a fixed shape (height = width = 128)
                # transforms.Resize((128, 128)),
                # transforms.RandomGrayscale(0.5),  # 随机灰度化
                # transforms.RandomSolarize(threshold=192.0),
                # transforms.ColorJitter(
                #     brightness=0.5, hue=0.5
                # ),  # 改变图像的亮度和饱和度
                # transforms.RandomRotation(degrees=(0, 180)),  # 图像随机旋转
                transforms.ToTensor(),
            ]
        )

        img = Image.open("HW3/ut_data/2_3.jpg")
        img = train_img_transformer(img)
        print(f"type(img):{type(img)}")

        # train_ds = train.ImgClassifierDataSet(
        #     "HW3/ut_data/", train_img_transformer, "train"
        # )
        # return_transform = transforms.ToPILImage()
        # for img, lable in train_ds:
        #     print(f"lable:{lable},img.type:{type(img)}")
        #     assert lable == 2
        #     plt.imshow(img)
        #     img_pil = return_transform(img)
        #     img_pil.save("HW3/ut_data/" + str(lable) + "_saved_image.jpg")

        # # 保存图像
        # plt.show()


if __name__ == "__main__":
    unittest.main()
