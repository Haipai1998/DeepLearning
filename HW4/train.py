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
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import figure
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.transforms as transforms
import ttach
import json

config = {
    "model_save_path": "HW4/model.pth",
    "validation_ratio": 0.2,
    "seed": 1998,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "n_epochs": 500,
    "stop_train_count": 100,
    "max_seg_len": 128,
}


def open_json_file(json_data_path):
    with open(json_data_path, "r") as f:
        json_file = json.load(f)
        return json_file


# From Class Transform comments
#  - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
#               `(N, S, E)` if `batch_first=True`.
torch.nn.TransformerEncoderLayer()
torch.nn.Transformer()

# class SpeakerClassifierModel(torch.nn.Module):
#     def


class SpeakerDataset(torch.utils.data.Dataset):
    # id_to_speaker and speaker_to_id
    def __init__(self, mode, json_data_path, mapping_file):
        # 存feature path和对应的ground truth(id)
        self.data = []
        self.mode = mode
        if mode == "train":
            train_data = open_json_file(json_data_path)
            for id in train_data["speakers"]:
                # print(len(train_data["speakers"][id]))
                for id_to_dict in train_data["speakers"][id]:
                    feature_name = id_to_dict["feature_path"]
                    id_to_class = mapping_file["speaker2id"][id]
                    # print(type(id_to_class))
                    self.data.append(
                        [feature_name, torch.tensor(id_to_class, dtype=torch.int)]
                    )
                    feature_tensor = torch.load(
                        os.path.join("HW4/Dataset", feature_name)
                    )
                    print(feature_tensor.size())
                    # print(f"feature_name:{feature_name},id_to_class:{id_to_class}")

    def __len__(self):
        return len(self.data)

    def cut_seg_if_too_long(seg, max_seg_len):
        if len(seg) <= max_seg_len:
            return seg

        st = random.randint(0, len(seg) - max_seg_len)
        return seg[st, st + max_seg_len]

    def __getitem__(self, index):
        if self.mode == "train":
            # 读取self.data[index]对应的feature
            feature_name, id_to_class = self.data[index]
            feature_tensor = torch.load(os.path.join("HW4/Dataset", feature_name))
            feature_tensor = self.cut_seg_if_too_long(
                feature_tensor, config["max_seg_len"]
            )
            # [undefined, 40]
            return feature_tensor, id_to_class

    # 每个sample各个维度都要等长. TODO:是否必须？
    def collate_batch(batch):
        mels, lables = zip(*batch)
        # mel: [batch_size, undefined, 40] -> [batch_size, max_len, 40]
        mels = torch.nn.utils.rnn.pad_sequence(
            sequences=mels, batch_first=True, padding_value=0
        )
        return mels, lables


def get_train_and_val_ld():
    mapping_file = open_json_file("HW4/Dataset/mapping.json")
    SpeakerDataset("train", "HW4/Dataset/metadata.json", mapping_file)


def train():
    get_train_and_val_ld()
    # train_loader, validation_loader = get_train_and_val_ld()
    # loss_record = train_model(
    #     train_loader,
    #     validation_loader,
    # )
    # plot_learning_curve(loss_record, title="deep model")


if __name__ == "__main__":
    train()
