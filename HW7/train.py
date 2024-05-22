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
import json
import transformers


# todo: 如何判断读取格式？
def open_json_file(json_data_path):
    with open(json_data_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
        return json_file


tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-chinese")
print(dir(tokenizer))


# chinese extractive question answer
class CNEQADataSet(torch.utils.data.Dataset):
    # 必须得一次性Load进来，因为数据只在一个文件中，与图像分类不一样
    def __init__(self, mode, data_json_path):
        json_data = open_json_file(data_json_path)
        self.mode = mode
        self.tokenized_questions = []
        self.question_id_to_paragraph_id = {}
        for question_info in json_data["questions"]:
            self.tokenized_questions.append(tokenizer(question_info["question_text"]))
            self.question_id_to_paragraph_id[question_info["id"]] = question_info[
                "paragraph_id"
            ]
        self.tokenized_paragraphs = []
        for paragraph in json_data["paragraphs"]:
            self.tokenized_paragraphs.append(tokenizer(paragraph))

    def __getitem__(self, index):
        tokenized_question = self.tokenized_questions[index]
        tokenized_paragraph = self.tokenized_paragraphs[
            self.question_id_to_paragraph_id[index]
        ]
        # tokenized_paragraph.
        # getitem返回：tokenize question和paragrah后进行拼接padding, ans_token_{start_index/end_index}
        if self.mode == "train":
            return


def get_train_and_val_ld():
    train_ds = CNEQADataSet("train", "HW7/hw7_train.json")


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
