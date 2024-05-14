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

config = {
    "model_save_path": "HW4/model.pth",
    "validation_ratio": 0.1,
    "seed": 1998,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "n_epochs": 500,
    "stop_train_count": 100,
    "max_seg_len": 128,
}

args = None


def open_json_file(json_data_path):
    with open(json_data_path, "r") as f:
        json_file = json.load(f)
        return json_file


# model必须调__init__因为其基类需要初始化
class SpeakerClassifierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # input: [40]

        # Since 128 features could learn more from 40 features, and
        # self-attention prefers using high dimension input since it consider global data

        # Shape:
        # - Input: :math:`(*, H_{in})` where :math:`*` means any number of
        #   dimensions including none and :math:`H_{in} = \text{in\_features}`.
        # - Output: :math:`(*, H_{out})` where all but the last dimension
        #   are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
        self.pre_layer = torch.nn.Linear(40, 128)

        # From Class Transform comments
        #  - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
        #               `(N, S, E)` if `batch_first=True`.
        # N: batch_size, S:seqence_len, E: embedding length特征维度

        #  - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
        #   `(N, T, E)` if `batch_first=True`.
        # input and output have same dimensions
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=128, nhead=2, dim_feedforward=2048, batch_first=True
        )

        self.predict_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 600),
        )

    def forward(self, x):
        # print(f"x.size:{x.size()}")
        # [batch_size, seq_max_len, 40] -> [batch_size, seq_max_len, 128]
        pre_layer_output = self.pre_layer(x)
        # print(f"pre_layer_output.size:{pre_layer_output.size()}")
        # [batch_size, seq_max_len, 128] -> [batch_size, seq_max_len, 128]
        encoder_layer_output = self.encoder_layer(pre_layer_output)
        # print(f"encoder_layer_output.size:{encoder_layer_output.size()}")

        # 在音频任务中，通常会沿着时间维度进行池化而不是特征维度上的池化。
        # 这是因为音频数据通常是时间序列数据，其关键信息往往分布在时间轴上，
        # 例如音频中的语音信号或音乐节奏。因此，对音频数据进行池化时，更关注
        # 时间维度上的特征提取，以捕获时间序列中的重要模式和结构，而不是特征
        # 维度上的池化。
        # [batch_size, seq_max_len, 128] -> [batch_size, 128]
        encoder_layer_output_mean_pooling = encoder_layer_output.mean(dim=1)
        # print(
        #     f"encoder_layer_output_mean_pooling.size:{encoder_layer_output_mean_pooling.size()}"
        # )
        # [batch_size, 128]-> [batch_size, 600]
        return self.predict_layer(encoder_layer_output_mean_pooling)


# 由于Dataset是抽象基类，基类没有__init__方法定义
class SpeakerDataset(torch.utils.data.Dataset):
    # id_to_speaker and speaker_to_id
    def __init__(self, mode, json_data_path, mapping_file):
        # 存feature path和对应的ground truth(id)
        self.data = []
        self.mode = mode
        if mode == "train":
            train_data = open_json_file(json_data_path)
            for id in tqdm.tqdm(train_data["speakers"], desc="Processing init data"):
                # print(len(train_data["speakers"][id]))
                for id_to_dict in train_data["speakers"][id]:
                    feature_name = id_to_dict["feature_path"]
                    id_to_class = mapping_file["speaker2id"][id]
                    # print(type(id_to_class))
                    self.data.append(
                        [feature_name, torch.tensor(id_to_class, dtype=torch.int)]
                    )
                    # print(feature_tensor.size())
                    # print(f"feature_name:{feature_name},id_to_class:{id_to_class}")
        elif mode == "inference":
            test_data = open_json_file(json_data_path)
            for utterance in test_data["utterances"]:
                feature_name, mel_len = utterance["feature_path"], utterance["mel_len"]
                # print(f"feature_name:{feature_name},mel_len:{mel_len}")
                self.data.append(feature_name)

    def __len__(self):
        return len(self.data)

    def cut_seg_if_too_long(self, seg, max_seg_len):
        if len(seg) <= max_seg_len:
            return seg

        st = random.randint(0, len(seg) - max_seg_len)
        return seg[st : st + max_seg_len]

    def __getitem__(self, index):
        if self.mode == "train":
            # 读取self.data[index]对应的feature
            feature_name, id_to_class = self.data[index]
            # print(f"type(id_to_class):{type(id_to_class)}")
            feature_tensor = torch.load(os.path.join("HW4/Dataset", feature_name))
            feature_tensor = self.cut_seg_if_too_long(
                feature_tensor, config["max_seg_len"]
            )
            # [undefined, 40]
            return feature_tensor, id_to_class
        elif self.mode == "inference":
            feature_name = self.data[index]
            feature_tensor = torch.load(os.path.join("HW4/Dataset", feature_name))
            # print(f"inf::feature_tensor:size:{feature_tensor.size()}")
            return feature_name, feature_tensor
            # inference不需要cut

    # 每个sample各个维度都要等长
    def train_collate_batch(batch):
        mels, lables = zip(*batch)
        # print(f"type(lable):{type(lables)},lables:{lables}")
        # mel: [batch_size, undefined, 40] -> [batch_size, max_len, 40]
        mels = torch.nn.utils.rnn.pad_sequence(
            sequences=mels, batch_first=True, padding_value=0
        )
        return mels, torch.tensor(lables, dtype=int)


def get_train_and_val_ld():
    mapping_file = open_json_file("HW4/Dataset/mapping.json")
    train_data_set = SpeakerDataset("train", "HW4/Dataset/metadata.json", mapping_file)

    tot_data_length = len(train_data_set)
    train_data_length = int((1 - config["validation_ratio"]) * tot_data_length)
    validation_data_length = tot_data_length - train_data_length
    train_data_set, validation_data_set = torch.utils.data.random_split(
        train_data_set,
        [train_data_length, validation_data_length],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    train_data_ld = torch.utils.data.DataLoader(
        dataset=train_data_set,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        collate_fn=SpeakerDataset.train_collate_batch,
    )

    val_data_ld = torch.utils.data.DataLoader(
        dataset=validation_data_set,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        collate_fn=SpeakerDataset.train_collate_batch,
    )
    return train_data_ld, val_data_ld


def train_model(train_loader, validation_loader):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    best_acc = 0.0
    model = SpeakerClassifierModel().to(device=device)
    if args.load_model_to_train == "true":
        try:
            model = SpeakerClassifierModel().to(device)
            ckpt = torch.load(config["model_save_path"], map_location="cpu")
            model.load_state_dict(ckpt)
            with open("HW4/best_accuracy.pkl", "rb") as f:
                best_acc = pickle.load(f)
            print(f"Use local model, best_acc:{best_acc}")
        except FileNotFoundError:
            print("No local model")

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
    )

    n_epoch = config["n_epochs"]
    loss_record = {"train": [], "validation": []}
    bad_acc_cnt = 0
    for epoch in range(n_epoch):
        model.train()
        each_epoch_loss_record = []
        for x, y in tqdm.tqdm(train_loader, desc=f"train_epoch:{epoch}"):
            optimizer.zero_grad()
            # print(f"type(x):{type(x)}, type(y):{type(y)}")
            x = x.to(device)
            y = y.to(device)

            predicted_res = model(x)
            loss_res = loss_func(predicted_res, y)
            loss_res.backward()
            optimizer.step()

            each_epoch_loss_record.append(loss_res.detach().cpu().item())

        loss_record["train"].append(
            sum(each_epoch_loss_record) / len(each_epoch_loss_record)
        )

        model.eval()
        each_epoch_loss_record = []
        eval_acc = []
        for x, y in tqdm.tqdm(validation_loader, desc=f"val_epoch:{epoch}"):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                predicted_res = model(x)
                loss_res = loss_func(predicted_res, y)
                each_epoch_loss_record.append(loss_res.detach().cpu().item())
                _, test_pred = torch.max(predicted_res, 1)
                eval_acc.append((test_pred.detach() == y.detach()).sum().item())
        validation_mean_loss = sum(each_epoch_loss_record) / len(each_epoch_loss_record)
        validation_mean_acc = sum(eval_acc) / len(validation_loader.dataset)
        print(
            f"epoch:{epoch},validation_mean_loss:{validation_mean_loss},validation_mean_acc:{validation_mean_acc}"
        )
        loss_record["validation"].append(validation_mean_loss)
        if validation_mean_acc > best_acc:
            best_acc = validation_mean_acc
            with open("HW4/best_accuracy.pkl", "wb") as f:
                pickle.dump(best_acc, f)
            torch.save(model.state_dict(), config["model_save_path"])
            bad_acc_cnt = 0
        else:
            bad_acc_cnt += 1
        if bad_acc_cnt >= config["stop_train_count"]:
            print("break")
            break
    return loss_record


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


def train():
    train_loader, validation_loader = get_train_and_val_ld()
    loss_record = train_model(
        train_loader,
        validation_loader,
    )
    plot_learning_curve(loss_record, title="deep model")


def inference():
    mapping_file = open_json_file("HW4/Dataset/mapping.json")
    test_ds = SpeakerDataset("inference", "HW4/Dataset/testdata.json", mapping_file)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    print(f"test_ds:{len(test_ds)}")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = SpeakerClassifierModel().to(device)
    ckpt = torch.load(config["model_save_path"], map_location="cpu")
    model.load_state_dict(ckpt)

    feature_names = []
    preds = []
    model.eval()
    for feature_name, x in tqdm.tqdm(test_loader):
        # print(f"x.size():{x.size()}")
        x = x.to(device)
        feature_names.append(feature_name)
        # print(
        #     f"type(feature_name):{type(feature_name)},len(feature_name):{len(feature_name)}"
        # )

        with torch.no_grad():
            pred = model(x)
            _, test_pred = torch.max(pred, 1)
            preds.append(test_pred.detach().cpu())
            # break
    preds = torch.cat(preds, dim=0).numpy()
    print(len(preds))
    assert len(preds) == len(feature_names)
    save_pred(preds, feature_names, "HW4/pred.csv", mapping_file)


def save_pred(preds, feature_names, file, mapping_file):
    """Save predictions to specified file"""
    print("Saving results to {}".format(file))
    with open(file, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Id", "Category"])
        for i, p in enumerate(preds):
            p = mapping_file["id2speaker"][str(p)]
            writer.writerow([feature_names[i][0], p])


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
    elif args.mode == "inference":
        inference()
