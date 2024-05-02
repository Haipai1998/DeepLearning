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
    "batch_size": 512,
    "learning_rate": 1e-4,
    "n_epochs": 20,
    "train_id_lable_path": "HW2/libriphone/train_labels.txt",
    "model_save_path": "HW2/model.pth",
    "stop_train_count": 400,
    "concat_num_each_side": 5,
    "frame_dimension": 39,
}

# feature
# get from feat/train and merge adjacent data by the file_id in lables.txt

# lable
# get from lables.txt


class LibriphoneModel(torch.nn.Module):
    def __init__(self, input_dimension) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(64),
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(input_dimension, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 41),
        )

    def forward(self, x):
        return self.layers(x)


class LibriphoneDataSet(torch.utils.data.Dataset):
    def __init__(self, feature, ground_truth):
        if ground_truth is None:
            self.ground_truth = None
        else:
            self.ground_truth = torch.LongTensor(ground_truth)
        self.feature = feature

    def __getitem__(self, index):
        if self.ground_truth is None:
            return self.feature[index]
        else:
            return self.feature[index], self.ground_truth[index]

    def __len__(self):
        return len(self.feature)


def read_feature_with_path(path):
    return torch.load(path)


def get_train_and_val_ld(
    id_lable_path, feature_root_path, concat_num_each_side, sample_feature_num
):
    with open(id_lable_path, "r") as file:
        content = file.readlines()

    feature_tensors = []
    ground_truth = []
    print(f"content.len:{len(content)}")
    for line in content:
        line = line.strip().split(" ")
        file_name = line[0]
        file_feature = read_feature_with_path(feature_root_path + file_name + ".pt")
        sample_num = len(file_feature)
        for i, one_sample_feature in enumerate(file_feature):
            # print(one_sample_feature)
            for j in range(-concat_num_each_side, concat_num_each_side + 1):
                if j == 0:
                    continue
                merge_target_index = i + j
                # [0,sample_num-1]
                if merge_target_index < 0 or merge_target_index >= sample_num:
                    merge_targe_data = torch.zeros(sample_feature_num)
                else:
                    merge_targe_data = file_feature[merge_target_index]
                # print(f"merge_targe_data_type:{type(merge_targe_data)}, torch.dim:{merge_targe_data.size()}")
                one_sample_feature = torch.cat(
                    (one_sample_feature, merge_targe_data[0:sample_feature_num])
                )
            feature_tensors.append(one_sample_feature.unsqueeze(0))
        for y in line[1:]:
            ground_truth.append(int(y))
    merged_tensors = torch.cat(feature_tensors, 0)
    print(
        f"feature_tensors.sz:{merged_tensors.size()},ground_truth.len:{len(ground_truth)}"
    )
    # print(feature_tensors)
    ds = LibriphoneDataSet(merged_tensors, ground_truth)

    tot_data_length = len(ds)
    train_data_length = int((1 - config["validation_ratio"]) * tot_data_length)
    validation_data_length = tot_data_length - train_data_length
    train_data_set, validation_data_set = torch.utils.data.random_split(
        ds,
        [train_data_length, validation_data_length],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=config["batch_size"], shuffle=True, pin_memory=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_data_set,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )

    return train_loader, validation_loader

    # break


def train_model(
    train_loader, validation_loader, concat_num_each_side, sample_feature_num
):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = LibriphoneModel((concat_num_each_side * 2 + 1) * sample_feature_num).to(
        device
    )

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=0.95,
        weight_decay=0.001,
    )

    # optimizer = torch.optim.AdamW(
    #    model.parameters(), lr=config["learning_rate"], weight_decay=0.08
    # )

    n_epoch = config["n_epochs"]
    loss_record = {"train": [], "validation": []}
    min_loss = math.inf
    bad_loss_cnt = 0

    for epoch in range(n_epoch):
        # train
        model.train()
        each_epoch_loss_record = []
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            # print("-------")
            # print(f"x={x}")
            # print(f"y={y}")
            # finish forward，计算偏导
            predicted_res = model(x)
            # print(f"predicted_res={predicted_res}")
            # 输入类型: tensor
            loss_res = loss_func(predicted_res, y)
            # print(f"loss_res={loss_res}")
            # print("-------")
            # backpropagation，计算偏导
            loss_res.backward()
            # 按照参数更新算法(例如gradient descent), 更新参数weights and bias
            optimizer.step()
            each_epoch_loss_record.append(loss_res.detach().cpu().item())
        loss_record["train"].append(
            sum(each_epoch_loss_record) / len(each_epoch_loss_record)
        )
        # eval
        model.eval()
        each_epoch_loss_record = []
        for x, y in validation_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                predicted_res = model(x)
                loss_res = loss_func(predicted_res, y)
                each_epoch_loss_record.append(loss_res.detach().cpu().item())
                # loss_record["validation"].append(loss_res.detach().cpu().item())
        validation_mean_loss = sum(each_epoch_loss_record) / len(each_epoch_loss_record)
        loss_record["validation"].append(validation_mean_loss)
        # loss_record["validation"].append(validation_mean_loss)
        print(f"epoch:{epoch},validation_mean_loss:{validation_mean_loss}")
        if validation_mean_loss < min_loss:
            min_loss = validation_mean_loss
            torch.save(model.state_dict(), config["model_save_path"])
            bad_loss_cnt = 0
        else:
            bad_loss_cnt += 1
        if bad_loss_cnt >= config["stop_train_count"]:
            print("break")
            break
    print("Finish training")

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
    plt.ylim(0.0, 4)
    plt.xlabel("Training steps")
    plt.ylabel("MSE loss")
    plt.title("Learning curve of {}".format(title))
    plt.legend()
    plt.show()


if __name__ == "__main__":

    train_loader, validation_loader = get_train_and_val_ld(
        config["train_id_lable_path"],
        "HW2/libriphone/feat/train/",
        config["concat_num_each_side"],
        config["frame_dimension"],
    )
    loss_record = train_model(
        train_loader,
        validation_loader,
        config["concat_num_each_side"],
        config["frame_dimension"],
    )
    plot_learning_curve(loss_record, title="deep model")
