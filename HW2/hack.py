import os
import torch
import random
from tqdm import tqdm


def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    print(x[0])
    print(x[1])
    print(x[2])
    x = x.repeat(1, concat_n)
    torch.set_printoptions(threshold=float("inf"), linewidth=100)
    print("-------------run1")
    x = x.view(seq_len, concat_n, feature_dim).permute(
        1, 0, 2
    )  # concat_n, seq_len, feature_dim
    # print(x[0])
    # print("-------------run2")
    mid = concat_n // 2
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)
    x = x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)
    print(x[1])
    return x


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8):
    class_num = 41  # NOTE: pre-computed, should not need change

    if split == "train" or split == "val":
        mode = "train"
    elif split == "test":
        mode = "test"
    else:
        raise ValueError("Invalid 'split' argument for dataset: PhoneDataset!")

    label_dict = {}
    if mode == "train":
        for line in open(os.path.join(phone_path, f"{mode}_labels.txt")).readlines():
            line = line.strip("\n").split(" ")
            label_dict[line[0]] = [int(p) for p in line[1:]]

        # split training and validation data
        usage_list = open(os.path.join(phone_path, "train_split.txt")).readlines()
        # random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = (
            usage_list[:train_len] if split == "train" else usage_list[train_len:]
        )

    elif mode == "test":
        usage_list = open(os.path.join(phone_path, "test_split.txt")).readlines()

    usage_list = [line.strip("\n") for line in usage_list]
    print(
        "[Dataset] - # phone classes: "
        + str(class_num)
        + ", number of utterances for "
        + split
        + ": "
        + str(len(usage_list))
    )

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode == "train":
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f"{fname}.pt"))
        cur_len = len(feat)
        # print(f"feat_len:{feat.size()}")
        feat = concat_feat(feat, concat_nframes)
        if mode == "train":
            label = torch.LongTensor(label_dict[fname])
            # print(f"label.size:{len(label)},cur_len:{cur_len},feat_len:{feat.size()}")
            break

        X[idx : idx + cur_len, :] = feat
        if mode == "train":
            y[idx : idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == "train":
        y = y[:idx]


train_X, train_y = preprocess_data(
    split="train",
    feat_dir="HW2/libriphone/feat",
    phone_path="HW2/libriphone",
    concat_nframes=3,
    train_ratio=0.75,
)
