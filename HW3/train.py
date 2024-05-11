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
import ttach

# torch.nn.Conv2d()
args = None

config = {
    "train_data_root_path": "HW3/train/",
    "val_data_root_path": "HW3/valid/",
    "test_data_root_path": "HW3/test/",
    "model_save_path": "HW3/model.pth",
    "validation_ratio": 0.2,
    "seed": 1998,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "n_epochs": 500,
    "stop_train_count": 100,
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
        # Use padding to make the output of convolutional layer is multiple of 2,
        # and the out_channels will be bigger with the deeper convolutional layer
        # which will help network learn more deep and valid feature.
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, padding=1
            ),  # [3,128,128] ->  [64,128,128]
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  #  [64,128,128] -> [64,64,64]
            #
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),  # [64,64,64] -> [128,64,64]
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  #  [128,64,64] -> [128,32,32]
            #
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),  # [128,32,32] -> [256,32,32]
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # [256,32,32] -> [256,16,16]
            #
            torch.nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # [256,16,16] -> [512,16,16]
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # [512,16,16] -> [512,8,8]
            #
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),  # [512,8,8] -> [512,8,8]
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # [512,8,8] -> [512,4,4]
        )

        # Input 512*4*4 feature from CNN to fully connected network
        # There are 11 categories in total
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 * 4 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 11),
        )

    # x is tensor and its size:[3,128,128]
    def forward(self, x):
        cnn_output = self.cnn_layers(x)
        # print(f"cnn_output.size:{cnn_output.size()}")
        # Flatten 512*4*4 into a one-dim tensor and send it to fc network
        # [batch_size, 512*4*4]
        flatten_cnn_output = cnn_output.view(cnn_output.size()[0], 512 * 4 * 4)
        # print(f"flatten_cnn_output.size:{flatten_cnn_output.size()}")
        return self.fc(flatten_cnn_output)


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

    val_transformer = transforms.Compose(
        [
            # Resize the image into a fixed shape (height = width = 128)
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    # TODO: we shouldn't use train_img_transformer for validation set.
    # The model is training 12hours, but the accuracy is only 0.64(simple baseline),
    # it demonstrate we need use right tsm for validation
    val_ds = ImgClassifierDataSet(
        config["val_data_root_path"], val_transformer, "train"
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    print(f"val_loader.dataset:{len(val_loader.dataset)}")
    return train_loader, val_loader


def train_model(train_loader, validation_loader):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = ImgClassifierModel().to(device)
    if args.load_model_to_train == "true":
        try:
            model = ImgClassifierModel().to(device)
            ckpt = torch.load(config["model_save_path"], map_location="cpu")
            model.load_state_dict(ckpt)
            print("Use local model")
        except FileNotFoundError:
            print("No local model")

    loss_func = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config["learning_rate"],
    #     momentum=0.95,
    #     weight_decay=0.001,
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
    )
    # cos退火
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=8, T_mult=2, eta_min=config["learning_rate"] / 2
    # )

    n_epoch = config["n_epochs"]
    loss_record = {"train": [], "validation": []}
    best_acc = 0.0
    bad_acc_cnt = 0
    for epoch in range(n_epoch):
        # train
        model.train()
        each_epoch_loss_record = []
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            # finish forward，计算偏导
            predicted_res = model(x)
            # 输入类型: tensor
            loss_res = loss_func(predicted_res, y)
            # backpropagation，计算偏导
            loss_res.backward()
            # 按照参数更新算法(例如gradient descent), 更新参数weights and bias
            optimizer.step()
            each_epoch_loss_record.append(loss_res.detach().cpu().item())
        loss_record["train"].append(
            sum(each_epoch_loss_record) / len(each_epoch_loss_record)
        )
        # scheduler.step()

        model.eval()
        each_epoch_loss_record = []
        eval_acc = []
        for x, y in validation_loader:
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
            checkpoint = {"model_state_dict": model.state_dict()}
            torch.save(model.state_dict(), config["model_save_path"])
            bad_acc_cnt = 0
        else:
            bad_acc_cnt += 1
        if bad_acc_cnt >= config["stop_train_count"]:
            print("break")
            break
    return loss_record


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


def save_pred(preds, file):
    """Save predictions to specified file"""
    print("Saving results to {}".format(file))
    with open(file, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Id", "Category"])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


def inference():
    test_transformer = transforms.Compose(
        [
            # Resize the image into a fixed shape (height = width = 128)
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    test_ds = ImgClassifierDataSet(
        config["test_data_root_path"], test_transformer, "test"
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
    )
    print(f"test_ds:{len(test_ds)}")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = ImgClassifierModel().to(device)
    ckpt = torch.load(config["model_save_path"], map_location="cpu")
    model.load_state_dict(ckpt)

    # multi inference
    model = ttach.ClassificationTTAWrapper(
        model, transforms=ttach.aliases.d4_transform(), merge_mode="mean"
    )

    model.eval()
    preds = []
    # 研究model(x)的输出: [batch_size, model_output_dimension]
    for x in tqdm.tqdm(test_loader):
        x = x.to(device)
        # print(f"len(x):{len(x)}")
        with torch.no_grad():
            pred = model(x)
            # print(f"len(pred):{len(pred)}")
            _, test_pred = torch.max(pred, 1)
            preds.append(test_pred.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    print(len(preds))
    save_pred(preds, "HW3/pred1.csv")


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
