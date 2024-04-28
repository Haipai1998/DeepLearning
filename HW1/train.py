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
    "covid_train_path": "HW1/covid_train.csv",
    "covid_test_path": "HW1/covid_test.csv",
    "seed": 1998,
    "batch_size": 256,
    "learning_rate": 1e-6,
    "n_epochs": 30000,
    "model_save_path": "HW1/model.pth",
    "stop_train_count": 400,
}


class COVID19DataSet(torch.utils.data.Dataset):
    def __init__(self, feature, res):
        if res is None:
            self.res = None
        else:
            self.res = torch.FloatTensor(res)
        self.feature = torch.FloatTensor(feature)

    def __getitem__(self, index):
        if self.res is None:
            return self.feature[index]
        else:
            # print(f"feature:{self.feature[index]},self.res:{self.res[index]}")
            return self.feature[index], self.res[index]

    def __len__(self):
        return len(self.feature)


class COVIDModel(torch.nn.Module):
    def __init__(self, input_dimension) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            # torch.nn.Linear(input_dimension, 64),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 16),
            torch.nn.Linear(input_dimension, 64),
            # torch.nn.BatchNorm1d(64),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
        )

    def forward(self, x):
        # todo: why?
        x = self.layers(x).squeeze(1)
        return x


def SplitTrainAndValidationData():
    origin_data_set = pandas.read_csv(config["covid_train_path"]).values
    # print(numpy.array(origin_data_set))
    # print(type(pandas.read_csv(config["covid_train_path"])).values)
    tot_data_length = len(origin_data_set)
    train_data_length = int((1 - config["validation_ratio"]) * tot_data_length)
    validation_data_length = tot_data_length - train_data_length
    train_data_set, validation_data_set = torch.utils.data.random_split(
        origin_data_set,
        [train_data_length, validation_data_length],
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    # print(
    #     f"train_data_length:{train_data_length},validation_data_length:{validation_data_length}"
    # )
    # numpy.array(train_data_set)
    # print(f"train_data_set:{numpy.array(train_data_set).shape}")
    return numpy.array(train_data_set), numpy.array(validation_data_set)


def FeatureSelection(train_data, validation_data):
    print(f"train_data:{train_data[:, :-1]}")
    return (
        train_data[:, 1:-1],
        train_data[:, -1],
        validation_data[:, 1:-1],
        validation_data[:, -1],
    )


def trainer(dimension, train_loader, validation_loader):
    # Move model and data(todo, why?) to cuda as first priority, otherwise cpu
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = COVIDModel(dimension).to(device)
    # define loss function
    loss_func = torch.nn.MSELoss(reduction="mean")

    # SGD, todo: 第一个 参数是指什么？
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=0.95,
        weight_decay=0.001,
    )

    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=config["learning_rate"], weight_decay=0.08
    # )

    n_epoch = config["n_epochs"]
    loss_record = {"train": [], "validation": []}
    min_loss = math.inf
    bad_loss_cnt = 0
    # todo：为何这样写直接可以访问到data & lable? 以下一些函数怎么确定输入类型？
    for epoch in range(n_epoch):
        # train
        model.train()
        each_loss_record = []
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            predicted_res = model(x)
            # 输入类型？
            loss_res = loss_func(predicted_res, y)
            # 没有补全提醒？
            loss_res.backward()
            optimizer.step()
            # print(f"train-loss:{loss_res.detach().cpu().item()}")
            each_loss_record.append(loss_res.detach().cpu().item())
            # loss_record["train"].append(loss_res.detach().cpu().item())
        loss_record["train"].append(sum(each_loss_record) / len(each_loss_record))
        # eval
        model.eval()
        each_loss_record = []
        for x, y in validation_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                predicted_res = model(x)
                loss_res = loss_func(predicted_res, y)
                each_loss_record.append(loss_res.detach().cpu().item())
                # loss_record["validation"].append(loss_res.detach().cpu().item())
        validation_mean_loss = sum(each_loss_record) / len(each_loss_record)
        loss_record["validation"].append(validation_mean_loss)
        print(f"epoch:{epoch},validation_mean_loss:{validation_mean_loss}")
        # loss_record["validation"].append(validation_mean_loss)
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

    # InputModelForinf_train_data(model,"HW1/use_finished_model_direct_run_train.csv")

    return loss_record


def InputModelForinf_train_data(model, save_path):
    origin_test_data_set = pandas.read_csv(config["covid_train_path"]).values
    origin_test_data_set = numpy.array(origin_test_data_set)
    origin_test_data_set = origin_test_data_set[:, :-1]
    print(f"number of features: {origin_test_data_set.shape[1]}")
    test_dataset = COVID19DataSet(origin_test_data_set, None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.eval()
    preds = []
    for x in test_loader:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    print(f"preds:{len(preds)}")
    pred_res = torch.cat(preds, dim=0).numpy()
    print(len(pred_res))
    save_pred(pred_res, save_path)


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
    plt.ylim(0.0, 100)
    plt.xlabel("Training steps")
    plt.ylabel("MSE loss")
    plt.title("Learning curve of {}".format(title))
    plt.legend()
    plt.show()


def try_train():
    print("train")
    # Read Data and feature selection
    train_data, validation_data = SplitTrainAndValidationData()
    print(f"train_data:{train_data.shape},validation_data:{validation_data.shape}")
    train_data, train_res, validation_data, validation_res = FeatureSelection(
        train_data, validation_data
    )
    print(f"number of features: {train_data.shape[1]},{validation_data.shape[1]}")

    # Create COVID19DataSet
    train_dataset = COVID19DataSet(train_data, train_res)
    validation_dataset = COVID19DataSet(validation_data, validation_res)
    print(
        f"len of train_dataset:{len(train_dataset)},len of validation_dataset:{len(validation_dataset)}"
    )

    # Use DataLodaer help load data to main memory with batching
    # epoch: input all data to dl netowrk and finish forward and backward calculation
    # todo: 如何验证数据是正确的内容？
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )

    loss_record = trainer(train_data.shape[1], train_loader, validation_loader)
    plot_learning_curve(loss_record, title="deep model")


def doInterface(data_loader, device, model, save_path):
    model.eval()
    preds = []
    for x in data_loader:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    print(f"preds:{len(preds)}")
    pred_res = torch.cat(preds, dim=0).numpy()
    print(len(pred_res))
    save_pred(pred_res, save_path)


def GetTestDataLoader():
    # hack code below
    # origin_test_data_set = pandas.read_csv(config["covid_train_path"]).values

    origin_test_data_set = pandas.read_csv(config["covid_test_path"]).values
    origin_test_data_set = numpy.array(origin_test_data_set)
    origin_test_data_set = origin_test_data_set[:, 1:]
    print(f"number of features: {origin_test_data_set.shape[1]}")
    test_dataset = COVID19DataSet(origin_test_data_set, None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
    )
    return test_loader, origin_test_data_set.shape[1]


def inference():
    print("inference")
    # load test data
    test_loader, input_dimension = GetTestDataLoader()
    # load model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = COVIDModel(input_dimension).to(device)
    ckpt = torch.load(config["model_save_path"], map_location="cpu")
    model.load_state_dict(ckpt)
    # InputModelForinf_train_data(model,"HW1/use_loaded_model_run_train1.csv")
    # InputModelForinf_train_data(model,"HW1/use_loaded_model_run_train2.csv")
    doInterface(test_loader, device, model, "HW1/test_data_res.csv")
    # hack below


def save_pred(preds, file):
    """Save predictions to specified file"""
    print("Saving results to {}".format(file))
    with open(file, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "tested_positive"])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or perform inference")
    # parser.add_argument("--train", action="store_true", help="Perform model training")
    args = parser.parse_args()

    confirmation = input(
        "Are you sure you want to perform model training? This will overwrite existing model. (train/inf): "
    )
    if confirmation.lower() == "train":
        try_train()
    elif confirmation.lower() == "inf":
        inference()
    else:
        print("Error op.")
