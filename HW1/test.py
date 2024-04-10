import pandas
import numpy
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

config = {
    "validation_ratio": 0.2,
    "covid_train_path": "HW1/covid_train.csv",
    "seed": 1998,
    "batch_size": 256,
}


class COVID19DataSet(torch.utils.data.Dataset):
    def __init__(self, feature, res):
        if res is None:
            self.res = None
        else:
            self.res = res
        self.feature = feature

    def __getitem__(self, index):
        if self.res is None:
            return self.feature[index]
        else:
            return self.feature[index], self.res[index]

    def __len__(self):
        return len(self.feature)


class COVIDModel(torch.nn.Module):
    def __init__(self, input_dimension) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dimension, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dimension, 1),
        )
    def forward(self,)


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
    train_res = train_data[:, -1]
    validation_res = validation_data[:, -1]
    # print(train_res)
    # print(validation_res)
    return train_data[:, :-1], train_res, validation_data[:, :-1], validation_res


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
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=config["batch_size"], shuffle=True, pin_memory=True
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
)
