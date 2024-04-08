import pandas

import torch
import torch.utils.data
from torch.utils.data import random_split

config = {
    "validation_ratio": 0.2,
    "covid_train_path": "HW1/covid_train.csv",
    "seed": 1998,
}


def GetRealTrainAndValidationData():
    original_train_data = pandas.read_csv(config["covid_train_path"])
    tot_data_length = len(original_train_data)
    train_data_length = int(config["validation_ratio"] * tot_data_length)
    validation_data_length = tot_data_length - train_data_length
    train_data, validation_data = random_split(
        original_train_data,
        [train_data_length, validation_data_length],
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    return train_data, validation_data


train_data, validation_data = GetRealTrainAndValidationData()
print(train_data)
# train_data = pd.read_csv('test.csv')
# print(train_data.values)
