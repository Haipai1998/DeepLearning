import pandas
import numpy
import torch

config = {
    "validation_ratio": 0.2,
    "covid_train_path": "HW1/covid_train.csv",
    "seed": 1998,
}


def GetRealTrainAndValidationData():
    origin_data_set = pandas.read_csv(config["covid_train_path"]).values
    print(numpy.array(origin_data_set))
    # print(type(pandas.read_csv(config["covid_train_path"])).values)
    tot_data_length = len(origin_data_set)
    train_data_length = int((1 - config["validation_ratio"]) * tot_data_length)
    validation_data_length = tot_data_length - train_data_length
    train_data_set, validation_data_set = torch.utils.data.random_split(
        origin_data_set,
        [train_data_length, validation_data_length],
        generator=torch.Generator().manual_seed(0),
    )
    # print(
    #     f"train_data_length:{train_data_length},validation_data_length:{validation_data_length}"
    # )
    # numpy.array(train_data_set)
    # print(f"train_data_set:{numpy.array(train_data_set).shape}")
    return numpy.array(train_data_set), numpy.array(validation_data_set)


train_data, validation_data = GetRealTrainAndValidationData()
print(f"train_data:{train_data.shape},validation_data:{validation_data.shape}")
print(train_data)
