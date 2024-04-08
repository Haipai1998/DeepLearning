import torch
import pandas
import torch.utils.data


# Pytorch
from torch.utils.data import random_split

config = {
    'validation_ratio':0.2,
    'covid_train_path':'HW1/covid_train.csv'
}

def GetRealTrainAndValidationData():
    original_train_data = pandas.read_csv(config['covid_train_path'])
    print(f'original_train_data len: {len(original_train_data)}')
    # random_split()

GetRealTrainAndValidationData()
# train_data = pd.read_csv('test.csv')
# print(train_data.values)
