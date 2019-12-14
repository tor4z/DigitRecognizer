import torch
import copy
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class DigitDataset(Dataset):
    def __init__(self, data, test=False):
        self.data = data
        self.len = len(data)
        self.test = test
        self.eye = torch.eye(10)

    def __len__(self):
        return self.len

    def one_hot(self, label):
        if label > 9 or label < 0:
            raise ValueError(f'({label})label out of range.')
        return self.eye[label]

    def __getitem__(self, index):
        if self.test:
            label = 0                   # to be predict
            image = self.data[index, :]
        else:
            label = self.data[index, 0]
            image = self.data[index, 1:]
        
        image = torch.tensor(image).type(torch.float32).view(28, 28) / 255.0
        image = image.unsqueeze(0)
        label = self.one_hot(label)

        return index, image, label


def get_train_val_test_dataset(opt):
    all_train_data = data_reader(opt.train_path)
    test_data = data_reader(opt.test_path)
    
    train_len = len(all_train_data)
    test_len = len(test_data)

    ids = [i for i in range(train_len)]
    np.random.shuffle(ids)

    train_data_len = int(train_len * opt.train_rate)
    validate_data_len = train_len - train_data_len

    validate_ids = copy.copy(ids[: validate_data_len])
    del ids[:validate_data_len]
    train_ids = ids

    train_data = all_train_data[train_ids, :]
    validate_data = all_train_data[validate_ids, :]

    train_dataset = DigitDataset(train_data)
    validate_dataset = DigitDataset(validate_data)
    test_dataset = DigitDataset(test_data, test=True)

    return train_dataset, validate_dataset, test_dataset


def data_reader(path):
    df = pd.read_csv(path)
    return df.to_numpy()


if __name__ == "__main__":
    data_reader('/data/cwj/data/digit/train.csv')
