import torch
from torchvision import transforms
import copy
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class DigitDataset(Dataset):
    def __init__(self, opt, data, test=False, validate=False):
        self.data = data
        self.data_size = opt.data_size
        self.data_len = len(data)
        self.test = test
        self.validate = validate
        self.eye = torch.eye(10)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(20),
            transforms.RandomGrayscale(),
            transforms.RandomRotation(20),
            transforms.RandomPerspective(),
            transforms.ToTensor()
        ])

    def __len__(self):
        if not (self.test or self.validate):
            return self.data_size
        else:
            return self.data_len

    def one_hot(self, label):
        if label > 9 or label < 0:
            raise ValueError(f'({label})label out of range.')
        return self.eye[label]
    
    def transforms(self, image):
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        if not (self.test or self.validate):
            index = np.random.randint(0, self.data_len)

        if self.test:
            label = 0                   # to be predict
            image = self.data[index, :]
        else:
            label = self.data[index, 0]
            image = self.data[index, 1:]
        
        image = torch.tensor(image).type(torch.float32).view(28, 28) / 255.0
        if not (self.test or self.validate):
            image = self.transforms(image)
        else:
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

    train_dataset = DigitDataset(opt, train_data)
    validate_dataset = DigitDataset(opt, validate_data, validate=True)
    test_dataset = DigitDataset(opt, test_data, test=True)

    return train_dataset, validate_dataset, test_dataset


def data_reader(path):
    df = pd.read_csv(path)
    return df.to_numpy()


if __name__ == "__main__":
    data_reader('/data/cwj/data/digit/train.csv')
