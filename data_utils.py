import pandas
import numpy as np
import torch
from torch.utils import data


class MnistKaggleTrainDataset(data.Dataset):

    def __init__(self, csv_path='data/train.csv', transform=None):
        self.data_frame = pandas.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, index):
        # label = torch.from_numpy(self.data_frame.iloc[index].values[[0]])
        # image = torch.from_numpy(self.data_frame.iloc[index].values[1:])
        label = torch.from_numpy(np.array(self.data_frame.values[index][0]))
        image = torch.from_numpy(self.data_frame.values[index][1:])
        image = image.view(1, 28, 28)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MnistKaggleTestDataset(data.Dataset):

    def __init__(self, csv_path='data/test.csv', transform=None):
        self.data_frame = pandas.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, index):
        # image = torch.from_numpy(self.data_frame.iloc[index].values)
        image = torch.from_numpy(self.data_frame.values[index])
        image = image.view(1, 28, 28)

        if self.transform is not None:
            image = self.transform(image)

        return image
