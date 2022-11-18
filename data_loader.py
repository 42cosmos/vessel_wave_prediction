import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset

from sklearn.preprocessing import MinMaxScaler, RobustScaler


class Loader:
    def __init__(self, config):
        self.config = config
        self.window_size = config.window_size
        self.raw_datasets = pd.read_csv(os.path.join(self.config.data_dir, 'vessel_data.csv'))
        scaler = MinMaxScaler()
        quartile_1 = self.raw_datasets['WIND_WAVE_DIRECTION'].quantile(0.25)
        quartile_3 = self.raw_datasets['WIND_WAVE_DIRECTION'].quantile(0.75)

        IQR = quartile_3 - quartile_1

        search_raw_datasets = self.raw_datasets[
            (self.raw_datasets['WIND_WAVE_DIRECTION'] < (quartile_1 - 1.5 * IQR)) | (
                    self.raw_datasets['WIND_WAVE_DIRECTION'] > (quartile_3 + 1.5 * IQR))]

        self.raw_datasets = self.raw_datasets.drop(search_raw_datasets.index, axis=0)

        self.raw_datasets = pd.DataFrame(scaler.fit_transform(self.raw_datasets), columns=self.raw_datasets.columns)

    def get_dataset(self, evaluate=False):
        # load data
        length_dataset = self.raw_datasets.shape[0]
        train_ratio = int(length_dataset * self.config.train_ratio)

        # split data
        datasets = self.raw_datasets.iloc[train_ratio:] if evaluate else self.raw_datasets.iloc[:train_ratio]
        column_names = datasets.columns

        train_columns = [col for col in datasets.columns if col != "WIND_WAVE_HEIGHT"]

        data = datasets[train_columns].values
        label = datasets["WIND_WAVE_HEIGHT"].values.reshape(-1, 1)

        data, label = self._make_window_size(data, label)

        # convert to tensor
        tensor_data = Variable(torch.from_numpy(data).float())

        tensor_label = Variable(torch.from_numpy(label).float())

        features = TensorDataset(tensor_data, tensor_label)

        return features

    def _make_window_size(self, data, label):
        data_x, data_y = [], []
        for i in range(len(data) - self.window_size):
            sample_x = data[i:(i + self.window_size), :]
            sample_y = label[i + self.window_size, :]
            data_x.append(sample_x)
            data_y.append(sample_y)
        data_x = np.array(data_x)  # (batch_size, time_steps, input_dimension)
        data_y = np.array(data_y)  # (N, 1)
        return data_x, data_y
