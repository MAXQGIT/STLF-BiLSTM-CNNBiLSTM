import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def split_dataset(data):
    # split into train validation and test sets
    train, val, test = data[12:13932], data[13932:18300], data[18300:22716]
    # restructure into samples of daily data shape is [samples, hours, feature]
    train = np.array(np.split(train, len(train) / 24))
    val = np.array(np.split(val, len(val) / 24))
    test = np.array(np.split(test, len(test) / 24))
    return train, val, test


def convert_train_val(train, n_input, n_out=24):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step of 1 hour
        in_start += 1
    return np.array(X), np.array(y)


dataset = pd.read_csv('data/data_spatial_TotalKW.csv', header=0,
                      parse_dates=['datetime'], index_col=['datetime'])

trans = MinMaxScaler()
dataset = trans.fit_transform(dataset)
tran_scale = trans.scale_

# split into train and test
train, val, test = split_dataset(dataset)
N_actual = test[1:184] / tran_scale
N_predicted = test[0:183] / tran_scale


class BiLSTM_model(nn.Module):
    def __init__(self):
        super(BiLSTM_model, self).__init__()
        self.BiLSTM1 = nn.LSTM(1, 200, batch_first=True)
        self.relu = nn.ReLU()
        self.BiLSTM2 = nn.LSTM(200, 100, batch_first=True)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        x, _ = self.BiLSTM1(x)
        x = self.relu(x)
        x, _ = self.BiLSTM2(x)
        x = self.relu(x)
        x = self.linear1(x)
        # x = self.relu(x)
        x = self.linear2(x)
        return x


def mape(y_pred, y_real):
    y_pred = y_pred.detach().cpu().numpy()
    y_real = y_real.detach().cpu().numpy()
    absolute_errors = np.abs(y_real - y_pred)
    rmase_value = np.mean(absolute_errors)
    return rmase_value


def fit(train_data, val_data, epochs):
    model = BiLSTM_model().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.02)
    loss = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        train_sum_loss, i, train_mape = 0, 0, 0
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            pre = model(x.float())
            train_loss = loss(pre, y.float())
            train_sum_loss += train_loss
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            train_m = mape(pre, y)
            train_mape += train_m
            i += 1
        model.eval()
        eval_sum_loss, j, val_mape = 0, 0, 0
        for x, y in val_data:
            x, y = x.to(device), y.to(device)
            pre = model(x.float())
            val_loss = loss(pre, y.float())
            eval_sum_loss += val_loss
            val_m = mape(pre, y)
            val_mape += val_m
            j += 1
        print('epoch:{},train_loss:{:.5},train_mae:{:.5f},val_loss:{:.5f},val_mae:{:.5f}'.format
              (epoch + 1, train_sum_loss / i,train_mape/i, eval_sum_loss / j,val_mape/j))


def bulid_model_BiLSTM(train, val, n_input):
    train_x, train_y = convert_train_val(train, n_input)
    val_x, val_y = convert_train_val(val, n_input)
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    val_y = val_y.reshape((val_y.shape[0], val.shape[1], 1))
    train_x, train_y = torch.tensor(train_x), torch.tensor(train_y)
    val_x, val_y = torch.tensor(val_x), torch.tensor(val_y)
    train_data = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_data, batch_size=64)
    val_data = TensorDataset(val_x, val_y)
    val_dataloader = DataLoader(val_data, batch_size=64)
    fit(train_dataloader, val_dataloader, 30)


n_input = 24
bulid_model_BiLSTM(train, val, n_input)
