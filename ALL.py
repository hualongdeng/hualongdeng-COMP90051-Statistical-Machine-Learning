import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.nn as nn


def encode(item):
    index = [0, 3, 4, 5]
    year = np.zeros((3))
    year[item[0] - 1] = 1
    band = np.zeros((3))
    band[item[3] - 1] = 1
    group = np.zeros((11))
    group[item[4] - 1] = 1
    denomination = np.zeros((3))
    denomination[item[5] - 1] = 1
    new_item = np.delete(item, index)
    newdata = np.concatenate((year,band,group,denomination,new_item),axis=0)
    return newdata


def encode_data(data):
    new = []
    for item in data:
        new.append(encode(item))
    return new


def read_data(file):
    raw_data = pd.read_csv(file)
    data_len = len(raw_data)
    data = []
    for i in range(0, data_len):
        row = np.array(raw_data.iloc[i])
        data.append(row)
    data = np.array(data)
    return data


def split_data(data):
    label = data[:, 22]
    data = data[:, :22]
    pre_x_train, x_test, pre_y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)
    x_train, x_dev, y_train, y_dev = train_test_split(pre_x_train, pre_y_train, test_size=0.25, random_state=0)
    return x_train, x_dev, x_test, y_train, y_dev, y_test


def get_lr_mse(train_x, dev_x, test_x, train_y, dev_y, test_y):
    _, dev_x_100, _, dev_y_100 = train_test_split(dev_x, dev_y, test_size=100, random_state=0)
    logreg = Lasso()
    logreg.fit(train_x, train_y)
    prediction = logreg.predict(test_x)
    result = mean_squared_error(test_y, prediction)
    print(result)
    return result


class MyClassifier(nn.Module):
    def __init__(self, hidden_layer):
        super(MyClassifier, self).__init__()
        self.fc1 = nn.Linear(22, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def get_mse_net(train_x, dev_x, test_x, train_y, dev_y, test_y):
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    dev_x = torch.from_numpy(dev_x).type(torch.FloatTensor)
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
    dev_y = torch.from_numpy(dev_y).type(torch.FloatTensor)
    test_y = torch.from_numpy(test_y).type(torch.FloatTensor)
    _, dev_x_100, _, dev_y_100 = train_test_split(dev_x, dev_y, test_size=100, random_state=0)
    model = MyClassifier(20)
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(200):
        y_pred = model(train_x)
        loss = loss_fn(y_pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    prediction = model(test_x)
    result = mean_squared_error(test_y, prediction.detach().numpy())
    print(result)
    return result


# read data.
femaleData = np.array(encode_data(read_data("FEMALE.csv")))
maleData = np.array(encode_data(read_data("MALE.csv")))
mixedData = np.array(encode_data(read_data("MIXED.csv")))

# split data
male_train_x, male_dev_x, male_test_x, male_train_y, male_dev_y, male_test_y = split_data(maleData)
female_train_x, female_dev_x, female_test_x, female_train_y, female_dev_y, female_test_y = split_data(femaleData)
mix_train_x, mix_dev_x, mix_test_x, mix_train_y, mix_dev_y, mix_test_y = split_data(mixedData)

_, t_male_train_x, _, t_male_train_y = train_test_split(male_train_x, male_train_y, test_size=1000, random_state=0)
_, t_female_train_x, _, t_female_train_y = train_test_split(female_train_x, female_train_y, test_size=1000, random_state=0)
_, t_mix_train_x, _, t_mix_train_y = train_test_split(mix_train_x, mix_train_y, test_size=1000, random_state=0)

mix_target_x = np.append(male_train_x, female_train_x, axis=0)
mix_target_x = np.append(mix_target_x, t_mix_train_x, axis=0)
mix_target_y = np.append(male_train_y, female_train_y, axis=0)
mix_target_y = np.append(mix_target_y, t_mix_train_y, axis=0)

male_target_x = np.append(mix_train_x, female_train_x, axis=0)
male_target_x = np.append(male_target_x, t_male_train_x, axis=0)
male_target_y = np.append(mix_train_y, female_train_y, axis=0)
male_target_y = np.append(male_target_y, t_male_train_y, axis=0)

female_target_x = np.append(mix_train_x, male_train_x, axis=0)
female_target_x = np.append(female_target_x, t_female_train_x, axis=0)
female_target_y = np.append(mix_train_y, male_train_y, axis=0)
female_target_y = np.append(female_target_y, t_female_train_y, axis=0)

# calculate LogisticRegression result
reg_result = 0
reg_result = get_lr_mse(mix_target_x, mix_dev_x, mix_test_x, mix_target_y, mix_dev_y, mix_test_y)
reg_result = reg_result + get_lr_mse(male_target_x, male_dev_x, male_test_x, male_target_y, male_dev_y, male_test_y)
reg_result = reg_result + get_lr_mse(female_target_x, female_dev_x, female_test_x, female_target_y, female_dev_y, female_test_y)
print(reg_result/3)

# calculate Neural Network result
# net_result = 0
# net_result = get_mse_net(mix_target_x, mix_dev_x, mix_test_x, mix_target_y, mix_dev_y, mix_test_y)
# net_result = net_result + get_mse_net(male_target_x, male_dev_x, male_test_x, male_target_y, male_dev_y, male_test_y)
# net_result = net_result + get_mse_net(female_target_x, female_dev_x, female_test_x, female_target_y, female_dev_y, female_test_y)
# print(net_result/3)