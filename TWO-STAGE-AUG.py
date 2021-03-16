"""
An example showing the usage of the TwoStageTrAdaBoostR2 algorithm.

"""
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2  # import the two-stage algorithm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor

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
##=============================================================================

#                                Example 1

##=============================================================================

# 1. define the data generating function
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

# read data.
femaleData = np.array(encode_data(read_data("FEMALE.csv")))
maleData = np.array(encode_data(read_data("MALE.csv")))
mixedData = np.array(encode_data(read_data("MIXED.csv")))

# split data
male_train_x, male_dev_x, male_test_x, male_train_y, male_dev_y, male_test_y = split_data(maleData)
female_train_x, female_dev_x, female_test_x, female_train_y, female_dev_y, female_test_y = split_data(femaleData)
mix_train_x, mix_dev_x, mix_test_x, mix_train_y, mix_dev_y, mix_test_y = split_data(mixedData)

_, t_male_train_x, _, t_male_train_y = train_test_split(male_train_x, male_train_y, test_size=100, random_state=0)
_, t_female_train_x, _, t_female_train_y = train_test_split(female_train_x, female_train_y, test_size=100, random_state=0)
_, t_mix_train_x, _, t_mix_train_y = train_test_split(mix_train_x, mix_train_y, test_size=100, random_state=0)

def transfer2source (sourceData):
    new_source = []
    for item in sourceData:
        new_data = np.append(item, item)
        new_source.append(new_data)
    new_source = np.array(new_source)
    return new_source

def transfer2target (targetData):
    new_target = []
    zero_data = [0] * 22
    for item in targetData:
        new_data = np.append(item, zero_data)
        new_target.append(new_data)
    new_target = np.array(new_target)
    return new_target

t_male_train_x = np.tile(t_male_train_x, (35, 1))
t_male_train_y = np.reshape(t_male_train_y, (100,1))
t_male_train_y = np.tile(t_male_train_y, (35, 1))
t_male_train_y = np.reshape(t_male_train_y, (3500,))

t_female_train_x = np.tile(t_female_train_x, (33, 1))
t_female_train_y = np.reshape(t_female_train_y, (100,1))
t_female_train_y = np.tile(t_female_train_y, (33, 1))
t_female_train_y = np.reshape(t_female_train_y, (3300,))

t_mix_train_x = np.tile(t_mix_train_x, (24, 1))
t_mix_train_y = np.reshape(t_mix_train_y, (100,1))
t_mix_train_y = np.tile(t_mix_train_y, (24, 1))
t_mix_train_y = np.reshape(t_mix_train_y, (2400,))

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

n_estimators = 100
steps = 10
fold = 5
random_state = np.random.RandomState(1)

def get_lr_mse(train_x, dev_x, test_x, train_y, dev_y, test_y):
    _, dev_x_100, _, dev_y_100 = train_test_split(dev_x, dev_y, test_size=100, random_state=0)
    logreg = TwoStageTrAdaBoostR2(Lasso(),
                              n_estimators=n_estimators, sample_size=[train_x.shape[0] - 100, 100],
                              steps=steps, fold=fold,
                              random_state=random_state)
    logreg.fit(train_x, train_y)
    prediction = logreg.predict(test_x)
    result = mean_squared_error(test_y, prediction)
    print(result)
    return result

# calculate LogisticRegression result
reg_result = 0
reg_result = get_lr_mse(transfer2source(mix_target_x), transfer2target(mix_dev_x), transfer2target(mix_test_x), mix_target_y, mix_dev_y, mix_test_y)
reg_result = reg_result + get_lr_mse(transfer2source(male_target_x), transfer2target(male_dev_x), transfer2target(male_test_x), male_target_y, male_dev_y, male_test_y)
reg_result = reg_result + get_lr_mse(transfer2source(female_target_x), transfer2target(female_dev_x), transfer2target(female_test_x), female_target_y, female_dev_y, female_test_y)
print(reg_result/3)

