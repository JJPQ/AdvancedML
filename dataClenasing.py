import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn import functional as F

train_data = pd.read_csv('C:/Users/WJJ/Desktop/NJU/研一下/高级机器学习/Assignment_2/Churn/train.csv')
test_data = pd.read_csv('C:/Users/WJJ/Desktop/NJU/研一下/高级机器学习/Assignment_2/Churn/test.csv')
# 删除无用列，填充nan
train_data = train_data.iloc[:, 3:]
test_data = test_data.iloc[:, 3:]
train_data.fillna(method='ffill', axis=0, inplace=True)
test_data.fillna(method='ffill', axis=0, inplace=True)
data = pd.concat((train_data, test_data))
# 给Gender, geography, Surname编号，方便后续Embedding
labels, uniques = pd.factorize(data['Gender'])
data['Gender'] = labels
labels, uniques = pd.factorize(data['Geography'])
data['Geography'] = labels
train_data = data[:9000]
test_data = data[9000:].drop('Exited', axis=1)
device = 'cuda'
# 构造数据
features = torch.from_numpy(train_data.values[:, :-1]).type(torch.float).to(device)
label = torch.from_numpy(train_data.values[:, -1]).reshape(-1).type(torch.float).to(device)
train_dataset = TensorDataset(features[:7000], label[:7000])
valid_dataset = TensorDataset(features[7000:], label[7000:])
train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_iter = DataLoader(valid_dataset, batch_size=64, shuffle=True)
