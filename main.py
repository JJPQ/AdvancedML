import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn import functional as F
from imblearn.over_sampling import SMOTE

train_data = pd.read_csv('Churn/train.csv')
test_data = pd.read_csv('Churn/test.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 删除无用列，对10000个样本的数值数据归一化，填充nan
train_data = train_data.iloc[:, 3:]
test_data = test_data.iloc[:, 3:]
all_data = pd.concat((train_data, test_data))
numeric_features = all_data.dtypes[all_data.dtypes != 'object'].index.drop('Exited')
all_data[numeric_features] = all_data[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_data.iloc[:, :-1] = all_data.iloc[:, :-1].fillna(method='ffill', axis=0)
all_data = pd.get_dummies(all_data, dummy_na=False)
all_data = all_data.astype(float)

# 对9000个样本上采样
train_data_original = all_data[:9000].drop('Exited', axis=1)
train_labels_original = all_data.Exited[:9000]
smote = SMOTE(random_state=42)
train_data_resampled, train_labels_resampled = smote.fit_resample(train_data_original, train_labels_original)

# 划分数据集为8:2
# 0.58,0.78 : 4,1
train_data = torch.tensor(train_data_resampled.values, dtype=torch.float32)
train_labels = torch.tensor(train_labels_resampled, dtype=torch.float32).reshape(-1, 1)
all_data = all_data.drop('Exited', axis=1)
test_data = torch.tensor(all_data[9000:].values, dtype=torch.float32)
n_train = 14284
valid_data_tensor = train_data[int(np.ceil(n_train * 0.58)):int(np.ceil(n_train * 0.78))]
train_data_tensor = torch.concat((train_data[:int(np.ceil(n_train * 0.58))], train_data[int(np.ceil(n_train * 0.78)):]))
valid_label_tensor = train_labels[int(np.ceil(n_train * 0.58)):int(np.ceil(n_train * 0.78))]
train_label_tensor = torch.concat((train_labels[:int(np.ceil(n_train * 0.58))], train_labels[int(np.ceil(n_train * 0.78)):]))
train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
valid_dataset = TensorDataset(valid_data_tensor, valid_label_tensor)
train_iter = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_iter = DataLoader(valid_dataset, batch_size=128, shuffle=True)


# MLP
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 16)
        self.linear5 = nn.Linear(16, num_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        y1 = self.relu(self.linear1(x))
        y2 = self.dropout1(self.relu(self.linear2(y1)))
        y3 = self.dropout2(self.relu(self.linear3(y2)))
        y4 = self.relu(self.linear4(y3))
        y5 = self.linear5(y4)
        return F.sigmoid(y5)


# 设置超参数
num_inputs = train_data.shape[1]
num_outputs = 1
net = MLP(num_inputs, num_outputs).to(device)
loss = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.01)
epochs = 200

# 训练网络，保存在验证集上macro f1分数最高的模型
for name in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    max_test_f1 = 0
    epoch_index = 0
    for epoch in range(epochs):
        net.train()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            tp_t = 0
            fn_t = 0
            fp_t = 0
            tn_t = 0
            for x, y in valid_iter:
                x = x.to(device)
                y = y.to(device)
                y_hat = net(x)
                l = loss(y_hat, y)
                y_pre = torch.where(y_hat > 0.5, 1, 0)
                tp_t += y_pre[y.bool()].sum()
                fn_t += (y_pre[y.bool()].shape[0] - y_pre[y.bool()].sum())
                fp_t += y_pre[~y.bool()].sum()
                tn_t += (y_pre[~y.bool()].shape[0] - y_pre[~y.bool()].sum())
            f1_positive = 2 * tp_t / (2 * tp_t + fn_t + fp_t)
            f1_negative = 2 * tn_t / (2 * tn_t + fn_t + fp_t)
            macro_f1 = (f1_positive + f1_negative) / 2
            acc = (tp_t + tn_t) / (tp_t + fn_t + fp_t + tn_t)
            if macro_f1 > max_test_f1:
                max_test_f1 = macro_f1
                epoch_index = epoch
                if macro_f1 > 0.7:
                    torch.save(
                        net.state_dict(),
                        f'{name}_{epoch}_{acc}_{macro_f1}.pt'
                    )
                    print(
                        f'valid epoch:{epoch},tp:{tp_t}, fp:{fp_t}, fn:{fn_t}, tn:{tn_t}, num:{tp_t + fn_t + fp_t + tn_t}'
                    )
                    print(
                        f'f1_positive:{f1_positive},f1_negative:{f1_negative},macro_f1:{macro_f1},acc:{acc}\n'
                    )
    print(epoch_index, max_test_f1)