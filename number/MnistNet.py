import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # 计算损失（负对数似然损失）
        self.loss = nn.NLLLoss()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.log_softmax(self.fc4(x))
        return x

    def cal_loss(self, x, y):
        return self.loss(x, y)
