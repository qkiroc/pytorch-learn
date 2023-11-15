import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )
        # 计算损失（负对数似然损失）
        self.loss = nn.NLLLoss()

    def forward(self, x):
        return self.net(x)

    def cal_loss(self, x, y):
        return self.loss(x, y)
