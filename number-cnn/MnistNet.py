import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 计算输入大小
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # 计算损失（负对数似然损失）
        self.loss = nn.NLLLoss()

    def forward(self, x):
        # 卷积层和池化层
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层
        x = self.relu3(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x

    def cal_loss(self, x, y):
        return self.loss(x, y)
