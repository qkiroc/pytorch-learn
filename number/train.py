import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from MnistNet import Net
import tools.drawing as drawing

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    # 使用 torch.no_grad() 来关闭梯度计算，因为在评估阶段我们不需要计算梯度
    with torch.no_grad():
        for (x, y) in test_data:
            # 将输入数据 x 通过网络进行前向传播，得到输出结果
            outputs = net.forward(x.view(-1, 28*28))
            # 遍历每个样本的输出
            for i, output in enumerate(outputs):
                # 检查模型预测的类别是否与真实类别 y[i] 一致
                if torch.argmax(output) == y[i]:
                    n_correct += 1 # 如果一致，则正确分类数加一
                n_total += 1 # 总样本数加一
    # 返回准确率（正确分类数除以总样本数）
    return n_correct / n_total


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))
    # 定义优化器（Adam优化器）和学习率
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    lossInfo =[]
    for epoch in range(2):
        net.train()
        for (x, y) in train_data:
            # 梯度清零
            net.zero_grad()
            # 前向传播
            output = net.forward(x.view(-1, 28*28))
            # 计算损失（负对数似然损失）
            loss = net.cal_loss(output, y)
            # 反向传播
            loss.backward()
             # 更新参数
            optimizer.step()
            print("epoch", epoch, "loss:", loss.item())
            lossInfo.append(loss.item())
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))
    torch.save(net.state_dict(), "mnist_net.pth")
    drawing.plotLearningCurve(lossInfo, 'MNIST')
    # for (n, (x, _)) in enumerate(test_data):
    #     if n > 3:
    #         break
    #      # 获取模型预测结果
    #     predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
    #     plt.figure(n)
    #     plt.imshow(x[0].view(28, 28))
    #     plt.title("prediction: " + str(int(predict)))
    # plt.show()


if __name__ == "__main__":
    main()